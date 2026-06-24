"""
Local browser-based gold-labeling app for candidate pairs.

Usage from the notebook (cwd = gold_labeling/):

    from labeler import launch_labeler, load_labels, apply_labels, label_summary, stop_labeler

    subset = pairs[(pairs.SSN_clean_A == pairs.SSN_clean_B) & (pairs.silver_label == False)]
    launch_labeler(subset)          # opens a browser tab; writes data/gold_labels/gold_labels.csv
    ...
    stop_labeler()                  # shut the server down when done

Design
------
* The app reads/writes a **sparse** store ``gold_labels.csv`` -- one row per pair you have
  *touched*, keyed by ``(PATID_A, PATID_B)``. Untouched pairs never appear in the file.
* Each click POSTs to a tiny Flask server (127.0.0.1 only) which updates the in-memory
  store and atomically rewrites the CSV. State survives kernel restarts / browser crashes.
* The store CSV is the source of truth. The notebook can read it back with ``load_labels`` /
  ``apply_labels`` and edit it with plain pandas if needed.

Columns of the store: ``PATID_A, PATID_B, gold_label, ambiguous_pair, reviewed_at``
  * ``gold_label``      -> 'match' | 'no_match' | (absent = undecided)
  * ``ambiguous_pair``  -> True | False  (default False)
  * ``reviewed_at``     -> ISO timestamp of the last click on that pair

PHI: the rendered page contains real patient values in the browser DOM. The server binds to
127.0.0.1 only -- never run this on a public interface, and keep gold_labels.csv out of git.
"""
from __future__ import annotations

import ast
import datetime
import html
import json
import os
import socket
import sys
import threading
import webbrowser

import pandas as pd

# Default comparison fields (base column names, without the _A / _B suffix). The caller can
# override; launch_labeler also auto-drops any field not present in the passed DataFrame.
DEFAULT_FIELDS = [
    'FirstNM_clean', 'LastNM_clean', 'MiddleNM_clean', 'SuffixNM_clean', 'BirthDT_clean',
    'SSN_clean', 'last_4_SSN', 'AddressLine1_clean', 'AddressLine2_clean', 'CityNM_clean',
    'ZipCD_clean_base', 'ZipCD_clean_ext', 'StateCD_clean', 'Email_clean',
    'SexAtBirthDSC_clean', 'Phones_set',
]
STORE_COLS = ['PATID_A', 'PATID_B', 'gold_label', 'ambiguous_pair', 'reviewed_at']

# Default store lives in data/gold_labels/ (anchored to this file, so it is the same
# location no matter what the notebook's working directory happens to be).
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STORE = os.path.normpath(os.path.join(_HERE, '..', 'data', 'gold_labels', 'gold_labels.csv'))

# Track running servers by port so re-launching cleanly replaces the old one. Stashed on the
# `sys` module (never reloaded) so the registry survives `importlib.reload(labeler)` in the
# notebook -- otherwise a reloaded module would orphan the old server still holding the port.
_RUNNING: dict = getattr(sys, '_anymatch_labeler_servers', None)
if _RUNNING is None:
    _RUNNING = {}
    sys._anymatch_labeler_servers = _RUNNING


def _port_free(host: str, port: int) -> bool:
    """True if `port` can be bound right now (nothing actively listening on it)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# --------------------------------------------------------------------------- helpers
def _as_collection(v):
    """Return v as a list of items if it's collection-like, else None.

    Handles list / tuple / set, numpy & pandas arrays (anything with .tolist()), and
    stringified collections like "{'7735551234', '7735555678'}" (parquet -> array, but
    CSV round-trips them to strings). Plain scalars / strings return None.
    """
    if isinstance(v, (list, tuple, set, frozenset)):
        return list(v)
    if isinstance(v, str):
        s = v.strip()
        if len(s) >= 2 and s[0] in '[{(' and s[-1] in ']})':
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set, frozenset)):
                    return list(parsed)
            except (ValueError, SyntaxError):
                pass
        return None
    if isinstance(v, bytes):
        return None
    if hasattr(v, 'tolist'):  # numpy / pandas array
        try:
            out = v.tolist()
            return out if isinstance(out, list) else None
        except Exception:
            return None
    return None


def _is_missing(v) -> bool:
    if v is None:
        return True
    coll = _as_collection(v)
    if coll is not None:  # collection: missing iff it has no non-missing element
        return all(_is_missing(x) for x in coll)
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str):
        return v.strip() == ''
    return False


def _to_bool(v) -> bool:
    """Robust truthiness for values that may round-trip through CSV as the string 'False'."""
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    return str(v).strip().lower() in ('true', '1', 'yes')


def _fmt(v) -> str:
    coll = _as_collection(v)
    if coll is not None:
        return ', '.join(str(x) for x in coll if not _is_missing(x))
    if _is_missing(v):
        return ''
    return str(v)


def _cell_class(a, b) -> str:
    """Same colour rules as the notebook's show_pairs, returned as a CSS class name.

    Collection fields (e.g. Phones_set) are GREEN if A and B share *any* element -- they do
    not all have to agree.
    """
    am, bm = _is_missing(a), _is_missing(b)
    if am and bm:
        return 'both-missing'
    if am or bm:
        return 'one-missing'
    ca, cb = _as_collection(a), _as_collection(b)
    if ca is not None or cb is not None:
        sa = {str(x) for x in ca} if ca is not None else {str(a)}
        sb = {str(x) for x in cb} if cb is not None else {str(b)}
        return 'equal' if sa & sb else 'diff'
    return 'equal' if str(a) == str(b) else 'diff'


# --------------------------------------------------------------------------- store
class LabelStore:
    """In-memory dict of decisions backed by an atomically-rewritten sparse CSV."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.lock = threading.Lock()
        self.data: dict[tuple[str, str], dict] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        df = pd.read_csv(self.path, dtype={'PATID_A': 'string', 'PATID_B': 'string'})
        for _, r in df.iterrows():
            gl = r.get('gold_label')
            self.data[(str(r['PATID_A']), str(r['PATID_B']))] = {
                'gold_label': gl if isinstance(gl, str) else None,
                'ambiguous_pair': _to_bool(r.get('ambiguous_pair')),
                'reviewed_at': None if pd.isna(r.get('reviewed_at')) else str(r.get('reviewed_at')),
            }

    def get(self, a: str, b: str) -> dict:
        return self.data.get((a, b), {'gold_label': None, 'ambiguous_pair': False, 'reviewed_at': None})

    def set(self, a: str, b: str, gold_label=..., ambiguous_pair=...) -> dict:
        with self.lock:
            cur = dict(self.data.get((a, b), {'gold_label': None, 'ambiguous_pair': False}))
            if gold_label is not ...:
                cur['gold_label'] = gold_label
            if ambiguous_pair is not ...:
                cur['ambiguous_pair'] = bool(ambiguous_pair)
            cur['reviewed_at'] = datetime.datetime.now().isoformat(timespec='seconds')
            self.data[(a, b)] = cur
            self._flush()
            return cur

    def _flush(self):
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        rows = [{'PATID_A': a, 'PATID_B': b,
                 'gold_label': v.get('gold_label'),
                 'ambiguous_pair': bool(v.get('ambiguous_pair', False)),
                 'reviewed_at': v.get('reviewed_at')}
                for (a, b), v in self.data.items()]
        df = pd.DataFrame(rows, columns=STORE_COLS)
        tmp = self.path + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.path)  # atomic: never leaves a half-written file


# --------------------------------------------------------- notebook-side convenience
def load_labels(store_path: str = DEFAULT_STORE) -> pd.DataFrame:
    """Read the sparse store as a DataFrame (empty frame with the right columns if absent)."""
    if os.path.exists(store_path):
        return pd.read_csv(store_path, dtype={'PATID_A': 'string', 'PATID_B': 'string'})
    return pd.DataFrame(columns=STORE_COLS)


def apply_labels(df: pd.DataFrame, store_path: str = DEFAULT_STORE) -> pd.DataFrame:
    """Left-merge gold_label + ambiguous_pair from the store onto a pairs DataFrame.

    Idempotent and order-independent -- run any time to refresh `df` from the store.
    """
    store = load_labels(store_path)[['PATID_A', 'PATID_B', 'gold_label', 'ambiguous_pair']]
    out = df.copy()
    for c in ('PATID_A', 'PATID_B'):
        out[c] = out[c].astype('string')
    out = out.merge(store, on=['PATID_A', 'PATID_B'], how='left')
    out['ambiguous_pair'] = out['ambiguous_pair'].map(_to_bool)
    return out


def label_summary(store_path: str = DEFAULT_STORE) -> dict:
    """Counts of match / no_match / ambiguous in the store."""
    s = load_labels(store_path)
    return {
        'touched': len(s),
        'match': int((s['gold_label'] == 'match').sum()) if len(s) else 0,
        'no_match': int((s['gold_label'] == 'no_match').sum()) if len(s) else 0,
        'ambiguous': int(s['ambiguous_pair'].map(_to_bool).sum()) if len(s) else 0,
    }


# --------------------------------------------------------------------------- payload
def build_payload(df: pd.DataFrame, fields: list[str], store: LabelStore):
    """Build the JSON payload (one dict per pair) the front-end renders."""
    display_names = [f.replace('_clean', '') for f in fields]
    pairs = []
    for _, r in df.iterrows():
        a_id, b_id = str(r['PATID_A']), str(r['PATID_B'])
        rec = store.get(a_id, b_id)
        flds = []
        for f, name in zip(fields, display_names):
            a, b = r.get(f'{f}_A'), r.get(f'{f}_B')
            flds.append({'name': name, 'a': _fmt(a), 'b': _fmt(b), 'cls': _cell_class(a, b)})
        sv = r.get('silver_label')
        pairs.append({
            'patid_a': a_id, 'patid_b': b_id,
            'silver_label': (None if _is_missing(sv) else bool(sv)),
            'fields': flds,
            'gold_label': rec['gold_label'],
            'ambiguous_pair': bool(rec['ambiguous_pair']),
        })
    return pairs, display_names


# --------------------------------------------------------------------------- page
def render_page(pairs, columns, store_path) -> str:
    data_json = json.dumps(pairs).replace('</', '<\\/')
    cols_json = json.dumps(columns)
    store_disp = html.escape(os.path.abspath(store_path))
    return _PAGE.replace('__DATA__', data_json).replace('__COLS__', cols_json).replace('__STORE__', store_disp)


_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gold labeling</title>
<style>
 body { font-family: -apple-system, Arial, sans-serif; margin: 0; background:#f5f5f5; }
 header { position: sticky; top:0; background:#222; color:#fff; padding:8px 14px; z-index:10; }
 header .counts span { margin-right:16px; font-size:13px; }
 header .help { font-size:11px; color:#aaa; margin-top:3px; }
 .pair { display:flex; align-items:stretch; border:1px solid #ddd; margin:8px; background:#fff; }
 .pair.focused { outline:3px solid #1e88e5; outline-offset:-1px; }
 .controls { width:200px; padding:8px; border-right:1px solid #eee; flex:0 0 auto; }
 .btnrow { display:flex; gap:4px; margin-bottom:5px; }
 button.lab { flex:1; padding:6px 4px; font-size:12px; border:1px solid #bbb; background:#eee;
              cursor:pointer; border-radius:4px; }
 button.lab.match.active   { background:#00C853; color:#fff; border-color:#00C853; }
 button.lab.no_match.active{ background:#D50000; color:#fff; border-color:#D50000; }
 button.lab.amb.active     { background:#FB8C00; color:#fff; border-color:#FB8C00; }
 button.lab.ok.active      { background:#607D8B; color:#fff; border-color:#607D8B; }
 .silverbar { flex:0 0 auto; width:16px; display:flex; align-items:center; justify-content:center; }
 .silverbar span { writing-mode:vertical-rl; transform:rotate(180deg);
                   font-size:10px; font-weight:bold; color:#fff; letter-spacing:1px; }
 .silverbar.silver-true  { background:#00E676; }
 .silverbar.silver-false { background:#FF1744; }
 .silverbar.silver-null  { background:#bdbdbd; }
 .comp { overflow-x:auto; flex:1; }
 table.cmp { border-collapse:collapse; font-size:11px; }
 table.cmp th { font-size:10px; color:#666; padding:2px 6px; text-align:left; white-space:nowrap; }
 table.cmp td { padding:2px 6px; white-space:nowrap; }
 td.equal { background:#a8e6a3; } td.diff { background:#f4a8a8; }
 td.one-missing { background:#fff3a3; } td.both-missing { background:#d3d3d3; }
 td.patid { background:#e0e0e0; color:#777; font-size:10px; }
 th.patid { color:#999; }
 td.side { font-weight:bold; background:#fff; }
</style></head>
<body>
<header>
 <div class="counts" id="counts"></div>
 <div class="help">keys: &uarr;/&darr; or j/k move &middot; 1 match &middot; 2 no-match &middot; 3 toggle ambiguous &middot; store: __STORE__</div>
</header>
<div id="list"></div>
<script>
const PAIRS = __DATA__;
const COLS = __COLS__;
let focus = 0;

function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}

function summary(){
  let m=0,n=0,amb=0,rev=0;
  for(const p of PAIRS){
    if(p.gold_label==='match') m++; else if(p.gold_label==='no_match') n++;
    if(p.ambiguous_pair) amb++;
    if(p.gold_label!==null || p.ambiguous_pair) rev++;
  }
  document.getElementById('counts').innerHTML =
    `<span><b>${PAIRS.length}</b> in subset</span>`+
    `<span>&#10003; match: <b>${m}</b></span>`+
    `<span>&#10007; no-match: <b>${n}</b></span>`+
    `<span>? ambiguous: <b>${amb}</b></span>`+
    `<span>reviewed: <b>${rev}</b></span>`+
    `<span>remaining: <b>${PAIRS.length-rev}</b></span>`;
}

function post(p, body){
  return fetch('/label', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(Object.assign({patid_a:p.patid_a, patid_b:p.patid_b}, body))})
    .catch(e => alert('Save failed: '+e+'\nIs the notebook server still running?'));
}

function setGold(i, val){
  const p = PAIRS[i];
  p.gold_label = (p.gold_label===val) ? null : val;   // clicking the active one clears it
  post(p, {gold_label: p.gold_label}); renderRow(i); summary();
}
function setAmb(i, val){
  const p = PAIRS[i]; p.ambiguous_pair = val;
  post(p, {ambiguous_pair: val}); renderRow(i); summary();
}

function rowHTML(p, i){
  const head = '<tr><td class="side"></td>' + COLS.map(c=>`<th>${esc(c)}</th>`).join('') +
    '<th class="patid">PATID</th></tr>';
  const side = s => `<tr><td class="side">${s}</td>` +
    p.fields.map(f=>`<td class="${f.cls}">${esc(s==='A'?f.a:f.b)}</td>`).join('') +
    `<td class="patid">${esc(s==='A'?p.patid_a:p.patid_b)}</td></tr>`;
  const svCls = p.silver_label===true ? 'silver-true' : p.silver_label===false ? 'silver-false' : 'silver-null';
  const svTxt = p.silver_label===null ? '?' : ('silver ' + p.silver_label);
  return `
  <div class="controls">
    <div class="btnrow">
      <button class="lab match ${p.gold_label==='match'?'active':''}" onclick="setGold(${i},'match')">Match</button>
      <button class="lab no_match ${p.gold_label==='no_match'?'active':''}" onclick="setGold(${i},'no_match')">No match</button>
    </div>
    <div class="btnrow">
      <button class="lab amb ${p.ambiguous_pair?'active':''}" onclick="setAmb(${i},true)">Ambiguous</button>
      <button class="lab ok ${!p.ambiguous_pair?'active':''}" onclick="setAmb(${i},false)">OK</button>
    </div>
  </div>
  <div class="silverbar ${svCls}"><span>${svTxt}</span></div>
  <div class="comp"><table class="cmp">${head}${side('A')}${side('B')}</table></div>`;
}

function renderRow(i){
  const el = document.getElementById('pair-'+i);
  el.innerHTML = rowHTML(PAIRS[i], i);
  el.className = 'pair' + (i===focus ? ' focused' : '');
}
function renderAll(){
  document.getElementById('list').innerHTML =
    PAIRS.map((p,i)=>`<div class="pair" id="pair-${i}"></div>`).join('');
  PAIRS.forEach((p,i)=>renderRow(i));
  summary();
}
function setFocus(i){
  if(i<0||i>=PAIRS.length) return;
  const old=focus; focus=i;
  document.getElementById('pair-'+old)?.classList.remove('focused');
  const el=document.getElementById('pair-'+focus);
  el.classList.add('focused'); el.scrollIntoView({block:'nearest'});
}
document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='ArrowDown'||e.key==='j'){ setFocus(focus+1); e.preventDefault(); }
  else if(e.key==='ArrowUp'||e.key==='k'){ setFocus(focus-1); e.preventDefault(); }
  else if(e.key==='1'){ setGold(focus,'match'); }
  else if(e.key==='2'){ setGold(focus,'no_match'); }
  else if(e.key==='3'){ setAmb(focus, !PAIRS[focus].ambiguous_pair); }
});
renderAll();
</script>
</body></html>"""


# --------------------------------------------------------------------------- server
def launch_labeler(df: pd.DataFrame, fields=None, store_path: str = DEFAULT_STORE,
                   host: str = '127.0.0.1', port: int = 8765, open_browser: bool = True,
                   max_pairs: int = 3000):
    """Start the labeling web app for `df` (a pairs-shaped DataFrame) and open a browser tab.

    `df` needs PATID_A, PATID_B, optional silver_label, and `<field>_A` / `<field>_B` columns.
    Re-running replaces any app already on `port`. Call ``stop_labeler(port)`` to shut down.
    Returns the URL.
    """
    from flask import Flask, jsonify, request, Response
    from werkzeug.serving import make_server

    fields = fields or [f for f in DEFAULT_FIELDS if f'{f}_A' in df.columns]
    if not fields:
        raise ValueError("No comparison fields found in df (expected '<field>_A'/'<field>_B' columns).")

    if len(df) > max_pairs:
        print(f"[labeler] subset has {len(df)} pairs; showing first {max_pairs}. "
              f"Pass a tighter filter or raise max_pairs.")
        df = df.head(max_pairs)

    store = LabelStore(store_path)

    app = Flask(__name__)

    @app.route('/')
    def index():
        # rebuilt per request so reopening the tab reflects the current store state
        pairs, cols = build_payload(df, fields, store)
        return Response(render_page(pairs, cols, store_path), mimetype='text/html')

    @app.route('/label', methods=['POST'])
    def label():
        d = request.get_json(force=True)
        a, b = str(d['patid_a']), str(d['patid_b'])
        kw = {}
        if 'gold_label' in d:
            kw['gold_label'] = d['gold_label']
        if 'ambiguous_pair' in d:
            kw['ambiguous_pair'] = d['ambiguous_pair']
        return jsonify(store.set(a, b, **kw))

    # replace any app already running on this port
    if port in _RUNNING:
        try:
            _RUNNING[port].shutdown()
        except Exception:
            pass
        del _RUNNING[port]

    # find a free port starting at `port`. Probe with a plain socket first: werkzeug's
    # make_server prints to stderr and calls sys.exit(1) (raising SystemExit, not OSError)
    # when the port is busy, so we avoid calling it on an occupied port.
    server = None
    last_err = None
    for p in range(port, port + 20):
        if not _port_free(host, p):
            continue
        try:
            server = make_server(host, p, app, threaded=True)
            port = p
            break
        except (OSError, SystemExit) as e:  # lost a race for the port; try the next one
            last_err = e
            continue
    if server is None:
        raise RuntimeError(f'No free port in {port}..{port + 19} ({last_err})')

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    server._thread = t  # keep a ref
    _RUNNING[port] = server

    url = f'http://{host}:{port}/'
    if open_browser:
        webbrowser.open(url)
    print(f'[labeler] running at {url}')
    print(f'[labeler] store: {os.path.abspath(store_path)}  ({len(df)} pairs loaded)')
    print(f'[labeler] call stop_labeler({port}) to shut it down')
    return url


def stop_labeler(port: int = None):
    """Shut down the labeling server on `port` (default: all running servers)."""
    ports = list(_RUNNING) if port is None else [port]
    if not ports:
        print('[labeler] no servers running')
        return
    for p in ports:
        srv = _RUNNING.pop(p, None)
        if srv is not None:
            try:
                srv.shutdown()
                print(f'[labeler] stopped server on port {p}')
            except Exception as e:
                print(f'[labeler] error stopping port {p}: {e}')
