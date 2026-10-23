// AML Consortium dashboard: four views on hash routes, all data from the FastAPI app (dashboard/app.py).
// Alert decisions (escalate / dismiss) are kept in this browser only (localStorage); they are a reviewer's
// convenience for the demo, not part of the audit trail.

const view = document.getElementById('view');
const DASH = '\u2014'; // taste-skill-ignore: the grey dash for a missing value in a data cell (app-screens §11)
const nf = (d = 0) => new Intl.NumberFormat('en-GB', { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = {
  int: (v) => nf(0).format(v),
  dec: (v, d = 3) => nf(d).format(v),
  score: (v) => (v >= 0.9995 ? '>0.999' : v < 0.0005 && v > 0 ? '<0.001' : nf(3).format(v)),
  usd: (v) => '$' + nf(2).format(v),
  usd0: (v) => '$' + nf(0).format(v),
  pct: (v, d = 1) => nf(d).format(v * 100) + '%',
  signed: (v, d = 2) => (v > 0 ? '+' : v < 0 ? '−' : '') + nf(d).format(Math.abs(v)),
  when: (ts) => {
    const d = new Date(ts.replace(' ', 'T'));
    return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) + ', ' + ts.slice(11, 16);
  },
  date: (ts) => new Date(ts.replace(' ', 'T')).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }),
  short: (h, n = 10) => (h ? h.slice(0, n) + '…' : DASH),
};
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
// Fictional display names from the config (partition.bank_names).
let BANK_NAMES = {};
const bank = (c) => BANK_NAMES[c] || 'Bank ' + String(c).replace('client_', '');
const h = (html) => { const t = document.createElement('template'); t.innerHTML = html.trim(); return t.content; };

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return r.json();
}

// ---------- local decisions ----------
const store = {
  get() { try { return JSON.parse(localStorage.getItem('aml-decisions') || '{}'); } catch { return {}; } },
  set(all) { try { localStorage.setItem('aml-decisions', JSON.stringify(all)); } catch {} },
};

// ---------- feature names in the analyst's words ----------
const yesNo = (v) => (v ? 'Yes' : 'No');
const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
const duration = (s) => (s >= 2592000 ? 'none in 30 days' : s < 3600 ? `${Math.round(s / 60)} min` : s < 86400 ? `${nf(1).format(s / 3600)} h` : `${nf(1).format(s / 86400)} days`);
const FEATURES = {
  log_amount: ['Amount', (v) => fmt.usd0(Math.expm1(v))],
  payment_format: ['Payment format', String],
  payment_currency: ['Paid in', String],
  receiving_currency: ['Received in', String],
  cross_currency: ['Currency converted', yesNo],
  cross_bank: ['Sent to another bank', yesNo],
  is_round: ['Round amount', yesNo],
  hour: ['Hour of day', (v) => `${String(v).padStart(2, '0')}:00`], // taste-skill-ignore: a clock, not a counter
  dayofweek: ['Day of week', (v) => days[v] ?? v],
  src_txn_count: ["Sender's earlier payments", fmt.int],
  src_amount_mean: ["Sender's typical amount", (v) => (v ? fmt.usd0(Math.expm1(v)) : 'no history')],
  src_amount_std: ["Spread of sender's amounts", (v) => (v ? `×${nf(1).format(Math.exp(v))}` : 'no history')],
  src_fan_out: ['Accounts the sender has paid', fmt.int],
  src_num_currencies: ['Currencies the sender used', fmt.int],
  src_secs_since_prev: ["Since sender's last payment", duration],
  src_burst: ["Sender's payments, last 24 h", fmt.int],
  dst_fan_in: ['Accounts that paid the receiver', fmt.int],
  dst_received_volume: ['Received by the receiver so far', (v) => fmt.usd0(Math.expm1(v))],
  dst_burst: ["Receiver's payments, last 24 h", fmt.int],
  account_age_days: ['Sender account age', (v) => `${fmt.int(v)} days`],
  customer_type: ['Customer type', String],
  kyc_risk_level: ['KYC risk level', String],
  occupation_category: ['Occupation', (v) => String(v).replace('_', ' ')],
  declared_annual_income: ['Declared income', fmt.usd0],
  branch_region: ['Branch region', String],
  account_type: ['Account type', String],
  num_linked_accounts: ['Linked accounts', fmt.int],
  onboarding_channel: ['Opened via', String],
};
const featureLabel = (f) => (FEATURES[f] ? FEATURES[f][0] : f);
const featureValue = (f, v) => (v == null ? 'not held by this bank' : FEATURES[f] ? FEATURES[f][1](v) : String(v));
const PATTERNS = { fan_out: 'Fan-out', fan_in: 'Fan-in', cycle: 'Cycle', scatter_gather: 'Scatter-gather', gather_scatter: 'Gather-scatter',
                   bipartite: 'Bipartite', stack: 'Stack', random: 'Random', unlisted: 'Not in a listed pattern' };

// ---------- shell ----------
function setNav(name) {
  document.querySelectorAll('.rail-item').forEach((a) => {
    if (a.dataset.view === name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
  document.title = `AML Consortium · ${name[0].toUpperCase()}${name.slice(1)}`;
}

function errorBlock(where, e, retry) {
  const node = h(`<section class="panel err"><span>Couldn't load ${esc(where)}: ${esc(e.message)}</span><button class="btn ghost" type="button">Retry</button></section>`);
  node.querySelector('button').addEventListener('click', retry);
  return node;
}

// ---------- alerts ----------
const alertsState = { rows: null, meta: null, bank: 'all', tab: 'open', shown: 100, selected: null };

async function renderAlerts(txn) {
  setNav('alerts');
  if (!alertsState.rows) {
    view.replaceChildren(h(`<div class="title-row"><div><h1>Alerts</h1><p class="sub">Loading the flagged transactions…</p></div></div>
      <div class="triage"><section class="panel queue"><div class="queue-head"><div class="skeleton loading-bar"></div></div></section><section class="panel"><div class="skeleton loading-bar"></div></section></div>`));
    try {
      const data = await api('/api/alerts?limit=100000');
      alertsState.rows = data.alerts;
      alertsState.meta = data;
    } catch (e) {
      view.replaceChildren(h('<div class="title-row"><div><h1>Alerts</h1></div></div>'), errorBlock('the alerts', e, () => renderAlerts(txn)));
      return;
    }
  }
  const { rows, meta } = alertsState;
  const banks = Object.keys(meta.per_bank).sort();
  const top = Object.entries(meta.per_bank).sort((a, b) => b[1].alerts - a[1].alerts)[0];
  const stamps = rows.map((r) => r.timestamp).sort();
  const [d0, d1] = stamps.length ? [new Date(stamps[0].replace(' ', 'T')), new Date(stamps.at(-1).replace(' ', 'T'))] : [];
  const period = !d0 ? '' : d0.getMonth() === d1.getMonth()
    ? `${d0.getDate()}\u2013${d1.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })}`
    : `${fmt.date(stamps[0])} \u2013 ${fmt.date(stamps.at(-1))}`;
  view.replaceChildren(h(`
    <div class="title-row">
      <div>
        <h1>Alerts</h1>
        <p class="sub">${fmt.int(meta.total)} transactions flagged by 5 banks, ${esc(period)}. ${top ? `${bank(top[0])} raised the most (${fmt.int(top[1].alerts)}).` : ''}</p>
      </div>
    </div>
    <div class="triage" id="triage">
      <section class="panel queue" aria-label="Alert queue">
        <div class="queue-head">
          <label class="count" for="qbank"><span class="muted">Highest score first</span><span class="muted" id="qcount"></span></label>
          <select class="input" id="qbank" aria-label="Bank">
            <option value="all">All banks (${fmt.int(rows.length)})</option>
            ${banks.map((b) => `<option value="${b}">${bank(b)} (${fmt.int(meta.per_bank[b].alerts)})</option>`).join('')}
          </select>
          <div class="queue-tabs" role="tablist">
            <button type="button" role="tab" data-tab="open">Open</button>
            <button type="button" role="tab" data-tab="escalated">Escalated</button>
            <button type="button" role="tab" data-tab="dismissed">Dismissed</button>
          </div>
        </div>
        <ol id="qlist"></ol>
        <div class="queue-foot" id="qfoot"></div>
      </section>
      <section class="detail" id="detail" aria-label="Open alert"></section>
    </div>`));
  const sel = view.querySelector('#qbank');
  sel.value = alertsState.bank;
  sel.addEventListener('change', () => { alertsState.bank = sel.value; alertsState.shown = 100; drawQueue(); });
  view.querySelectorAll('.queue-tabs button').forEach((b) => b.addEventListener('click', () => { alertsState.tab = b.dataset.tab; alertsState.shown = 100; drawQueue(); }));
  drawQueue();
  const first = txn ?? alertsState.selected ?? currentQueue()[0]?.txn_id;
  if (first != null) openAlert(Number(first), txn != null);
  else view.querySelector('#detail').replaceChildren(h('<section class="panel"><div class="empty">No alerts in this view.</div></section>'));
}

function currentQueue() {
  const decided = store.get();
  return alertsState.rows.filter((r) => (alertsState.bank === 'all' || r.bank === alertsState.bank)
    && (alertsState.tab === 'open' ? !decided[r.txn_id] : decided[r.txn_id]?.decision === alertsState.tab));
}

function drawQueue() {
  const decided = store.get();
  const inBank = alertsState.rows.filter((r) => alertsState.bank === 'all' || r.bank === alertsState.bank);
  const counts = { open: 0, escalated: 0, dismissed: 0 };
  inBank.forEach((r) => { counts[decided[r.txn_id]?.decision || 'open'] += 1; });
  view.querySelectorAll('.queue-tabs button').forEach((b) => {
    b.setAttribute('aria-selected', b.dataset.tab === alertsState.tab);
    b.textContent = `${b.dataset.tab[0].toUpperCase()}${b.dataset.tab.slice(1)} ${fmt.int(counts[b.dataset.tab])}`;
  });
  const list = currentQueue();
  view.querySelector('#qcount').textContent = `${fmt.int(list.length)} in view`;
  const ol = view.querySelector('#qlist');
  if (!list.length) {
    ol.replaceChildren(h(`<li style="padding:16px"><div class="empty">${alertsState.tab === 'open' ? 'Every alert in this bank has a decision.' : `No ${alertsState.tab} alerts yet.`}</div></li>`));
  } else {
    ol.replaceChildren(h(list.slice(0, alertsState.shown).map((r) => `
      <li><button type="button" class="alert-row" data-txn="${r.txn_id}" aria-selected="${r.txn_id === alertsState.selected}">
        <span class="who">${fmt.usd(r.amount_usd)}</span><span class="when">${esc(fmt.when(r.timestamp))}</span>
        <span class="what">${bank(r.bank)} · ${esc(r.payment_format)} · ${esc(r.payment_currency)}${r.payment_currency !== r.receiving_currency ? ' → ' + esc(r.receiving_currency) : ''}</span>
        <span class="score">${fmt.score(r.score)}</span>
      </button></li>`).join('')));
    ol.querySelectorAll('.alert-row').forEach((b) => b.addEventListener('click', () => { location.hash = `#/alerts/${b.dataset.txn}`; }));
  }
  const foot = view.querySelector('#qfoot');
  const shown = Math.min(alertsState.shown, list.length);
  foot.replaceChildren(h(`<div class="err"><span class="stamp">${fmt.int(shown ? 1 : 0)}–${fmt.int(shown)} of ${fmt.int(list.length)}</span>${list.length > shown ? '<button type="button" class="card-link" id="more">Show 100 more</button>' : ''}</div>`));
  foot.querySelector('#more')?.addEventListener('click', () => { alertsState.shown += 100; drawQueue(); });
}

async function openAlert(txn, push) {
  alertsState.selected = txn;
  view.querySelectorAll('.alert-row').forEach((b) => b.setAttribute('aria-selected', Number(b.dataset.txn) === txn));
  const pane = view.querySelector('#detail');
  view.querySelector('#triage')?.classList.toggle('open', push);
  pane.replaceChildren(h('<section class="panel"><div class="skeleton loading-bar"></div><div class="skeleton loading-bar"></div></section>'));
  let a;
  try { a = await api(`/api/alerts/${txn}`); } catch (e) { pane.replaceChildren(errorBlock('this alert', e, () => openAlert(txn, push))); return; }
  if (alertsState.selected !== txn) return;
  const decision = store.get()[txn];
  const max = Math.max(...a.top_features.map((f) => Math.abs(f.contribution)), 1e-9);
  const total = a.top_features.reduce((s, f) => s + f.contribution, 0);
  const logit = a.logit ?? Math.log(a.score / (1 - a.score));
  pane.replaceChildren(h(`
    <section class="panel">
      <button type="button" class="card-link back" id="back">All alerts</button>
      <div class="detail-head">
        <div>
          <h2>${fmt.usd(a.amount_usd)} ${esc(a.payment_format)} payment</h2>
          <p class="sub">${bank(a.bank)} · ${esc(fmt.when(a.timestamp))} ${esc(a.timestamp.slice(0, 4))} · transaction ${fmt.int(a.txn_id)}</p>
        </div>
        <div class="actions">
          <button type="button" class="btn" data-decide="dismissed">Dismiss</button>
          <button type="button" class="btn primary" data-decide="escalated">Escalate</button>
        </div>
      </div>
      <div class="gauge" role="img" aria-label="Score ${fmt.score(a.score)} against threshold ${fmt.dec(a.threshold)}">
        <div class="gauge-label"><span>Risk score <strong class="num">${fmt.score(a.score)}</strong></span><span class="muted">${bank(a.bank)}'s threshold ${fmt.dec(a.threshold)}</span></div>
        <div class="gauge-track"><div class="gauge-fill" style="width:${(a.score * 100).toFixed(2)}%"></div><div class="gauge-tick" style="left:calc(${(a.threshold * 100).toFixed(2)}% - 1px)"></div></div>
        <div class="gauge-scale"><span>0</span><span>1</span></div>
      </div>
      ${decision ? `<p class="decision-note">${decision.decision === 'escalated' ? 'Escalated' : 'Dismissed'} on this browser, ${esc(new Date(decision.at).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' }))}. <button type="button" class="card-link" id="undo">Undo</button></p>` : ''}
    </section>
    <div class="split">
      <div class="detail">
        <section class="panel">
          <header><h2>Why it was flagged</h2><span class="meta">SHAP, top ${a.top_features.length} features</span></header>
          <div class="why">
            ${a.top_features.map((f) => {
              const w = (Math.abs(f.contribution) / max) * 50;
              const left = f.contribution >= 0 ? 50 : 50 - w;
              return `<div class="feat">${esc(featureLabel(f.feature))}<small>${esc(featureValue(f.feature, f.value))}</small></div>
                <div class="bar" aria-hidden="true"><i class="${f.contribution >= 0 ? '' : 'down'}" style="left:${left.toFixed(2)}%;width:${Math.max(w, 0.6).toFixed(2)}%"></i></div>
                <div class="val">${fmt.signed(f.contribution)}</div>`;
            }).join('')}
          </div>
          <p class="why-note">Blue pushes towards an alert, grey away from one, in log-odds against the bank's average transaction (${fmt.signed(a.base_value)}). These ${a.top_features.length} features add ${fmt.signed(total)}; the rest of the bank's features make up the remaining ${fmt.signed(logit - a.base_value - total)}. Computed by the bank on its own model and data.</p>
        </section>
        <section class="panel">
          <header><h2>Dataset label</h2><span class="meta">evaluation only</span></header>
          <p>${a.label ? `Laundering${a.pattern && a.pattern !== 'none' ? `, ${esc(PATTERNS[a.pattern] || a.pattern)} pattern` : ''}` : 'Not laundering'}</p>
          <p class="truth">From the synthetic dataset's ground truth. A real bank would not know this when the alert is raised.</p>
        </section>
      </div>
      <div class="detail">
        <section class="panel">
          <header><h2>Transaction</h2></header>
          <dl class="kv">
            <dt>Amount</dt><dd>${fmt.usd(a.amount_usd)}</dd>
            <dt>Payment format</dt><dd>${esc(a.payment_format)}</dd>
            <dt>Sender account</dt><dd class="mono">${esc(a.src_key)}</dd>
            <dt>Receiver account</dt><dd class="mono">${esc(a.dst_key)}</dd>
            <dt>Paid in</dt><dd>${esc(a.payment_currency)}</dd>
            <dt>Received in</dt><dd>${esc(a.receiving_currency)}</dd>
            <dt>Time</dt><dd>${esc(a.timestamp.slice(0, 16))}</dd>
          </dl>
        </section>
        <section class="panel">
          <header><h2>Ledger record</h2><button type="button" class="btn" id="verify">Verify on chain</button></header>
          <div class="receipt">
            <div class="kv" style="grid-template-columns:auto minmax(0,1fr)">
              <dt>Alert hash</dt><dd class="mono" title="${esc(a.alert_hash)}">${fmt.short(a.alert_hash, 16)}</dd>
              <dt>Model version</dt><dd class="mono" title="${esc(a.model_version)}">${fmt.short(a.model_version, 16)}</dd>
            </div>
            <div id="proof" class="receipt"><p class="muted">Checks that this alert is unchanged and was logged by ${bank(a.bank)}.</p></div>
          </div>
        </section>
      </div>
    </div>`));
  pane.querySelectorAll('[data-decide]').forEach((b) => b.addEventListener('click', () => decide(txn, b.dataset.decide)));
  pane.querySelector('#undo')?.addEventListener('click', () => decide(txn, null));
  pane.querySelector('#back').addEventListener('click', () => { location.hash = '#/alerts'; });
  pane.querySelector('#verify').addEventListener('click', () => verifyAlert(a));
}

function decide(txn, decision) {
  const all = store.get();
  if (decision) all[txn] = { decision, at: new Date().toISOString() };
  else delete all[txn];
  store.set(all);
  const list = currentQueue();
  drawQueue();
  if (decision && alertsState.tab === 'open') {
    const next = list.find((r) => r.txn_id !== txn);
    if (next) { location.hash = `#/alerts/${next.txn_id}`; return; }
  }
  openAlert(txn, false);
}

async function verifyAlert(a) {
  const proof = view.querySelector('#proof');
  const btn = view.querySelector('#verify');
  btn.disabled = true; btn.textContent = 'Checking…';
  try {
    const v = await api(`/api/chain/alerts/${a.txn_id}`);
    const lines = [`<p class="check ${v.intact ? 'ok' : 'bad'}">${v.intact ? 'Unchanged: the stored alert still hashes to its recorded hash' : 'Changed: the stored alert no longer matches its hash'}</p>`];
    if (!v.online) {
      lines.push('<p class="notice">No chain node is running, so the ledger could not be checked. Start one with <span class="mono">npx hardhat node</span> in src/aml_fl/chain/hardhat, then run <span class="mono">build_alerts.py --chain</span>.</p>');
    } else if (!v.logged) {
      lines.push('<p class="check bad">Not on the ledger: no bank logged this hash</p>');
    } else {
      lines.push(`<p class="check ${v.verified ? 'ok' : 'bad'}">Logged on the ledger by ${esc(v.logged_by_bank ? bank(v.logged_by_bank) : 'an unknown account')}${v.verified ? '' : `, not by ${bank(a.bank)}`}</p>`);
    }
    proof.replaceChildren(h(lines.join('')));
  } catch (e) {
    proof.replaceChildren(errorBlock('the ledger check', e, () => verifyAlert(a)));
  } finally {
    btn.disabled = false; btn.textContent = 'Verify on chain';
  }
}

// ---------- models ----------
const SETUPS = [
  ['centralized', 'xgboost', 'Centralized XGBoost', 'all data pooled, not private', 'bound'],
  ['centralized', 'mlp', 'Centralized MLP', 'all data pooled, not private', 'bound'],
  ['local', 'xgboost', 'Each bank alone, XGBoost', 'no sharing', ''],
  ['fl_hybrid', 'mlp', 'Federated hybrid (ours)', 'shared core encoder, private encoder', 'ours'],
  ['fl_zero_filled', 'mlp', 'Federated zero-filled', 'banks reveal their column lists', ''],
  ['local', 'mlp', 'Each bank alone, MLP', 'no sharing', ''],
  ['fl_encoder_head', 'mlp', 'Federated, shared head only', 'private encoder, shared head', ''],
  ['fl_naive', 'mlp', 'Federated, common columns', 'whole model on 9 shared columns', ''],
];

async function renderModels() {
  setNav('models');
  view.replaceChildren(h('<div class="title-row"><div><h1>Models</h1><p class="sub">Loading results…</p></div></div>'));
  let o;
  try { o = await api('/api/overview'); } catch (e) { view.replaceChildren(h('<div class="title-row"><div><h1>Models</h1></div></div>'), errorBlock('the results', e, renderModels)); return; }
  const pick = (setup, model) => o.setups.find((s) => s.setup === setup && s.model === model && (s.client === 'pooled' || s.client === 'all'));
  const rows = SETUPS.map(([setup, model, name, sub, kind]) => ({ r: pick(setup, model), name, sub, kind })).filter((x) => x.r);
  const hy = pick('fl_hybrid', 'mlp'), alone = pick('local', 'mlp'), cen = pick('centralized', 'mlp');
  const axis = Math.ceil(Math.max(...rows.map((x) => x.r.pr_auc_mean + (x.r.pr_auc_std || 0))) * 10) / 10;
  const seeds = Math.max(...rows.map((x) => x.r.seeds));
  const per = (setup, model) => o.per_client.find((p) => p.setup === setup && p.model === model) || {};
  const pa = per('local', 'mlp'), ph = per('fl_hybrid', 'mlp');
  const banks = Object.keys(o.dataset.banks).sort();
  const d = o.dataset;
  view.replaceChildren(h(`
    <div class="title-row">
      <div>
        <h1>Models</h1>
        <p class="sub">${hy && alone && cen ? `Federated hybrid reaches a PR-AUC of ${fmt.dec(hy.pr_auc_mean)}, ${fmt.pct(hy.pr_auc_mean / alone.pr_auc_mean - 1, 0)} above each bank alone (${fmt.dec(alone.pr_auc_mean)}) and ${fmt.pct(hy.pr_auc_mean / cen.pr_auc_mean, 0)} of pooling all data (${fmt.dec(cen.pr_auc_mean)}), without sharing data or column lists.` : 'Run summarize_seeds.py to compare the setups.'}</p>
      </div>
    </div>
    <div class="grid top">
      <section class="panel" style="grid-column: span 8">
        <header><h2>Test PR-AUC, mean of ${seeds} seeds</h2><span class="meta">pooled over the 5 banks · bar ± one standard deviation</span></header>
        <div class="ranking">
          ${rows.map(({ r, name, sub, kind }) => `
            <div class="rank-row">
              <div class="name">${esc(name)}<small>${esc(sub)}</small></div>
              <div class="rank-track" aria-hidden="true">
                <i class="${kind}" style="width:${(r.pr_auc_mean / axis * 100).toFixed(2)}%"></i>
                ${r.pr_auc_std ? `<b style="left:${((r.pr_auc_mean - r.pr_auc_std) / axis * 100).toFixed(2)}%;width:${(2 * r.pr_auc_std / axis * 100).toFixed(2)}%"></b>` : ''}
              </div>
              <div class="v">${fmt.dec(r.pr_auc_mean)} <small>±${fmt.dec(r.pr_auc_std || 0)}</small></div>
            </div>`).join('')}
        </div>
      </section>
      <section class="panel" style="grid-column: span 4">
        <header><h2>Data</h2><span class="meta">IBM AML, HI-Small</span></header>
        <dl class="kv">
          <dt>Transactions</dt><dd>${fmt.int(d.rows)}</dd>
          <dt>Laundering</dt><dd>${fmt.int(d.laundering_rows)} <span class="muted">(${fmt.pct(d.laundering_rate, 3)})</span></dd>
          <dt>Accounts</dt><dd>${fmt.int(d.num_accounts)}</dd>
          <dt>Sending banks</dt><dd>${fmt.int(d.num_src_banks)} <span class="muted">in 5 clients</span></dd>
          <dt>Period</dt><dd>${esc(fmt.date(d.time_start))} – ${esc(fmt.date(d.time_end))}</dd>
          <dt>Split</dt><dd>by time, 60 / 20 / 20</dd>
          <dt>Test transactions</dt><dd>${fmt.int(banks.reduce((s, b) => s + d.banks[b].per_split.test[0], 0))}</dd>
          <dt>Laundering in test</dt><dd>${fmt.int(banks.reduce((s, b) => s + d.banks[b].per_split.test[1], 0))}</dd>
          <dt>Columns per bank</dt><dd>${Math.min(...banks.map((b) => d.banks[b].num_columns))}\u2013${Math.max(...banks.map((b) => d.banks[b].num_columns))} <span class="muted">(9 shared)</span></dd>
        </dl>
      </section>
      <section class="panel" style="grid-column: 1 / -1">
        <header><h2>Per bank</h2><span class="meta">test PR-AUC, mean of ${seeds} seeds</span></header>
        <div class="table-wrap"><table>
          <thead><tr><th>Bank</th><th class="n">Transactions</th><th class="n">Laundering in test</th><th class="n">Columns</th><th class="n">Alone (MLP)</th><th class="n">Federated hybrid</th><th class="n">Change</th></tr></thead>
          <tbody>${banks.map((b) => {
            const info = d.banks[b], a = pa[`${b}_mean`], f = ph[`${b}_mean`];
            const ch = a != null && f != null ? f - a : null;
            return `<tr><td>${bank(b)}</td><td class="n">${fmt.int(info.rows)}</td><td class="n">${fmt.int(info.per_split.test[1])}</td><td class="n">${fmt.int(info.num_columns)}</td>
              <td class="n">${a != null ? fmt.dec(a) : DASH}</td><td class="n">${f != null ? fmt.dec(f) : DASH}</td>
              <td class="n ${ch > 0 ? 'pos' : ch < 0 ? 'neg' : ''}">${ch != null ? fmt.signed(ch, 3) : DASH}</td></tr>`;
          }).join('')}</tbody>
        </table></div>
      </section>
    </div>`));
}

// ---------- training ----------
const COMPARE = [['zero_filled', 'Zero-filled'], ['encoder_head', 'Shared head only'], ['naive', 'Common columns']];
const trainingState = { other: 'zero_filled' };

async function renderTraining() {
  setNav('training');
  view.replaceChildren(h('<div class="title-row"><div><h1>Training</h1><p class="sub">Loading the training history…</p></div></div>'));
  let hist, pats;
  try { [hist, pats] = await Promise.all([api('/api/history?seed=42'), api('/api/patterns')]); }
  catch (e) { view.replaceChildren(h('<div class="title-row"><div><h1>Training</h1></div></div>'), errorBlock('the training history', e, renderTraining)); return; }
  const modes = hist.modes;
  const hy = modes.hybrid;
  const last = (m) => modes[m]?.at(-1)?.val_pr_auc;
  const recall = (setup, model, p) => pats.rows.find((r) => r.setup === setup && r.model === model && r.pattern === p);
  const patterns = [...new Set(pats.rows.map((r) => r.pattern))].filter((p) => p !== 'none' && PATTERNS[p])
    .sort((a, b) => (a === 'unlisted') - (b === 'unlisted') || (recall('fl_hybrid', 'mlp', b)?.positives || 0) - (recall('fl_hybrid', 'mlp', a)?.positives || 0));
  view.replaceChildren(h(`
    <div class="title-row">
      <div>
        <h1>Training</h1>
        <p class="sub">${hy ? `After 100 rounds the hybrid reaches a validation PR-AUC of ${fmt.dec(last('hybrid'))}, against ${fmt.dec(last('zero_filled') ?? NaN)} for zero-filled FedAvg (seed 42).` : 'No hybrid run for seed 42 yet.'}</p>
      </div>
    </div>
    <div class="grid top">
      <section class="panel" style="grid-column: 1 / -1">
        <div class="chart-head"><h3>Validation PR-AUC by round</h3><span id="legend" class="legend"></span></div>
        <div class="filters"><div class="segmented" role="group" aria-label="Compare the hybrid with">
          ${COMPARE.filter(([m]) => modes[m]).map(([m, label]) => `<button type="button" data-mode="${m}" aria-pressed="${m === trainingState.other}">${label}</button>`).join('')}
        </div></div>
        <div class="chart" id="curve"></div>
      </section>
      <section class="panel" style="grid-column: 1 / -1">
        <header><h2>Recall by laundering pattern</h2><span class="meta">test set, mean of 3 seeds, each bank's own threshold</span></header>
        <div class="table-wrap"><table>
          <thead><tr><th>Pattern</th><th class="n">In test</th><th class="n">Federated hybrid</th><th class="n">Federated zero-filled</th><th class="n">Each bank alone (MLP)</th><th class="n">Centralized XGBoost</th></tr></thead>
          <tbody>${patterns.map((p) => {
            const cell = (s, m) => { const r = recall(s, m, p); return r ? fmt.pct(r.recall, 0) : DASH; };
            const n = recall('fl_hybrid', 'mlp', p)?.positives;
            return `<tr><td>${esc(PATTERNS[p])}</td><td class="n">${n != null ? fmt.int(Math.round(n)) : DASH}</td><td class="n">${cell('fl_hybrid', 'mlp')}</td><td class="n">${cell('fl_zero_filled', 'mlp')}</td><td class="n">${cell('local', 'mlp')}</td><td class="n">${cell('centralized', 'xgboost')}</td></tr>`;
          }).join('')}</tbody>
        </table></div>
      </section>
    </div>`));
  const draw = () => {
    const el = view.querySelector('#curve');
    el.replaceChildren();
    const other = modes[trainingState.other];
    const series = [{ name: 'Federated hybrid', values: hy.map((r) => r.val_pr_auc) }];
    if (other) series.push({ name: COMPARE.find(([m]) => m === trainingState.other)[1], values: other.map((r) => r.val_pr_auc), colour: '#8a8a8a' });
    const xAt = hy.map((r, i) => (r.round % 10 === 0 ? i : -1)).filter((i) => i >= 0);
    window.Charts.line(el, { labels: hy.map((r) => `Round ${r.round}`), series, format: (v) => nf(2).format(v), height: 240, xAt, legend: view.querySelector('#legend') });
  };
  view.querySelectorAll('.segmented button').forEach((b) => b.addEventListener('click', () => {
    trainingState.other = b.dataset.mode;
    view.querySelectorAll('.segmented button').forEach((x) => x.setAttribute('aria-pressed', x === b));
    draw();
  }));
  if (hy) draw(); else view.querySelector('#curve').replaceChildren(h('<div class="empty">Run python scripts/run_fl.py --mode hybrid --seed 42 to draw this chart.</div>'));
}

// ---------- audit ----------
const auditState = { run: null };

async function renderAudit() {
  setNav('audit');
  view.replaceChildren(h('<div class="title-row"><div><h1>Audit</h1><p class="sub">Checking the ledger…</p></div></div>'));
  let runs, status;
  try { [runs, status] = await Promise.all([api('/api/runs'), api('/api/chain/status')]); }
  catch (e) { view.replaceChildren(h('<div class="title-row"><div><h1>Audit</h1></div></div>'), errorBlock('the audit', e, renderAudit)); return; }
  auditState.run = auditState.run || (runs.runs.find((r) => r.run === runs.default) ? runs.default : runs.runs[0]?.run);
  view.replaceChildren(h(`
    <div class="title-row">
      <div><h1>Audit</h1><p class="sub" id="finding">${status.online ? 'Comparing each run’s recorded hashes with the ledger…' : 'No chain node is running, so nothing can be checked against the ledger.'}</p></div>
      <div class="actions">
        <select class="input" id="run" aria-label="Run" style="height:36px">${runs.runs.map((r) => `<option value="${esc(r.run)}">${esc(r.run)}${r.bank_hashes ? '' : ' (model hashes only)'}</option>`).join('')}</select>
        <button type="button" class="btn" id="recheck">Check again</button>
      </div>
    </div>
    <div class="grid top">
      <section class="panel" style="grid-column: 1 / -1" id="ledger-panel">
        <header><h2>Ledger: global model and each bank\u2019s update, per round</h2><div class="legend"><span><i></i>matches</span><span><i class="missing"></i>not recorded</span><span><i class="bad"></i>does not match</span></div></header>
        <div id="ledger"></div>
      </section>
      <section class="panel" style="grid-column: span 7">
        <header><h2>Contract</h2><span class="meta">${status.online ? `local chain ${status.chain_id} · block ${fmt.int(status.block)}` : 'offline'}</span></header>
        ${status.online ? `<dl class="kv">
          <dt>Network</dt><dd>Hardhat, local</dd>
          <dt>FLAudit contract</dt><dd class="mono" title="${esc(status.address)}">${esc(status.address)}</dd>
          <dt>Operator (FL server)</dt><dd class="mono" title="${esc(status.operator)}">${esc(status.operator)}</dd>
          <dt>Global model hashes</dt><dd>${fmt.int(status.counts.global_models)}</dd>
          <dt>Bank update hashes</dt><dd>${fmt.int(status.counts.updates)}</dd>
          <dt>Alert hashes</dt><dd>${fmt.int(status.counts.alerts)}</dd>
        </dl>` : `<div class="notice">Start a local chain and record a run:<br><span class="mono">cd src/aml_fl/chain/hardhat && npx hardhat node</span><br><span class="mono">python scripts/audit_to_chain.py ${esc(auditState.run || 'fl_hybrid_s42')}</span></div>`}
      </section>
      <section class="panel" style="grid-column: span 5">
        <header><h2>Registered banks</h2><span class="meta">${status.online ? `${status.banks.filter((b) => b.registered).length} of ${status.banks.length}` : DASH}</span></header>
        ${status.online ? `<div class="rows">${status.banks.map((b) => `<div class="row"><span style="flex:1">${bank(b.bank)}</span><span class="mono muted">${fmt.short(b.address, 12)}</span><span class="pill ${b.registered ? '' : 'neg'}">${b.registered ? 'Registered' : 'Not registered'}</span></div>`).join('')}</div>`
          : '<div class="empty">Bank accounts appear once the chain is running.</div>'}
      </section>
    </div>`));
  const sel = view.querySelector('#run');
  sel.value = auditState.run;
  sel.addEventListener('change', () => { auditState.run = sel.value; drawLedger(status.online); });
  view.querySelector('#recheck').addEventListener('click', renderAudit);
  drawLedger(status.online);
}

async function drawLedger(online) {
  const el = view.querySelector('#ledger');
  const finding = view.querySelector('#finding');
  if (!auditState.run) { el.replaceChildren(h('<div class="empty">No FL run with an audit.json yet.</div>')); return; }
  el.replaceChildren(h('<div class="skeleton loading-bar"></div>'));
  let v;
  try { v = await api(`/api/chain/runs/${encodeURIComponent(auditState.run)}`); } catch (e) { el.replaceChildren(errorBlock('the ledger', e, () => drawLedger(online))); return; }
  if (!v.online) {
    el.replaceChildren(h(`<div class="empty">${fmt.int(v.rounds)} rounds recorded in ${esc(v.run)}/audit.json. Start the chain to compare them with the ledger.</div>`));
    return;
  }
  const n = v.rounds.length;
  const banks = [...new Set(v.rounds.flatMap((r) => Object.keys(r.updates)))].sort();
  const cell = (state, title) => `<i class="${state}" title="${esc(title)}"></i>`;
  const global = v.rounds.map((r) => cell(!r.recorded ? 'missing' : r.match ? '' : 'bad', `Round ${r.round}: global model ${r.recorded ? (r.match ? 'matches' : 'does not match') : 'not recorded'}`));
  const rows = [`<span class="who">Global model</span><div class="cells" style="--n:${n}">${global.join('')}</div>`];
  for (const b of banks) {
    rows.push(`<span class="who" title="${esc(bank(b))}\u2019s update">${bank(b)}</span><div class="cells" style="--n:${n}">${v.rounds.map((r) => (b in r.updates
      ? cell(r.updates[b] ? '' : 'bad', `Round ${r.round}: ${bank(b)}'s update ${r.updates[b] ? 'matches' : 'does not match or is not recorded'}`)
      : cell('none', `Round ${r.round}: no update hash in this run`))).join('')}</div>`);
  }
  el.replaceChildren(h(`<div class="ledger">${rows.join('')}<span></span><div class="ledger-axis"><span>Round 1</span><span>Round ${n}</span></div></div>
    ${banks.length ? '' : '<p class="why-note">This run was made before per-bank update hashes were added, so only the global model is on the ledger.</p>'}`));
  const checked = n + v.rounds.reduce((s, r) => s + Object.keys(r.updates).length, 0);
  const bad = v.rounds.filter((r) => !r.match || Object.values(r.updates).some((x) => !x)).map((r) => r.round);
  finding.innerHTML = bad.length
    ? `<span class="neg">${fmt.int(bad.length)} of ${fmt.int(n)} rounds of ${esc(v.run)} do not match the ledger</span> (first: round ${bad[0]}). The audit file was changed after it was recorded, or was never recorded.`
    : `All ${fmt.int(n)} rounds of ${esc(v.run)} match the ledger: ${fmt.int(checked)} hashes checked.`;
}

// ---------- router ----------
async function route() {
  const [name, arg] = location.hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent);
  const known = { alerts: renderAlerts, models: renderModels, training: renderTraining, audit: renderAudit };
  const page = known[name] ? name : 'alerts';
  if (page === 'alerts' && view.querySelector('#triage') && arg) { openAlert(Number(arg), true); return; }
  if (page === 'alerts' && view.querySelector('#triage') && !arg) { view.querySelector('#triage').classList.remove('open'); return; }
  await known[page](arg);
}

async function loadScope() {
  try {
    const r = await api('/api/runs');
    BANK_NAMES = r.bank_names || {};
    document.getElementById('scope').textContent = `${Object.keys(BANK_NAMES).length || 5} banks · alerts from ${r.default}`;
  } catch {}
}

window.addEventListener('hashchange', route);
loadScope().then(route);
