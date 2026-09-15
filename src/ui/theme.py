"""Bizmax Risk Command visual shell for the Streamlit application.

The tokens and component language are adapted from the standalone frontend in
``Desktop/teamupperechelons/frontend``.  Keeping the theme here lets app.py stay
focused on orchestration and analysis.
"""

from __future__ import annotations

from html import escape

import streamlit as st


BIZMAX_THEME = r"""
<style>
  :root {
    --bz-bg: #060a16;
    --bz-sidebar: #080d1b;
    --bz-panel: rgba(14, 22, 42, .88);
    --bz-panel-2: #111c33;
    --bz-line: rgba(137, 159, 204, .15);
    --bz-line-strong: rgba(137, 159, 204, .28);
    --bz-text: #f4f7ff;
    --bz-muted: #8e9ab6;
    --bz-muted-2: #586581;
    --bz-mint: #37e8c2;
    --bz-cyan: #35c7ff;
    --bz-violet: #7c5cff;
    --bz-pink: #ff5f8f;
    --bz-amber: #ffbf3f;
    --bz-red: #ff5277;
    --bz-green: #37e8a6;
    --bz-radius: 15px;
  }

  html, body, [class*="css"] {
    font-family: "Segoe UI Variable", "Segoe UI", sans-serif;
  }
  [data-testid="stAppViewContainer"] {
    color: var(--bz-text);
    background:
      linear-gradient(rgba(55, 232, 194, .025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(55, 232, 194, .025) 1px, transparent 1px),
      radial-gradient(circle at 82% -8%, rgba(53, 93, 205, .17), transparent 35%),
      radial-gradient(circle at 15% 80%, rgba(55, 232, 194, .055), transparent 32%),
      var(--bz-bg);
    background-size: 42px 42px, 42px 42px, auto, auto, auto;
  }
  [data-testid="stHeader"] { background: rgba(6, 10, 22, .72); }
  [data-testid="stMainBlockContainer"] {
    max-width: 1540px;
    padding-top: 1.1rem;
    padding-bottom: 4rem;
  }

  /* Sidebar from the supplied Risk Command frontend. */
  [data-testid="stSidebar"] {
    background: rgba(8, 13, 27, .97);
    border-right: 1px solid var(--bz-line);
  }
  [data-testid="stSidebar"] > div:first-child { padding-top: .8rem; }
  [data-testid="stSidebar"] h3 {
    margin: 1.15rem 0 .45rem;
    color: var(--bz-muted-2);
    font-family: "Cascadia Mono", Consolas, monospace;
    font-size: .69rem;
    font-weight: 800;
    letter-spacing: .16em;
    text-transform: uppercase;
  }
  .bz-side-brand {
    display: flex; align-items: center; gap: 11px; margin: 0 0 13px;
    padding: 5px 3px 15px; border-bottom: 1px solid var(--bz-line);
  }
  .bz-mark {
    width: 35px; height: 35px; display: grid; place-items: center;
    border: 1px solid rgba(55,232,194,.32); border-radius: 10px;
    color: var(--bz-mint); background: linear-gradient(145deg, rgba(55,232,194,.14), rgba(124,92,255,.12));
    box-shadow: inset 0 0 18px rgba(55,232,194,.08); font-weight: 950;
  }
  .bz-side-brand strong { display:block; font-size:.93rem; letter-spacing:.14em; }
  .bz-side-brand small { display:block; margin-top:2px; color:var(--bz-mint); font-size:.57rem; font-weight:800; letter-spacing:.17em; }
  .bz-workspace {
    display:grid; grid-template-columns:34px 1fr auto; align-items:center; gap:9px;
    padding:9px; margin:0 0 11px; border:1px solid var(--bz-line); border-radius:10px;
    background:linear-gradient(130deg,rgba(55,232,194,.055),rgba(124,92,255,.035));
  }
  .bz-workspace-icon { width:32px; height:32px; display:grid; place-items:center; border-radius:8px; color:#061611; background:var(--bz-mint); font-weight:950; }
  .bz-workspace strong { display:block; color:var(--bz-text); font-size:.67rem; }
  .bz-workspace small { display:block; margin-top:2px; color:var(--bz-muted); font-size:.57rem; }
  .bz-workspace > span { color:var(--bz-muted-2); font-size:.72rem; }
  .bz-system-card { margin-top:14px; padding:11px; border:1px solid var(--bz-line); border-radius:11px; background:rgba(17,28,51,.58); }
  .bz-system-title { display:flex; align-items:center; gap:7px; margin-bottom:8px; color:var(--bz-muted); font:800 .58rem/1 "Cascadia Mono",Consolas,monospace; letter-spacing:.14em; }
  .bz-system-grid { display:grid; grid-template-columns:1fr auto; gap:6px 10px; color:var(--bz-muted); font-size:.59rem; }
  .bz-system-grid b { color:var(--bz-text); font-weight:700; text-align:right; }
  .bz-system-grid b.live { color:var(--bz-mint); }

  /* Native controls, made to read as one dark operations console. */
  [data-baseweb="radio"] > div,
  [data-baseweb="select"] > div,
  [data-baseweb="base-input"],
  [data-testid="stFileUploaderDropzone"] {
    border-color: var(--bz-line) !important;
    background: rgba(255,255,255,.025) !important;
    border-radius: 10px !important;
  }
  [data-testid="stFileUploaderDropzone"] { border-style: dashed !important; }
  div[role="radiogroup"] { gap: .35rem; }
  div[role="radiogroup"] label {
    border: 1px solid transparent; border-radius: 9px; padding: .34rem .48rem;
  }
  div[role="radiogroup"] label:hover { background: rgba(55,232,194,.045); border-color: var(--bz-line); }
  .stButton > button, .stDownloadButton > button {
    min-height: 2.35rem; border-radius: 9px; border: 1px solid var(--bz-line-strong);
    color: var(--bz-text); background: rgba(255,255,255,.035);
    font-size: .78rem; font-weight: 750; transition: .18s ease;
  }
  .stButton > button:hover, .stDownloadButton > button:hover {
    border-color: rgba(55,232,194,.45); color: var(--bz-mint);
    background: rgba(55,232,194,.08); transform: translateY(-1px);
  }
  .stButton > button[kind="primary"] {
    color: #031611; border-color: rgba(55,232,194,.45);
    background: linear-gradient(135deg, #50f3cf, #27cfae);
    box-shadow: 0 8px 24px rgba(55,232,194,.14);
  }
  .stButton > button[kind="primary"]:hover { color:#031611; box-shadow:0 10px 30px rgba(55,232,194,.24); }
  textarea, input { color: var(--bz-text) !important; }
  textarea:focus, input:focus { border-color: var(--bz-mint) !important; box-shadow: 0 0 0 1px rgba(55,232,194,.2) !important; }

  /* Header, copied as a real component rather than a decorative title. */
  .bz-command-header {
    display:grid; grid-template-columns:minmax(220px,1fr) auto; align-items:center; gap:20px;
    min-height:74px; margin:0 0 14px; padding:13px 17px;
    border:1px solid var(--bz-line); border-radius:14px;
    background:linear-gradient(100deg, rgba(17,28,51,.93), rgba(10,17,32,.78));
    box-shadow:0 18px 55px rgba(0,0,0,.18); backdrop-filter:blur(20px);
  }
  .bz-eyebrow { color:var(--bz-mint); font:800 .62rem/1 "Cascadia Mono",Consolas,monospace; letter-spacing:.17em; }
  .bz-command-header h1 { margin:5px 0 2px; color:var(--bz-text); font:690 1.35rem/1.2 "Bahnschrift SemiCondensed","Segoe UI",sans-serif; letter-spacing:.015em; }
  .bz-command-header p { margin:0; color:var(--bz-muted); font-size:.7rem; }
  .bz-header-actions { display:flex; align-items:center; justify-content:flex-end; gap:8px; flex-wrap:wrap; }
  .bz-chip {
    display:inline-flex; align-items:center; gap:6px; height:31px; padding:0 9px;
    border:1px solid var(--bz-line); border-radius:8px; color:var(--bz-muted);
    background:rgba(255,255,255,.025); font:650 .64rem/1 "Cascadia Mono",Consolas,monospace;
  }
  .bz-chip.live { color:var(--bz-mint); border-color:rgba(55,232,194,.2); background:rgba(55,232,194,.055); }
  .bz-dot { width:6px; height:6px; border-radius:50%; background:var(--bz-mint); box-shadow:0 0 9px var(--bz-mint); animation:bz-pulse 2s ease-in-out infinite; }
  @keyframes bz-pulse { 50% { opacity:.45; box-shadow:0 0 16px var(--bz-mint); } }

  .bz-status-ribbon {
    display:grid; grid-template-columns:1.25fr 1fr 1fr auto; align-items:center; gap:18px;
    min-height:58px; margin:0 0 18px; padding:10px 14px;
    border:1px solid rgba(55,232,194,.13); border-radius:14px;
    background:linear-gradient(90deg, rgba(55,232,194,.06), rgba(124,92,255,.045), transparent);
  }
  .bz-pulse-unit { display:flex; align-items:center; gap:10px; min-width:0; }
  .bz-orb { width:31px; height:31px; position:relative; flex:0 0 auto; border:1px solid rgba(55,232,194,.35); border-radius:50%; }
  .bz-orb:before,.bz-orb:after { content:""; position:absolute; border:1px solid rgba(55,232,194,.22); border-radius:50%; animation:bz-ring 2s infinite; }
  .bz-orb:before { inset:5px; }.bz-orb:after { inset:10px; animation-delay:.45s; }
  @keyframes bz-ring { 50% { opacity:.12; transform:scale(1.45); } }
  .bz-pulse-copy strong { display:block; color:var(--bz-text); font-size:.73rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bz-pulse-copy small { display:block; margin-top:3px; color:var(--bz-muted); font-size:.62rem; }
  .bz-signal label { display:flex; justify-content:space-between; margin-bottom:7px; color:var(--bz-muted); font-size:.61rem; }
  .bz-track { height:4px; overflow:hidden; border-radius:10px; background:rgba(255,255,255,.055); }
  .bz-track span { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--bz-mint),var(--bz-cyan)); box-shadow:0 0 12px rgba(55,232,194,.5); }
  .bz-track.risk span { background:linear-gradient(90deg,var(--bz-amber),var(--bz-pink)); }
  .bz-mode { color:var(--bz-mint); font:800 .64rem/1 "Cascadia Mono",Consolas,monospace; letter-spacing:.08em; }

  .bz-section {
    display:flex; align-items:end; justify-content:space-between; gap:16px;
    margin:1.25rem 0 .7rem; padding-bottom:.7rem; border-bottom:1px solid var(--bz-line);
  }
  .bz-section .kicker { color:var(--bz-mint); font:800 .6rem/1 "Cascadia Mono",Consolas,monospace; letter-spacing:.15em; }
  .bz-section h2 { margin:5px 0 0; color:var(--bz-text); font:680 1.03rem/1.2 "Bahnschrift SemiCondensed","Segoe UI",sans-serif; }
  .bz-section p { margin:0; max-width:620px; color:var(--bz-muted); font-size:.69rem; text-align:right; }

  /* Metric cards, panels, tabs and tables from the supplied frontend. */
  [data-testid="stMetric"] {
    min-height:112px; position:relative; padding:15px !important;
    border:1px solid var(--bz-line); border-radius:12px;
    background:linear-gradient(145deg, rgba(17,28,51,.9), rgba(10,17,32,.76));
    overflow:hidden; transition:transform .18s,border-color .18s;
  }
  [data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:var(--bz-line-strong); }
  [data-testid="stMetric"]:after { content:""; position:absolute; width:95px; height:95px; right:-48px; top:-50px; border-radius:50%; background:var(--bz-cyan); opacity:.055; filter:blur(13px); }
  [data-testid="stMetricLabel"] { color:var(--bz-muted); font-size:.69rem; font-weight:700; letter-spacing:.045em; }
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] p {
    color:var(--bz-text); font-size:clamp(1.2rem,1.6vw,1.65rem) !important;
    line-height:1.12 !important; font-variant-numeric:tabular-nums; letter-spacing:-.035em;
  }
  [data-testid="stMetricDelta"] { font-size:.65rem; }
  [data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--bz-line); }
  [data-testid="stTabs"] button { color:var(--bz-muted); font-size:.71rem; font-weight:700; letter-spacing:.025em; }
  [data-testid="stTabs"] button[aria-selected="true"] { color:var(--bz-mint); }
  [data-testid="stExpander"] { border:1px solid var(--bz-line); border-radius:12px; background:rgba(14,22,42,.52); }
  [data-testid="stDataFrame"] { border:1px solid var(--bz-line); border-radius:12px; overflow:hidden; }
  [data-testid="stAlert"] { border-radius:12px; border:1px solid var(--bz-line); background:rgba(17,28,51,.84); }
  [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--bz-line) !important; }
  hr { border-color:var(--bz-line) !important; }
  h1,h2,h3,h4 { color:var(--bz-text); }
  h1,h2 { font-family:"Bahnschrift SemiCondensed","Segoe UI",sans-serif !important; }
  p, label, [data-testid="stCaptionContainer"] { color:var(--bz-muted); }
  code { color:var(--bz-mint); background:rgba(55,232,194,.07); }

  .bz-agent-strip { display:grid; grid-template-columns:repeat(6,1fr); gap:7px; margin:7px 0 16px; }
  .bz-agent-pill { padding:8px 7px; border:1px solid var(--bz-line); border-radius:9px; color:var(--bz-muted); background:rgba(255,255,255,.018); text-align:center; font:700 .58rem/1.25 "Cascadia Mono",Consolas,monospace; letter-spacing:.04em; }
  .bz-agent-pill b { display:block; color:var(--bz-mint); font-size:.66rem; margin-bottom:2px; }

  /* Post-analysis workspace, based on the supplied AI Actuary reference screens. */
  .bz-workspace-header {
    position:sticky; top:.25rem; z-index:20; display:flex; align-items:center; justify-content:space-between; gap:18px;
    margin:-.35rem 0 12px; padding:10px 14px; min-height:61px;
    border:1px solid var(--bz-line); border-radius:13px; background:rgba(8,14,29,.92);
    box-shadow:0 14px 45px rgba(0,0,0,.22); backdrop-filter:blur(18px);
  }
  .bz-workspace-title { display:flex; align-items:center; gap:13px; min-width:0; }
  .bz-menu-glyph { color:var(--bz-muted); font:700 1.35rem/1 monospace; }
  .bz-page-code { color:var(--bz-cyan); font:800 .52rem/1 "Cascadia Mono",Consolas,monospace; letter-spacing:.16em; }
  .bz-workspace-title h1 { margin:3px 0 1px; font-size:1.08rem; line-height:1.15; }
  .bz-workspace-title p { margin:0; color:var(--bz-muted); font-size:.65rem; }
  .bz-workspace-actions { display:flex; align-items:center; justify-content:flex-end; gap:7px; min-width:0; }
  .bz-chip.offline { color:var(--bz-amber); border-color:rgba(255,191,63,.22); }
  .bz-chip.model { max-width:195px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .bz-portfolio { max-width:190px; padding-left:10px; border-left:1px solid var(--bz-line); color:var(--bz-text); font-size:.68rem; font-weight:750; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

  .bz-portfolio-bar {
    display:grid; grid-template-columns:minmax(190px,1fr) minmax(260px,2.1fr) auto auto auto; align-items:center; gap:15px;
    padding:10px 13px; margin-bottom:15px; border:1px solid var(--bz-line); border-radius:12px;
    background:linear-gradient(90deg,rgba(53,199,255,.055),rgba(55,232,194,.035),transparent);
  }
  .bz-portfolio-bar > div:first-child span { display:block; color:var(--bz-cyan); font:800 .5rem/1 monospace; letter-spacing:.14em; }
  .bz-portfolio-bar > div:first-child strong { display:block; margin-top:4px; color:var(--bz-text); font-size:.72rem; }
  .bz-portfolio-bar p { margin:0; color:var(--bz-muted); font-size:.62rem; }
  .bz-portfolio-facts { color:var(--bz-muted); font-size:.58rem; white-space:nowrap; }
  .bz-portfolio-facts b { display:block; color:var(--bz-text); font-size:.72rem; font-variant-numeric:tabular-nums; }
  .bz-mode-pill { padding:7px 9px; color:var(--bz-mint); border:1px solid rgba(55,232,194,.23); border-radius:999px; background:rgba(55,232,194,.06); font:800 .53rem/1 monospace; }

  .bz-kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:4px 0 18px; }
  .bz-kpi-card { min-height:112px; position:relative; overflow:hidden; padding:15px; border:1px solid var(--bz-line); border-radius:13px; background:linear-gradient(145deg,rgba(17,28,51,.94),rgba(10,17,32,.84)); }
  .bz-kpi-card:after { content:""; position:absolute; width:100px; height:100px; right:-48px; top:-50px; border-radius:50%; background:var(--accent,var(--bz-cyan)); opacity:.09; filter:blur(14px); }
  .bz-kpi-card.cyan { --accent:var(--bz-cyan); }.bz-kpi-card.green { --accent:var(--bz-green); }.bz-kpi-card.amber { --accent:var(--bz-amber); }.bz-kpi-card.red { --accent:var(--bz-red); }
  .bz-kpi-top { display:flex; align-items:center; gap:8px; color:var(--bz-muted); font-size:.61rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
  .bz-kpi-icon { width:29px; height:29px; display:grid; place-items:center; border-radius:8px; color:var(--accent); background:color-mix(in srgb,var(--accent) 13%,transparent); }
  .bz-kpi-card > strong { display:block; margin:12px 0 4px; color:var(--bz-text); font:730 1.45rem/1 "Cascadia Mono",Consolas,monospace; letter-spacing:-.05em; }
  .bz-kpi-card > small { color:var(--bz-muted); font-size:.59rem; }

  .bz-panel-heading { display:grid; grid-template-columns:36px 1fr auto; align-items:center; gap:10px; min-height:58px; margin:15px 0 0; padding:10px 12px; border:1px solid var(--bz-line); border-bottom:0; border-radius:13px 13px 0 0; background:rgba(17,28,51,.78); }
  .bz-panel-icon { width:34px; height:34px; display:grid; place-items:center; border-radius:9px; color:var(--bz-cyan); background:rgba(53,199,255,.11); font-weight:900; }
  .bz-panel-icon.green { color:var(--bz-green); background:rgba(55,232,166,.11); }.bz-panel-icon.amber { color:var(--bz-amber); background:rgba(255,191,63,.11); }.bz-panel-icon.red { color:var(--bz-red); background:rgba(255,82,119,.11); }
  .bz-panel-heading h3 { margin:0; font-size:.76rem; }.bz-panel-heading p { margin:2px 0 0; color:var(--bz-muted); font-size:.6rem; }
  .bz-panel-badge { padding:5px 8px; border:1px solid rgba(53,199,255,.3); border-radius:999px; color:var(--bz-cyan); font:750 .55rem/1 monospace; }
  .bz-panel-badge.green { color:var(--bz-green); border-color:rgba(55,232,166,.3); }.bz-panel-badge.amber { color:var(--bz-amber); border-color:rgba(255,191,63,.3); }.bz-panel-badge.red { color:var(--bz-red); border-color:rgba(255,82,119,.3); }

  .bz-risk-list { padding:10px; border:1px solid var(--bz-line); border-radius:0 0 13px 13px; background:rgba(14,22,42,.62); }
  .bz-risk-row { display:flex; align-items:center; justify-content:space-between; gap:14px; padding:10px 11px; margin:5px 0; border:1px solid rgba(137,159,204,.08); border-radius:9px; background:rgba(6,10,22,.28); }
  .bz-risk-row > span { color:var(--bz-muted); font-size:.64rem; }.bz-risk-row small { margin-left:7px; padding:2px 5px; border:1px solid var(--bz-line); border-radius:999px; font-size:.48rem; }
  .bz-risk-row b { color:var(--bz-text); font:720 .72rem/1 monospace; font-variant-numeric:tabular-nums; }

  .bz-agent-network { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:9px; padding:12px; border:1px solid var(--bz-line); border-radius:0 0 13px 13px; background:rgba(14,22,42,.62); }
  .bz-agent-card { display:grid; grid-template-columns:34px 1fr 7px auto; align-items:center; gap:9px; min-height:77px; padding:10px; border:1px solid rgba(137,159,204,.09); border-radius:10px; background:rgba(6,10,22,.25); }
  .bz-agent-icon { width:32px; height:32px; display:grid; place-items:center; border-radius:8px; color:var(--bz-cyan); background:rgba(53,199,255,.1); font:800 .62rem/1 monospace; }
  .bz-agent-copy strong,.bz-agent-copy span,.bz-agent-copy small { display:block; }.bz-agent-copy strong { color:var(--bz-text); font-size:.68rem; }.bz-agent-copy span { margin-top:2px; color:var(--bz-muted); font-size:.55rem; }.bz-agent-copy small { margin-top:6px; color:var(--bz-muted-2); font:500 .53rem/1.25 monospace; }
  .bz-agent-card > i { width:7px; height:7px; border-radius:50%; background:var(--bz-muted-2); }.bz-agent-card > em { color:var(--bz-muted-2); font:700 .5rem/1 monospace; text-transform:uppercase; font-style:normal; }
  .bz-agent-card.done { border-color:rgba(55,232,166,.2); }.bz-agent-card.done > i { background:var(--bz-green); box-shadow:0 0 10px var(--bz-green); }.bz-agent-card.done > em { color:var(--bz-green); }
  .bz-agent-card.error { border-color:rgba(255,82,119,.22); }.bz-agent-card.error > i { background:var(--bz-red); }.bz-agent-card.error > em { color:var(--bz-red); }

  .bz-trace-list { padding:12px 15px; border:1px solid var(--bz-line); border-radius:0 0 13px 13px; background:rgba(14,22,42,.62); }
  .bz-trace-step { position:relative; display:grid; grid-template-columns:29px 1fr 10px; align-items:center; gap:10px; min-height:61px; }
  .bz-trace-step:not(:last-child):after { content:""; position:absolute; left:14px; top:45px; bottom:-15px; width:1px; background:var(--bz-line); }
  .bz-trace-step > span { width:29px; height:29px; display:grid; place-items:center; border:1px solid var(--bz-line); border-radius:50%; color:var(--bz-muted); background:var(--bz-panel-2); font:700 .58rem/1 monospace; }
  .bz-trace-step strong { display:block; color:var(--bz-text); font-size:.66rem; }.bz-trace-step p { margin:3px 0 0; color:var(--bz-muted); font-size:.57rem; }
  .bz-trace-step > i { width:8px; height:8px; border-radius:50%; background:var(--bz-muted-2); }.bz-trace-step.done > i,.bz-trace-step.ready > i { background:var(--bz-green); box-shadow:0 0 9px rgba(55,232,166,.6); }.bz-trace-step.attention > i { background:var(--bz-amber); }

  .bz-layer-stack { padding:14px 15px 17px; border:1px solid var(--bz-line); border-radius:0 0 13px 13px; background:rgba(14,22,42,.62); }
  .bz-layer-label { display:flex; justify-content:space-between; gap:10px; margin:9px 0 6px; }.bz-layer-label strong { color:var(--bz-text); font-size:.65rem; }.bz-layer-label span { color:var(--bz-muted); font:500 .56rem/1 monospace; }
  .bz-layer-track { height:42px; overflow:hidden; border-radius:8px; background:rgba(6,10,22,.55); }
  .bz-layer { height:100%; display:flex; align-items:center; justify-content:center; padding:0 8px; border:1px dashed var(--bz-line-strong); border-radius:8px; color:var(--bz-muted); background:rgba(137,159,204,.05); font:700 .57rem/1 monospace; white-space:nowrap; }
  .bz-layer.retained { color:var(--bz-red); border-color:rgba(255,82,119,.45); background:rgba(255,82,119,.16); }.bz-layer.ceded { color:var(--bz-green); border-color:rgba(55,232,166,.4); background:rgba(55,232,166,.15); }.bz-layer.available { color:var(--bz-muted); }
  .bz-layer-flow { display:grid; grid-template-columns:1fr auto 1fr auto 1fr; align-items:center; gap:12px; margin-top:18px; padding-top:16px; border-top:1px solid var(--bz-line); text-align:center; }
  .bz-layer-flow span { display:block; color:var(--bz-muted); font-size:.5rem; letter-spacing:.12em; }.bz-layer-flow b { display:block; margin-top:5px; color:var(--bz-red); font:750 1rem/1 monospace; }.bz-layer-flow b.green { color:var(--bz-green); }.bz-layer-flow b.amber { color:var(--bz-amber); }.bz-layer-flow em { color:var(--bz-muted-2); font-style:normal; font-size:1.25rem; }

  .bz-clause-card { margin:0 0 8px; padding:11px 12px; border:1px solid var(--bz-line); border-radius:10px; background:rgba(17,28,51,.65); }.bz-clause-card.selected { border-color:rgba(53,199,255,.42); box-shadow:0 0 22px rgba(53,199,255,.06); }
  .bz-clause-card > div { display:flex; justify-content:space-between; gap:8px; }.bz-clause-card > div span { color:var(--bz-cyan); font-size:.55rem; font-weight:800; }.bz-clause-card > div small { color:var(--bz-muted-2); font-size:.52rem; }
  .bz-clause-card p { margin:8px 0; color:var(--bz-text); font-size:.61rem; line-height:1.48; }.bz-clause-card footer { color:var(--bz-muted); font:500 .52rem/1.3 monospace; }
  .bz-evidence-detail { padding:14px; border:1px solid var(--bz-line); border-radius:0 0 13px 13px; background:rgba(14,22,42,.62); }
  .bz-evidence-head { display:flex; align-items:center; justify-content:space-between; color:var(--bz-cyan); font:800 .55rem/1 monospace; letter-spacing:.1em; }.bz-evidence-head b { color:var(--bz-green); }
  .bz-evidence-detail blockquote { margin:14px 0; padding:14px; border:1px solid rgba(137,159,204,.08); border-left:2px solid var(--bz-cyan); border-radius:8px; color:var(--bz-text); background:rgba(6,10,22,.3); font-size:.67rem; line-height:1.6; }
  .bz-evidence-grid { display:grid; grid-template-columns:1fr 1fr; gap:9px; }.bz-evidence-grid > div { padding:10px; border:1px solid rgba(137,159,204,.08); border-radius:8px; }.bz-evidence-grid span { display:block; color:var(--bz-muted-2); font-size:.49rem; letter-spacing:.1em; }.bz-evidence-grid b { display:block; margin-top:5px; color:var(--bz-text); font-size:.59rem; }
  .bz-evidence-detail footer { margin-top:12px; color:var(--bz-muted-2); font:500 .52rem/1.4 monospace; }

  .bz-decision { margin:0 0 14px; padding:15px 17px; border:1px solid rgba(55,232,194,.22); border-radius:12px; background:linear-gradient(125deg,rgba(55,232,194,.075),rgba(53,199,255,.035)); box-shadow:0 0 35px rgba(55,232,194,.035); }
  .bz-decision-label { margin-bottom:8px; color:var(--bz-mint); font:800 .54rem/1 monospace; letter-spacing:.13em; }

  /* The first sidebar radio is the workspace navigation. */
  [data-testid="stSidebar"] div[role="radiogroup"] label { width:100%; min-height:38px; align-items:center; padding:.45rem .55rem; }
  [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) { border-color:rgba(53,199,255,.16); background:rgba(53,199,255,.09); color:var(--bz-text); }
  [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):before { content:""; width:3px; height:24px; margin-left:-.58rem; border-radius:0 4px 4px 0; background:var(--bz-cyan); box-shadow:0 0 10px rgba(53,199,255,.6); }

  @media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
    .bz-command-header { grid-template-columns:1fr; }
    .bz-header-actions { justify-content:flex-start; }
    .bz-status-ribbon { grid-template-columns:1fr 1fr; }
    .bz-agent-strip { grid-template-columns:repeat(3,1fr); }
    .bz-workspace-header { position:relative; top:0; align-items:flex-start; }
    .bz-workspace-actions { max-width:52%; flex-wrap:wrap; }
    .bz-portfolio-bar { grid-template-columns:1fr 1fr; }
    .bz-portfolio-bar p { grid-column:1 / -1; grid-row:2; }
    .bz-kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
  }
  @media (max-width: 620px) {
    .bz-status-ribbon { grid-template-columns:1fr; }
    .bz-signal { display:none; }
    .bz-section { display:block; }
    .bz-section p { margin-top:6px; text-align:left; }
    .bz-agent-strip { grid-template-columns:repeat(2,1fr); }
    .bz-workspace-header { display:block; }
    .bz-workspace-actions { max-width:none; justify-content:flex-start; margin-top:10px; }
    .bz-workspace-actions .model,.bz-portfolio { display:none; }
    .bz-portfolio-bar { grid-template-columns:1fr auto; }
    .bz-portfolio-bar p,.bz-portfolio-facts { display:none; }
    .bz-kpi-grid,.bz-agent-network { grid-template-columns:1fr; }
    .bz-layer-label { display:block; }.bz-layer-label span { display:block; margin-top:3px; }
    .bz-layer-flow { grid-template-columns:1fr; }.bz-layer-flow em { transform:rotate(90deg); }
    .bz-evidence-grid { grid-template-columns:1fr; }
  }
  @media (prefers-reduced-motion: reduce) {
    *, *:before, *:after { animation-duration:.001ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; }
  }
</style>
"""


def apply_theme() -> None:
    """Inject the application-wide visual system once per Streamlit rerun."""
    st.markdown(BIZMAX_THEME, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        '<div class="bz-side-brand"><div class="bz-mark">B</div><div>'
        '<strong>BIZMAX</strong><small>RISK COMMAND</small></div></div>'
        '<div class="bz-workspace"><div class="bz-workspace-icon">R</div><div>'
        '<strong>Risk Intelligence Lab</strong><small>4 portfolios · 6 specialists</small>'
        '</div><span>⌄</span></div>',
        unsafe_allow_html=True,
    )


def sidebar_system_card(*, model: str, traced: bool) -> None:
    trace_label = "CONNECTED" if traced else "OFFLINE"
    trace_class = "live" if traced else ""
    st.markdown(
        f'<div class="bz-system-card"><div class="bz-system-title"><i class="bz-dot"></i>SYSTEM STATUS</div>'
        f'<div class="bz-system-grid"><span>LLM</span><b>{escape(model)}</b>'
        f'<span>Actuarial engine</span><b>READY</b><span>ML models</span><b>2 LOADED</b>'
        f'<span>PRISM</span><b class="{trace_class}">{trace_label}</b></div></div>',
        unsafe_allow_html=True,
    )


def command_header(*, mode: str, model: str, traced: bool) -> None:
    trace_label = "PRISM LIVE" if traced else "PRISM OFF"
    trace_class = "live" if traced else ""
    st.markdown(
        f"""
        <header class="bz-command-header">
          <div>
            <div class="bz-eyebrow">AI ACTUARIAL INTELLIGENCE</div>
            <h1>Risk command center</h1>
            <p>Claims, capital, policy and reinsurance evidence reconciled in one decision surface.</p>
          </div>
          <div class="bz-header-actions">
            <span class="bz-chip {trace_class}"><i class="bz-dot"></i>{trace_label}</span>
            <span class="bz-chip">{escape(model)}</span>
            <span class="bz-chip">{escape(mode.upper())}</span>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def status_ribbon(*, business: str, source_count: int, claim_count: int, mode: str, severity_shift: float) -> None:
    readiness_pct = 100 if claim_count else 0
    risk_pct = max(4, min(100, int(abs(severity_shift))))
    st.markdown(
        f"""
        <section class="bz-status-ribbon">
          <div class="bz-pulse-unit">
            <div class="bz-orb"></div>
            <div class="bz-pulse-copy"><strong>{escape(business)}</strong>
              <small>{claim_count:,} claims · {source_count} source{'s' if source_count != 1 else ''} online</small></div>
          </div>
          <div class="bz-signal"><label><span>Claims model readiness</span><b>{readiness_pct}%</b></label>
            <div class="bz-track"><span style="width:{readiness_pct}%"></span></div></div>
          <div class="bz-signal"><label><span>Latest severity shift</span><b>{severity_shift:+.1f}%</b></label>
            <div class="bz-track risk"><span style="width:{risk_pct}%"></span></div></div>
          <div class="bz-mode">{escape(mode.upper())} MODE</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="bz-section"><div><div class="kicker">{escape(kicker)}</div>'
        f'<h2>{escape(title)}</h2></div><p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def agent_strip() -> None:
    agents = (
        ("01", "ACTUARY"), ("02", "CLAIMS"), ("03", "REINSURANCE"),
        ("04", "CAPITAL"), ("05", "RISK"), ("06", "POLICY"),
    )
    boxes = "".join(f'<div class="bz-agent-pill"><b>{n}</b>{name}</div>' for n, name in agents)
    st.markdown(f'<div class="bz-agent-strip">{boxes}</div>', unsafe_allow_html=True)
