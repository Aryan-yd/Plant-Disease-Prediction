import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroVision AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — design tokens + all component styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ══════════════════════════════════════════════════════
   DESIGN TOKENS
══════════════════════════════════════════════════════ */
:root {
    --bg:           #0a1a0e;
    --bg2:          #0d1f11;
    --bg3:          #122016;
    --surface:      #162b1a;
    --sidebar-bg:   #0b1a0f;
    --border:       rgba(255,255,255,.07);
    --border-green: rgba(74,180,100,.28);

    --green:        #4ab464;
    --green-bright: #6dd88a;
    --green-dim:    rgba(74,180,100,.12);
    --green-glow:   rgba(74,180,100,.18);
    --amber:        #d4954a;
    --amber-dim:    rgba(212,149,74,.15);
    --red:          #c0524a;

    --txt:          #e0ede2;
    --txt-muted:    rgba(224,237,226,.5);
    --txt-faint:    rgba(224,237,226,.25);

    --font-serif:   'Cormorant Garamond', Georgia, serif;
    --font-sans:    'Outfit', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;

    --shadow-sm:    0 2px 10px rgba(0,0,0,.3);
    --shadow-md:    0 8px 32px rgba(0,0,0,.4);
    --shadow-lg:    0 20px 60px rgba(0,0,0,.55);

    --radius:       12px;
    --radius-lg:    18px;
}

/* ══════════════════════════════════════════════════════
   RESETS & GLOBAL
══════════════════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: var(--font-sans);
    color: var(--txt);
    background: var(--bg) !important;
}
.main { background: var(--bg) !important; }
.block-container { padding: 2rem 2.5rem 2rem 2rem !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ══════════════════════════════════════════════════════
   SIDEBAR — vertical nav matching reference design
══════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 0 !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 0 !important;
}

/* Hide the default sidebar toggle arrow */
button[data-testid="collapsedControl"] { display: none !important; }

/* ── Sidebar Logo Block ── */
.sb-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0;
}
.sb-logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #1a5c2a, #2e8b46);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    box-shadow: 0 4px 16px rgba(74,180,100,.25);
    flex-shrink: 0;
}
.sb-logo-text .sb-title {
    font-family: var(--font-serif);
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--green-bright);
    line-height: 1.1;
    letter-spacing: -.01em;
}
.sb-logo-text .sb-sub {
    font-size: .6rem;
    text-transform: uppercase;
    letter-spacing: .28em;
    color: var(--txt-faint);
    font-weight: 600;
    margin-top: 2px;
}

/* ── Nav Section Header ── */
.sb-nav-label {
    font-size: .58rem;
    text-transform: uppercase;
    letter-spacing: .28em;
    color: var(--txt-faint);
    font-weight: 700;
    padding: 20px 24px 8px;
}

/* ── Nav Buttons ── */
section[data-testid="stSidebar"] .stButton {
    margin: 2px 12px !important;
}
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    width: 100% !important;
    text-align: left !important;
    padding: 11px 16px !important;
    font-family: var(--font-sans) !important;
    font-size: .875rem !important;
    font-weight: 500 !important;
    color: var(--txt-muted) !important;
    letter-spacing: .01em !important;
    transition: all .18s ease !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    gap: 10px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--green-dim) !important;
    color: var(--txt) !important;
    transform: none !important;
}
section[data-testid="stSidebar"] .stButton > button:focus,
section[data-testid="stSidebar"] .stButton > button:active {
    background: var(--green-glow) !important;
    color: var(--green-bright) !important;
    box-shadow: none !important;
    border-left: 2px solid var(--green) !important;
}

/* Active nav item highlight */
.nav-active > button {
    background: var(--green-glow) !important;
    color: var(--green-bright) !important;
    border-left: 2px solid var(--green) !important;
}

/* ── Sidebar Status Badge ── */
.sb-status {
    position: absolute;
    bottom: 24px;
    left: 12px; right: 12px;
    background: var(--green-dim);
    border: 1px solid var(--border-green);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sb-status-dot {
    width: 8px; height: 8px;
    background: var(--green);
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: .4; transform: scale(.75); }
}
.sb-status-text {
    font-size: .72rem;
    color: var(--green);
    font-weight: 600;
    letter-spacing: .04em;
}

/* ══════════════════════════════════════════════════════
   PAGE HEADER (replaces topbar — no longer needed as strip)
══════════════════════════════════════════════════════ */
.page-header {
    margin-bottom: 28px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border);
}
.page-header-eyebrow {
    font-size: .6rem;
    text-transform: uppercase;
    letter-spacing: .28em;
    color: var(--green);
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.page-header-eyebrow::before {
    content: '';
    width: 20px; height: 1px;
    background: var(--green);
}
.page-header-title {
    font-family: var(--font-serif);
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--txt);
    line-height: 1.1;
    letter-spacing: -.02em;
}
.page-header-title em { color: var(--green-bright); font-style: italic; }

/* ══════════════════════════════════════════════════════
   HERO BANNER
══════════════════════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, var(--bg3) 0%, var(--surface) 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 48px 44px 44px;
    position: relative;
    overflow: hidden;
    margin-bottom: 28px;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 55% 70% at 95% 50%, rgba(74,180,100,.1) 0%, transparent 60%),
        radial-gradient(ellipse 35% 45% at 5% 85%, rgba(212,149,74,.07) 0%, transparent 50%);
    pointer-events: none;
}
.hero-line {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, transparent, var(--green) 25%, var(--green) 75%, transparent);
    opacity: .5;
    border-radius: 0 2px 2px 0;
}
.hero-content { padding-left: 20px; position: relative; }
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 10px;
    font-size: .64rem; text-transform: uppercase; letter-spacing: .3em;
    color: var(--green); font-weight: 700; margin-bottom: 16px;
}
.hero-eyebrow-line { width: 24px; height: 1px; background: var(--green); }
.hero-h1 {
    font-family: var(--font-serif);
    font-size: clamp(2.2rem, 4vw, 3.4rem);
    font-weight: 700; color: var(--txt);
    line-height: 1.08; letter-spacing: -.02em; margin-bottom: 18px;
}
.hero-h1 em { color: var(--green-bright); font-style: italic; }
.hero-desc {
    font-size: .97rem; color: var(--txt-muted); line-height: 1.75;
    max-width: 500px; font-weight: 300; margin-bottom: 28px;
}
.hero-tags { display: flex; flex-wrap: wrap; gap: 8px; }
.hero-tag {
    background: rgba(0,0,0,.25);
    border: 1px solid var(--border);
    color: var(--txt-muted);
    padding: 5px 14px; border-radius: 100px;
    font-size: .73rem; font-weight: 500; letter-spacing: .03em;
}

/* ══════════════════════════════════════════════════════
   STAT CARDS
══════════════════════════════════════════════════════ */
.stat-row {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 14px; margin: 24px 0;
}
.stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 26px 22px;
    position: relative; overflow: hidden;
    transition: border-color .2s, transform .2s;
}
.stat-card:hover { border-color: var(--border-green); transform: translateY(-2px); }
.stat-card:nth-child(1)::after { background: linear-gradient(90deg, var(--amber), #e8b87a); }
.stat-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: linear-gradient(90deg, var(--green), var(--green-bright));
}
.stat-n {
    font-family: var(--font-serif); font-size: 2.8rem; font-weight: 700;
    color: var(--green-bright); line-height: 1; margin-bottom: 8px;
}
.stat-l {
    font-size: .67rem; text-transform: uppercase; letter-spacing: .18em;
    color: var(--txt-faint); font-weight: 600;
}

/* ══════════════════════════════════════════════════════
   INFO CARDS
══════════════════════════════════════════════════════ */
.icard {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 26px; height: 100%;
    transition: border-color .2s;
}
.icard:hover { border-color: var(--border-green); }
.icard h3 { font-family: var(--font-serif); font-size: 1.1rem; font-weight: 700; color: var(--txt); margin-bottom: 10px; }
.icard p, .icard li { font-size: .86rem; color: var(--txt-muted); line-height: 1.72; }
.icard li { margin-bottom: 4px; }
.icard strong { color: var(--txt); font-weight: 600; }

/* ══════════════════════════════════════════════════════
   HOW IT WORKS
══════════════════════════════════════════════════════ */
.hiw {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius-lg); padding: 36px; margin: 24px 0;
    position: relative; overflow: hidden;
}
.hiw::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(-45deg, rgba(255,255,255,.008) 0px, rgba(255,255,255,.008) 1px, transparent 1px, transparent 24px);
    pointer-events: none;
}
.hiw-title {
    font-family: var(--font-serif); font-size: 1.45rem; font-weight: 700;
    color: var(--txt); text-align: center; margin-bottom: 28px;
}
.steps {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; position: relative;
}
.steps::before {
    content: ''; position: absolute; top: 26px; left: 17%; right: 17%;
    height: 1px; border-top: 2px dashed rgba(74,180,100,.2);
}
.step {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px 18px 20px;
    text-align: center; transition: border-color .2s, transform .2s;
}
.step:hover { border-color: var(--border-green); transform: translateY(-3px); }
.step-n {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--green), var(--green-bright));
    color: var(--bg); font-family: var(--font-serif); font-size: 1.2rem;
    font-weight: 700; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 14px; box-shadow: 0 4px 14px rgba(74,180,100,.25);
}
.step-t { font-weight: 700; font-size: .88rem; color: var(--txt); margin-bottom: 7px; }
.step-d { font-size: .78rem; color: var(--txt-muted); line-height: 1.55; }

/* ══════════════════════════════════════════════════════
   CTA BANNER
══════════════════════════════════════════════════════ */
.cta-wrap {
    background: linear-gradient(120deg, #102414 0%, #183020 50%, #0e1e10 100%);
    border: 1px solid rgba(74,180,100,.22);
    border-radius: var(--radius-lg); padding: 32px 40px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 18px; margin: 24px 0; position: relative; overflow: hidden;
}
.cta-wrap::before {
    content: ''; position: absolute; right: -50px; top: -50px;
    width: 180px; height: 180px; border-radius: 50%;
    border: 1px solid rgba(74,180,100,.1);
}
.cta-wrap h3 {
    font-family: var(--font-serif); font-size: 1.45rem; font-weight: 700;
    color: var(--txt); margin-bottom: 5px;
}
.cta-wrap p { color: var(--txt-muted); font-size: .86rem; line-height: 1.5; }
.cta-pill {
    background: linear-gradient(135deg, var(--green), var(--green-bright));
    color: var(--bg); font-weight: 700; font-size: .82rem;
    padding: 12px 28px; border-radius: 100px; letter-spacing: .06em;
    white-space: nowrap; box-shadow: 0 6px 22px rgba(74,180,100,.3);
}

/* ══════════════════════════════════════════════════════
   ABOUT — SPECIES GRID
══════════════════════════════════════════════════════ */
.species-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 14px; margin-top: 8px;
}
.sp-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px 18px 16px;
    transition: border-color .2s, transform .2s; position: relative; overflow: hidden;
}
.sp-card:hover { border-color: var(--border-green); transform: translateY(-3px); }
.sp-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px; background: linear-gradient(180deg, var(--green), var(--green-bright));
}
.sp-icon { font-size: 1.5rem; margin-bottom: 7px; padding-left: 4px; }
.sp-name {
    font-family: var(--font-serif); font-size: 1.05rem; font-weight: 700;
    color: var(--txt); margin-bottom: 7px; padding-left: 4px;
}
.sp-counts { display: flex; gap: 7px; margin-bottom: 10px; padding-left: 4px; flex-wrap: wrap; }
.sp-badge {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 3px 8px; border-radius: 100px;
    font-size: .65rem; font-weight: 700; letter-spacing: .05em;
}
.sp-badge-g { background: rgba(74,180,100,.12); color: var(--green); border: 1px solid rgba(74,180,100,.2); }
.sp-badge-r { background: rgba(192,82,74,.12); color: #e07878; border: 1px solid rgba(192,82,74,.2); }
.sp-list { list-style: none; padding: 0; margin: 0; }
.sp-list li {
    font-size: .76rem; color: var(--txt-muted); padding: 4px 4px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 6px; line-height: 1.3;
}
.sp-list li:last-child { border-bottom: none; }
.dot-h { width: 5px; height: 5px; border-radius: 50%; background: var(--green); flex-shrink: 0; }
.dot-d { width: 5px; height: 5px; border-radius: 50%; background: var(--red); flex-shrink: 0; }

.acard {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 24px; height: 100%;
    transition: border-color .2s;
}
.acard:hover { border-color: var(--border-green); }
.acard h3 { font-family: var(--font-serif); font-size: 1.1rem; font-weight: 700; color: var(--txt); margin-bottom: 10px; }
.acard p, .acard li { font-size: .86rem; color: var(--txt-muted); line-height: 1.72; }
.acard strong { color: var(--txt); }

.tech-row { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 4px; }
.tech-pill {
    background: var(--bg3); color: var(--green-bright);
    padding: 6px 13px; border-radius: 8px;
    font-family: var(--font-mono); font-size: .73rem; font-weight: 500;
    border: 1px solid rgba(74,180,100,.15); letter-spacing: .02em;
}

/* ══════════════════════════════════════════════════════
   SCAN PAGE
══════════════════════════════════════════════════════ */
.scan-panel-header {
    background: linear-gradient(90deg, var(--bg3), var(--surface));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    padding: 20px 28px;
    display: flex; align-items: center; gap: 14px;
    border-bottom: 1px solid var(--border-green);
}
.scan-panel-icon { font-size: 1.7rem; }
.scan-panel-title { font-family: var(--font-serif); font-size: 1.25rem; font-weight: 700; color: var(--txt); }
.scan-panel-sub { font-size: .72rem; color: var(--txt-faint); margin-top: 2px; }

/* Upload zone override */
.stFileUploader > div > div {
    background: var(--surface) !important;
    border: 2px dashed rgba(74,180,100,.3) !important;
    border-radius: 12px !important;
}
.stFileUploader label, .stFileUploader span,
.stFileUploader p, .stFileUploader small,
.stFileUploader div { color: var(--txt-muted) !important; }
.stFileUploader button {
    background: var(--green) !important; color: var(--bg) !important;
    border: none !important; border-radius: 6px !important; font-weight: 600 !important;
}

.upload-deco {
    background: var(--green-dim); border: 2px dashed rgba(74,180,100,.3);
    border-radius: 12px; padding: 20px 16px; text-align: center; margin-bottom: 12px;
}
.ud-icon { font-size: 2rem; margin-bottom: 7px; }
.ud-text { font-size: .83rem; color: var(--txt-muted); font-weight: 500; }
.ud-sub { font-size: .7rem; color: var(--txt-faint); margin-top: 3px; }

.col-label {
    font-size: .62rem; text-transform: uppercase; letter-spacing: .24em;
    color: var(--green); font-weight: 700; margin-bottom: 7px;
    display: flex; align-items: center; gap: 6px;
}
.col-label::before { content: ''; width: 14px; height: 1px; background: var(--green); }
.col-title { font-family: var(--font-serif); font-size: 1.05rem; font-weight: 700; color: var(--txt); margin-bottom: 14px; }

.dcard { border-radius: var(--radius); overflow: hidden; margin: 10px 0; box-shadow: var(--shadow-md); border: 1px solid var(--border); }
.dcard-header { padding: 18px 22px; display: flex; align-items: center; gap: 14px; }
.dcard-header.healthy { background: linear-gradient(135deg, #0a3d22 0%, #145c34 100%); border-bottom: 1px solid rgba(74,180,100,.2); }
.dcard-header.diseased { background: linear-gradient(135deg, #3d0f0f 0%, #7a2020 100%); border-bottom: 1px solid rgba(192,82,74,.2); }
.dcard-emoji { font-size: 1.9rem; }
.dcard-badge { font-size: .58rem; text-transform: uppercase; letter-spacing: .2em; color: rgba(255,255,255,.5); font-weight: 700; margin-bottom: 3px; }
.dcard-name { font-family: var(--font-serif); font-size: 1.45rem; font-weight: 700; color: #fff; line-height: 1.1; }
.dcard-meta { background: var(--bg3); padding: 14px 22px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }
.dm-lbl { font-size: .58rem; text-transform: uppercase; letter-spacing: .15em; color: var(--txt-faint); font-weight: 700; margin-bottom: 4px; }
.dm-val { font-size: .95rem; font-weight: 700; color: var(--txt); }
.conf-track { background: rgba(255,255,255,.1); border-radius: 100px; height: 5px; margin-top: 7px; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, var(--green), var(--green-bright)); }

.await-state { text-align: center; padding: 44px 18px; color: var(--txt-faint); }
.await-icon { font-size: 2.6rem; margin-bottom: 14px; opacity: .5; }
.await-title { font-family: var(--font-serif); font-size: 1.05rem; font-weight: 600; color: var(--txt-muted); margin-bottom: 5px; }
.await-sub { font-size: .78rem; line-height: 1.6; }

.tips-row { display: flex; gap: 7px; flex-wrap: wrap; margin-top: 14px; }
.tip-chip {
    background: var(--green-dim); border: 1px solid rgba(74,180,100,.18);
    color: rgba(224,237,226,.6); padding: 4px 11px; border-radius: 100px;
    font-size: .7rem; font-weight: 500; display: flex; align-items: center; gap: 4px;
}
.tip-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--green-bright); flex-shrink: 0; }

.code-block { background: var(--bg3); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-top: 10px; }
.code-lbl { font-size: .58rem; text-transform: uppercase; letter-spacing: .18em; color: var(--txt-faint); font-weight: 700; margin-bottom: 7px; }
.code-val { font-family: var(--font-mono); font-size: .78rem; color: var(--green-bright); background: rgba(74,180,100,.07); padding: 5px 10px; border-radius: 6px; display: block; word-break: break-all; }

.mini-sp { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 10px 13px; margin: 4px 0; transition: border-color .18s; }
.mini-sp:hover { border-color: var(--border-green); }
.mini-sp-name { font-weight: 600; font-size: .85rem; color: var(--txt); }
.mini-sp-cnt { font-size: .7rem; color: var(--txt-faint); margin-top: 2px; }

/* ══════════════════════════════════════════════════════
   GLOBAL BUTTON OVERRIDES
══════════════════════════════════════════════════════ */
.stButton > button {
    border-radius: 8px !important; font-family: var(--font-sans) !important;
    font-weight: 600 !important; font-size: .86rem !important;
    transition: all .18s !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--green), var(--green-bright)) !important;
    border: none !important; color: var(--bg) !important;
    font-weight: 700 !important; box-shadow: 0 4px 14px rgba(74,180,100,.25) !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 22px rgba(74,180,100,.35) !important;
}

.stExpander {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
}
.stExpander summary { color: var(--txt) !important; font-weight: 600 !important; }
.stExpander p, .stExpander div { color: var(--txt-muted) !important; }

.divider { border: none; border-top: 1px solid var(--border); margin: 32px 0; }
.eyebrow { font-size: .62rem; text-transform: uppercase; letter-spacing: .26em; color: var(--green); font-weight: 700; margin-bottom: 5px; }
.section-title { font-family: var(--font-serif); font-size: 1.6rem; font-weight: 700; color: var(--txt); margin-bottom: 24px; line-height: 1.2; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
DISEASE_CLASSES = {
    "Apple":      {"Apple___Apple_scab":"Apple Scab","Apple___Black_rot":"Black Rot","Apple___Cedar_apple_rust":"Cedar Apple Rust","Apple___healthy":"Healthy"},
    "Blueberry":  {"Blueberry___healthy":"Healthy"},
    "Cherry":     {"Cherry_(including_sour)___Powdery_mildew":"Powdery Mildew","Cherry_(including_sour)___healthy":"Healthy"},
    "Corn":       {"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot":"Cercospora / Gray Leaf Spot","Corn_(maize)___Common_rust_":"Common Rust","Corn_(maize)___Northern_Leaf_Blight":"Northern Leaf Blight","Corn_(maize)___healthy":"Healthy"},
    "Grape":      {"Grape___Black_rot":"Black Rot","Grape___Esca_(Black_Measles)":"Esca (Black Measles)","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":"Leaf Blight","Grape___healthy":"Healthy"},
    "Orange":     {"Orange___Haunglongbing_(Citrus_greening)":"Huanglongbing (Citrus Greening)"},
    "Peach":      {"Peach___Bacterial_spot":"Bacterial Spot","Peach___healthy":"Healthy"},
    "Pepper":     {"Pepper,_bell___Bacterial_spot":"Bacterial Spot","Pepper,_bell___healthy":"Healthy"},
    "Potato":     {"Potato___Early_blight":"Early Blight","Potato___Late_blight":"Late Blight","Potato___healthy":"Healthy"},
    "Raspberry":  {"Raspberry___healthy":"Healthy"},
    "Soybean":    {"Soybean___healthy":"Healthy"},
    "Squash":     {"Squash___Powdery_mildew":"Powdery Mildew"},
    "Strawberry": {"Strawberry___Leaf_scorch":"Leaf Scorch","Strawberry___healthy":"Healthy"},
    "Tomato":     {"Tomato___Bacterial_spot":"Bacterial Spot","Tomato___Early_blight":"Early Blight","Tomato___Late_blight":"Late Blight","Tomato___Leaf_Mold":"Leaf Mold","Tomato___Septoria_leaf_spot":"Septoria Leaf Spot","Tomato___Spider_mites Two-spotted_spider_mite":"Spider Mites","Tomato___Target_Spot":"Target Spot","Tomato___Tomato_Yellow_Leaf_Curl_Virus":"Yellow Leaf Curl Virus","Tomato___Tomato_mosaic_virus":"Mosaic Virus","Tomato___healthy":"Healthy"},
}

FLAT_CLASSES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot',
    'Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight',
    'Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy',
    'Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

PLANT_ICONS = {
    "Apple":"🍎","Blueberry":"🫐","Cherry":"🍒","Corn":"🌽","Grape":"🍇",
    "Orange":"🍊","Peach":"🍑","Pepper":"🌶️","Potato":"🥔","Raspberry":"🍓",
    "Soybean":"🌱","Squash":"🎃","Strawberry":"🍓","Tomato":"🍅"
}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    confidence   = np.max(prediction) * 100
    return result_index, confidence


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — vertical nav matching reference screenshot
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-icon">🌿</div>
        <div class="sb-logo-text">
            <div class="sb-title">AgroVision AI</div>
            <div class="sb-sub">Disease Recognition</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Nav label
    st.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)

    # Nav buttons
    if st.button("🏠  Home", use_container_width=True):
        st.session_state.current_page = "Home"

    if st.button("🔬  Disease Recognition", use_container_width=True):
        st.session_state.current_page = "Disease Recognition"

    if st.button("ℹ️  About", use_container_width=True):
        st.session_state.current_page = "About"

    # Spacer + status badge
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-status">
        <div class="sb-status-dot"></div>
        <div class="sb-status-text">Model Ready · 38 Classes</div>
    </div>
    """, unsafe_allow_html=True)


app_mode = st.session_state.current_page


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if app_mode == "Home":

    st.markdown("""
    <div class="hero">
        <div class="hero-line"></div>
        <div class="hero-content">
            <div class="hero-eyebrow">
                <span class="hero-eyebrow-line"></span>
                AI-Powered Plant Pathology
            </div>
            <div class="hero-h1">
                Detect crop disease<br>before it <em>spreads.</em>
            </div>
            <div class="hero-desc">
                Upload a photo of any plant leaf — our deep learning CNN identifies
                the disease and confidence score within seconds.
            </div>
            <div class="hero-tags">
                <span class="hero-tag">🌱 13 Plant Species</span>
                <span class="hero-tag">🦠 38 Disease Classes</span>
                <span class="hero-tag">📸 87,000+ Training Images</span>
                <span class="hero-tag">⚡ Sub-second Inference</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        st.image("home_page.jpeg", use_column_width=True)
    except:
        pass

    st.markdown("""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-n">38</div>
            <div class="stat-l">Disease Classes</div>
        </div>
        <div class="stat-card">
            <div class="stat-n">87K+</div>
            <div class="stat-l">Training Images</div>
        </div>
        <div class="stat-card">
            <div class="stat-n">13</div>
            <div class="stat-l">Plant Species</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="icard">
            <h3>🎯 Our Mission</h3>
            <p>Equip farmers and agronomists with an instant, accurate disease-detection
            tool — turning a smartphone photo into actionable crop health intelligence.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="icard">
            <h3>⚡ Why AgroVision?</h3>
            <ul>
                <li><strong>Accuracy</strong> — State-of-the-art CNN trained on augmented data</li>
                <li><strong>Speed</strong> — Results in under a second</li>
                <li><strong>Simplicity</strong> — No expertise required</li>
                <li><strong>Breadth</strong> — 13 crops, 38 conditions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hiw">
        <div class="hiw-title">How It Works</div>
        <div class="steps">
            <div class="step">
                <div class="step-n">1</div>
                <div class="step-t">Upload Image</div>
                <div class="step-d">Navigate to Disease Recognition and upload a clear photo of the plant leaf.</div>
            </div>
            <div class="step">
                <div class="step-n">2</div>
                <div class="step-t">AI Analysis</div>
                <div class="step-d">Our CNN processes the 128×128 image through deep learned convolutional features.</div>
            </div>
            <div class="step">
                <div class="step-n">3</div>
                <div class="step-t">Get Results</div>
                <div class="step-d">Receive the predicted disease class and confidence score instantly.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="cta-wrap">
        <div>
            <h3>Ready to scan a leaf?</h3>
            <p>Click <strong>Disease Recognition</strong> in the sidebar to begin.</p>
        </div>
        <div class="cta-pill">🔬 Start Scanning →</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "About":

    st.markdown("""
    <div class="hero" style="padding:44px 40px 38px;">
        <div class="hero-line"></div>
        <div class="hero-content">
            <div class="hero-eyebrow">
                <span class="hero-eyebrow-line"></span>
                About This Project
            </div>
            <div class="hero-h1" style="font-size:2.2rem;">
                Dataset, Model &amp;<br><em>Methodology</em>
            </div>
            <div class="hero-desc">
                A deep dive into the data, architecture, and capabilities powering AgroVision AI.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="acard">
            <h3>📚 Dataset Overview</h3>
            <p>Approximately <strong>87,000 RGB images</strong> of healthy and diseased crop
            leaves spanning <strong>38 classes</strong> across <strong>13 plant species</strong>.
            Offline augmentation was applied to maximise model robustness.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="acard">
            <h3>📊 Dataset Split</h3>
            <ul>
                <li><strong>Training:</strong> 70,295 images</li>
                <li><strong>Validation:</strong> 17,572 images</li>
                <li><strong>Test:</strong> 33 images</li>
                <li><strong>Ratio:</strong> 80/20 train–val split</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Technical Stack</div><div class="section-title">Architecture &amp; Framework</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="acard">
        <div class="tech-row">
            <span class="tech-pill">TensorFlow / Keras</span>
            <span class="tech-pill">CNN Deep Learning</span>
            <span class="tech-pill">128 × 128 RGB Input</span>
            <span class="tech-pill">Multi-class Softmax</span>
            <span class="tech-pill">Streamlit UI</span>
            <span class="tech-pill">NumPy · Pillow</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Coverage</div><div class="section-title">Plant Species &amp; Disease Conditions</div>', unsafe_allow_html=True)

    st.markdown('<div class="species-grid">', unsafe_allow_html=True)
    for plant, diseases in DISEASE_CLASSES.items():
        icon         = PLANT_ICONS.get(plant, "🌿")
        total        = len(diseases)
        healthy_cnt  = sum(1 for k in diseases if "healthy" in k.lower())
        disease_cnt  = total - healthy_cnt
        li_items = ""
        for code, name in diseases.items():
            is_h    = "healthy" in code.lower()
            dot_cls = "dot-h" if is_h else "dot-d"
            li_items += f'<li><span class="{dot_cls}"></span>{name}</li>'
        st.markdown(f"""
        <div class="sp-card">
            <div class="sp-icon">{icon}</div>
            <div class="sp-name">{plant}</div>
            <div class="sp-counts">
                <span class="sp-badge sp-badge-g">✓ {healthy_cnt} healthy</span>
                <span class="sp-badge sp-badge-r">⚠ {disease_cnt} disease{'s' if disease_cnt != 1 else ''}</span>
            </div>
            <ul class="sp-list">{li_items}</ul>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DISEASE RECOGNITION
# ══════════════════════════════════════════════════════════════════════════════
elif app_mode == "Disease Recognition":

    st.markdown("""
    <div class="hero" style="padding:42px 40px 36px; margin-bottom:24px;">
        <div class="hero-line"></div>
        <div class="hero-content">
            <div class="hero-eyebrow">
                <span class="hero-eyebrow-line"></span>
                Disease Recognition
            </div>
            <div class="hero-h1" style="font-size:2rem;">
                Upload a leaf.<br>Get an instant <em>diagnosis.</em>
            </div>
            <div class="hero-desc">
                Upload a JPEG or PNG image of any supported plant leaf.
                Our CNN analyses it in under a second and returns a disease prediction with confidence score.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="scan-panel-header">
        <div class="scan-panel-icon">🔬</div>
        <div>
            <div class="scan-panel-title">Leaf Scan Interface</div>
            <div class="scan-panel-sub">Upload on the left · Analysis results appear on the right</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="small")

    with left_col:
        st.markdown("""
        <div style="background:var(--bg3);padding:22px 20px 6px;border:1px solid var(--border);border-top:none;">
            <div class="col-label">📁 Step 1 — Upload</div>
            <div class="col-title">Select Your Leaf Photo</div>
            <div class="upload-deco">
                <div class="ud-icon">🍃</div>
                <div class="ud-text">Drop image here or click Browse</div>
                <div class="ud-sub">JPEG · PNG · Any resolution</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        test_image = st.file_uploader(
            "Upload leaf image",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

        if test_image is not None:
            st.image(test_image, use_column_width=True, caption="📷 Uploaded leaf — ready to scan")
            st.success("✅ Image loaded — click Analyse on the right")

        st.markdown("""
        <div class="tips-row">
            <div class="tip-chip"><span class="tip-dot"></span>Natural daylight</div>
            <div class="tip-chip"><span class="tip-dot"></span>Single leaf</div>
            <div class="tip-chip"><span class="tip-dot"></span>Show affected area</div>
            <div class="tip-chip"><span class="tip-dot"></span>Sharp focus</div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("""
        <div style="background:var(--bg2);padding:22px 20px 6px;border:1px solid var(--border);border-top:none;border-left:none;">
            <div class="col-label">🧠 Step 2 — Prediction</div>
            <div class="col-title">AI Analysis Output</div>
        </div>
        """, unsafe_allow_html=True)

        if test_image is not None:
            run = st.button("🚀  Analyse Image", use_container_width=True, type="primary")
            if run:
                with st.spinner("Running CNN inference…"):
                    try:
                        result_index, confidence = model_prediction(test_image)
                        predicted_class = FLAT_CLASSES[result_index]
                        plant_type   = predicted_class.split("___")[0]
                        disease_name = DISEASE_CLASSES[plant_type][predicted_class]
                        is_healthy   = "healthy" in predicted_class.lower()
                        hdr_cls      = "healthy" if is_healthy else "diseased"
                        emoji        = "✅" if is_healthy else "⚠️"
                        status       = "Healthy" if is_healthy else "Diseased"
                        status_color = "#6abb6a" if is_healthy else "#e07878"

                        st.markdown(f"""
                        <div class="dcard">
                            <div class="dcard-header {hdr_cls}">
                                <div class="dcard-emoji">{emoji}</div>
                                <div>
                                    <div class="dcard-badge">Detected Condition</div>
                                    <div class="dcard-name">{disease_name}</div>
                                </div>
                            </div>
                            <div class="dcard-meta">
                                <div>
                                    <div class="dm-lbl">Plant Type</div>
                                    <div class="dm-val">{plant_type}</div>
                                </div>
                                <div>
                                    <div class="dm-lbl">Confidence</div>
                                    <div class="dm-val">{confidence:.1f}%</div>
                                    <div class="conf-track">
                                        <div class="conf-fill" style="width:{min(confidence,100):.1f}%"></div>
                                    </div>
                                </div>
                                <div>
                                    <div class="dm-lbl">Status</div>
                                    <div class="dm-val" style="color:{status_color};">{status}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="code-block">
                            <div class="code-lbl">Full Class Code</div>
                            <code class="code-val">{predicted_class}</code>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Prediction failed: {e}")
                        st.info("Ensure `trained_model.keras` exists and the image is a valid leaf photo.")
        else:
            st.markdown("""
            <div class="await-state">
                <div class="await-icon">🌾</div>
                <div class="await-title">Awaiting leaf image…</div>
                <div class="await-sub">Upload a photo on the left<br>to begin AI analysis</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    with st.expander("📚 View All Supported Plant Species"):
        cols = st.columns(4)
        for idx, (plant, diseases) in enumerate(DISEASE_CLASSES.items()):
            icon = PLANT_ICONS.get(plant, "🌿")
            with cols[idx % 4]:
                st.markdown(f"""
                <div class="mini-sp">
                    <div class="mini-sp-name">{icon} {plant}</div>
                    <div class="mini-sp-cnt">{len(diseases)} class{'es' if len(diseases) > 1 else ''}</div>
                </div>
                """, unsafe_allow_html=True)