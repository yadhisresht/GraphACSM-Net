import streamlit as st
import streamlit.components.v1 as components
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, TransformerConv, global_mean_pool, global_max_pool, GINEConv, GraphNorm, GlobalAttention
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import io, os, base64, time
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

st.set_page_config(page_title="GraphACSM-Net", page_icon="🧬",
                   layout="wide", initial_sidebar_state="collapsed")

# ── Read page from URL query params ──────────────────────────────────────────
params   = st.query_params
PAGE     = params.get("page", "home")

# ── CSS ───────────────────────────────────────────────────────────────────────
# ── Toggle Switch ─────────────────────────────────────────────────────────────
# ── Persisted Theme Logic via URL Params ───────────────────────────────────────
q = st.query_params
cur_theme = q.get("dark", "1")
if cur_theme == "0":
    dark_mode = False
else:
    dark_mode = True

# URLs to toggle theme
theme_val = "0" if dark_mode else "1"
theme_href = f"?page={PAGE}&dark={theme_val}"
theme_icon = "☀️" if dark_mode else "🌙"

theme_vars = """
    --bg-color: #0A0F1C;
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --card-bg: rgba(30, 41, 59, 0.4);
    --card-border: rgba(14, 165, 233, 0.2);
    --nav-bg: rgba(10, 15, 28, 0.7);
    --accent-blue: #0ea5e9;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-orange: #FB923C;
    --glow-blue: 0 0 15px rgba(14, 165, 233, 0.3);
    --grid-color: rgba(56, 189, 248, 0.08);
    --hero-gradient: linear-gradient(135deg, #1a0800 0%, #7c2d12 30%, #c2410c 60%, #1a0800 100%);
""" if dark_mode else """
    --bg-color: #EEF2FF;
    --bg-gradient: linear-gradient(135deg, #EEF2FF 0%, #F0FDF4 40%, #FFF7ED 80%, #EEF2FF 100%);
    --text-primary: #0F172A;
    --text-secondary: #475569;
    --card-bg: rgba(255, 255, 255, 0.60);
    --card-border: rgba(99, 102, 241, 0.22);
    --nav-bg: rgba(238, 242, 255, 0.90);
    --accent-blue: #4F46E5;
    --accent-green: #059669;
    --accent-purple: #7C3AED;
    --accent-teal: #0D9488;
    --accent-orange: #FB923C;
    --accent-red: #dc2626;
    --glow-blue: 0 4px 20px rgba(79, 70, 229, 0.22);
    --glow-green: 0 4px 20px rgba(5, 150, 105, 0.22);
    --glow-purple: 0 4px 20px rgba(124, 58, 237, 0.22);
    --glass-blur: blur(20px);
    --grid-color: rgba(79, 70, 229, 0.05);
    --hero-gradient: linear-gradient(135deg, #431407 0%, #9a3412 30%, #ea580c 60%, #431407 100%);
"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400&family=Sora:wght@400;600;700;800&family=Cinzel:wght@700;900&family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {{
{theme_vars}
}}

*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="block-container"]{{
    background:var(--bg-gradient, var(--bg-color))!important;color:var(--text-primary)!important;
    font-family:'Plus Jakarta Sans',sans-serif!important; transition: background 0.3s ease;
    font-size: 15px !important;
}}
[data-testid="block-container"]{{padding:0!important;max-width:100%!important;}}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}

/* ── NAVBAR ─────────────────────────────────────────────────────────────── */
.navbar{{
    background:var(--nav-bg);
    backdrop-filter: var(--glass-blur);
    padding:0 2.5rem;display:flex;align-items:center;justify-content:space-between;
    height:70px; border-bottom: 1px solid var(--card-border);
    position:sticky;top:0;z-index:1000;
}}
.navbar-brand{{display:flex;align-items:center;gap:.8rem;}}
.navbar-logo{{width:40px;height:40px;background:linear-gradient(135deg, var(--accent-blue), var(--accent-green));border-radius:12px;
    display:flex;align-items:center;justify-content:center;font-size:1.4rem;box-shadow:var(--glow-blue);}}
.navbar-title{{font-family:\'Sora\',sans-serif;font-weight:700;font-size:1.4rem;color:var(--text-primary);letter-spacing:-.01em;}}
.navbar-title span{{color:var(--accent-green);}}
.navbar-subtitle{{font-size:.7rem;color:var(--text-secondary);font-weight:400;display:block;margin-top:-4px;}}
.navbar-links{{display:flex;gap:.5rem;}}
.nav-btn{{
    color:var(--text-secondary)!important;background:transparent;border:none;
    padding:.5rem 1rem;border-radius:8px;font-size:.9rem;font-weight:600;
    text-decoration:none;display:inline-block;transition:all .3s;font-family:'Plus Jakarta Sans';}}
.nav-btn:hover{{background:rgba(120, 120, 120, 0.1);color:var(--text-primary)!important;}}
.nav-btn.active{{background:rgba(120, 120, 120, 0.15);color:var(--text-primary)!important;
    border-bottom:3px solid var(--accent-green);border-radius:8px 8px 0 0;}}

/* ── HERO ────────────────────────────────────────────────────────────────── */
.hero-banner{{
    background: linear-gradient(135deg, #1c0700 0%, #7c2d12 25%, #c2410c 55%, #431407 100%) !important;
    padding:5.5rem 3rem 4rem;text-align:center;position:relative;overflow:hidden;
    border-bottom: 1px solid rgba(251,146,60,0.4);
}}
.hero-banner .hero-sub{{color:rgba(255,237,213,0.85) !important;}}
.hero-title{{font-family:'Cinzel',serif;font-weight:900;font-size:2rem;
    letter-spacing:.08em;margin-bottom:.5rem;}}
.hero-title span{{color:var(--accent-green);}}
.hero-sub{{font-size:0.95rem;color:var(--text-secondary);margin-bottom:1.5rem;font-weight:300;}}
.hero-badges{{display:flex;justify-content:center;gap:.8rem;flex-wrap:wrap;}}
.hero-badge{{background:rgba(255,255,255,0.55);border:1.5px solid rgba(99,102,241,0.35);
    border-radius:20px;padding:.4rem 1rem;font-size:.85rem;color:#3730a3;font-weight:700;
    backdrop-filter:var(--glass-blur);box-shadow:0 2px 12px rgba(99,102,241,0.1);}}
.hero-badge.green{{background:rgba(255,255,255,0.55);border-color:rgba(5,150,105,0.45);color:#065f46;
    box-shadow:0 2px 12px rgba(5,150,105,0.12);}}

/* ── PAGE SECTION ────────────────────────────────────────────────────────── */
.page-section{{padding:2rem 2.5rem;}}
.section-heading{{font-family:\'Sora\',sans-serif;font-weight:700;font-size:1.4rem;
    color:var(--text-primary);margin-bottom:.4rem;}}
.section-sub{{font-size:.95rem;color:var(--text-secondary);margin-bottom:1.5rem;}}

/* ── CARDS ───────────────────────────────────────────────────────────────── */
.card{{background:var(--card-bg);border-radius:16px;padding:1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);border:1px solid var(--card-border);
    backdrop-filter: var(--glass-blur);}}
.card-blue{{border-top:4px solid var(--accent-blue);}}
.card-green{{border-top:4px solid var(--accent-green);}}

/* ── STAT PILLS ──────────────────────────────────────────────────────────── */
.stat-grid{{display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0;}}
.stat-pill{{background:rgba(120,120,120,0.05);border:1px solid var(--card-border);border-radius:12px;
    padding:.8rem 1.2rem;text-align:center;flex:1;min-width:90px;}}
.stat-val{{font-family:\'Space Grotesk\';font-weight:700;font-size:1.4rem;color:var(--accent-blue);}}
.stat-val.green{{color:var(--accent-green);}}
.stat-lbl{{font-size:.75rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.08em;font-weight:600;}}

/* ── BUTTONS ─────────────────────────────────────────────────────────────── */
.stButton>button{{
    background:linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 50%, var(--accent-green) 100%)!important;
    color:white!important;border:none!important;border-radius:10px!important;
    padding:.7rem 2rem!important;font-family:\'Sora\'!important;font-weight:700!important;
    font-size:1rem!important;width:100%!important; box-shadow: var(--glow-blue); transition:all .3s!important;}}
.stButton>button:hover{{transform:translateY(-2px); box-shadow: var(--glow-green)!important;}}

/* ── INPUTS ──────────────────────────────────────────────────────────────── */
div[data-testid="stTextInputRootElement"], div[data-testid="stTextAreaRootElement"]{{
    background:var(--card-bg)!important;
    background-color:var(--card-bg)!important;
    border:1px solid var(--card-border)!important;
    border-radius:10px!important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}}
div[data-testid="stTextInputRootElement"] *, div[data-testid="stTextAreaRootElement"] *{{
    background:transparent!important;
    background-color:transparent!important;
}}
div[data-testid="stTextInputRootElement"] input, div[data-testid="stTextAreaRootElement"] textarea{{
    color:var(--text-primary)!important;
    background:transparent!important;
    background-color:transparent!important;
    font-family:\'JetBrains Mono\',monospace!important;
    font-size:.9rem!important;
}}
div[data-testid="stTextInputRootElement"] input::placeholder, div[data-testid="stTextAreaRootElement"] textarea::placeholder{{
    color:var(--text-secondary)!important;
    opacity:0.6;
}}
div[data-testid="stTextInputRootElement"]:focus-within, div[data-testid="stTextAreaRootElement"]:focus-within{{
    border-color:var(--accent-blue)!important;
    box-shadow:var(--glow-blue)!important;
}}
label{{color:var(--text-secondary)!important;font-weight:600!important;font-size:.87rem!important;}}

.stSelectbox>div>div, .stSelectbox div[data-baseweb="select"], .stSelectbox [data-baseweb="select"]>div{{
    background:var(--card-bg)!important;
    background-color:var(--card-bg)!important;
    color:var(--text-primary)!important;
}}
.stSelectbox>div>div{{
    border:1px solid var(--card-border)!important;
    border-radius:10px!important;
}}

/* ── RESULT ──────────────────────────────────────────────────────────────── */
@keyframes pulseGlow {{
    0% {{ box-shadow: 0 0 5px rgba(16, 185, 129, 0.2), inset 0 0 5px rgba(16, 185, 129, 0.1); }}
    50% {{ box-shadow: 0 0 20px rgba(16, 185, 129, 0.5), inset 0 0 10px rgba(16, 185, 129, 0.2); }}
    100% {{ box-shadow: 0 0 5px rgba(16, 185, 129, 0.2), inset 0 0 5px rgba(16, 185, 129, 0.1); }}
}}
.result-active{{background:rgba(16,185,129,0.1);border:2px solid var(--accent-green);
    border-radius:16px;padding:1.5rem;text-align:center;margin:1rem 0; 
    animation: pulseGlow 2s infinite alternate;}}
.result-inactive{{background:rgba(239,68,68,0.1);border:2px solid var(--accent-red);
    border-radius:16px;padding:1.5rem;text-align:center;margin:1rem 0; box-shadow: 0 0 15px rgba(239,68,68,0.3);}}
.result-label{{font-family:\'Space Grotesk\';font-weight:700;font-size:1.3rem;margin-bottom:.3rem;}}
.result-sub{{font-size:.9rem;color:var(--text-secondary);}}

.conf-wrap{{margin:.8rem 0;}}
.conf-row{{display:flex;justify-content:space-between;font-size:.85rem;color:var(--text-secondary);margin-bottom:.4rem;font-weight:600;}}
.conf-track{{height:10px;background:rgba(120,120,120,0.1);border-radius:5px;overflow:hidden;border:1px solid var(--card-border);}}
.conf-fill{{height:100%;border-radius:5px;}}

.desc-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin-top:1rem;}}
.desc-item{{background:rgba(120,120,120,0.05);border:1px solid var(--card-border);border-radius:10px;padding:.8rem 1rem;}}
.desc-key{{font-size:.7rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.08em;font-weight:600;}}
.desc-val{{font-family:\'Space Grotesk\';font-weight:700;font-size:1.1rem;color:var(--accent-blue);margin-top:.2rem;}}

/* ── TABLE ───────────────────────────────────────────────────────────────── */
.perf-table{{width:100%;border-collapse:collapse;font-size:.9rem;}}
.perf-table th{{background:rgba(120,120,120,0.1);color:var(--text-primary);padding:.8rem 1rem;text-align:left;
    font-family:\'Space Grotesk\';font-weight:700;font-size:.85rem;letter-spacing:.05em;border-bottom:1px solid var(--card-border);}}
.perf-table td{{padding:.8rem 1rem;border-bottom:1px solid var(--card-border);color:var(--text-secondary);}}
.perf-table tr:hover td{{background:rgba(120,120,120,0.05);}}
.perf-table .best{{color:var(--accent-green);font-weight:700;}}
.perf-table .model-col{{font-family:\'Space Grotesk\';font-weight:700;color:var(--text-primary);}}

/* ── TABS ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"]{{background:transparent!important;border-bottom:2px solid var(--card-border)!important;gap:0!important;padding:0 1rem!important;}}
.stTabs [data-baseweb="tab"]{{font-family:\'Sora\'!important;font-weight:600!important;
    color:var(--text-secondary)!important;font-size:1rem!important;padding:1rem 1.5rem!important;
    border-radius:0!important;background:transparent!important;}}
.stTabs [aria-selected="true"]{{color:var(--accent-blue)!important;border-bottom:3px solid var(--accent-blue)!important;}}
.stTabs [data-baseweb="tab-panel"]{{background:transparent!important;padding:1.5rem 0!important;}}

[data-testid="stFileUploader"]{{background:rgba(120,120,120,0.05)!important;border:2px dashed var(--accent-blue)!important;border-radius:12px!important;}}
[data-testid="stDataFrame"]{{border:1px solid var(--card-border)!important;border-radius:10px!important;}}
[data-testid="stSidebar"]{{background:var(--bg-color)!important;border-right:1px solid var(--card-border)!important;}}

.info-box{{background:rgba(14,165,233,0.1);border-left:4px solid var(--accent-blue);border-radius:0 10px 10px 0;
    padding:1rem 1.2rem;font-size:.9rem;color:var(--text-primary);margin:.5rem 0;}}
.warn-box{{background:rgba(239,68,68,0.1);border-left:4px solid var(--accent-red);border-radius:0 10px 10px 0;
    padding:1rem 1.2rem;font-size:.9rem;color:var(--text-primary);margin:.5rem 0;}}

.site-footer{{background:rgba(120,120,120,0.05);color:var(--text-secondary);text-align:center;
    padding:2rem;font-size:.85rem;margin-top:3rem;border-top:1px solid var(--card-border);}}
.site-footer a{{color:var(--accent-blue);text-decoration:none;font-weight:600;}}
::-webkit-scrollbar{{width:6px;}}
::-webkit-scrollbar-thumb{{background:var(--card-border);border-radius:4px;}}

/* ── MOBILE RESPONSIVENESS ───────────────────────────────────────────────── */
@media screen and (max-width: 768px) {{
    .navbar {{ flex-direction: column; height: auto; padding: 1rem; gap: 0.8rem; }}
    .navbar-links {{ width: 100%; overflow-x: auto; white-space: nowrap; padding-bottom: 0.5rem; }}
    .hero-title {{ font-size: 1.8rem; }}
    .hero-sub {{ font-size: 0.95rem; }}
    .page-section {{ padding: 1.5rem 1rem; }}
    .hero-banner {{ padding: 2rem 1rem 1.5rem; }}
    .stat-pill {{ min-width: 45%; }}
}}

/* ── CUSTOM ADDITIONS FOR FAB & DRUG CARDS ────────────────────────────────── */
.mol-svg-wrap{{background:#fff;border-radius:12px;padding:8px;text-align:center;border:1px solid var(--card-border);}}
.mol-svg-wrap img{{max-width:100%;height:auto;}}
.drug-info-card{{background:var(--card-bg);border:1px solid var(--card-border);border-radius:14px;
    padding:1rem 1.2rem;margin-top:.8rem;backdrop-filter:var(--glass-blur);}}
.drug-name{{font-family:'Space Grotesk',sans-serif;font-weight:800;font-size:1.2rem;letter-spacing:-.02em;margin-bottom:.2rem;}}
.drug-desc{{font-size:.85rem;color:var(--text-secondary);line-height:1.5;margin-bottom:.5rem;}}
.drug-tag{{display:inline-block;padding:.22rem .7rem;border-radius:20px;font-size:.7rem;
    font-weight:700;letter-spacing:.05em;text-transform:uppercase;}}
.theme-fab{{position:fixed;bottom:1.5rem;right:1.5rem;width:48px;height:48px;border-radius:50%;
    background:linear-gradient(135deg,#38BDF8,#34D399);display:flex;align-items:center;
    justify-content:center;font-size:1.4rem;text-decoration:none;z-index:9999;
    box-shadow:0 4px 20px rgba(56,189,248,.4);transition:all .3s;}}
.theme-fab:hover{{transform:scale(1.12) rotate(20deg);}}

/* ── ENHANCED TYPOGRAPHY ──────────────────────────────────────────────────── */
:root {{
    --font-sans: 'Plus Jakarta Sans', 'Inter', sans-serif;
    --font-display: 'Sora', 'Plus Jakarta Sans', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --radius-sm: 10px;
    --radius-md: 16px;
    --radius-lg: 24px;
    --radius-xl: 32px;
}}

.hero-title {{
    font-size: clamp(1.4rem, 3vw, 2.2rem) !important;
    font-family: var(--font-display) !important;
    font-weight: 900 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.04em !important;
}}
.hero-sub {{
    font-size: clamp(0.8rem, 1.3vw, 0.95rem) !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    max-width: 680px;
    margin-left: auto;
    margin-right: auto;
}}

/* ── 3D CARD EFFECTS ──────────────────────────────────────────────────────── */
.card {{
    transform-style: preserve-3d !important;
    perspective: 1000px !important;
    transition: transform 0.4s cubic-bezier(0.34, 1.56, 0.64, 1),
                box-shadow 0.4s ease !important;
}}
.card:hover {{
    transform: translateY(-6px) rotateX(1.5deg) rotateY(-1deg) !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15),
                0 0 0 1px var(--card-border),
                inset 0 1px 0 rgba(255,255,255,0.1) !important;
}}
.card-blue {{
    box-shadow: 0 0 0 1px rgba(56,189,248,0.2),
                0 4px 24px rgba(56,189,248,0.08) !important;
}}
.card-blue:hover {{
    box-shadow: 0 20px 60px rgba(56,189,248,0.15),
                0 0 30px rgba(56,189,248,0.12),
                0 0 0 1px rgba(56,189,248,0.4) !important;
}}
.card-green:hover {{
    box-shadow: 0 20px 60px rgba(52,211,153,0.15),
                0 0 30px rgba(52,211,153,0.12),
                0 0 0 1px rgba(52,211,153,0.4) !important;
}}

/* ── NATIVE STREAMLIT CARD WRAPPERS ──────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    backdrop-filter: var(--glass-blur) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05) !important;
    transform-style: preserve-3d !important;
    perspective: 1000px !important;
    transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
}}
[data-testid="stVerticalBlockBorderWrapper"]:hover {{
    transform: translateY(-6px) rotateX(1.5deg) rotateY(-1deg) !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.15), 
                0 0 0 1px var(--card-border), 
                inset 0 1px 0 rgba(255,255,255,0.1) !important;
}}

/* ── STICKY NAVBAR WRAPPER ────────────────────────────────────────────────── */
div[data-testid="stVerticalBlock"]:has(.sticky-nav-marker) {{
    position: sticky !important;
    top: 0 !important;
    z-index: 9999 !important;
    background: var(--nav-bg) !important;
    backdrop-filter: var(--glass-blur) !important;
    border-bottom: 1px solid var(--card-border) !important;
    padding: 0.5rem 2.5rem !important;
    margin: 0 0 1.5rem 0 !important;
}}
/* Hide empty container padding for navbar wrap */
div[data-testid="stVerticalBlock"]:has(.sticky-nav-marker) > div {{
    gap: 0 !important;
}}

/* Style all snappy buttons inside the navbar to look like topbar links */
div[data-testid="stVerticalBlock"]:has(.sticky-nav-marker) button {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    height: 46px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease !important;
    margin: 0 !important;
    padding: 0.5rem !important;
    border-radius: 8px !important;
}}
div[data-testid="stVerticalBlock"]:has(.sticky-nav-marker) button:hover {{
    color: var(--text-primary) !important;
    background: rgba(120, 120, 120, 0.08) !important;
}}
/* Active page marker */
div[data-testid="stVerticalBlock"]:has(.sticky-nav-marker) button[kind="primary"] {{
    color: var(--accent-green) !important;
    border-bottom: 3px solid var(--accent-green) !important;
    border-radius: 8px 8px 0 0 !important;
    background: rgba(120, 120, 120, 0.12) !important;
}}

/* ── FLOATING FAB OVERRIDE ───────────────────────────────────────────────── */
div[data-testid="stElementContainer"]:has(.theme-fab-marker) + div {{
    position: fixed !important;
    bottom: 2.5rem !important;
    right: 2.5rem !important;
    z-index: 10000 !important;
    width: auto !important;
}}
div[data-testid="stElementContainer"]:has(.theme-fab-marker) + div button {{
    width: 56px !important;
    height: 56px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-green) 100%) !important;
    color: white !important;
    font-size: 1.6rem !important;
    border: none !important;
    box-shadow: 0 8px 32px rgba(14, 165, 233, 0.3) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease !important;
    padding: 0 !important;
}}
div[data-testid="stElementContainer"]:has(.theme-fab-marker) + div button:hover {{
    transform: scale(1.12) rotate(12deg) !important;
    box-shadow: 0 12px 40px rgba(14, 165, 233, 0.5) !important;
}}

/* ── CINZEL GOLD TYPEWRITER HERO TITLE ──────────────────────────────────── */
@keyframes goldShift {{
    0% {{ background-position: 0%; }}
    100% {{ background-position: 200%; }}
}}
@keyframes caretBlink {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0; }}
}}
.hero-title {{
    font-family: 'Cinzel', serif !important;
    font-weight: 900 !important;
    font-size: clamp(1.4rem, 3vw, 2.2rem) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: none !important;
    animation: none !important;
    letter-spacing: .08em !important;
    text-shadow: 0 2px 12px rgba(0,0,0,0.4) !important;
    display: inline-block !important;
}}
.hero-title span {{
    -webkit-text-fill-color: #ffffff !important;
    background: none !important;
}}
#hero-title-tw {{
    font-family: 'Cinzel', serif;
    font-weight: 900;
    font-size: clamp(1.4rem, 3vw, 2.2rem);
    color: #ffffff;
    -webkit-text-fill-color: #ffffff;
    background: none;
    animation: none;
    letter-spacing: .08em;
    text-shadow: 0 2px 12px rgba(0,0,0,0.4);
    display: inline-block;
    margin-bottom: .5rem;
}}
#hero-title-tw::after {{
    content: '|';
    -webkit-text-fill-color: #ffffff;
    animation: caretBlink 1s step-end infinite;
    margin-left: 2px;
    font-weight: 400;
}}
#hero-title-tw.done::after {{
    animation: caretBlink 1s step-end infinite;
}}

/* ── LIGHT MODE EXTRA COLOUR ─────────────────────────────────────────────── */
.stat-pill {{
    background: linear-gradient(135deg, rgba(79,70,229,0.07) 0%, rgba(5,150,105,0.07) 100%) !important;
    border-color: rgba(79,70,229,0.2) !important;
}}
.stat-val {{ color: var(--accent-blue) !important; }}
.stat-val.green {{ color: var(--accent-green) !important; }}
.info-box {{
    background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(13,148,136,0.08)) !important;
    border-left-color: var(--accent-blue) !important;
}}
.hero-badge {{
    background: rgba(251,146,60,0.18) !important;
    border: 1.5px solid rgba(251,146,60,0.5) !important;
    color: #fff7ed !important;
    box-shadow: 0 2px 12px rgba(251,146,60,0.2) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}}
.hero-badge.green {{
    background: rgba(249,115,22,0.2) !important;
    border-color: rgba(249,115,22,0.5) !important;
    color: #ffedd5 !important;
    box-shadow: 0 2px 12px rgba(249,115,22,0.2) !important;
}}
.navbar-logo {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-green)) !important;
}}
.perf-table th {{
    background: linear-gradient(135deg, rgba(79,70,229,0.08), rgba(13,148,136,0.08)) !important;
}}
.site-footer {{
    background: linear-gradient(135deg, rgba(79,70,229,0.06), rgba(5,150,105,0.06)) !important;
    border-top-color: rgba(79,70,229,0.2) !important;
}}
</style>
<script>
(function() {{
    function runTypewriter() {{
        var el = document.getElementById('hero-title-tw');
        if (!el) {{ setTimeout(runTypewriter, 300); return; }}
        var full = el.getAttribute('data-text') || el.textContent.trim();
        el.setAttribute('data-text', full);
        el.textContent = '';
        var i = 0;
        function type() {{
            if (i <= full.length) {{
                el.textContent = full.slice(0, i);
                i++;
                setTimeout(type, i === 1 ? 600 : 90);
            }} else {{
                el.classList.add('done');
                setTimeout(function() {{
                    i = 0;
                    el.classList.remove('done');
                    setTimeout(function() {{
                        el.textContent = '';
                        type();
                    }}, 400);
                }}, 2800);
            }}
        }}
        type();
    }}
    document.addEventListener('DOMContentLoaded', runTypewriter);
    setTimeout(runTypewriter, 800);
}})();
</script>""",
    unsafe_allow_html=True)




# ── Constants ─────────────────────────────────────────────────────────────────
NODE_DIM=21; DESC_DIM=13; THRESHOLD=7.0
Y_REG_MEAN=6.408  # UPDATE with Cell 4 value
Y_REG_STD =1.208   # UPDATE with Cell 4 value

# ── Model classes ─────────────────────────────────────────────────────────────
class SimpleGAT(nn.Module):
    def __init__(self,node_dim=21,desc_dim=13,hidden=128,heads=4,dropout=.2,num_layers=4,**kw):
        super().__init__()
        self.embed=nn.Sequential(nn.Linear(node_dim,hidden),nn.LayerNorm(hidden),nn.GELU(),nn.Dropout(dropout))
        self.eproj=nn.Linear(7,32)
        self.convs=nn.ModuleList([GATv2Conv(hidden,hidden,heads=heads,concat=False,edge_dim=32) for _ in range(num_layers)])
        self.norms=nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])
        self.desc_bn=nn.BatchNorm1d(desc_dim); p=hidden*2+desc_dim
        self.cls_head=nn.Sequential(nn.Linear(p,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(64,1))
        self.reg_head=nn.Sequential(nn.Linear(p,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,64),nn.GELU(),nn.Dropout(dropout*.5),nn.Linear(64,1))
    def forward(self,data):
        x=self.embed(data.x); ea=self.eproj(data.edge_attr)
        for conv,norm in zip(self.convs,self.norms): x=x+norm(F.gelu(conv(x,data.edge_index,ea)))
        desc=data.desc
        if desc.size(0)>1: desc=self.desc_bn(desc)
        g=torch.cat([global_mean_pool(x,data.batch),global_max_pool(x,data.batch),desc],-1)
        return self.cls_head(g).squeeze(-1),self.reg_head(g).squeeze(-1)

class HighPerfMSMP(nn.Module):
    def __init__(self, node_dim, desc_dim=13, hidden=192, heads=4,
                 dropout=.21, num_layers=4, **kw):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.LayerNorm(hidden),
            nn.GELU(), nn.Dropout(dropout))
        self.edge_proj_gine = nn.Linear(7, hidden)
        self.edge_proj_attn = nn.Linear(7, 32)
        self.desc_bn = nn.BatchNorm1d(desc_dim)
        self.layers = nn.ModuleList()
        self.residual_scales = nn.ParameterList()
        self.num_layers = num_layers
        for i in range(num_layers):
            dp = dropout * (1 + .1*i)
            self.layers.append(nn.ModuleDict({
                "gine":  GINEConv(nn.Sequential(
                    nn.Linear(hidden, hidden*2), nn.GELU(),
                    nn.Dropout(dp), nn.Linear(hidden*2, hidden))),
                "gat":   GATv2Conv(hidden, hidden, heads=heads, concat=False, edge_dim=32),
                "trans": TransformerConv(hidden, hidden, heads=heads, concat=False, edge_dim=32),
                "norm":  GraphNorm(hidden)}))
            self.residual_scales.append(nn.Parameter(torch.tensor(.5)))
        self.layer_pool_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.attn_pool = GlobalAttention(nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Linear(hidden//2, 1)))
        pool_dim = hidden*4 + desc_dim
        self.pool_norm = nn.LayerNorm(pool_dim)
        self.fusion = nn.Sequential(
            nn.Linear(pool_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(),
            nn.Dropout(dropout*.5), nn.Linear(hidden//2, 128))
        self.cls_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(.3), nn.Linear(64, 1))
        self.reg_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(.3), nn.Linear(64, 1))

    def forward(self, data, return_embedding=False):
        x  = self.node_embed(data.x)
        ei = data.edge_index; ea = data.edge_attr; bat = data.batch
        desc = data.desc
        if desc.size(0) > 1: desc = self.desc_bn(desc)
        eg = self.edge_proj_gine(ea); ea2 = self.edge_proj_attn(ea); lp = []
        for i, layer in enumerate(self.layers):
            h = (layer["gine"](x, ei, eg) +
                 layer["gat"](x, ei, ea2) +
                 layer["trans"](x, ei, ea2))
            h = layer["norm"](h, bat)
            x = F.gelu(x + torch.sigmoid(self.residual_scales[i]) * h)
            lp.append(global_mean_pool(h, bat))
        lw = F.softmax(self.layer_pool_weights, 0)
        g = torch.cat([
            global_mean_pool(x, bat), global_max_pool(x, bat),
            self.attn_pool(x, bat),
            (torch.stack(lp, 1) * lw.view(1,-1,1)).sum(1), desc], -1)
        f = self.fusion(self.pool_norm(g))
        if return_embedding: return f
        return self.cls_head(f).squeeze(-1), self.reg_head(f).squeeze(-1)

# ── Feature functions ─────────────────────────────────────────────────────────
def atom_features(atom):
    atom_types=['C','N','O','S','F','Cl','Br','I','P','B']
    f=[1. if atom.GetSymbol()==t else 0. for t in atom_types]
    f+=[atom.GetAtomicNum()/100.,atom.GetDegree()/8.,atom.GetFormalCharge()/5.,
        atom.GetTotalNumHs()/8.,atom.GetTotalValence()/8.,float(atom.GetIsAromatic()),
        float(atom.IsInRing()),float(atom.GetChiralTag()!=Chem.rdchem.ChiralType.CHI_UNSPECIFIED)]
    hyb=atom.GetHybridization()
    for ht in [Chem.rdchem.HybridizationType.SP,Chem.rdchem.HybridizationType.SP2,Chem.rdchem.HybridizationType.SP3]:
        f.append(1. if hyb==ht else 0.)
    return f

def bond_features(bond):
    bt=bond.GetBondType()
    return [float(bt==Chem.rdchem.BondType.SINGLE),float(bt==Chem.rdchem.BondType.DOUBLE),
            float(bt==Chem.rdchem.BondType.TRIPLE),float(bt==Chem.rdchem.BondType.AROMATIC),
            float(bond.GetIsConjugated()),float(bond.IsInRing()),
            float(bond.GetStereo()!=Chem.rdchem.BondStereo.STEREONONE)]

def smiles_to_graph(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    x=[atom_features(a) for a in mol.GetAtoms()]
    ei,ea=[],[]
    for bond in mol.GetBonds():
        i,j=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        bf=bond_features(bond); ei+=[[i,j],[j,i]]; ea+=[bf,bf]
    if not ei: ei=[[0,0],[0,0]]; ea=[[0.]*7,[0.]*7]
    try:
        desc=[Descriptors.MolWt(mol)/1000.,Descriptors.MolLogP(mol)/10.,
              Descriptors.TPSA(mol)/200.,Descriptors.NumRotatableBonds(mol)/20.,
              QED.qed(mol),Descriptors.NumHDonors(mol)/10.,Descriptors.NumHAcceptors(mol)/10.,
              float(rdMolDescriptors.CalcNumAromaticRings(mol))/5.,Descriptors.FractionCSP3(mol),
              float(mol.GetNumHeavyAtoms())/50.,float(rdMolDescriptors.CalcNumRings(mol))/10.,
              min(Descriptors.BertzCT(mol)/1000.,3.),float(rdMolDescriptors.CalcNumHeteroatoms(mol))/20.]
    except: desc=[0.]*13
    g=Data(x=torch.tensor(x,dtype=torch.float),
           edge_index=torch.tensor(ei,dtype=torch.long).t().contiguous(),
           edge_attr=torch.tensor(ea,dtype=torch.float),
           desc=torch.tensor(desc,dtype=torch.float).view(1,-1),
           y_cls=torch.zeros(1),y_reg=torch.zeros(1))
    g.batch=torch.zeros(g.x.size(0),dtype=torch.long)
    return g


def svg_img_tag(svg):
    import base64
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<div class="mol-svg-wrap"><img src="data:image/svg+xml;base64,{b64}" style="max-height:280px;"/></div>'

def mol_to_svg(smiles,size=(420,280), dark=False):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return None
    AllChem.Compute2DCoords(mol)
    drawer=rdMolDraw2D.MolDraw2DSVG(size[0],size[1])
    opts = drawer.drawOptions()
    
    if dark:
        # #0A0F1C (deep dark space blue)
        opts.backgroundColour = (0.04, 0.06, 0.11, 1.0)
    else:
        # Clean pure white
        opts.backgroundColour = (1.0, 1.0, 1.0, 1.0)
        
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    # Ensure all hardcoded black paths in XML are safely converted for legibility
    if dark:
        svg = svg.replace("stroke:#000000;", "stroke:#F8FAFC;")
        svg = svg.replace("fill:#000000;", "fill:#F8FAFC;")
    return svg

def make_3d_viewer(smiles, width="100%", height="250px", dark=True):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return ""
    mol = Chem.AddHs(mol)
    try:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        pass
    block = Chem.MolToMolBlock(mol)
    
    # Escape for safe injection into JavaScript string literal
    js_block = block.replace("\\", "\\\\").replace("\n", "\\n").replace("`", "\\`").replace("$", "\\$")
    bg_hex = '#0A0F1C' if dark else '#FFFFFF'
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
        <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; overflow: hidden; background: {bg_hex}; }}
            #container-3d {{ width: {width}; height: {height}; position: relative; border-radius: 8px; overflow: hidden; }}
        </style>
    </head>
    <body>
        <div id="container-3d"></div>
        <script>
            $(document).ready(function() {{
                let element = $('#container-3d');
                let config = {{ backgroundColor: '{bg_hex}' }};
                let viewer = $3Dmol.createViewer(element, config);
                let mol_data = `{js_block}`;
                viewer.addModel(mol_data, "sdf");
                viewer.setStyle({{stick: {{radius: 0.15}}, sphere: {{scale: 0.3}}}});
                viewer.zoomTo();
                viewer.render();
            }});
        </script>
    </body>
    </html>
    """
    return html

def compute_props(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None: return {}
    try:
        return {"Mol. Weight":f"{Descriptors.MolWt(mol):.2f} Da",
                "LogP":f"{Descriptors.MolLogP(mol):.3f}",
                "TPSA":f"{Descriptors.TPSA(mol):.2f} Å²",
                "HB Donors":int(Descriptors.NumHDonors(mol)),
                "HB Acceptors":int(Descriptors.NumHAcceptors(mol)),
                "Rotatable Bonds":int(Descriptors.NumRotatableBonds(mol)),
                "QED":f"{QED.qed(mol):.4f}",
                "Aromatic Rings":int(rdMolDescriptors.CalcNumAromaticRings(mol)),
                "Heavy Atoms":int(mol.GetNumHeavyAtoms())}
    except: return {}

@st.cache_resource
def load_models():
    gat_random=SimpleGAT(); gat_random_ok=False
    msmp_scaffold=HighPerfMSMP(node_dim=NODE_DIM); msmp_scaffold_ok=False
    if os.path.exists("gat_random_aug_fold3_seed42.pt"):
        chk=torch.load("gat_random_aug_fold3_seed42.pt",map_location="cpu")
        gat_random.load_state_dict(chk.get("model_state_dict",chk)); gat_random_ok=True
    if os.path.exists("msmp_scaffold_aug_fold4_seed42.pt"):
        chk=torch.load("msmp_scaffold_aug_fold4_seed42.pt",map_location="cpu")
        msmp_scaffold.load_state_dict(chk.get("model_state_dict",chk)); msmp_scaffold_ok=True
    gat_random.eval(); msmp_scaffold.eval()
    return gat_random,gat_random_ok,msmp_scaffold,msmp_scaffold_ok

gat_random_model,gat_random_loaded,msmp_scaffold_model,msmp_scaffold_loaded=load_models()

# ── FDA Approved Drugs Examples Metadata ───────────────────────────────────────
EXAMPLES = {
    "Gefitinib": "COc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCCCN1CCOCC1",
    "Lenvatinib": "COc1cc2c(Nc3ccc(Cl)c(C(=O)NC)c3)ncnc2cc1C(=O)N",
    "Imatinib": "CN1CCN(Cc2ccc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cnccn5)n4)c3)cc2)CC1",
    "Sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
    "Erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
}

EXAMPLES_META = {
    "Gefitinib": {
        "desc": "FDA-approved for non-small cell lung cancer (NSCLC)",
        "tag": "EGFR inhibitor", "color": "#38BDF8"
    },
    "Lenvatinib": {
        "desc": "Approved for thyroid, liver, and kidney cancers",
        "tag": "Multikinase inhibitor", "color": "#34D399"
    },
    "Imatinib": {
        "desc": "Used for chronic myeloid leukemia (CML)",
        "tag": "Targeted Therapy", "color": "#A78BFA"
    },
    "Sorafenib": {
        "desc": "Approved for liver, kidney, and thyroid cancers",
        "tag": "Anti-angiogenic", "color": "#F59E0B"
    },
    "Erlotinib": {
        "desc": "Used in NSCLC and pancreatic cancer",
        "tag": "EGFR inhibitor", "color": "#EC4899"
    }
}

def run_predict(model,smiles):
    g=smiles_to_graph(smiles.strip())
    if g is None: return None
    model.eval()
    with torch.no_grad():
        co,ro=model(g)
        prob=float(torch.sigmoid(co).item())
        pic50=float(ro.item())*Y_REG_STD+Y_REG_MEAN
    return {"prob":prob,"pic50":round(pic50,4),
            "active":bool(prob>=0.5 and pic50>=THRESHOLD),
            "ic50":round(10**(6-pic50),4) if pic50<12 else ">1M"}

# NSAIDs Overridden
# EXAMPLES={
#     "Ibuprofen":   "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
#     "Celecoxib":   "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(N)cc2)cc1S(N)(=O)=O",
#     "Naproxen":    "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
#     "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
#     "Diclofenac":  "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
#     "Indomethacin":"COc1ccc2c(c1)c(CC(=O)O)c(C)n2C(=O)c1ccc(Cl)cc1",
# }


# ── Helper: result block ───────────────────────────────────────────────────────
def show_result(r):
    css="result-active" if r["active"] else "result-inactive"
    color="#0D7A5F" if r["active"] else "#C0392B"
    icon="✅" if r["active"] else "❌"
    bar_bg="linear-gradient(90deg,#0D7A5F,#5DDEAE)" if r["active"] else "linear-gradient(90deg,#C0392B,#E74C3C)"
    st.markdown(f"""
    <div class="{css}">
        <div class="result-label" style="color:{color};">{icon} {"ACTIVE" if r["active"] else "INACTIVE"}</div>
        <div class="result-sub">Confidence: {r["prob"]*100:.1f}% &nbsp;·&nbsp; Threshold: pIC₅₀ ≥ 7.0</div>
    </div>
    <div class="conf-wrap">
        <div class="conf-row"><span>Activity Probability</span><span>{r["prob"]:.4f}</span></div>
        <div class="conf-track"><div class="conf-fill" style="width:{r["prob"]*100:.1f}%;background:{bar_bg};"></div></div>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f'<div class="stat-pill"><div class="stat-val">{r["pic50"]}</div><div class="stat-lbl">pIC₅₀</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-pill"><div class="stat-val" style="font-size:1rem;">{r["ic50"]}</div><div class="stat-lbl">IC₅₀ (µM)</div></div>',unsafe_allow_html=True)
    with c3:
        tm="✓ Met" if r["active"] else "✗ Not Met"
        tc="#0D7A5F" if r["active"] else "#C0392B"
        st.markdown(f'<div class="stat-pill"><div class="stat-val" style="font-size:.95rem;color:{tc};">{tm}</div><div class="stat-lbl">Threshold ≥7.0</div></div>',unsafe_allow_html=True)
    props=compute_props(st.session_state.get("last_smi",""))
    if props:
        items="".join([f'<div class="desc-item"><div class="desc-key">{k}</div><div class="desc-val">{v}</div></div>' for k,v in props.items()])
        st.markdown(f'<div style="margin-top:1rem;"><div style="font-family:\'Space Grotesk\';font-weight:600;font-size:.85rem;color:var(--text-primary);margin-bottom:.5rem;">Physicochemical Properties</div><div class="desc-grid">{items}</div></div>',unsafe_allow_html=True)

# ── Helper for Dropdown ────────────────────────────────────────────────────────
def on_example_change(smi_key, ex_key):
    ex = st.session_state[ex_key]
    if ex != "— Custom input —":
        st.session_state[smi_key] = EXAMPLES[ex]
    else:
        st.session_state[smi_key] = ""

# ── Prediction panel (shared) ──────────────────────────────────────────────────
def predict_panel(model, model_name, btn_label, smi_key, btn_key):
    left,right=st.columns([1,1.2],gap="large")
    with left:
        with st.container(border=True):
            st.markdown(f'<div style="font-family:\'Space Grotesk\';font-weight:600;font-size:1rem;color:var(--text-primary);margin-bottom:.8rem;">🔬 Single Molecule Input</div>',unsafe_allow_html=True)
            
            # Dropdown with a callback to force the text area to update
            st.selectbox("Load example", ["— Custom input —"] + list(EXAMPLES.keys()), 
                         key=smi_key+"_ex", 
                         on_change=on_example_change, 
                         args=(smi_key, smi_key+"_ex"))
            
            # Text area tied directly to the session state key
            smi=st.text_area("SMILES string", height=90, placeholder="Paste SMILES here…", key=smi_key)
            
            st.session_state["last_smi"]=smi
            run=st.button(btn_label,key=btn_key)
            if smi.strip():
                mol=Chem.MolFromSmiles(smi.strip())
                if mol:
                    t2d, t3d = st.tabs(["2D View", "3D View"])
                    with t2d:
                        svg=mol_to_svg(smi.strip(),(400,250), dark=dark_mode)
                        if svg:
                            st.markdown(svg_img_tag(svg), unsafe_allow_html=True)
                    with t3d:
                        components.html(make_3d_viewer(smi.strip(), height="250px", dark=dark_mode), height=265)
                else:
                    st.markdown('<div class="warn-box">⚠️ Invalid SMILES</div>',unsafe_allow_html=True)
    with right:
        with st.container(border=True):
            st.markdown('<div style="font-family:\'Space Grotesk\';font-weight:600;font-size:1rem;color:var(--text-primary);margin-bottom:.8rem;">📊 Prediction Results</div>',unsafe_allow_html=True)
            if run and smi.strip():
                mol=Chem.MolFromSmiles(smi.strip())
                if not mol: st.error("Invalid SMILES.")
                else:
                    with st.spinner(f"Running {model_name} inference…"):
                        time.sleep(0.25); r=run_predict(model,smi.strip())
                    if r: show_result(r)
            else:
                st.markdown(f'<div style="height:360px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;color:var(--text-secondary);gap:.8rem;"><div style="font-size:2.5rem;opacity:.3;">⬡</div><div style="font-family:\'Space Grotesk\';font-size:1rem;">Enter a SMILES string<br>and click {btn_label}</div></div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PURE ENGINEERED NATIVE NAVIGATION  — zero reload snappy transitions
# ══════════════════════════════════════════════════════════════════════════════

# Navbar Block (Made sticky via .sticky-nav-marker in CSS)
with st.container():
    st.markdown('<div class="sticky-nav-marker"></div>', unsafe_allow_html=True)
    nav_l, nav_r = st.columns([1.8, 5.2])
    
    with nav_l:
        st.markdown(f"""
        <div class="navbar-brand" style="margin-top: 8px; pointer-events: none;">
            <div class="navbar-logo">🧬</div>
            <div>
                <div class="navbar-title" style="font-size:1.3rem; letter-spacing:-0.02em;">Graph<span style="color:var(--accent-green);">ACSM</span>-Net</div>
                <span class="navbar-subtitle" style="font-size:0.65rem; color:var(--text-secondary);">Anticancer Activity Platform</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with nav_r:
        c_nav = st.columns(8)
        pages_list = [
            ("home", "Home"),
            ("predict_random", "Random"),
            ("predict_scaffold", "Scaffold"),
            ("batch", "Batch"),
            ("algorithm", "Model"),
            ("dataset", "Dataset"),
            ("help", "Help"),
            ("contact", "Contact")
        ]
        for idx, (p_id, p_label) in enumerate(pages_list):
            b_type = "primary" if PAGE == p_id else "secondary"
            with c_nav[idx]:
                if st.button(p_label, key=f"nav_snappy_{p_id}", type=b_type, use_container_width=True):
                    st.query_params["page"] = p_id
                    st.query_params["dark"] = cur_theme
                    st.rerun()

# Snappy Native FAB Theme Switcher
st.markdown('<div class="theme-fab-marker"></div>', unsafe_allow_html=True)
if st.button(theme_icon, key="theme_fab_btn"):
    st.query_params["page"] = PAGE
    st.query_params["dark"] = theme_val
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if PAGE == "home":
    st.markdown("""
    <div class="hero-banner">
        <div id="hero-title-tw" data-text="GraphACSM-Net">GraphACSM-Net</div>
        <div class="hero-sub">A Web Server for Graph Neural Network-Based Anticancer Activity Prediction</div>
        <div class="hero-badges">
            <span class="hero-badge green">✓ GCN · GAT · GT · MSMP</span>
            <span class="hero-badge">Random Split (GT) · Scaffold Split (GAT)</span>
            <span class="hero-badge">Classification + Regression</span>
            <span class="hero-badge green">3,600 Compounds · PubChem</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="page-section">', unsafe_allow_html=True)

    # ── HOME: 3-COLUMN LAYOUT ─────────────────────────────────────────────────
    col_fda, col_welcome, col_predict = st.columns([2.8, 3.5, 3.7], gap="large")

    # ── LEFT: FDA Drug Explorer ───────────────────────────────────────────────
    with col_fda:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.9rem;">
                <span style="font-size:1.2rem;">🧪</span>
                <div style="font-family:'Space Grotesk';font-weight:800;font-size:0.95rem;color:var(--text-primary);">
                    FDA Example Drugs
                </div>
            </div>
            <div style="font-size:0.75rem;color:var(--text-secondary);margin-bottom:0.8rem;line-height:1.5;">
                Select a reference compound to explore its structure and auto-fill the prediction panel.
            </div>
            """, unsafe_allow_html=True)

            # Drug selector — callback writes SMILES into session state before text_input renders
            def _home_fda_changed():
                chosen = st.session_state["home_fda_sel"]
                st.session_state["home_smi_bento"] = EXAMPLES.get(chosen, "")

            sel_home = st.selectbox(
                "Reference Compound",
                list(EXAMPLES.keys()),
                key="home_fda_sel",
                on_change=_home_fda_changed,
            )
            # Seed session state on very first load
            if "home_smi_bento" not in st.session_state:
                st.session_state["home_smi_bento"] = EXAMPLES[list(EXAMPLES.keys())[0]]

            smi_ex = EXAMPLES[sel_home]
            meta = EXAMPLES_META.get(sel_home, {})
            drug_color = meta.get("color", "#FB923C")

            st.markdown(f"""
            <div style="background:{drug_color}18; border:1px solid {drug_color}40; border-radius:12px; padding:0.75rem; margin-bottom:0.8rem;">
                <div style="font-family:'Space Grotesk';font-weight:800;font-size:1rem;color:{drug_color};">{sel_home}</div>
                <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:3px;">{meta.get('desc','')}</div>
                <div style="display:inline-block;margin-top:0.5rem;padding:0.2rem 0.6rem;border-radius:20px;font-size:0.65rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;background:{drug_color}22;border:1px solid {drug_color}55;color:{drug_color};">{meta.get('tag','')}</div>
                <div style="margin-top:0.65rem;padding-top:0.65rem;border-top:1px solid {drug_color}22;">
                    <div style="font-size:0.65rem;color:var(--text-secondary);font-weight:600;letter-spacing:.06em;text-transform:uppercase;margin-bottom:0.3rem;">SMILES</div>
                    <div style="font-family:'JetBrains Mono';font-size:0.62rem;color:var(--text-primary);word-break:break-all;line-height:1.5;background:rgba(0,0,0,0.25);border-radius:6px;padding:0.35rem 0.5rem;">{smi_ex}</div>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.4rem;margin-top:0.5rem;">
                    <div style="background:{drug_color}10;border:1px solid {drug_color}22;border-radius:8px;padding:0.4rem 0.5rem;text-align:center;">
                        <div style="font-size:0.6rem;color:var(--text-secondary);font-weight:600;text-transform:uppercase;letter-spacing:.05em;">Mol. Weight</div>
                        <div style="font-family:'Sora',sans-serif;font-weight:700;font-size:0.9rem;color:{drug_color};margin-top:1px;">{Descriptors.MolWt(Chem.MolFromSmiles(smi_ex)):.2f} <span style="font-size:0.6rem;font-weight:400;">Da</span></div>
                    </div>
                    <div style="background:{drug_color}10;border:1px solid {drug_color}22;border-radius:8px;padding:0.4rem 0.5rem;text-align:center;">
                        <div style="font-size:0.6rem;color:var(--text-secondary);font-weight:600;text-transform:uppercase;letter-spacing:.05em;">LogP</div>
                        <div style="font-family:'Sora',sans-serif;font-weight:700;font-size:0.9rem;color:{drug_color};margin-top:1px;">{Descriptors.MolLogP(Chem.MolFromSmiles(smi_ex)):.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            t2d_h, t3d_h = st.tabs(["2D View", "3D View"])
            with t2d_h:
                svg_h = mol_to_svg(smi_ex, (320, 200), dark=dark_mode)
                if svg_h: st.markdown(svg_img_tag(svg_h), unsafe_allow_html=True)
            with t3d_h:
                components.html(make_3d_viewer(smi_ex, height="200px", dark=dark_mode), height=212)

    # ── MIDDLE: Welcome / About ───────────────────────────────────────────────
    with col_welcome:
        with st.container(border=True):
            st.markdown("""
            <div style="margin-bottom:1rem;">
                <div style="font-family:'Cinzel',serif;font-weight:900;font-size:1.25rem;
                     color:#000000;letter-spacing:.05em;">
                    Welcome to GraphACSM-Net
                </div>
            </div>
            <div style="font-size:0.82rem;color:var(--text-primary);line-height:1.75;margin-bottom:1rem;">
                GraphACSM-Net is an advanced web-based platform designed for the rapid prediction and evaluation
                of anticancer small molecules using integrated machine learning and graph deep learning approaches.
                The platform combines molecular descriptors, graph-based molecular representations, and multi-head
                predictive modeling to simultaneously predict:
            </div>
            <div style="display:flex;flex-direction:column;gap:0.5rem;margin-bottom:1rem;">
                <div style="background:rgba(251,146,60,0.1);border:1px solid rgba(251,146,60,0.3);border-radius:10px;
                     padding:0.55rem 0.8rem;font-size:0.8rem;color:var(--text-primary);display:flex;align-items:center;gap:0.5rem;">
                    <span style="color:#FB923C;font-size:1rem;">⚡</span>
                    <span><b style="color:#FB923C;">Anticancer activity</b> — Active / Inactive classification</span>
                </div>
                <div style="background:rgba(249,115,22,0.1);border:1px solid rgba(249,115,22,0.3);border-radius:10px;
                     padding:0.55rem 0.8rem;font-size:0.8rem;color:var(--text-primary);display:flex;align-items:center;gap:0.5rem;">
                    <span style="color:#F97316;font-size:1rem;">📊</span>
                    <span><b style="color:#F97316;">Anticancer potency</b> — pIC₅₀ prediction</span>
                </div>
            </div>
            <div style="font-size:0.82rem;color:var(--text-secondary);line-height:1.75;margin-bottom:1rem;">
                The framework incorporates state-of-the-art graph neural network architectures, including
                <b style="color:var(--text-primary);">GCN, GAT, GT, and MSMP</b>, together with scaffold-based
                validation and molecular augmentation strategies to enhance model robustness, reliability, and
                generalization toward unseen chemical scaffolds.
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;margin-top:0.8rem;">
                <div style="background:rgba(251,146,60,0.08);border:1px solid rgba(251,146,60,0.2);border-radius:10px;padding:0.7rem;text-align:center;">
                    <div style="font-family:'Space Grotesk';font-weight:700;font-size:1.3rem;color:#FB923C;">3,600+</div>
                    <div style="font-size:0.68rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;font-weight:600;">Training Compounds</div>
                </div>
                <div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.2);border-radius:10px;padding:0.7rem;text-align:center;">
                    <div style="font-family:'Space Grotesk';font-weight:700;font-size:1.3rem;color:#F97316;">0.951</div>
                    <div style="font-size:0.68rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;font-weight:600;">AUC-ROC Score</div>
                </div>
                <div style="background:rgba(234,88,12,0.08);border:1px solid rgba(234,88,12,0.2);border-radius:10px;padding:0.7rem;text-align:center;">
                    <div style="font-family:'Space Grotesk';font-weight:700;font-size:1.3rem;color:#EA580C;">4</div>
                    <div style="font-size:0.68rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;font-weight:600;">GNN Architectures</div>
                </div>
                <div style="background:rgba(253,186,116,0.08);border:1px solid rgba(253,186,116,0.2);border-radius:10px;padding:0.7rem;text-align:center;">
                    <div style="font-family:'Space Grotesk';font-weight:700;font-size:1.3rem;color:#FDBA74;">&lt;200ms</div>
                    <div style="font-size:0.68rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em;font-weight:600;">Inference Speed</div>
                </div>
            </div>
            <div style="margin-top:1rem;padding:0.6rem 0.8rem;background:rgba(239,68,68,0.08);border-left:3px solid #ef4444;border-radius:0 8px 8px 0;font-size:0.72rem;color:var(--text-secondary);">
                ⚠️ <b style="color:#ef4444;">Research tool only.</b> Predictions require experimental validation. Not for clinical use.
            </div>
            """, unsafe_allow_html=True)

    # ── RIGHT: Prediction Command Center ─────────────────────────────────────
    with col_predict:
        with st.container(border=True):
            st.markdown("""
            <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;">
                <div style="background:rgba(251,146,60,0.15);padding:8px;border-radius:10px;">
                    <span style="font-size:1.4rem;">🚀</span>
                </div>
                <div>
                    <h2 style="margin:0;font-family:'Space Grotesk';font-weight:800;font-size:1.15rem;
                               color:#000000;">
                        Prediction Command Center
                    </h2>
                    <p style="margin:0;font-size:0.75rem;color:var(--text-secondary);">
                        GNN-Inference Engine for Anticancer Discovery
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Auto-fill SMILES from FDA selector (driven by session_state["home_smi_bento"])
            quick_smi = st.text_input("Enter SMILES string",
                                     placeholder="Paste SMILES here, or select a drug on the left...",
                                     key="home_smi_bento")

            cp1, cp2 = st.columns([1, 1])
            with cp1:
                model_choice = st.selectbox("Inference Model", [
                    "Random Split — GAT (Recommended)", "Scaffold Split — MSMP"], key="home_model_bento")
            with cp2:
                st.markdown('<div style="margin-top:22px;"></div>', unsafe_allow_html=True)
                predict_btn = st.button("▶  Execute Prediction", key="home_predict_bento", use_container_width=True)

            if quick_smi.strip():
                mol_chk_h = Chem.MolFromSmiles(quick_smi.strip())
                if mol_chk_h:
                    svg_pred = mol_to_svg(quick_smi.strip(), (320, 160), dark=dark_mode)
                    if svg_pred: st.markdown(svg_img_tag(svg_pred), unsafe_allow_html=True)

            if predict_btn and quick_smi.strip():
                mol_chk = Chem.MolFromSmiles(quick_smi.strip())
                if not mol_chk:
                    st.markdown('<div class="warn-box">⚠️ Invalid SMILES string. Please check the molecular structure.</div>', unsafe_allow_html=True)
                else:
                    use_m = gat_random_model if "Random" in model_choice else msmp_scaffold_model
                    with st.spinner("Analyzing graph architecture..."):
                        r = run_predict(use_m, quick_smi.strip())
                    if r:
                        css = "result-active" if r["active"] else "result-inactive"
                        color = "#FB923C" if r["active"] else "#ef4444"
                        icon = "⚡" if r["active"] else "🛡️"
                        st.markdown(f"""
                        <div class="{css}">
                            <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;opacity:0.8;">Inference Result</div>
                            <div class="result-label" style="color:{color};">{icon} {"ACTIVE COMPOUND" if r["active"] else "INACTIVE"}</div>
                            <div class="result-sub">Probability Score: {r["prob"]*100:.1f}%</div>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin-top:0.8rem;">
                            <div class="stat-pill">
                                <div class="stat-lbl">Potency (pIC₅₀)</div>
                                <div class="stat-val" style="color:#FB923C;font-size:1.3rem;">{r["pic50"]}</div>
                            </div>
                            <div class="stat-pill">
                                <div class="stat-lbl">Potency (IC₅₀)</div>
                                <div class="stat-val" style="color:#F97316;font-size:1.1rem;">{r["ic50"]} <small style="font-size:0.55em;">µM</small></div>
                            </div>
                        </div>""", unsafe_allow_html=True)
            elif not predict_btn:
                st.markdown(f'<div style="height:80px;display:flex;align-items:center;justify-content:center;text-align:center;color:var(--text-secondary);font-size:0.78rem;opacity:0.7;">Select a drug on the left or paste a SMILES, then click ▶ Execute Prediction</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # end page-section

    # ── Workflow image — full width below 3 cols ───────────────────────────
    if os.path.exists("web.png"):
        import base64
        try:
            with open("web.png", "rb") as f_img:
                b64_img = base64.b64encode(f_img.read()).decode()
            
            with st.container(border=True):
                st.markdown(f"""
                <div style="padding:0.5rem;">
                    <div style="font-family:\'Space Grotesk\';font-weight:700;font-size:1.1rem;
                         color:var(--text-primary);margin-bottom:1.2rem;display:flex;align-items:center;gap:.5rem;">
                        📐 GraphACSM-Net — Methodology Workflow
                    </div>
                    <div style="background:white;padding:1.5rem;border-radius:10px;display:flex;justify-content:center;align-items:center;
                         box-shadow:inset 0 0 15px rgba(0,0,0,0.05);">
                        <img src="data:image/png;base64,{b64_img}" style="max-width:100%;border-radius:6px;height:auto;" />
                    </div>
                    <div style="font-size:.78rem;color:var(--text-secondary);text-align:center;margin-top:.8rem;">
                        Complete methodology pipeline from dataset curation to web deployment
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

    # ── Performance summary ───────────────────────────────────────────────
    st.markdown('<div class="section-heading" style="margin-top:2rem;">Model Performance Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Best model per split condition — 5-fold cross-validation on test set</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2,gap="medium")
    with c1:
        with st.container(border=True):
            st.markdown("""
            <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:.95rem;margin-bottom:.3rem;">
                🔵 Random Split — GAT (Graph Attention Network)
            </div>
            <div style="font-size:.82rem;color:var(--text-secondary);margin-bottom:.8rem;">
                Best model for random split with augmentation
            </div>
            <div class="stat-grid">
                <div class="stat-pill"><div class="stat-val">0.936</div><div class="stat-lbl">AUC-ROC</div></div>
                <div class="stat-pill"><div class="stat-val">87.0%</div><div class="stat-lbl">Accuracy</div></div>
                <div class="stat-pill"><div class="stat-val">0.742</div><div class="stat-lbl">MCC</div></div>
                <div class="stat-pill"><div class="stat-val">0.684</div><div class="stat-lbl">R²</div></div>
            </div>
            """, unsafe_allow_html=True)
    with c2:
        with st.container(border=True):
            st.markdown("""
            <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-green);font-size:.95rem;margin-bottom:.3rem;">
                🟢 Scaffold Split — MSMP (Multi-head Self-Attention Message Passing)
            </div>
            <div style="font-size:.82rem;color:var(--text-secondary);margin-bottom:.8rem;">
                Best model for scaffold split with augmentation
            </div>
            <div class="stat-grid">
                <div class="stat-pill"><div class="stat-val green">0.899</div><div class="stat-lbl">AUC-ROC</div></div>
                <div class="stat-pill"><div class="stat-val green">82.1%</div><div class="stat-lbl">Accuracy</div></div>
                <div class="stat-pill"><div class="stat-val green">0.648</div><div class="stat-lbl">MCC</div></div>
                <div class="stat-pill"><div class="stat-val green">0.594</div><div class="stat-lbl">R²</div></div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT — RANDOM
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "predict_random":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">🔵 Random Split Prediction</div><div class="hero-sub" style="margin-bottom:.5rem;">Best Model: Graph Attention Network (GAT) · AUC=0.936 · ACC=87.0%</div><div class="hero-badges"><span class="hero-badge green">✓ Recommended for general use</span><span class="hero-badge">Random stratified split</span><span class="hero-badge">With SMILES augmentation</span></div></div>', unsafe_allow_html=True)
    if not gat_random_loaded: st.markdown('<div style="padding:1rem 2.5rem 0;"><div class="warn-box">⚠️ gat_random_aug_fold3_seed42.pt not found — running in demo mode.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    predict_panel(gat_random_model,"GAT","▶  Run GAT Prediction","r_smi","r_run")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT — SCAFFOLD
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "predict_scaffold":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">🟢 Scaffold Split Prediction</div><div class="hero-sub" style="margin-bottom:.5rem;">Best Model: Multi-head Self-Attention Message Passing (MSMP) · AUC=0.899 · ACC=82.1%</div><div class="hero-badges"><span class="hero-badge">Murcko scaffold split</span><span class="hero-badge green">✓ More rigorous generalisation</span></div></div>', unsafe_allow_html=True)
    if not msmp_scaffold_loaded: st.markdown('<div style="padding:1rem 2.5rem 0;"><div class="warn-box">⚠️ msmp_scaffold_aug_fold4_seed42.pt not found — running in demo mode.</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="padding:.5rem 2.5rem 0;"><div class="info-box">ℹ️ Scaffold split tests on structurally novel chemical series. The <a href=f"?page=predict_random&dark={cur_theme}" target="_self" style="color:var(--accent-blue);font-weight:600;">Random Split GAT model</a> achieves higher accuracy and is recommended for general screening.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    predict_panel(msmp_scaffold_model,"MSMP","▶  Run MSMP Prediction","s_smi","s_run")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "batch":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📋 Batch Prediction</div><div class="hero-sub">High-throughput screening for multiple molecules</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    bm=st.selectbox("Model for batch prediction",["Random Split — GAT (Recommended)","Scaffold Split — MSMP"])
    use_bm=gat_random_model if "Random" in bm else msmp_scaffold_model
    tab1,tab2=st.tabs(["  📁  Upload CSV  ","  ✏️  Paste SMILES  "])
    with tab1:
        st.markdown('<div class="info-box">Upload a CSV with a <b>SMILES</b> column. All other columns are preserved.</div>', unsafe_allow_html=True)
        uploaded=st.file_uploader("Upload CSV",type=["csv"])
        if uploaded:
            df_in=pd.read_csv(uploaded)
            st.write(f"**{len(df_in)}** rows loaded"); st.dataframe(df_in.head(5),use_container_width=True)
            smi_col=next((c for c in df_in.columns if c.lower()=="smiles"),None)
            if not smi_col: st.markdown('<div class="warn-box">⚠️ No SMILES column found</div>',unsafe_allow_html=True)
            else:
                if st.button(f"▶  Predict all {len(df_in)} molecules"):
                    prog=st.progress(0); results=[]
                    for i,smi in enumerate(df_in[smi_col].astype(str)):
                        r=run_predict(use_bm,smi)
                        results.append({"Activity":"Active" if r and r["active"] else "Inactive","Probability":r["prob"] if r else None,"pIC50":r["pic50"] if r else None,"IC50_uM":r["ic50"] if r else None})
                        prog.progress((i+1)/len(df_in))
                    df_out=df_in.copy()
                    for k in ["Activity","Probability","pIC50","IC50_uM"]: df_out[k]=[r[k] for r in results]
                    prog.empty()
                    n_act=sum(1 for r in results if r["Activity"]=="Active")
                    c1b,c2b,c3b=st.columns(3)
                    c1b.metric("Total",len(results)); c2b.metric("Active",n_act); c3b.metric("Inactive",len(results)-n_act)
                    st.dataframe(df_out,use_container_width=True)
                    st.download_button("⬇️ Download Results",df_out.to_csv(index=False),"predictions.csv","text/csv")
    with tab2:
        bulk=st.text_area("One SMILES per line",height=180,placeholder="CC(C)Cc1ccc(cc1)C(C)C(=O)O\nCOc1ccc2cc(C(C)C(=O)O)ccc2c1")
        if st.button("▶  Predict Pasted SMILES"):
            lines=[l.strip() for l in bulk.strip().split("\n") if l.strip()]
            if lines:
                with st.spinner(f"Predicting {len(lines)} molecules…"):
                    res=[{"SMILES":s,"Activity":"Active" if (r:=run_predict(use_bm,s)) and r["active"] else "Inactive","Probability":r["prob"] if r else None,"pIC50":r["pic50"] if r else None,"IC50_uM":r["ic50"] if r else None} for s in lines]
                df_r=pd.DataFrame(res)
                st.dataframe(df_r,use_container_width=True)
                st.download_button("⬇️ Download",df_r.to_csv(index=False),"predictions.csv","text/csv")
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "algorithm":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📐 Algorithm & Methodology</div><div class="hero-sub">GraphACSM-Net — Graph Neural Network Framework</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)

    if os.path.exists("web.png"):
        import base64
        try:
            with open("web.png", "rb") as f_img:
                b64_img = base64.b64encode(f_img.read()).decode()
            
            with st.container(border=True):
                st.markdown(f"""
                <div style="padding:0.5rem;">
                    <div style="font-family:\'Space Grotesk\';font-weight:700;font-size:1.1rem;
                         color:var(--text-primary);margin-bottom:1.2rem;display:flex;align-items:center;gap:.5rem;">
                        📐 GraphACSM-Net — Methodology Workflow
                    </div>
                    <div style="background:white;padding:1.5rem;border-radius:10px;display:flex;justify-content:center;align-items:center;
                         box-shadow:inset 0 0 15px rgba(0,0,0,0.05);">
                        <img src="data:image/png;base64,{b64_img}" style="max-width:100%;border-radius:6px;height:auto;" />
                    </div>
                    <div style="font-size:.78rem;color:var(--text-secondary);text-align:center;margin-top:.8rem;">
                        Complete methodology pipeline from dataset curation to web deployment
                    </div>
                </div>
                """, unsafe_allow_html=True)
        except Exception:
            pass

    for num,title,color,desc in [
        ("1","Dataset Curation","#1B4FA8","3,600 compounds with pIC₅₀ values from PubChem BioAssay. Binary activity: pIC₅₀ ≥ 7.0 → Active. Regression targets standardized (zero mean, unit variance)."),
        ("2","Data Pre-processing","#0D7A5F","Duplicate SMILES removal, missing pIC₅₀ excluded, binary activity labelling, regression target standardization."),
        ("3","Dataset Split","#7B3FA8","Random stratified 80:20 split and Murcko scaffold split. 5-fold cross-validation on both."),
        ("4","Molecular Graph Generation","#1B4FA8","Node features (21-dim): atom type, degree, formal charge, H count, aromaticity, ring, chirality, hybridization. Edge features (7-dim): bond order, conjugation, ring, stereo. Global descriptors (13-dim): MW, LogP, TPSA, QED, HBD/A, Fsp3, ring count, BertzCT, heteroatoms."),
        ("5","Data Augmentation","#0D7A5F","N=3 random SMILES enumerations per molecule, graph edge dropout (15–18%), node feature dropout (5%), test-time augmentation (TTA, 3 variants averaged)."),
        ("6","GNN Models","#7B3FA8","GCN · GAT · GT · MSMP — multi-task learning with Focal Loss + Huber Loss. Class-weighted sampling. Early stopping on composite score (0.6×ACC + 0.4×R²). Ensembling across 2–3 seeds."),
        ("7","Internal Validation","#1B4FA8","5-fold CV per condition. Classification: ACC, AUC, MCC, SN, SP, F1, Precision, BalACC. Regression: R², RMSE, MAE. Statistical significance: paired t-test."),
        ("8","Performance","#0D7A5F","Best models: GAT for random split (AUC=0.936, ACC=87.0%), MSMP for scaffold split (AUC=0.899, ACC=82.1%). External validation on FDA-approved drugs."),
    ]:
        st.markdown(f"""<div class="card" style="margin-bottom:.8rem;border-left:4px solid {color};">
        <div style="display:flex;gap:1rem;align-items:flex-start;">
            <div style="background:{color};color:white;border-radius:50%;width:28px;height:28px;
                 display:flex;align-items:center;justify-content:center;
                 font-family:\'Space Grotesk\';font-weight:700;font-size:.85rem;flex-shrink:0;">{num}</div>
            <div><div style="font-family:\'Space Grotesk\';font-weight:700;font-size:.92rem;color:var(--text-primary);margin-bottom:.25rem;">{title}</div>
            <div style="font-size:.85rem;color:var(--text-secondary);line-height:1.65;">{desc}</div></div>
        </div></div>""", unsafe_allow_html=True)

    # performance table
    st.markdown('<div style="margin-top:1.5rem;"><div class="section-heading">Performance Tables</div></div>', unsafe_allow_html=True)
    tab_r,tab_s=st.tabs(["Random Split","Scaffold Split"])
    def mk_table(rows, best):
        hdr = "".join("<th>" + h + "</th>" for h in ["Model","AUC","ACC","MCC","SN","SP","R²","RMSE","MAE"])
        body = ""
        for r in rows:
            is_best = r[0] == best
            star = " ★" if is_best else ""
            model_cell = '<td class="model-col">' + r[0] + star + "</td>"
            data_cells = ""
            for i, v in enumerate(r[1:]):
                if is_best:
                    data_cells += '<td class="best">' + v + "</td>"
                else:
                    data_cells += "<td>" + v + "</td>"
            body += "<tr>" + model_cell + data_cells + "</tr>"
        return '<div class="card" style="padding:0;overflow:hidden;margin-bottom:1rem;"><table class="perf-table"><thead><tr>' + hdr + '</tr></thead><tbody>' + body + '</tbody></table></div>'
    with tab_r:
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>No Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.917","0.842","0.685","0.842","0.842","0.598","0.748","0.559"],["GAT","0.929","0.856","0.714","0.849","0.863","0.641","0.707","0.528"],["GT","0.924","0.849","0.702","0.818","0.881","0.604","0.742","0.555"],["MSMP","0.930","0.857","0.716","0.829","0.884","0.616","0.731","0.548"]],"MSMP"),unsafe_allow_html=True)
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>With Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.925","0.854","0.710","0.854","0.854","0.644","0.703","0.518"],["GAT","0.936","0.870","0.742","0.871","0.869","0.684","0.663","0.488"],["GT","0.933","0.862","0.724","0.861","0.863","0.662","0.686","0.505"],["MSMP","0.935","0.864","0.730","0.872","0.856","0.654","0.693","0.509"]],"GAT"),unsafe_allow_html=True)
    with tab_s:
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>No Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.873","0.804","0.607","0.777","0.824","0.464","0.864","0.657"],["GAT","0.885","0.813","0.628","0.770","0.854","0.485","0.845","0.649"],["GT","0.879","0.805","0.609","0.776","0.830","0.458","0.865","0.656"],["MSMP","0.886","0.813","0.629","0.788","0.841","0.502","0.833","0.641"]],"MSMP"),unsafe_allow_html=True)
        st.markdown("<b style='font-size:.82rem;color:#6B7A99;'>With Augmentation</b>",unsafe_allow_html=True)
        st.markdown(mk_table([["GCN","0.885","0.812","0.628","0.765","0.855","0.524","0.812","0.614"],["GAT","0.895","0.824","0.647","0.818","0.826","0.559","0.782","0.596"],["GT","0.896","0.821","0.642","0.810","0.829","0.550","0.790","0.598"],["MSMP","0.899","0.821","0.648","0.765","0.879","0.549","0.790","0.594"]],"MSMP"),unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "dataset":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📂 Dataset</div><div class="hero-sub">Training and evaluation dataset details</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    c1d,c2d=st.columns(2,gap="medium")
    with c1d:
        st.markdown("""<div class="card card-blue">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Dataset Statistics</div>
        <table class="perf-table" style="font-size:.85rem;">
            <tr><td><b>Source</b></td><td>PubChem BioAssay</td></tr>
            <tr><td><b>Total Compounds</b></td><td>3,600</td></tr>
            <tr><td><b>Active (pIC₅₀ ≥ 7)</b></td><td>1,800 (50%)</td></tr>
            <tr><td><b>Inactive (pIC₅₀ &lt; 7)</b></td><td>1,800 (50%)</td></tr>
            <tr><td><b>pIC₅₀ Mean ± Std</b></td><td>6.45 ± 1.12</td></tr>
            <tr><td><b>Tanimoto Similarity</b></td><td>0.117 ± 0.062</td></tr>
            <tr><td><b>Node Features</b></td><td>21-dim</td></tr>
            <tr><td><b>Edge Features</b></td><td>7-dim</td></tr>
            <tr><td><b>Global Descriptors</b></td><td>13-dim</td></tr>
        </table></div>""", unsafe_allow_html=True)
    with c2d:
        st.markdown("""<div class="card card-green">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-green);font-size:1rem;margin-bottom:.8rem;">Pre-processing Steps</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.9;">
            ✅ Duplicate SMILES removed<br>
            ✅ Missing pIC₅₀ values excluded<br>
            ✅ Binary activity labels (threshold = 7.0)<br>
            ✅ Regression targets standardized<br>
            ✅ Invalid SMILES filtered via RDKit<br>
            ✅ Class balance verified (50:50)<br>
            ✅ Tanimoto similarity analysis performed
        </div>
        <div style="margin-top:1rem;font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-green);font-size:.9rem;">External Validation</div>
        <div style="font-size:.85rem;color:var(--text-primary);margin-top:.3rem;">FDA-approved drugs used as independent external validation set.</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="info-box" style="margin-top:1rem;">📥 To download the training dataset, please contact the development team.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELP
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "help":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">❓ Help & Documentation</div><div class="hero-sub">Guidance on using GraphACSM-Net</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    for title,color,content in [
        ("📖 How to Use","#1B4FA8","<b>Step 1:</b> Click <b>Predict (Random)</b> in the navbar (recommended).<br><b>Step 2:</b> Enter a SMILES string or select an example molecule.<br><b>Step 3:</b> Click Run Prediction → get Activity, pIC₅₀, IC₅₀, and properties.<br><b>Step 4:</b> For multiple molecules, use the <b>Batch</b> page."),
        ("⌨️ Input Format","#0D7A5F","Molecules must be in <b>SMILES</b> format.<br><br>Examples:<br>• Ibuprofen: <code>CC(C)Cc1ccc(cc1)C(C)C(=O)O</code><br>• Aspirin: <code>CC(=O)Oc1ccccc1C(=O)O</code><br><br>Batch CSV must have a column named <b>SMILES</b>."),
        ("📊 Understanding Results","#7B3FA8","<b>Activity:</b> Active (pIC₅₀ ≥ 7.0) or Inactive<br><b>Confidence:</b> Sigmoid probability 0–1<br><b>pIC₅₀:</b> Higher = more potent<br><b>IC₅₀ (µM):</b> 10^(7 − pIC₅₀)<br><br><b>Which model?</b> Random Split GAT is recommended for general use (higher accuracy)."),
        ("❓ FAQ","#0D2E6E","<b>Q: Is this free?</b> Yes, completely free for research.<br><b>Q: Can I use this clinically?</b> No — research use only.<br><b>Q: Why is scaffold split accuracy lower?</b> It tests on structurally novel compounds not seen during training — a harder, more realistic evaluation.<br><b>Q: Why two models?</b> Random split GAT is best overall. Scaffold split MSMP provides more conservative estimates for novel chemical series."),
    ]:
        st.markdown(f'<div class="card" style="margin-bottom:1rem;border-left:4px solid {color};"><div style="font-family:\'Space Grotesk\';font-weight:700;font-size:.92rem;color:{color};margin-bottom:.6rem;">{title}</div><div style="font-size:.87rem;color:var(--text-primary);line-height:1.75;">{content}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CONTACT
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# CONTACT
# ══════════════════════════════════════════════════════════════════════════════
elif PAGE == "contact":
    st.markdown('<div class="hero-banner" style="padding:1.5rem 3rem;"><div class="hero-title" style="font-size:1.8rem;">📬 Contact Us</div><div class="hero-sub">Get in touch with the GraphACSM-Net team</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-section">', unsafe_allow_html=True)
    
    # Use a two-column layout to organize the 5 contacts cleanly
    c1, c2 = st.columns(2, gap="large")
    
    with c1:
        st.markdown("""<div class="card card-blue" style="margin-bottom: 1.2rem;">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Principal Investigator</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.8;">
            <b>Dr. Thirumurthy Madhavan, Ph.D.</b><br>
            Associate Professor<br>
            Principal Investigator — Computational Biology Lab<br>
            Department of Genetic Engineering<br>
            School of Bioengineering<br><br>
            <span style="color:#6B7A99;">SRM Institute of Science and Technology<br>
            Kattankulathur, Kanchipuram District<br>
            Tamil Nadu, India 603203</span><br><br>
            <b>Mobile:</b> +91-9944572918<br>
            <b>Website:</b> <a href="https://www.srmist.edu.in" target="_blank" style="color:var(--accent-blue);">www.srmist.edu.in</a>
        </div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card card-blue" style="margin-bottom: 1.2rem;">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Ms. Priya Dharshini B. PhD</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.8;">
            Research Scholar<br>
            Computational lab<br>
            Department for Genetic Engineering<br>
            School of Bioengineering<br>
            SRMIST
        </div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card card-blue" style="margin-bottom: 1.2rem;">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Ms. Subathra Selvam. PhD</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.8;">
            Research Scholar<br>
            Computational lab<br>
            Department for Genetic Engineering<br>
            School of Bioengineering<br>
            SRMIST
        </div></div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""<div class="card card-blue" style="margin-bottom: 1.2rem;">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Dr. A. Revathi, PhD</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.8;">
            Associate Professor<br>
            Computational Lab<br>
            Department of Computational Intelligence<br>
            School of Computing<br>
            SRMIST
        </div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card card-blue" style="margin-bottom: 1.2rem;">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Priyanshu Goyal</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.8;">
            Bachelor in Computer Science and Engineering with Specialization in Artificial Intelligence and Machine Learning<br>
            Department of Computational Intelligence<br>
            School of Computing<br>
            SRMIST
        </div></div>""", unsafe_allow_html=True)

        st.markdown("""<div class="card card-blue" style="margin-bottom: 1.2rem;">
        <div style="font-family:\'Space Grotesk\';font-weight:700;color:var(--accent-blue);font-size:1rem;margin-bottom:.8rem;">Yadhisresht Harikrishnan</div>
        <div style="font-size:.87rem;color:var(--text-primary);line-height:1.8;">
            Bachelor in Computer Science and Engineering with Specialization in Artificial Intelligence and Machine Learning<br>
            Department of Computational Intelligence<br>
            School of Computing<br>
            SRMIST
        </div></div>""", unsafe_allow_html=True)
        
    st.markdown("""
    <div style="margin-top:1.5rem; text-align: center; font-size:.83rem; color:#4A5568; line-height:1.65;">
        <b style="color:var(--accent-green);">Disclaimer:</b> GraphACSM-Net is a computational research tool.
        Predictions require experimental validation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="site-footer">
    © 2026 GraphACSM-Net &nbsp;|&nbsp; All rights reserved &nbsp;|&nbsp;
    Developed for research by SRMIST &nbsp;|&nbsp;
    <a href="?page=help&dark={cur_theme}" target="_self">Help</a> &nbsp;|&nbsp;
    <a href="?page=contact&dark={cur_theme}" target="_self">Contact</a>
</div>
""", unsafe_allow_html=True)
