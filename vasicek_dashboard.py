# ============================================================
# VASICEK INTEREST RATE MODEL — STREAMLIT DASHBOARD
# Simulates future SBP (State Bank of Pakistan) policy rates
# using the Vasicek stochastic differential equation model.
# ============================================================

# ── Imports ──────────────────────────────────────────────────────────────────
# streamlit   : builds the interactive web dashboard UI
# numpy       : numerical computations (random numbers, statistics, arrays)
# pandas      : loading and manipulating the CSV data
# plotly      : interactive charts (line, histogram, fan chart)
# datetime    : used to timestamp the live rate and the navbar clock
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import requests
import re
from bs4 import BeautifulSoup

# ── Page Configuration ───────────────────────────────────────────────────────
# Sets the browser tab title, uses the full screen width,
# and hides the sidebar by default so the layout is clean on load.
st.set_page_config(page_title="Vasicek Rate Engine", layout="wide", initial_sidebar_state="collapsed")

# ── Dark / Light Mode State ───────────────────────────────────────────────────
# st.session_state persists values across reruns of the app.
# On first load, dark_mode is set to True (dark theme is the default).
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Shorthand: D is True when in dark mode, False when in light mode.
D = st.session_state.dark_mode

# ── Theme Token Dictionary ────────────────────────────────────────────────────
# T is a dictionary of colour hex codes used throughout the UI.
# Every colour has two values: one for dark mode (D=True) and one for light.
# Using a central dictionary makes it easy to switch themes without
# hunting down colours scattered across the entire file.
T = {
    "app_bg":        "#030308" if D else "#f0f2f6",   # main background
    "navbar_bg":     "#0a0a14" if D else "#ffffff",   # top nav bar background
    "navbar_border": "#1a1a2e" if D else "#e0e0e8",   # nav bar bottom border
    "brand_color":   "#ffffff" if D else "#0a0a14",   # brand name text
    "sub_color":     "#4444aa" if D else "#888888",   # subtitle / muted text
    "time_color":    "#333366" if D else "#aaaaaa",   # timestamp text in nav
    "badge_bg":      "#1a1a2e" if D else "#f0f0f8",   # "SBP · PKR · LIVE" pill background
    "badge_border":  "#2a2a4e" if D else "#ddddee",   # pill border
    "badge_color":   "#7777cc" if D else "#555588",   # pill text
    "card_bg":       "#0a0a14" if D else "#ffffff",   # stat/chart card background
    "card_border":   "#1a1a2e" if D else "#e0e0e8",   # card border
    "label_color":   "#444488" if D else "#999999",   # small uppercase labels
    "value_color":   "#ffffff" if D else "#0a0a14",   # large stat values
    "delta_neutral": "#333366" if D else "#aaaaaa",   # neutral delta text (no direction)
    "chart_title":   "#8888cc" if D else "#333333",   # chart section headings
    "chart_tag_bg":  "#1a1a2e" if D else "#f0f0f8",   # tag pill background (e.g. "VASICEK · 500 PATHS")
    "chart_tag_col": "#5555aa" if D else "#666688",   # tag pill text
    "slider_color":  "#7c3aed" if D else "#1a1a2e",   # slider thumb/track colour
    "btn_bg":        "#7c3aed" if D else "#1a1a2e",   # button gradient start
    "btn_hover":     "#6d28d9" if D else "#2d2d50",   # button hover colour (CSS only)
    "input_bg":      "#0a0a14" if D else "#ffffff",   # text input background
    "input_color":   "#aaaacc" if D else "#333333",   # text input font colour
    "input_border":  "#1a1a2e" if D else "#dddddd",   # text input border
    "plot_bg":       "#0a0a14" if D else "#ffffff",   # plotly chart plot area
    "paper_bg":      "#030308" if D else "#f0f2f6",   # plotly chart outer background
    "grid_color":    "#0f0f20" if D else "#f0f0f8",   # chart grid lines
    "tick_color":    "#444488" if D else "#999999",   # axis tick labels
    "legend_bg":     "#0a0a14" if D else "#ffffff",   # chart legend background
    "legend_border": "#1a1a2e" if D else "#e0e0e8",   # chart legend border
    "path_color":    "rgba(124,58,237,0.15)" if D else "rgba(37,99,235,0.12)",  # individual sim paths
    "ci_fill":       "rgba(124,58,237,0.08)" if D else "rgba(37,99,235,0.05)",  # 95% CI shaded band
    "ci_fill2":      "rgba(124,58,237,0.14)" if D else "rgba(37,99,235,0.10)",  # 68% CI shaded band (darker)
    "mean_color":    "#a78bfa" if D else "#1a1a2e",   # mean forecast line colour
    "current_color": "#06b6d4" if D else "#dc2626",   # current rate reference line
    "hist_fill":     "rgba(124,58,237,0.1)"  if D else "rgba(37,99,235,0.06)",  # historical area fill
    "hist_line":     "#7c3aed" if D else "#2563eb",   # historical line colour
    "hist_bar":      "rgba(167,139,250,0.6)" if D else "rgba(26,26,46,0.7)",    # histogram bar fill
    "hist_bar_line": "rgba(167,139,250,0.9)" if D else "rgba(26,26,46,0.9)",    # histogram bar edge
    "vline_color":   "#f59e0b" if D else "#dc2626",   # long-run mean reference line
    "footer_color":  "#1a1a2e" if D else "#cccccc",   # footer text colour
    "footer_border": "#1a1a2e" if D else "#e0e0e8",   # footer top border
    "toggle_icon":   "☀️" if D else "🌙",              # icon on the dark/light toggle button
    "toggle_label":  "Light" if D else "Dark",        # label on the toggle button
    "accent1":       "#7c3aed" if D else "#2563eb",   # primary accent (purple / blue)
    "accent2":       "#06b6d4" if D else "#dc2626",   # secondary accent (cyan / red)
}

# ── Global CSS + Animated Particle Canvas ────────────────────────────────────
# Injects a large block of CSS that:
#   1. Imports three Google Fonts (Bebas Neue for headings, Outfit for body,
#      JetBrains Mono for monospaced/code-like labels).
#   2. Resets margins/padding globally.
#   3. Styles the Streamlit app background, hides the default Streamlit header/footer.
#   4. Defines styles for: navbar, stat cards, chart wrappers, sliders,
#      buttons, download buttons, text inputs, insight boxes, and sensitivity table.
#
# Also injects a <canvas> element and a JavaScript animation that draws:
#   - 80 floating particles that bounce around the background.
#   - Lines between particles that are close together (< 120px apart),
#     creating a "connected network" visual effect.
# The canvas is fixed behind all other content (z-index:0, pointer-events:none).
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box;}}
.stApp{{background-color:{T['app_bg']} !important;color:{T['value_color']};}}
#MainMenu,footer,header{{visibility:hidden !important;}}
#particle-canvas{{position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;opacity:{'0.5' if D else '0.15'};}}
.navbar{{position:relative;z-index:10;display:flex;justify-content:space-between;align-items:center;background:{T['navbar_bg']};border-bottom:1px solid {T['navbar_border']};padding:14px 36px;margin-bottom:24px;{'box-shadow:0 1px 30px rgba(124,58,237,0.12);' if D else 'box-shadow:0 1px 10px rgba(0,0,0,0.06);'}}}
.navbar-left{{display:flex;align-items:center;gap:16px;}}
.navbar-brand{{font-family:'Bebas Neue',sans-serif;font-size:22px;color:{T['brand_color']};letter-spacing:2px;{'text-shadow:0 0 20px rgba(167,139,250,0.4);' if D else ''}}}
.navbar-brand span{{color:{T['accent1']};}}
.navbar-divider{{width:1px;height:18px;background:{T['navbar_border']};}}
.navbar-sub{{font-family:'Outfit',sans-serif;font-size:12px;color:{T['sub_color']};}}
.navbar-time{{font-family:'JetBrains Mono',monospace;font-size:11px;color:{T['time_color']};}}
.navbar-badge{{background:{T['badge_bg']};border:1px solid {T['badge_border']};color:{T['badge_color']};font-family:'JetBrains Mono',monospace;font-size:10px;padding:4px 12px;border-radius:20px;font-weight:600;letter-spacing:1px;}}
.stat-row{{display:flex;gap:14px;margin-bottom:24px;position:relative;z-index:5;}}
.stat-card{{flex:1;background:{T['card_bg']};border:1px solid {T['card_border']};border-radius:16px;padding:22px 26px;position:relative;overflow:hidden;{'box-shadow:0 0 0 1px rgba(124,58,237,0.08);' if D else 'box-shadow:0 2px 12px rgba(0,0,0,0.06);'}}}
.stat-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),transparent);}}
.stat-card::after{{content:'';position:absolute;top:-50%;right:-20%;width:120px;height:120px;border-radius:50%;background:radial-gradient(circle,var(--accent) 0%,transparent 70%);opacity:{'0.06' if D else '0.03'};}}
.stat-card.c1{{--accent:#7c3aed;}}.stat-card.c2{{--accent:#06b6d4;}}.stat-card.c3{{--accent:#10b981;}}.stat-card.c4{{--accent:#f59e0b;}}
.stat-label{{font-family:'JetBrains Mono',monospace;font-size:10px;color:{T['label_color']};text-transform:uppercase;letter-spacing:2px;margin-bottom:10px;}}
.stat-value{{font-family:'Bebas Neue',sans-serif;font-size:36px;color:{T['value_color']};line-height:1;letter-spacing:1px;}}
.stat-delta{{font-family:'Outfit',sans-serif;font-size:11px;margin-top:8px;font-weight:500;}}
.stat-delta.up{{color:#10b981;}}.stat-delta.down{{color:#ef4444;}}.stat-delta.neutral{{color:{T['delta_neutral']};}}
.chart-wrap{{background:{T['card_bg']};border:1px solid {T['card_border']};border-radius:16px;padding:20px;position:relative;z-index:5;margin-bottom:16px;{'box-shadow:0 0 0 1px rgba(124,58,237,0.06),0 8px 32px rgba(0,0,0,0.3);' if D else 'box-shadow:0 2px 16px rgba(0,0,0,0.06);'}}}
.chart-header{{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}}
.chart-title{{font-family:'Outfit',sans-serif;font-size:13px;font-weight:600;color:{T['chart_title']};}}
.chart-tag{{font-family:'JetBrains Mono',monospace;font-size:9px;background:{T['chart_tag_bg']};color:{T['chart_tag_col']};padding:3px 10px;border-radius:20px;font-weight:600;letter-spacing:1px;border:1px solid {T['card_border']};}}
.stSlider>div>div>div{{background:{T['slider_color']} !important;}}
.stSlider label{{font-family:'Outfit',sans-serif !important;color:{T['sub_color']} !important;font-size:11px !important;}}
.stButton>button{{background:linear-gradient(135deg,{T['btn_bg']},{T['accent2']}) !important;color:#ffffff !important;border:none !important;border-radius:10px !important;font-family:'Outfit',sans-serif !important;font-weight:600 !important;font-size:12px !important;padding:10px 16px !important;{'box-shadow:0 0 20px rgba(124,58,237,0.3) !important;' if D else 'box-shadow:0 2px 8px rgba(0,0,0,0.15) !important;'}}}
.stButton>button p,.stButton>button span,.stButton>button div{{color:#ffffff !important;}}
.stButton>button:hover{{opacity:0.88 !important;transform:translateY(-1px) !important;}}
.stDownloadButton>button{{background:linear-gradient(135deg,{T['btn_bg']},{T['accent2']}) !important;color:#ffffff !important;border:none !important;border-radius:10px !important;font-family:'Outfit',sans-serif !important;font-weight:600 !important;font-size:12px !important;padding:10px 16px !important;{'box-shadow:0 0 20px rgba(124,58,237,0.3) !important;' if D else 'box-shadow:0 2px 8px rgba(0,0,0,0.15) !important;'}}}
.stDownloadButton>button,.stDownloadButton>button *,.stDownloadButton>button p,.stDownloadButton>button span,.stDownloadButton>button div{{color:#ffffff !important;}}
.stDownloadButton>button:hover{{opacity:0.88 !important;transform:translateY(-1px) !important;}}
.stTextInput>div>div>input{{background:{T['input_bg']} !important;color:{T['input_color']} !important;border:1px solid {T['input_border']} !important;border-radius:8px !important;font-family:'JetBrains Mono',monospace !important;font-size:11px !important;}}
.control-wrap{{background:{T['card_bg']};border:1px solid {T['card_border']};border-radius:12px;padding:16px 20px;margin-bottom:20px;position:relative;z-index:5;}}
.insight-box{{background:{'rgba(124,58,237,0.06)' if D else 'rgba(37,99,235,0.04)'};border-left:3px solid {T['accent1']};border-radius:0 10px 10px 0;padding:12px 16px;margin-top:10px;margin-bottom:4px;}}
.insight-text{{font-family:'Outfit',sans-serif;font-size:12.5px;color:{T['chart_title']};line-height:1.7;font-weight:400;}}
.insight-text strong{{color:{T['value_color']};font-weight:600;}}
.sens-table{{width:100%;border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:11px;}}
.sens-table th{{background:{T['chart_tag_bg']};color:{T['chart_tag_col']};padding:8px 12px;text-align:center;letter-spacing:1px;font-size:10px;border:1px solid {T['card_border']};}}
.sens-table td{{padding:8px 12px;text-align:center;border:1px solid {T['card_border']};color:{T['value_color']};}}
.sens-table tr:hover td{{background:{'rgba(124,58,237,0.06)' if D else 'rgba(37,99,235,0.04)'};}}
.sens-table .highlight{{color:{T['accent1']};font-weight:600;}}
</style>
<canvas id="particle-canvas"></canvas>
<script>
(function(){{
  const c=document.getElementById('particle-canvas');
  if(!c)return;
  const ctx=c.getContext('2d');
  c.width=window.innerWidth;c.height=window.innerHeight;
  const pts=Array.from({{length:80}},()=>({{x:Math.random()*c.width,y:Math.random()*c.height,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*1.5+.3,a:Math.random()*.4+.1}}));
  function draw(){{
    ctx.clearRect(0,0,c.width,c.height);
    for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){{
      const dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);
      if(d<120){{ctx.beginPath();ctx.strokeStyle=`rgba(124,58,237,${{.12*(1-d/120)}})`;ctx.lineWidth=.4;ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.stroke();}}
    }}
    pts.forEach(p=>{{p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>c.width)p.vx*=-1;if(p.y<0||p.y>c.height)p.vy*=-1;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle=`rgba(124,58,237,${{p.a}})`;ctx.fill();}});
    requestAnimationFrame(draw);
  }}
  draw();
  window.addEventListener('resize',()=>{{c.width=window.innerWidth;c.height=window.innerHeight;}});
}})();
</script>
""", unsafe_allow_html=True)

# ── Data Loading Function ─────────────────────────────────────────────────────
# @st.cache_data means Streamlit caches the result; the CSV is only read from
# disk once per session (or until the file changes), not on every page rerun.
# Steps:
#   1. Read the CSV file.
#   2. Strip any accidental whitespace from column names.
#   3. Convert the 'Date' column to proper datetime objects.
#   4. Sort by date (oldest first) and reset the row index.
#   5. Convert 'Rate' from percentage (e.g. 13.0) to decimal (0.13),
#      because the Vasicek math works in decimal form.
@st.cache_data
def load_data(path):
    df=pd.read_csv(path); df.columns=df.columns.str.strip()
    df['Date']=pd.to_datetime(df['Date']); df=df.sort_values('Date').reset_index(drop=True)
    df['Rate']=df['Rate']/100; return df

# ── Live Rate Scraper ─────────────────────────────────────────────────────────
# @st.cache_data(ttl=86400) caches the result for 86400 seconds (24 hours),
# so the live scrape only fires once per day, not on every user interaction.
#
# Attempts to scrape the current SBP policy rate from the SBP website using:
#   - requests  : fetches the HTML of the SBP monetary policy page
#   - BeautifulSoup : parses the HTML into readable text
#   - re.search : looks for a pattern like "SBP Policy Rate  13.5 %"
#
# Returns a 3-tuple: (rate_as_decimal, date_string, error_message_or_None)
# If scraping fails for any reason, returns (None, None, error_string)
# so the rest of the app can fall back gracefully to the CSV data.
@st.cache_data(ttl=86400)
def fetch_live_sbp_rate():
    """Scrape latest policy rate from SBP. Returns (rate_decimal, date_str, error_msg)."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get("https://www.sbp.org.pk/m_policy/index.asp", headers=headers, timeout=8)
        soup = BeautifulSoup(r.text, 'html.parser')
        text = soup.get_text()
        match = re.search(r'SBP Policy\s*Rat\w*\s*[\|\s]*(\d+\.?\d*)\s*%', text, re.IGNORECASE)
        if match:
            rate = float(match.group(1)) / 100
            return rate, datetime.now().strftime("%d %b %Y"), None
        return None, None, "Rate not found on SBP page"
    except Exception as e:
        return None, None, str(e)

# ── Live Rate Merge Function ──────────────────────────────────────────────────
# Decides how to incorporate the live-scraped rate into the historical DataFrame.
#
# Case 1: The live rate is meaningfully different from the last CSV entry
#   → Append a brand-new row with today's date and the live rate.
#     (Indicates a rate change that hasn't been captured in the CSV yet.)
#
# Case 2: The live rate is essentially the same as the last CSV entry
#   → Just update the date of the last row to today.
#     (No new data, just refreshes the timestamp.)
#
# The threshold for "meaningfully different" is 0.0001 (i.e., 0.01% in decimal).
def append_live_rate(df, live_rate):
    """If live rate differs from last CSV row, append it as today's entry."""
    today = pd.Timestamp(datetime.now().date())
    last_rate = df['Rate'].iloc[-1]
    if abs(live_rate - last_rate) > 0.0001:
        new_row = pd.DataFrame({'Date': [today], 'Rate': [live_rate]})
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df.loc[df.index[-1], 'Date'] = today
    return df

# ── Core Vasicek Simulation ───────────────────────────────────────────────────
# Implements the Vasicek short-rate model using the discrete-time Euler-Maruyama
# approximation of the stochastic differential equation:
#
#   dr(t) = a * (b - r(t)) * dt + σ * sqrt(dt) * Z
#
# Where:
#   a     = mean reversion speed (how fast rates snap back toward b)
#   b     = long-run mean (derived as the historical average rate)
#   σ     = volatility (derived as the historical standard deviation of rates)
#   r(t)  = current rate at time t
#   Z     = random shock drawn from N(0,1)
#   dt    = time step = 1/12 (monthly steps, since rates are monthly data)
#
# Parameters:
#   rates       : pandas Series of historical rates (in decimal form)
#   a           : mean reversion speed (user-controlled via slider)
#   years       : forecast horizon in years
#   simulations : number of Monte Carlo paths to generate
#
# Process:
#   1. Fix the random seed to 42 so results are reproducible.
#   2. Compute steps = years / dt (e.g. 10 years × 12 months = 120 steps).
#   3. Calibrate b (long-run mean) and σ (volatility) from the historical data.
#   4. Set r0 = the last observed rate (starting point for all forecasts).
#   5. Initialise a 2D array: rows = simulation paths, columns = time steps.
#   6. At each time step, update every path simultaneously using the formula above.
#
# Returns:
#   sim    : 2D array (simulations × steps) of simulated rate paths
#   time   : 1D array of evenly-spaced time values from 0 to years
#   r0     : starting rate (current rate)
#   b      : long-run mean
#   sigma  : annualised volatility
def run_vasicek(rates,a,years,simulations):
    np.random.seed(42); dt=1/12; steps=int(years/dt)
    b=rates.mean(); sigma=rates.std(); r0=rates.iloc[-1]
    sim=np.zeros((simulations,steps)); sim[:,0]=r0
    for t in range(1,steps):
        z=np.random.normal(0,1,simulations)
        sim[:,t]=sim[:,t-1]+a*(b-sim[:,t-1])*dt+sigma*np.sqrt(dt)*z
    return sim,np.linspace(0,years,steps),r0,b,sigma

# ── Navbar ────────────────────────────────────────────────────────────────────
# Captures the current time to display in the nav bar ("28 Apr 2026 • 14:32").
# Splits the top of the page into two columns:
#   - Left (wide): renders the styled navbar HTML with brand name, subtitle, time, badge.
#   - Right (narrow): renders the dark/light mode toggle button.
# When the toggle button is clicked, it flips st.session_state.dark_mode and
# calls st.rerun() to immediately re-render the entire app in the new theme.
now=datetime.now().strftime("%d %b %Y  •  %H:%M")
nav_col,toggle_col=st.columns([11,1])
with nav_col:
    st.markdown(f"""<div class="navbar"><div class="navbar-left"><div class="navbar-brand">VASICEK <span>ENGINE</span></div><div class="navbar-divider"></div><div class="navbar-sub">State Bank of Pakistan — Policy Rate Simulation</div></div><div style="display:flex;align-items:center;gap:16px;"><div class="navbar-time">{now}</div><div class="navbar-badge">SBP · PKR · LIVE</div></div></div>""", unsafe_allow_html=True)
with toggle_col:
    st.markdown("<div style='margin-top:8px;'>",unsafe_allow_html=True)
    if st.button(f"{T['toggle_icon']} {T['toggle_label']}",use_container_width=True):
        st.session_state.dark_mode=not st.session_state.dark_mode; st.rerun()
    st.markdown("</div>",unsafe_allow_html=True)

# ── Live Rate Fetch & Status Badge ────────────────────────────────────────────
# Calls the scraper function defined above.
# Displays a fixed-position badge in the bottom-left corner of the screen:
#   - RED badge (⚠) if the scrape failed → tells the user the app is using
#     fallback data from the CSV and shows the truncated error message.
#   - GREEN badge (●) if the scrape succeeded → shows the live rate and
#     the source URL + timestamp.
# Also sets data_source_label and data_source_color for use in the footer.
live_rate, live_date, live_error = fetch_live_sbp_rate()

# Show live data status badge in a fixed corner
if live_error:
    st.markdown(f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
         background:{'#1a0a0a' if D else '#fff3f3'};
         border:1px solid #ef4444;border-radius:10px;
         padding:10px 16px;font-family:'JetBrains Mono',monospace;font-size:10px;
         color:#ef4444;max-width:280px;box-shadow:0 4px 20px rgba(239,68,68,0.2);">
        ⚠ Live data unavailable — showing last known rate<br>
        <span style="opacity:0.6;font-size:9px;">{live_error[:60]}</span>
    </div>""", unsafe_allow_html=True)
    data_source_label = "CSV FALLBACK"
    data_source_color = "#f59e0b"
else:
    st.markdown(f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
         background:{'#0a1a0a' if D else '#f0fff4'};
         border:1px solid #10b981;border-radius:10px;
         padding:10px 16px;font-family:'JetBrains Mono',monospace;font-size:10px;
         color:#10b981;box-shadow:0 4px 20px rgba(16,185,129,0.15);">
        ● LIVE · SBP Policy Rate: {live_rate*100:.2f}%<br>
        <span style="opacity:0.6;font-size:9px;">Source: sbp.org.pk · Updated: {live_date}</span>
    </div>""", unsafe_allow_html=True)
    data_source_label = "LIVE · SBP"
    data_source_color = "#10b981"

# ── Session-State Run Parameters ─────────────────────────────────────────────
# Stores the simulation parameters in session state so they only update
# when the user explicitly clicks "RUN ▶", not on every slider drag.
# This avoids expensive re-simulations every time a slider moves.
# Defaults: 10-year horizon, 500 simulation paths, show 10 sample paths, a=0.5.
if "run_params" not in st.session_state:
    st.session_state.run_params = dict(
        years=10, simulations=500, sample_paths=10, a_val=0.5
    )

# ── Control Panel (Sliders + Run Button) ─────────────────────────────────────
# Renders a styled control bar with 5 columns:
#   c1 - Years slider        : forecast horizon (1–20 years)
#   c2 - Sims slider         : number of Monte Carlo paths (100–2000)
#   c3 - Paths slider        : how many individual paths to draw on the chart (5–30)
#   c4 - Mean Rev. slider    : mean reversion speed 'a' (0.1–2.0)
#   c5 - RUN button          : commits slider values to session state and triggers
#                              a full re-computation of the simulation
with st.container():
    st.markdown('<div class="control-wrap">',unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns([1,1,1,1,0.8])
    with c1: years_input=st.slider("Years",1,20,st.session_state.run_params["years"])
    with c2: simulations_input=st.slider("Sims",100,2000,st.session_state.run_params["simulations"],step=100)
    with c3: sample_paths_input=st.slider("Paths",5,30,st.session_state.run_params["sample_paths"])
    with c4: a_val_input=st.slider("Mean Rev.",0.1,2.0,st.session_state.run_params["a_val"],step=0.05)
    with c5:
        st.markdown("<br>",unsafe_allow_html=True)
        run_clicked=st.button("RUN ▶",use_container_width=True)
    st.markdown('</div>',unsafe_allow_html=True)

# When RUN is clicked, save the current slider values into session state.
# These become the "committed" parameters used for all charts below.
if run_clicked:
    st.session_state.run_params = dict(
        years=years_input, simulations=simulations_input,
        sample_paths=sample_paths_input, a_val=a_val_input
    )

# Unpack the committed parameters into plain local variables for readability.
p = st.session_state.run_params
years       = p["years"]
simulations = p["simulations"]
sample_paths= p["sample_paths"]
a_val       = p["a_val"]

# ── Load CSV Data ─────────────────────────────────────────────────────────────
# Attempts to load the historical SBP rate CSV file.
# If the file is missing, shows a Streamlit error and stops the app entirely.
try:
    df = load_data("sbp_policy_rate.csv")
except Exception as e:
    st.error(f"Could not load CSV: {e}"); st.stop()

# Merge the live-scraped rate into the DataFrame if the scrape succeeded.
if live_rate is not None:
    df = append_live_rate(df, live_rate)

# ── Run the Vasicek Simulation ────────────────────────────────────────────────
# Runs the Monte Carlo simulation with the committed parameters.
# Then computes summary statistics across all simulation paths at each time step:
#   mean_path : average forecast rate across all simulations
#   upper_95 / lower_95 : 97.5th and 2.5th percentiles → 95% confidence interval
#   upper_68 / lower_68 : 84th and 16th percentiles    → 68% confidence interval
#   final_rates : all simulated terminal rates (in %) at the end of the horizon
#   rate_change : how much the mean forecast differs from today's rate
#   direction / arrow : used for the colour and arrow in the stat card delta label
sim,time,r0,b,sigma=run_vasicek(df['Rate'],a_val,years,simulations)
mean_path=sim.mean(axis=0)
upper_95=np.percentile(sim,97.5,axis=0); lower_95=np.percentile(sim,2.5,axis=0)
upper_68=np.percentile(sim,84,axis=0);   lower_68=np.percentile(sim,16,axis=0)
final_rates=sim[:,-1]*100
rate_change=mean_path[-1]-r0
direction="up" if rate_change>0 else "down"
arrow="▲" if rate_change>0 else "▼"

# ── Summary Stat Cards ────────────────────────────────────────────────────────
# Renders four metric cards in a horizontal row:
#   Card 1 (purple)  : Current Policy Rate — the starting point r0
#   Card 2 (cyan)    : Long-Run Mean — historical average, the model's gravity centre
#   Card 3 (green)   : Annualised Volatility — σ from historical data
#   Card 4 (amber)   : N-Year Forecast — mean projected rate at end of horizon,
#                      with a coloured delta arrow showing direction vs. current rate
st.markdown(f"""
<div class="stat-row">
  <div class="stat-card c1"><div class="stat-label">Current Policy Rate</div><div class="stat-value">{r0*100:.2f}%</div><div class="stat-delta neutral">Latest SBP decision</div></div>
  <div class="stat-card c2"><div class="stat-label">Long-Run Mean</div><div class="stat-value">{b*100:.2f}%</div><div class="stat-delta neutral">Historical average</div></div>
  <div class="stat-card c3"><div class="stat-label">Annualised Volatility</div><div class="stat-value">{sigma*100:.2f}%</div><div class="stat-delta neutral">σ from historical data</div></div>
  <div class="stat-card c4"><div class="stat-label">{years}Y Forecast</div><div class="stat-value">{mean_path[-1]*100:.2f}%</div><div class="stat-delta {direction}">{arrow} {abs(rate_change*100):.2f}% from current</div></div>
</div>""",unsafe_allow_html=True)

# ── Two-Column Layout for Main Charts ─────────────────────────────────────────
# Splits the page 3:2 — left column is wider for the simulation chart,
# right column is narrower for the historical chart and terminal distribution.
left,right=st.columns([3,2])

with left:
    # ── Stochastic Rate Simulation Chart ──────────────────────────────────────
    # Creates a Plotly figure showing the Monte Carlo simulation results.
    # Layers (drawn bottom to top):
    #   1. Individual sample paths: draws `sample_paths` random lines from the
    #      simulation array in a semi-transparent colour so density is visible.
    #   2. 95% CI band: a filled polygon using the upper/lower 95th percentile
    #      boundaries, drawn by concatenating the top boundary with the reversed
    #      bottom boundary to form a closed shape.
    #   3. 68% CI band: same technique, using 84th/16th percentiles (slightly darker).
    #   4. Glowing mean line: three wide, low-opacity traces of the mean path
    #      at widths 8, 4, 2 — stacked to create a soft glow/bloom effect.
    #   5. Crisp mean line: the actual thin mean path on top.
    #   6. Horizontal dashed line: current rate reference so users can see
    #      where rates started relative to the forecasts.
    st.markdown(f'<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Stochastic Rate Simulation</span><span class="chart-tag">VASICEK · {simulations} PATHS</span></div>',unsafe_allow_html=True)
    fig_sim=go.Figure()
    for i in range(sample_paths): fig_sim.add_trace(go.Scatter(x=time,y=sim[i]*100,line=dict(color=T['path_color'],width=0.6),showlegend=False,hoverinfo='skip'))
    fig_sim.add_trace(go.Scatter(x=np.concatenate([time,time[::-1]]),y=np.concatenate([upper_95*100,lower_95[::-1]*100]),fill='toself',fillcolor=T['ci_fill'],line=dict(color='rgba(0,0,0,0)'),name='95% CI'))
    fig_sim.add_trace(go.Scatter(x=np.concatenate([time,time[::-1]]),y=np.concatenate([upper_68*100,lower_68[::-1]*100]),fill='toself',fillcolor=T['ci_fill2'],line=dict(color='rgba(0,0,0,0)'),name='68% CI'))
    for w,op in [(8,0.04),(4,0.08),(2,0.2)]: fig_sim.add_trace(go.Scatter(x=time,y=mean_path*100,line=dict(color=T['mean_color'],width=w),opacity=op,showlegend=False,hoverinfo='skip'))
    fig_sim.add_trace(go.Scatter(x=time,y=mean_path*100,line=dict(color=T['mean_color'],width=2),name='Mean Path'))
    fig_sim.add_hline(y=r0*100,line=dict(color=T['current_color'],width=1.2,dash='dot'),annotation_text=f"  {r0*100:.1f}%",annotation_font_color=T['current_color'],annotation_font_size=11)
    fig_sim.update_layout(height=400,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color=T['tick_color'],size=11),legend=dict(bgcolor='rgba(0,0,0,0)',bordercolor=T['legend_border'],borderwidth=1,font=dict(size=10),orientation='h',y=-0.18),margin=dict(l=40,r=20,t=10,b=55),xaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color'],title='Years',title_font=dict(size=11)),yaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color'],ticksuffix='%'))
    st.plotly_chart(fig_sim,use_container_width=True)

    # ── Simulation Interpretation Box ─────────────────────────────────────────
    # Derives a plain-English interpretation from the computed statistics:
    #   - Whether the current rate is above or below the long-run mean.
    #   - The implied direction of mean-reversion pressure.
    #   - How wide the 95% CI is at the end of the forecast horizon.
    above_mean = r0 > b
    mean_direction = "above" if above_mean else "below"
    reversion_bias = "downward" if above_mean else "upward"
    ci_width = (upper_95[-1] - lower_95[-1]) * 100
    st.markdown(f"""<div class="insight-box"><div class="insight-text">
    The current SBP policy rate of <strong>{r0*100:.1f}%</strong> sits <strong>{mean_direction} the long-run mean of {b*100:.1f}%</strong>,
    creating a <strong>{reversion_bias} mean-reversion pressure</strong> over the forecast horizon.
    At {years} years, the 95% confidence interval spans <strong>{ci_width:.1f} percentage points</strong>,
    reflecting how uncertainty compounds over time under stochastic shocks.
    </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

with right:
    # ── Historical SBP Policy Rate Chart ──────────────────────────────────────
    # A compact area chart showing the entire historical rate series from the CSV
    # (supplemented by the live rate if available). The area is filled to zero
    # with a semi-transparent colour. A slightly thicker, low-opacity line is
    # drawn over the main line to create a soft glow effect.
    st.markdown('<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Historical SBP Policy Rates</span><span class="chart-tag">2018–PRESENT</span></div>',unsafe_allow_html=True)
    fig_hist=go.Figure()
    fig_hist.add_trace(go.Scatter(x=df['Date'],y=df['Rate']*100,fill='tozeroy',fillcolor=T['hist_fill'],line=dict(color=T['hist_line'],width=2)))
    fig_hist.add_trace(go.Scatter(x=df['Date'],y=df['Rate']*100,line=dict(color=T['hist_line'],width=6),opacity=0.08,showlegend=False,hoverinfo='skip'))
    fig_hist.update_layout(height=185,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color=T['tick_color'],size=10),showlegend=False,margin=dict(l=40,r=10,t=5,b=30),xaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color']),yaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color'],ticksuffix='%'))
    st.plotly_chart(fig_hist,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

    # ── Terminal Rate Distribution Chart ──────────────────────────────────────
    # Shows the distribution of all simulated rates at the very end of the
    # forecast horizon (the final column of the sim array).
    # Displayed as a histogram with 50 bins.
    # Two vertical lines are overlaid:
    #   - Dashed amber line : the mean terminal rate (μ)
    #   - Dotted coloured line : the current rate ("now") for reference
    # Below the chart, an insight box shows:
    #   - What percentage of paths end above the current rate.
    #   - The median terminal rate.
    #   - The 5th–95th percentile range (90% interval).
    st.markdown(f'<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Terminal Distribution — {years}Y</span><span class="chart-tag">MONTE CARLO</span></div>',unsafe_allow_html=True)
    fig_dist=go.Figure()
    fig_dist.add_trace(go.Histogram(x=final_rates,nbinsx=50,marker=dict(color=T['hist_bar'],line=dict(color=T['hist_bar_line'],width=0.3))))
    fig_dist.add_vline(x=mean_path[-1]*100,line=dict(color=T['vline_color'],width=2,dash='dash'),annotation_text=f"  μ={mean_path[-1]*100:.1f}%",annotation_font_color=T['vline_color'],annotation_font_size=10)
    fig_dist.add_vline(x=r0*100,line=dict(color=T['current_color'],width=1.5,dash='dot'),annotation_text="  now",annotation_font_color=T['current_color'],annotation_font_size=10,annotation_position="bottom right")
    fig_dist.update_layout(height=185,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color=T['tick_color'],size=10),showlegend=False,margin=dict(l=40,r=10,t=5,b=30),bargap=0.02,xaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color'],ticksuffix='%'),yaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color']))
    st.plotly_chart(fig_dist,use_container_width=True)
    prob_above_current = np.mean(sim[:,-1]*100 > r0*100) * 100
    prob_above_10      = np.mean(sim[:,-1]*100 > 10.0)   * 100
    st.markdown(f"""<div class="insight-box"><div class="insight-text">
    After <strong>{years} years</strong>, <strong>{prob_above_current:.0f}% of paths</strong> end above the current rate.
    The median terminal rate is <strong>{np.median(final_rates):.1f}%</strong>,
    with a <strong>{np.percentile(final_rates,5):.1f}% – {np.percentile(final_rates,95):.1f}%</strong> 90th-percentile range.
    </div></div>""", unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)

# ── Fan Chart ─────────────────────────────────────────────────────────────────
# A full-width chart that joins the historical rate series (left of "Today")
# with the forecast fan (right of "Today") into one continuous view.
#
# Historical section:
#   - Maps each historical data point to a negative time value so the
#     full rate history appears to the left of the t=0 vertical line.
#   - Drawn as a filled area, matching the style of the historical chart above.
#
# Forecast fan (4 nested bands):
#   Each band is a percentile range; together they form a "cone of uncertainty"
#   that widens as you move further into the future:
#     90% range (5th–95th percentile)  → most transparent
#     80% range (10th–90th percentile)
#     60% range (20th–80th percentile)
#     30% range (35th–65th percentile) → most opaque (tightest inner band)
#   Each band is rendered as a filled closed polygon, same technique as the CI
#   bands in the simulation chart.
#
# Overlays:
#   - Mean forecast line in purple.
#   - Dashed vertical line at t=0 labelled "Today".
#   - Dotted horizontal line at the long-run mean b.
#
# Below the chart, an insight box computes and explains:
#   - How wide the 90% CI is at 1 year.
#   - How wide it is at the end of the full horizon.
st.markdown(f'<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Rate Forecast Fan Chart — Uncertainty Widens Over Time</span><span class="chart-tag">PERCENTILE BANDS</span></div>',unsafe_allow_html=True)

fig_fan = go.Figure()

# Historical line leading into forecast
hist_years = np.linspace(-len(df)/12, 0, len(df))
fig_fan.add_trace(go.Scatter(
    x=hist_years, y=df['Rate']*100,
    line=dict(color=T['hist_line'], width=2),
    name='Historical Rate', fill='tozeroy', fillcolor=T['hist_fill']
))

# Fan bands — widening cones
bands = [
    (5,  95,  0.06, '90% Range'),
    (10, 90,  0.10, '80% Range'),
    (20, 80,  0.14, '60% Range'),
    (35, 65,  0.20, '30% Range'),
]
for lo, hi, alpha, label in bands:
    p_lo = np.percentile(sim, lo, axis=0) * 100
    p_hi = np.percentile(sim, hi, axis=0) * 100
    r, g, b_c = (124, 58, 237) if D else (37, 99, 235)
    fill_col = f'rgba({r},{g},{b_c},{alpha})'
    fig_fan.add_trace(go.Scatter(
        x=np.concatenate([time, time[::-1]]),
        y=np.concatenate([p_hi, p_lo[::-1]]),
        fill='toself', fillcolor=fill_col,
        line=dict(color='rgba(0,0,0,0)'),
        name=label, hoverinfo='skip'
    ))

# Mean path on top
fig_fan.add_trace(go.Scatter(
    x=time, y=mean_path*100,
    line=dict(color=T['mean_color'], width=2.5),
    name='Mean Forecast'
))

# Vertical line at t=0 (today)
fig_fan.add_vline(x=0, line=dict(color=T['current_color'], width=1.5, dash='dash'),
    annotation_text="  Today", annotation_font_color=T['current_color'], annotation_font_size=11)

# Long-run mean reference
fig_fan.add_hline(y=b*100, line=dict(color=T['vline_color'], width=1, dash='dot'),
    annotation_text=f"  Long-run mean {b*100:.1f}%",
    annotation_font_color=T['vline_color'], annotation_font_size=10)

fig_fan.update_layout(
    height=340,
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color=T['tick_color'], size=11),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor=T['legend_border'], borderwidth=1,
                font=dict(size=10), orientation='h', y=-0.22),
    margin=dict(l=40, r=20, t=10, b=65),
    xaxis=dict(gridcolor=T['grid_color'], zerolinecolor=T['grid_color'],
               title='Years (negative = past, positive = forecast)',
               title_font=dict(size=10)),
    yaxis=dict(gridcolor=T['grid_color'], zerolinecolor=T['grid_color'], ticksuffix='%')
)
st.plotly_chart(fig_fan, use_container_width=True)

# Fan chart interpretation — computes how wide the 90% band is at 1 year
# vs at the end of the full horizon to illustrate growing uncertainty.
fan_spread_1y  = (np.percentile(sim[:,11],95) - np.percentile(sim[:,11],5)) * 100
fan_spread_end = (np.percentile(sim[:,-1],95) - np.percentile(sim[:,-1],5)) * 100
st.markdown(f"""<div class="insight-box"><div class="insight-text">
The fan chart shows how forecast uncertainty <strong>compounds with time</strong>. The 90% confidence band is
<strong>{fan_spread_1y:.1f}pp wide at 1 year</strong>, expanding to <strong>{fan_spread_end:.1f}pp at {years} years</strong>.
The historical section (left of "Today") confirms the model is anchored to real SBP rate behaviour.
The long-run mean acts as a gravitational centre — all paths are statistically pulled toward it regardless of where they start.
</div></div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Parameter Sensitivity Table ────────────────────────────────────────────────
# Tests 6 different values of the mean reversion speed 'a' (0.1 to 2.0).
# For each value, a fresh Vasicek simulation is run (using the same seed),
# and the mean rate + 5th–95th percentile range is computed at 4 forecast
# horizons: 1, 3, 5, and 10 years.
#
# Results are rendered as an HTML table where:
#   - Columns = forecast horizons (1Y, 3Y, 5Y, 10Y)
#   - Rows = different values of 'a'
#   - The row matching the user's current 'a' slider value is highlighted
#     in the accent colour and labelled "← current"
#
# This lets users see at a glance how their choice of mean reversion speed
# affects both the forecast level and the spread of uncertainty.
st.markdown(f'<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Parameter Sensitivity — How Mean Reversion Speed (a) Affects the Forecast</span><span class="chart-tag">SCENARIO ANALYSIS</span></div>',unsafe_allow_html=True)

a_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
horizons = [1, 3, 5, 10]
rows_html = ""
for a_test in a_values:
    sim_test, time_test, _, _, _ = run_vasicek(df['Rate'], a_test, max(horizons), simulations)
    cells = ""
    for h in horizons:
        idx = int(h * 12) - 1          # convert years to month index
        m = sim_test[:, idx].mean() * 100
        hi = np.percentile(sim_test[:, idx], 95) * 100
        lo = np.percentile(sim_test[:, idx],  5) * 100
        is_current = abs(a_test - a_val) < 0.01
        cls = ' class="highlight"' if is_current else ''
        cells += f'<td{cls}>{m:.1f}% <span style="font-size:9px;opacity:0.5;">({lo:.0f}–{hi:.0f})</span></td>'
    label = f'<strong>{a_test}</strong> ← current' if abs(a_test - a_val) < 0.01 else str(a_test)
    rows_html += f"<tr><td>{label}</td>{cells}</tr>"

headers = "".join(f"<th>{h}Y Mean (5–95% range)</th>" for h in horizons)
st.markdown(f"""
<table class="sens-table">
  <thead><tr><th>Speed (a)</th>{headers}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<div style="font-family:'Outfit',sans-serif;font-size:11px;color:{T['label_color']};margin-top:10px;line-height:1.6;">
  <strong style="color:{T['chart_title']};">How to read this:</strong>
  Higher <em>a</em> = faster mean reversion → tighter confidence bands and faster convergence to {b*100:.1f}%.
  Lower <em>a</em> = slower reversion → wider uncertainty, paths drift further before snapping back.
  The highlighted row is your current simulation setting.
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── CSV Export ─────────────────────────────────────────────────────────────────
# Builds two DataFrames for export and provides download buttons for each:
#
#   export_df (Forecast Data):
#     A row for each time step with columns:
#       Year, Mean_Rate_%, CI_68_Low_%, CI_68_High_%, CI_95_Low_%, CI_95_High_%
#     Useful for importing forecast data into Excel or other tools.
#
#   meta_df (Model Parameters):
#     A two-column table (Parameter, Value) recording the exact model settings
#     used so the simulation is reproducible and documented.
#
# Two download buttons are placed side-by-side using st.columns.
# Each button encodes the DataFrame as a UTF-8 CSV bytes object and
# suggests a descriptive filename including the simulation settings.
st.markdown(f'<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Export Simulation Results</span><span class="chart-tag">DOWNLOAD</span></div>',unsafe_allow_html=True)

# Build export dataframe
export_df = pd.DataFrame({
    'Year':         np.round(time, 4),
    'Mean_Rate_%':  np.round(mean_path * 100, 4),
    'CI_68_Low_%':  np.round(lower_68 * 100, 4),
    'CI_68_High_%': np.round(upper_68 * 100, 4),
    'CI_95_Low_%':  np.round(lower_95 * 100, 4),
    'CI_95_High_%': np.round(upper_95 * 100, 4),
})
meta_df = pd.DataFrame({
    'Parameter': ['Current Rate (%)', 'Long-Run Mean (%)', 'Volatility (%)', 'Mean Reversion Speed', 'Simulations', 'Horizon (Years)'],
    'Value':     [round(r0*100,2), round(b*100,2), round(sigma*100,2), a_val, simulations, years]
})

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    csv_forecast = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download Forecast Data (CSV)",
        data=csv_forecast,
        file_name=f"vasicek_forecast_{years}y_{simulations}sims.csv",
        mime='text/csv',
        use_container_width=True
    )
with exp_col2:
    csv_meta = meta_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇ Download Model Parameters (CSV)",
        data=csv_meta,
        file_name="vasicek_parameters.csv",
        mime='text/csv',
        use_container_width=True
    )
st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
# A centred, monospaced footer bar at the bottom of the page displaying
# a summary of the current simulation run: model name, data source,
# number of paths simulated, and forecast horizon.
st.markdown(f'<div style="text-align:center;color:{T["footer_color"]};font-family:\'JetBrains Mono\',monospace;font-size:10px;margin-top:16px;border-top:1px solid {T["footer_border"]};padding-top:14px;letter-spacing:2px;">VASICEK MODEL  ·  SBP POLICY RATE  ·  {simulations} PATHS  ·  {years}Y HORIZON</div>',unsafe_allow_html=True)
