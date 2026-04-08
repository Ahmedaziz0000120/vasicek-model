import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="Vasicek Rate Engine", layout="wide", initial_sidebar_state="collapsed")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

D = st.session_state.dark_mode
T = {
    "app_bg":        "#030308" if D else "#f0f2f6",
    "navbar_bg":     "#0a0a14" if D else "#ffffff",
    "navbar_border": "#1a1a2e" if D else "#e0e0e8",
    "brand_color":   "#ffffff" if D else "#0a0a14",
    "sub_color":     "#4444aa" if D else "#888888",
    "time_color":    "#333366" if D else "#aaaaaa",
    "badge_bg":      "#1a1a2e" if D else "#f0f0f8",
    "badge_border":  "#2a2a4e" if D else "#ddddee",
    "badge_color":   "#7777cc" if D else "#555588",
    "card_bg":       "#0a0a14" if D else "#ffffff",
    "card_border":   "#1a1a2e" if D else "#e0e0e8",
    "label_color":   "#444488" if D else "#999999",
    "value_color":   "#ffffff" if D else "#0a0a14",
    "delta_neutral": "#333366" if D else "#aaaaaa",
    "chart_title":   "#8888cc" if D else "#333333",
    "chart_tag_bg":  "#1a1a2e" if D else "#f0f0f8",
    "chart_tag_col": "#5555aa" if D else "#666688",
    "slider_color":  "#7c3aed" if D else "#1a1a2e",
    "btn_bg":        "#7c3aed" if D else "#1a1a2e",
    "btn_hover":     "#6d28d9" if D else "#2d2d50",
    "input_bg":      "#0a0a14" if D else "#ffffff",
    "input_color":   "#aaaacc" if D else "#333333",
    "input_border":  "#1a1a2e" if D else "#dddddd",
    "plot_bg":       "#0a0a14" if D else "#ffffff",
    "paper_bg":      "#030308" if D else "#f0f2f6",
    "grid_color":    "#0f0f20" if D else "#f0f0f8",
    "tick_color":    "#444488" if D else "#999999",
    "legend_bg":     "#0a0a14" if D else "#ffffff",
    "legend_border": "#1a1a2e" if D else "#e0e0e8",
    "path_color":    "rgba(124,58,237,0.15)" if D else "rgba(37,99,235,0.12)",
    "ci_fill":       "rgba(124,58,237,0.08)" if D else "rgba(37,99,235,0.05)",
    "ci_fill2":      "rgba(124,58,237,0.14)" if D else "rgba(37,99,235,0.10)",
    "mean_color":    "#a78bfa" if D else "#1a1a2e",
    "current_color": "#06b6d4" if D else "#dc2626",
    "hist_fill":     "rgba(124,58,237,0.1)"  if D else "rgba(37,99,235,0.06)",
    "hist_line":     "#7c3aed" if D else "#2563eb",
    "hist_bar":      "rgba(167,139,250,0.6)" if D else "rgba(26,26,46,0.7)",
    "hist_bar_line": "rgba(167,139,250,0.9)" if D else "rgba(26,26,46,0.9)",
    "vline_color":   "#f59e0b" if D else "#dc2626",
    "footer_color":  "#1a1a2e" if D else "#cccccc",
    "footer_border": "#1a1a2e" if D else "#e0e0e8",
    "toggle_icon":   "☀️" if D else "🌙",
    "toggle_label":  "Light" if D else "Dark",
    "accent1":       "#7c3aed" if D else "#2563eb",
    "accent2":       "#06b6d4" if D else "#dc2626",
}

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
.stButton>button{{background:linear-gradient(135deg,{T['btn_bg']},{T['accent2']}) !important;color:white !important;border:none !important;border-radius:10px !important;font-family:'Outfit',sans-serif !important;font-weight:600 !important;font-size:12px !important;{'box-shadow:0 0 20px rgba(124,58,237,0.3) !important;' if D else ''}}}
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

@st.cache_data
def load_data(path):
    df=pd.read_csv(path); df.columns=df.columns.str.strip()
    df['Date']=pd.to_datetime(df['Date']); df=df.sort_values('Date').reset_index(drop=True)
    df['Rate']=df['Rate']/100; return df

@st.cache_data(ttl=86400)
def fetch_live_sbp_rate():
    """Scrape latest policy rate from SBP. Returns (rate_decimal, date_str, error_msg)."""
    try:
        import requests, re
        from bs4 import BeautifulSoup
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

def run_vasicek(rates,a,years,simulations):
    np.random.seed(42); dt=1/12; steps=int(years/dt)
    b=rates.mean(); sigma=rates.std(); r0=rates.iloc[-1]
    sim=np.zeros((simulations,steps)); sim[:,0]=r0
    for t in range(1,steps):
        z=np.random.normal(0,1,simulations)
        sim[:,t]=sim[:,t-1]+a*(b-sim[:,t-1])*dt+sigma*np.sqrt(dt)*z
    return sim,np.linspace(0,years,steps),r0,b,sigma

now=datetime.now().strftime("%d %b %Y  •  %H:%M")
nav_col,toggle_col=st.columns([11,1])
with nav_col:
    st.markdown(f"""<div class="navbar"><div class="navbar-left"><div class="navbar-brand">VASICEK <span>ENGINE</span></div><div class="navbar-divider"></div><div class="navbar-sub">State Bank of Pakistan — Policy Rate Simulation</div></div><div style="display:flex;align-items:center;gap:16px;"><div class="navbar-time">{now}</div><div class="navbar-badge">SBP · PKR · LIVE</div></div></div>""", unsafe_allow_html=True)
with toggle_col:
    st.markdown("<div style='margin-top:8px;'>",unsafe_allow_html=True)
    if st.button(f"{T['toggle_icon']} {T['toggle_label']}",use_container_width=True):
        st.session_state.dark_mode=not st.session_state.dark_mode; st.rerun()
    st.markdown("</div>",unsafe_allow_html=True)

# ── Live SBP rate fetch ────────────────────────────────────────────────────
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

# ── Init run params in session state (only update on RUN click) ────────────
if "run_params" not in st.session_state:
    st.session_state.run_params = dict(
        years=10, simulations=500, sample_paths=10, a_val=0.5
    )

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

if run_clicked:
    st.session_state.run_params = dict(
        years=years_input, simulations=simulations_input,
        sample_paths=sample_paths_input, a_val=a_val_input
    )

# Use committed params for all computation
p = st.session_state.run_params
years       = p["years"]
simulations = p["simulations"]
sample_paths= p["sample_paths"]
a_val       = p["a_val"]

try:
    df = load_data("sbp_policy_rate.csv")
except Exception as e:
    st.error(f"Could not load CSV: {e}"); st.stop()

# Merge live rate into dataframe if available
if live_rate is not None:
    df = append_live_rate(df, live_rate)

sim,time,r0,b,sigma=run_vasicek(df['Rate'],a_val,years,simulations)
mean_path=sim.mean(axis=0)
upper_95=np.percentile(sim,97.5,axis=0); lower_95=np.percentile(sim,2.5,axis=0)
upper_68=np.percentile(sim,84,axis=0);   lower_68=np.percentile(sim,16,axis=0)
final_rates=sim[:,-1]*100
rate_change=mean_path[-1]-r0
direction="up" if rate_change>0 else "down"
arrow="▲" if rate_change>0 else "▼"

st.markdown(f"""
<div class="stat-row">
  <div class="stat-card c1"><div class="stat-label">Current Policy Rate</div><div class="stat-value">{r0*100:.2f}%</div><div class="stat-delta neutral">Latest SBP decision</div></div>
  <div class="stat-card c2"><div class="stat-label">Long-Run Mean</div><div class="stat-value">{b*100:.2f}%</div><div class="stat-delta neutral">Historical average</div></div>
  <div class="stat-card c3"><div class="stat-label">Annualised Volatility</div><div class="stat-value">{sigma*100:.2f}%</div><div class="stat-delta neutral">σ from historical data</div></div>
  <div class="stat-card c4"><div class="stat-label">{years}Y Forecast</div><div class="stat-value">{mean_path[-1]*100:.2f}%</div><div class="stat-delta {direction}">{arrow} {abs(rate_change*100):.2f}% from current</div></div>
</div>""",unsafe_allow_html=True)

left,right=st.columns([3,2])
with left:
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
    # ── Interpretation
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
    st.markdown('<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Historical SBP Policy Rates</span><span class="chart-tag">2018–PRESENT</span></div>',unsafe_allow_html=True)
    fig_hist=go.Figure()
    fig_hist.add_trace(go.Scatter(x=df['Date'],y=df['Rate']*100,fill='tozeroy',fillcolor=T['hist_fill'],line=dict(color=T['hist_line'],width=2)))
    fig_hist.add_trace(go.Scatter(x=df['Date'],y=df['Rate']*100,line=dict(color=T['hist_line'],width=6),opacity=0.08,showlegend=False,hoverinfo='skip'))
    fig_hist.update_layout(height=185,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color=T['tick_color'],size=10),showlegend=False,margin=dict(l=40,r=10,t=5,b=30),xaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color']),yaxis=dict(gridcolor=T['grid_color'],zerolinecolor=T['grid_color'],ticksuffix='%'))
    st.plotly_chart(fig_hist,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

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

# ── Fan Chart: Uncertainty grows over time (easy to read) ──────────────────
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
# ── Fan chart interpretation
fan_spread_1y  = (np.percentile(sim[:,11],95) - np.percentile(sim[:,11],5)) * 100
fan_spread_end = (np.percentile(sim[:,-1],95) - np.percentile(sim[:,-1],5)) * 100
st.markdown(f"""<div class="insight-box"><div class="insight-text">
The fan chart shows how forecast uncertainty <strong>compounds with time</strong>. The 90% confidence band is
<strong>{fan_spread_1y:.1f}pp wide at 1 year</strong>, expanding to <strong>{fan_spread_end:.1f}pp at {years} years</strong>.
The historical section (left of "Today") confirms the model is anchored to real SBP rate behaviour.
The long-run mean acts as a gravitational centre — all paths are statistically pulled toward it regardless of where they start.
</div></div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ── Parameter Sensitivity Table ────────────────────────────────────────────
st.markdown(f'<div class="chart-wrap"><div class="chart-header"><span class="chart-title">Parameter Sensitivity — How Mean Reversion Speed (a) Affects the Forecast</span><span class="chart-tag">SCENARIO ANALYSIS</span></div>',unsafe_allow_html=True)

a_values = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]
horizons = [1, 3, 5, 10]
rows_html = ""
for a_test in a_values:
    sim_test, time_test, _, _, _ = run_vasicek(df['Rate'], a_test, max(horizons), simulations)
    cells = ""
    for h in horizons:
        idx = int(h * 12) - 1
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

# ── CSV Export ─────────────────────────────────────────────────────────────
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

st.markdown(f'<div style="text-align:center;color:{T["footer_color"]};font-family:\'JetBrains Mono\',monospace;font-size:10px;margin-top:16px;border-top:1px solid {T["footer_border"]};padding-top:14px;letter-spacing:2px;">VASICEK MODEL  ·  SBP POLICY RATE  ·  {simulations} PATHS  ·  {years}Y HORIZON</div>',unsafe_allow_html=True)
