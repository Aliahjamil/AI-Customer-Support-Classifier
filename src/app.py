"""
app.py
------
Streamlit web app: Customer Support Ticket Classifier (final polished UI).

Run from the project root:
    streamlit run src/app.py

Tabs:
    1. Single Ticket : category, confidence, probabilities, suggested reply
    2. Batch Mode    : classify many tickets at once, download CSV
    3. How it works  : short explanation of the model
"""

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from predict import MODEL_PATH, load_model
from preprocessing import clean_text
from responses import get_auto_reply

LOW_CONFIDENCE_LIMIT = 40  # percent

CATEGORY_STYLE = {
    "Login Issue":       {"color": "#6C5CE7", "icon": "🔐"},
    "Application Error": {"color": "#E74C3C", "icon": "🐞"},
    "Report":            {"color": "#00B894", "icon": "📊"},
    "Account Update":    {"color": "#F39C12", "icon": "👤"},
    "Performance":       {"color": "#0984E3", "icon": "⚡"},
}
DEFAULT_STYLE = {"color": "#636E72", "icon": "🎫"}

EXAMPLES = {
    "🔐  Login issue":    "I cannot sign in, my password is rejected.",
    "🐞  App error":      "The app crashes when I click the save button.",
    "📊  Report":         "I need a monthly sales report in Excel.",
    "👤  Account update": "Please update my email address on my profile.",
    "⚡  Slow app":       "The website is very slow and keeps lagging.",
}

st.set_page_config(page_title="Support Ticket Classifier", page_icon="🎫",
                   layout="wide")

# =====================================================================
# CSS  (every colour is explicit so it looks the same in light/dark mode)
# =====================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

    :root { color-scheme: light; }
    html, body, .stApp, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* ---------- page background + top bar ---------- */
    .stApp {
        background: linear-gradient(135deg, #f4f1ff 0%, #eaf4ff 50%, #fff4f8 100%);
        background-attachment: fixed;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    header[data-testid="stHeader"] * { color: #1e1b4b !important; }
    .block-container { max-width: 1150px; padding-top: 1.5rem; padding-bottom: 3rem; }

    /* ---------- base text colours ---------- */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp [data-testid="stHeading"] * {
        color: #1e1b4b !important; font-weight: 800 !important; opacity: 1 !important;
    }
    .stApp label, .stApp label *, .stApp .stMarkdown p, .stApp li,
    .stApp [data-testid="stWidgetLabel"] * { color: #1e1b4b !important; }
    .stApp [data-testid="stWidgetLabel"] p { font-weight: 700 !important; font-size: 1rem !important; }
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] * { color: #4b4f6b !important; opacity: 1 !important; }

    /* ---------- animations ---------- */
    @keyframes fadeUp  { from {opacity:0; transform: translateY(18px);} to {opacity:1; transform:none;} }
    @keyframes grow    { from {width:0;} }
    @keyframes shimmer { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }
    @keyframes floaty  { 0%,100% {transform: translateY(0);} 50% {transform: translateY(-6px);} }
    @keyframes pop     { 0% {transform: scale(.88); opacity:0;} 100% {transform: scale(1); opacity:1;} }

    /* ---------- SECTION TITLES (clear, bold, with accent) ---------- */
    .section-title {
        display:flex; align-items:center; gap:12px;
        font-size:1.35rem; font-weight:800; color:#1e1b4b !important;
        margin: 22px 0 12px 0; letter-spacing:.2px; line-height:1.2;
    }
    .section-title .ico {
        width:38px; height:38px; border-radius:12px; flex-shrink:0;
        display:flex; align-items:center; justify-content:center; font-size:1.2rem;
        background: linear-gradient(135deg,#6C5CE7,#0984E3);
        box-shadow: 0 6px 14px rgba(108,92,231,.35);
    }
    .section-title::after {
        content:""; flex:1; height:3px; border-radius:3px;
        background: linear-gradient(90deg, rgba(108,92,231,.45), rgba(108,92,231,0));
    }
    .section-title.small { font-size:1.12rem; margin-top:14px; }
    .section-title.small .ico { width:32px; height:32px; font-size:1rem; border-radius:10px; }

    /* ---------- hero ---------- */
    .hero {
        background: linear-gradient(120deg, #6C5CE7, #0984E3, #00B894, #6C5CE7);
        background-size: 300% 300%; animation: shimmer 10s ease infinite, fadeUp .7s ease;
        padding: 32px 36px; border-radius: 26px; color: #fff;
        box-shadow: 0 16px 38px rgba(108,92,231,.38); margin-bottom: 22px;
        display: flex; align-items: center; justify-content: space-between; gap: 18px; flex-wrap: wrap;
    }
    .stApp .hero h1 { color:#ffffff !important; margin:0; font-size:2.3rem; line-height:1.2; font-weight:800 !important;
                      text-shadow: 0 2px 10px rgba(0,0,0,.18); }
    .stApp .hero p  { color:#ffffff !important; margin:10px 0 0 0; font-size:1.05rem; font-weight:500; }
    .hero .emoji { font-size: 4rem; animation: floaty 3.5s ease-in-out infinite; }
    .chips { margin-top: 16px; }
    .chip {
        display:inline-block; background: rgba(255,255,255,.25); color:#fff !important;
        padding: 6px 15px; border-radius: 999px; font-size:.82rem; margin: 3px 6px 3px 0;
        backdrop-filter: blur(4px); font-weight:600; border:1px solid rgba(255,255,255,.35);
    }

    /* ---------- cards ---------- */
    .card {
        background:#fff; border-radius:20px; padding:22px 26px; margin: 10px 0 16px 0;
        box-shadow: 0 6px 22px rgba(60,60,120,.10); animation: fadeUp .5s ease;
        transition: transform .25s ease, box-shadow .25s ease; color:#2d3436;
    }
    .card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(60,60,120,.18); }
    .stApp .card h3 { margin:0 0 12px 0; font-size:1.25rem; color:#1e1b4b !important; font-weight:800 !important; }
    .card p { color:#2d3436 !important; }

    .result-card {
        border-radius: 24px; padding: 26px 30px; color:#fff; margin: 10px 0 16px 0;
        display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;
        box-shadow: 0 14px 32px rgba(0,0,0,.22); animation: pop .5s ease;
        transition: transform .25s ease;
    }
    .result-card:hover { transform: scale(1.012); }
    .result-card .small { font-size:.8rem; letter-spacing:1.8px; font-weight:700; color:#fff !important; opacity:1; }
    .result-card .cat   { font-size:2.2rem; font-weight:800; color:#fff !important; margin:4px 0 8px 0;
                          text-shadow: 0 2px 8px rgba(0,0,0,.2); }
    .donut {
        width:116px; height:116px; border-radius:50%; display:flex; align-items:center;
        justify-content:center; flex-shrink:0; box-shadow: 0 6px 16px rgba(0,0,0,.2);
    }
    .donut > div {
        width:86px; height:86px; border-radius:50%; display:flex; flex-direction:column;
        align-items:center; justify-content:center; color:#fff !important; font-weight:800;
        font-size:1.3rem; line-height:1.1;
    }
    .donut small { font-size:.6rem; font-weight:600; color:#fff !important; }

    /* ---------- probability bars ---------- */
    .prob-row { display:flex; align-items:center; margin:12px 0; gap:12px; }
    .prob-name  { width:190px; color:#1e1b4b; font-weight:700; font-size:.93rem; }
    .prob-track { flex:1; background:#e8ebfa; border-radius:10px; height:16px; overflow:hidden; }
    .prob-fill  { height:16px; border-radius:10px; animation: grow 1s ease; }
    .prob-val   { width:62px; text-align:right; color:#1e1b4b; font-weight:800; font-size:.92rem; }

    /* ---------- reply + history + badge ---------- */
    .reply-box {
        background:#fff; border-left:6px solid #00B894; border-radius:14px; padding:18px 22px;
        box-shadow: 0 4px 16px rgba(0,0,0,.08); color:#2d3436; line-height:1.7; animation: fadeUp .6s ease;
        font-size:.97rem;
    }
    .hist-item {
        background:#fff; border-radius:12px; padding:11px 15px; margin:9px 0; color:#2d3436;
        box-shadow: 0 2px 10px rgba(0,0,0,.08); font-size:.88rem; animation: fadeUp .4s ease;
        transition: transform .2s ease;
    }
    .hist-item:hover { transform: translateX(5px); }
    .badge { display:inline-block; color:#fff !important; padding:4px 13px; border-radius:999px;
             font-size:.8rem; font-weight:700; white-space:nowrap; }

    /* ---------- batch table ---------- */
    table.batch { width:100%; border-collapse:separate; border-spacing:0 8px; }
    table.batch th { text-align:left; color:#1e1b4b; font-size:.82rem; padding:6px 12px; font-weight:800;
                     text-transform:uppercase; letter-spacing:.8px; }
    table.batch td { background:#fff; color:#2d3436; padding:13px 12px; font-size:.9rem; }
    table.batch td:first-child { border-radius:12px 0 0 12px; font-weight:700; }
    table.batch td:last-child  { border-radius:0 12px 12px 0; }
    table.batch tr:hover td { background:#f6f4ff; }
    .mini-track { background:#e8ebfa; border-radius:8px; height:8px; width:90px; overflow:hidden; margin-top:5px; }
    .mini-fill  { height:8px; border-radius:8px; }
    .review { color:#c0392b; font-weight:700; font-size:.8rem; }
    .ok     { color:#00997a; font-weight:700; font-size:.8rem; }

    /* ---------- stat tiles ---------- */
    .stats { display:flex; gap:14px; flex-wrap:wrap; margin:8px 0 6px 0; }
    .stat {
        flex:1; min-width:140px; background:#fff; border-radius:18px; padding:16px 18px;
        box-shadow: 0 6px 18px rgba(60,60,120,.10); animation: fadeUp .5s ease;
        border-top:5px solid #6C5CE7;
    }
    .stat .num { font-size:1.8rem; font-weight:800; color:#1e1b4b; line-height:1.1; }
    .stat .lbl { font-size:.78rem; font-weight:700; color:#4b4f6b; letter-spacing:.8px; text-transform:uppercase; }

    /* ---------- inputs ---------- */
    div[data-baseweb="textarea"], div[data-baseweb="textarea"] > div {
        background:#ffffff !important; border-radius:16px !important;
    }
    .stApp textarea {
        background:#ffffff !important; color:#1e1b4b !important; border-radius:16px !important;
        font-family:'Poppins',sans-serif !important; font-size:.98rem !important;
        -webkit-text-fill-color:#1e1b4b !important;
    }
    .stApp textarea::placeholder { color:#8a90a8 !important; opacity:1 !important; -webkit-text-fill-color:#8a90a8 !important; }
    div[data-baseweb="textarea"] {
        border:2px solid #cfc9ff !important; box-shadow:0 3px 12px rgba(108,92,231,.10);
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    div[data-baseweb="textarea"]:focus-within {
        border-color:#6C5CE7 !important; box-shadow:0 0 0 4px rgba(108,92,231,.18);
    }

    /* ---------- buttons ---------- */
    .stApp .stButton > button, .stApp [data-testid="stBaseButton-secondary"],
    .stApp [data-testid="stDownloadButton"] button {
        background:#ffffff !important; color:#1e1b4b !important; border:1.5px solid #cfc9ff !important;
        border-radius:14px !important; font-weight:700 !important; padding:.55rem 1rem !important;
        transition: all .22s ease; box-shadow:0 2px 8px rgba(108,92,231,.10);
    }
    .stApp .stButton > button *, .stApp [data-testid="stBaseButton-secondary"] *,
    .stApp [data-testid="stDownloadButton"] button * { color:#1e1b4b !important; }
    .stApp .stButton > button:hover, .stApp [data-testid="stBaseButton-secondary"]:hover,
    .stApp [data-testid="stDownloadButton"] button:hover {
        transform: translateY(-3px); border-color:#6C5CE7 !important;
        box-shadow:0 8px 18px rgba(108,92,231,.25); background:#f6f4ff !important;
    }
    .stApp .stButton > button[kind="primary"], .stApp [data-testid="stBaseButton-primary"] {
        background: linear-gradient(120deg,#6C5CE7,#0984E3) !important; border:none !important;
        color:#ffffff !important; font-size:1.05rem !important; padding:.8rem 1rem !important;
        box-shadow:0 8px 20px rgba(108,92,231,.40);
    }
    .stApp .stButton > button[kind="primary"] *, .stApp [data-testid="stBaseButton-primary"] * { color:#ffffff !important; }
    .stApp .stButton > button[kind="primary"]:hover, .stApp [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(120deg,#5b4bd6,#0773c9) !important; transform: translateY(-3px) scale(1.01);
        box-shadow:0 12px 26px rgba(108,92,231,.55);
    }

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"] { gap:10px; background:transparent; }
    .stTabs [data-baseweb="tab"] {
        background:#ffffff; border-radius:14px; padding:10px 22px;
        box-shadow:0 2px 8px rgba(0,0,0,.07); transition: all .2s ease;
    }
    .stTabs [data-baseweb="tab"] p { color:#1e1b4b !important; font-weight:700 !important; font-size:1.02rem !important; }
    .stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); }
    .stTabs [aria-selected="true"] { background: linear-gradient(120deg,#6C5CE7,#0984E3) !important; }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] * { color:#ffffff !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

    /* ---------- TABS: force visible (beats Streamlit theme) ---------- */
    .stApp div[data-baseweb="tab-list"] { gap:10px !important; background:transparent !important; }
    .stApp button[data-baseweb="tab"] {
        background:#ffffff !important; border-radius:14px !important; padding:10px 22px !important;
        box-shadow:0 2px 8px rgba(0,0,0,.10) !important; border:1.5px solid #cfc9ff !important;
        opacity:1 !important;
    }
    .stApp button[data-baseweb="tab"] p,
    .stApp button[data-baseweb="tab"] span,
    .stApp button[data-baseweb="tab"] div {
        color:#1e1b4b !important; font-weight:700 !important; opacity:1 !important;
    }
    .stApp button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(120deg,#6C5CE7,#0984E3) !important; border:none !important;
    }
    .stApp button[data-baseweb="tab"][aria-selected="true"] p,
    .stApp button[data-baseweb="tab"][aria-selected="true"] span,
    .stApp button[data-baseweb="tab"][aria-selected="true"] div { color:#ffffff !important; }
    .stApp div[data-baseweb="tab-highlight"], .stApp div[data-baseweb="tab-border"] {
        display:none !important; height:0 !important; background:transparent !important;
    }

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] { background: linear-gradient(180deg,#ffffff,#ece7ff); }
    section[data-testid="stSidebar"] * { color:#1e1b4b; }
    section[data-testid="stSidebar"] hr { border-color:#d9d3ff; }
    .legend { font-weight:700; margin:8px 0; font-size:.95rem; padding:8px 12px; border-radius:12px;
              background:#ffffff; box-shadow:0 2px 8px rgba(0,0,0,.06); }

    /* ---------- expander / alerts ---------- */
    .stApp [data-testid="stExpander"] { background:#fff; border-radius:14px; border:1px solid #d9d3ff; }
    .stApp [data-testid="stExpander"] summary, .stApp [data-testid="stExpander"] summary * { color:#1e1b4b !important; font-weight:700; }
    .stApp [data-testid="stAlert"] { border-radius:14px; }

    /* ---------- footer ---------- */
    .footer { text-align:center; color:#4b4f6b; font-size:.82rem; margin-top:34px; font-weight:500; }

    /* ---------- responsive ---------- */
    @media (max-width: 700px) {
        .stApp .hero h1 { font-size:1.6rem; }
        .hero .emoji { display:none; }
        .result-card { flex-direction:column; align-items:flex-start; }
        .prob-name { width:125px; font-size:.8rem; }
        .block-container { padding-left:1rem; padding-right:1rem; }
        .section-title { font-size:1.15rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# Helpers
# =====================================================================
@st.cache_resource
def get_model():
    """Load the model once and reuse it (faster than loading on every click)."""
    return load_model()


def set_example(text):
    """Button callback: put an example into the text box."""
    st.session_state["ticket_text"] = text


def style_of(category):
    return CATEGORY_STYLE.get(category, DEFAULT_STYLE)


def predict_with_probabilities(model, text):
    probabilities = model.predict_proba([clean_text(text)])[0]
    best_index = probabilities.argmax()
    return model.classes_[best_index], probabilities[best_index] * 100, probabilities


def badge(category):
    info = style_of(category)
    return f"<span class='badge' style='background:{info['color']}'>{info['icon']} {category}</span>"


def section(icon, title, small=False):
    """A clearly visible section heading with an icon tile and accent line."""
    cls = "section-title small" if small else "section-title"
    st.markdown(
        f"<div class='{cls}'><span class='ico'>{icon}</span><span>{title}</span></div>",
        unsafe_allow_html=True,
    )


if "history" not in st.session_state:
    st.session_state["history"] = []

# =====================================================================
# Header
# =====================================================================
st.markdown(
    """
    <div class="hero">
        <div>
            <h1>Customer Support Ticket Classifier</h1>
            <p>Paste a support ticket and the AI predicts its category in a moment.</p>
            <div class="chips">
                <span class="chip">🧠 TF-IDF</span>
                <span class="chip">📈 Logistic Regression</span>
                <span class="chip">🐍 Python + Scikit-learn</span>
                <span class="chip">⚡ Streamlit</span>
            </div>
        </div>
        <div class="emoji">🎫</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Sidebar
# =====================================================================
with st.sidebar:
    section("⚡", "Try an example", small=True)
    for label, example_text in EXAMPLES.items():
        st.button(label, use_container_width=True,
                  on_click=set_example, args=(example_text,), key=f"ex_{label}")

    st.markdown("---")
    section("🎨", "Categories", small=True)
    for name, info in CATEGORY_STYLE.items():
        st.markdown(
            f"<div class='legend' style='color:{info['color']};border-left:5px solid {info['color']}'>"
            f"{info['icon']} {name}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("AI/ML fresher project. Trained on a small synthetic dataset, "
               "so treat predictions as a demo.")
    if st.session_state["history"]:
        if st.button("🗑️ Clear history", use_container_width=True, key="clear_hist"):
            st.session_state["history"] = []
            st.rerun()

# =====================================================================
# Model check
# =====================================================================
if not Path(MODEL_PATH).exists():
    st.error("Trained model not found. Please run `python src/train.py` first.")
    st.stop()

model = get_model()

tab_single, tab_batch, tab_info = st.tabs(["🎯 Single Ticket", "📦 Batch Mode", "📘 How it works"])

# =====================================================================
# Tab 1: single ticket
# =====================================================================
with tab_single:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        section("✍️", "Enter the ticket")
        ticket_text = st.text_area(
            "Ticket description",
            key="ticket_text",
            height=170,
            placeholder="Example: I forgot my password and the reset email is not coming.",
            label_visibility="collapsed",
        )
        st.caption(f"{len(ticket_text)} characters")
        predict_clicked = st.button("🚀 Predict Category", type="primary",
                                    use_container_width=True, key="predict_btn")

        if st.session_state["history"]:
            section("🕘", "Recent predictions", small=True)
            for item in st.session_state["history"]:
                info = style_of(item["category"])
                text = item["text"]
                short = html.escape(text if len(text) <= 80 else text[:77] + "...")
                st.markdown(
                    f"<div class='hist-item' style='border-left:5px solid {info['color']}'>"
                    f"{badge(item['category'])} &nbsp;<b>{item['confidence']:.1f}%</b>"
                    f"<br><span style='opacity:.9'>{short}</span></div>",
                    unsafe_allow_html=True,
                )

    with right:
        section("🎯", "Prediction")
        if predict_clicked:
            if not ticket_text.strip():
                st.warning("Please enter a ticket description first.")
            else:
                category, confidence, probabilities = predict_with_probabilities(model, ticket_text)
                info = style_of(category)
                pct = min(confidence, 100)

                # Result card with donut gauge
                st.markdown(
                    f"""
                    <div class="result-card" style="background: linear-gradient(135deg,
                         {info['color']}, {info['color']}b0);">
                        <div>
                            <div class="small">PREDICTED CATEGORY</div>
                            <div class="cat">{info['icon']} {category}</div>
                            <div class="small">MODEL CONFIDENCE</div>
                        </div>
                        <div class="donut" style="background: conic-gradient(#ffffff {pct}%, rgba(255,255,255,.28) 0);">
                            <div style="background:{info['color']}">{confidence:.0f}%<small>confidence</small></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if confidence < LOW_CONFIDENCE_LIMIT:
                    st.warning("⚠️ Low confidence. A human agent should review this ticket.")
                else:
                    st.success("✅ The model is reasonably confident about this prediction.")

                # Probability bars
                order = pd.Series(probabilities * 100, index=model.classes_).sort_values(ascending=False)
                rows = ""
                for name, value in order.items():
                    color = style_of(name)["color"]
                    rows += (
                        f"<div class='prob-row'>"
                        f"<div class='prob-name'>{style_of(name)['icon']} {name}</div>"
                        f"<div class='prob-track'><div class='prob-fill' "
                        f"style='width:{value:.1f}%;background:linear-gradient(90deg,{color},{color}aa)'></div></div>"
                        f"<div class='prob-val'>{value:.1f}%</div></div>"
                    )
                st.markdown(f"<div class='card'><h3>📈 Probability for each category</h3>{rows}</div>",
                            unsafe_allow_html=True)

                # Suggested reply
                reply = get_auto_reply(category)
                section("💬", "Suggested auto-reply", small=True)
                st.markdown(f"<div class='reply-box'>{html.escape(reply)}</div>", unsafe_allow_html=True)
                with st.expander("📋 Copy reply text"):
                    st.code(reply, language=None)

                # Save to history (latest 8)
                st.session_state["history"].insert(
                    0, {"text": ticket_text.strip(), "category": category, "confidence": confidence})
                st.session_state["history"] = st.session_state["history"][:8]
        else:
            st.markdown(
                """
                <div class="card" style="text-align:center">
                    <div style="font-size:3rem;animation:floaty 3.5s ease-in-out infinite">🤖</div>
                    <h3>Ready when you are</h3>
                    <p style="margin:0">Type a ticket (or click an example in the sidebar)
                    and press <b>Predict Category</b>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =====================================================================
# Tab 2: batch mode
# =====================================================================
with tab_batch:
    section("📦", "Batch classification")
    st.markdown("Paste several tickets, **one per line**, and classify them all at once.")
    batch_text = st.text_area(
        "Tickets (one per line)",
        key="batch_text",
        height=170,
        placeholder="I forgot my password\nThe app shows an error while saving\nNeed the monthly sales report",
        label_visibility="collapsed",
    )
    if st.button("⚡ Classify all", type="primary", use_container_width=True, key="batch_btn"):
        lines = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one ticket.")
        else:
            results = []
            for line in lines:
                category, confidence, _ = predict_with_probabilities(model, line)
                results.append({"Ticket": line, "Category": category, "Confidence": confidence})

            # --- stat tiles ---
            total = len(results)
            avg_conf = sum(r["Confidence"] for r in results) / total
            need_review = sum(1 for r in results if r["Confidence"] < LOW_CONFIDENCE_LIMIT)
            st.markdown(
                f"""
                <div class="stats">
                    <div class="stat"><div class="num">{total}</div><div class="lbl">Tickets</div></div>
                    <div class="stat" style="border-top-color:#00B894"><div class="num">{avg_conf:.1f}%</div>
                        <div class="lbl">Average confidence</div></div>
                    <div class="stat" style="border-top-color:#E74C3C"><div class="num">{need_review}</div>
                        <div class="lbl">Need review</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --- results table ---
            section("📋", "Results", small=True)
            table = ("<table class='batch'><tr><th>#</th><th>Ticket</th><th>Category</th>"
                     "<th>Confidence</th><th>Status</th></tr>")
            for number, row in enumerate(results, start=1):
                color = style_of(row["Category"])["color"]
                flag = ("<span class='review'>⚠️ Review</span>" if row["Confidence"] < LOW_CONFIDENCE_LIMIT
                        else "<span class='ok'>✅ OK</span>")
                table += (
                    f"<tr><td>{number}</td><td>{html.escape(row['Ticket'])}</td>"
                    f"<td>{badge(row['Category'])}</td>"
                    f"<td><b>{row['Confidence']:.1f}%</b>"
                    f"<div class='mini-track'><div class='mini-fill' "
                    f"style='width:{min(row['Confidence'], 100):.0f}%;background:{color}'></div></div></td>"
                    f"<td>{flag}</td></tr>"
                )
            table += "</table>"
            st.markdown(table, unsafe_allow_html=True)

            # --- category summary ---
            counts = pd.Series([r["Category"] for r in results]).value_counts()
            section("📊", "Summary by category", small=True)
            st.markdown(" ".join(f"{badge(name)} <b>×{count}</b>&nbsp;&nbsp;" for name, count in counts.items()),
                        unsafe_allow_html=True)

            csv = pd.DataFrame(results).to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results (CSV)", csv, "predictions.csv", "text/csv",
                               use_container_width=True, key="dl_btn")

# =====================================================================
# Tab 3: how it works
# =====================================================================
with tab_info:
    section("📘", "How it works")
    st.markdown(
        """
        <div class="card">
            <h3>🔄 Pipeline</h3>
            <p style="line-height:2.3">
            <span class="badge" style="background:#6C5CE7">Ticket text</span> →
            <span class="badge" style="background:#0984E3">Text cleaning</span> →
            <span class="badge" style="background:#00B894">TF-IDF features</span> →
            <span class="badge" style="background:#F39C12">Logistic Regression</span> →
            <span class="badge" style="background:#E74C3C">Category + confidence</span>
            </p>
        </div>
        <div class="card">
            <h3>🧠 In simple words</h3>
            <p><b>TF-IDF</b> turns each ticket into numbers: words that are special to a ticket
            (like <i>password</i>) get a high score, very common words get a low score.</p>
            <p><b>Logistic Regression</b> learned a weight for every word and every category.
            It adds up the weights of the words in your ticket and converts the totals into
            probabilities. The highest probability is the predicted category.</p>
            <p style="margin:0"><b>Note:</b> the model was trained on about 200 synthetic tickets,
            so confidence scores are modest and this is a demo, not a production system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='footer'>Built with Python · Scikit-learn · Streamlit</div>",
            unsafe_allow_html=True)