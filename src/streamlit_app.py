import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import feedparser
from textwrap import dedent
from datetime import date
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

st.set_page_config(
    page_title="FootballPulse AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Hanken+Grotesk:wght@400;500;600;700;800&display=swap');

/* Palette
   Primary   #A855F7   purple
   Secondary #7C3AED   deep violet
   Tertiary  #4ADE80   green accent
   Neutral   #E2E2F0   light text
   Bg0       #0D0D12   page
   Bg1       #13131A   card
   Bg2       #1A1A24   input / hover
   Border    #2A2A3A
*/

html, body, [class*="css"], .stApp {
    font-family: 'Hanken Grotesk', sans-serif !important;
    background-color: #0D0D12 !important;
    color: #E2E2F0 !important;
}
#MainMenu, header[data-testid="stHeader"], footer,
div[data-testid="stToolbar"], div[data-testid="stDecoration"],
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp > div { padding: 0 !important; }

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #1A1A24 !important;
    border: 0.5px solid #2A2A3A !important;
    color: #E2E2F0 !important;
    border-radius: 10px !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
}
div[data-baseweb="select"] span { color: #E2E2F0 !important; }
div[data-baseweb="popover"] { background: #1A1A24 !important; border: 0.5px solid #2A2A3A !important; }
li[role="option"] { background: #1A1A24 !important; color: #9090A8 !important; font-family: 'Hanken Grotesk', sans-serif !important; }
li[role="option"]:hover { background: #22223A !important; color: #E2E2F0 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #A855F7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: opacity .2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Radio pills */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
}
div[data-testid="stRadio"] label {
    background: #13131A !important;
    border: 0.5px solid #2A2A3A !important;
    border-radius: 20px !important;
    padding: 6px 16px !important;
    font-size: 12px !important;
    color: #5A5A7A !important;
    cursor: pointer !important;
    margin: 0 !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    transition: all .15s !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: rgba(168,85,247,0.12) !important;
    border-color: #A855F7 !important;
    color: #A855F7 !important;
}
div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 12px !important;
    margin: 0 !important;
}

/* Tabs — custom styled */
div[data-testid="stTabs"] > div:first-child {
    background: transparent !important;
    border-bottom: 0.5px solid #2A2A3A !important;
    gap: 0 !important;
    padding: 0 !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: #5A5A7A !important;
    font-family: 'Hanken Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    padding: 12px 20px !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all .2s !important;
}
button[data-baseweb="tab"]:hover { color: #E2E2F0 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #A855F7 !important;
    border-bottom-color: #A855F7 !important;
}
div[data-testid="stTabPanel"] { padding: 0 !important; }

.stAlert { border-radius: 10px !important; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }
@keyframes fadeIn { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
.fade-in { animation: fadeIn .4s ease forwards; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════
def _team_in_title(title, team):
    text_lower = title.lower()
    if team.lower() in text_lower:
        return True
    for word in team.split():
        if len(word) > 4 and word.lower() in text_lower:
            return True
    return False


def fetch_google_rss(team):
    queries = [f"{team} football", f"{team} soccer", f"{team} World Cup"]
    seen, results = set(), []
    for query in queries:
        url = f"https://news.google.com/rss/search?q={query.replace(' ','+')}&hl=en&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        for e in feed.entries:
            title = e.get("title", "")
            if not _team_in_title(title, team):
                continue
            key = title[:40].lower()
            if key in seen:
                continue
            seen.add(key)
            src = e.get("source", {})
            src_name = (
                (src.get("title") or src.get("name") or "Google News")
                if isinstance(src, dict)
                else (str(src) or "Google News")
            )
            results.append(
                {
                    "title": title,
                    "description": e.get("summary", ""),
                    "source": {"name": src_name},
                }
            )
            if len(results) >= 8:
                break
        if len(results) >= 8:
            break
    return results


def fetch_newsapi(team):
    try:
        from newsapi import NewsApiClient

        api_key = os.getenv("NEWS_API_KEY", "")
        if not api_key:
            return []
        client = NewsApiClient(api_key=api_key)
        response = client.get_everything(
            q=f'"{team}" football', language="en", sort_by="publishedAt", page_size=10
        )
        articles = response.get("articles", [])
        return [
            a
            for a in articles
            if _team_in_title(
                a.get("title", "") + " " + (a.get("description") or ""), team
            )
        ][:8]
    except:
        return []


def analyse_sentiment(articles):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    az = SentimentIntensityAnalyzer()
    sc = [
        az.polarity_scores(
            (a.get("title", "") or "") + " " + (a.get("description", "") or "")
        )["compound"]
        for a in articles
    ]
    if not sc:
        return 0.0, 60, 25, 15
    avg = round(float(np.mean(sc)), 3)
    pos = round(sum(1 for s in sc if s > 0.05) / len(sc) * 100)
    neg = round(sum(1 for s in sc if s < -0.05) / len(sc) * 100)
    return avg, pos, 100 - pos - neg, neg


# ══════════════════════════════════════════════════════════
# MODEL + DATA
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    base = os.path.join(os.path.dirname(__file__), "..", "models")
    model = joblib.load(os.path.join(base, "model.pkl"))
    features = joblib.load(os.path.join(base, "feature_columns.pkl"))
    return model, features


@st.cache_data
def load_data():
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", "clean_matches.csv"
    )
    df = pd.read_csv(path)
    return df.sort_values("Year").reset_index(drop=True)


model, features = load_model()
df = load_data()


@st.cache_data
def build_elo(_df):
    BASE_ELO = 1000
    K = 20
    elo = {}

    def get(t):
        return elo.get(t, BASE_ELO)

    def exp(a, b):
        return 1 / (1 + 10 ** ((b - a) / 400))

    def update(w, l, draw=False):
        ew = exp(get(w), get(l))
        sc = 0.5 if draw else 1.0
        elo[w] = get(w) + K * (sc - ew)
        elo[l] = get(l) + K * ((1 - sc) - (1 - ew))

    for _, r in _df.iterrows():
        h, a = r["Home Team Name"], r["Away Team Name"]
        if r["Winner"] == h:
            update(h, a)
        elif r["Winner"] == a:
            update(a, h)
        else:
            update(h, a, draw=True)
    return elo


elo_ratings = build_elo(df)


def attack(team):
    h = df[df["Home Team Name"] == team]["Home Team Goals"]
    a = df[df["Away Team Name"] == team]["Away Team Goals"]
    g = pd.concat([h, a])
    return round(g.mean(), 2) if len(g) >= 3 else 1.0


def defense(team):
    h = df[df["Home Team Name"] == team]["Away Team Goals"]
    a = df[df["Away Team Name"] == team]["Home Team Goals"]
    c = pd.concat([h, a])
    return round(c.mean(), 2) if len(c) >= 3 else 1.0


def win_rate(team):
    h = len(df[df["Home Team Name"] == team])
    a = len(df[df["Away Team Name"] == team])
    t = h + a
    if t == 0:
        return 0.5
    w = len(df[df["Winner"] == team])
    return round(w / t, 3)


def recent_form(team, n=5):
    m = df[(df["Home Team Name"] == team) | (df["Away Team Name"] == team)].tail(n)
    if len(m) == 0:
        return 0.5
    wts = np.array([0.1, 0.15, 0.2, 0.25, 0.3])[-len(m) :]
    wts = wts / wts.sum()
    sc = [
        1.0 if r["Winner"] == team else 0.5 if r["Winner"] == "Draw" else 0.0
        for _, r in m.iterrows()
    ]
    return round(float(np.dot(sc, wts)), 3)


STAGE_MAP = {
    "Group stage": 1,
    "Round of 16": 2,
    "Quarter-final": 3,
    "Semi-final": 4,
    "Third place": 5,
    "Final": 6,
}


def get_features(home, away, stage_val):
    he = elo_ratings.get(home, 1000)
    ae = elo_ratings.get(away, 1000)
    ha = attack(home)
    aa = attack(away)
    hd = defense(home)
    ad = defense(away)
    hg = ha - hd
    ag = aa - ad
    hf = recent_form(home)
    af = recent_form(away)
    hw = win_rate(home)
    aw = win_rate(away)
    return {
        "home_elo": he,
        "away_elo": ae,
        "elo_diff": he - ae,
        "elo_ratio": he / max(ae, 1),
        "home_attack": ha,
        "away_attack": aa,
        "home_defense": hd,
        "away_defense": ad,
        "home_goal_diff": hg,
        "away_goal_diff": ag,
        "attack_diff": ha - aa,
        "defense_diff": ad - hd,
        "goal_diff_diff": hg - ag,
        "home_recent_form": hf,
        "away_recent_form": af,
        "home_win_rate": hw,
        "away_win_rate": aw,
        "form_diff": hf - af,
        "stage_encoded": stage_val,
    }


def predict(team_a, team_b, stage_val):
    fa = pd.DataFrame([get_features(team_a, team_b, stage_val)]).reindex(
        columns=features, fill_value=0
    )
    fb = pd.DataFrame([get_features(team_b, team_a, stage_val)]).reindex(
        columns=features, fill_value=0
    )
    pa = model.predict_proba(fa)[0]
    pb = model.predict_proba(fb)[0]
    cl = list(model.classes_)
    draw = (pa[cl.index(0)] + pb[cl.index(0)]) / 2
    a_win = (pa[cl.index(1)] + pb[cl.index(2)]) / 2
    b_win = (pa[cl.index(2)] + pb[cl.index(1)]) / 2
    tot = draw + a_win + b_win
    return round(a_win / tot, 3), round(draw / tot, 3), round(b_win / tot, 3)


def get_h2h(team_a, team_b):
    mask = ((df["Home Team Name"] == team_a) & (df["Away Team Name"] == team_b)) | (
        (df["Home Team Name"] == team_b) & (df["Away Team Name"] == team_a)
    )
    return df[mask].sort_values("Year")


all_teams = sorted(set(df["Home Team Name"].tolist() + df["Away Team Name"].tolist()))
WC2026 = [
    "Argentina",
    "Australia",
    "Belgium",
    "Brazil",
    "Cameroon",
    "Canada",
    "Croatia",
    "Denmark",
    "Ecuador",
    "England",
    "France",
    "Germany",
    "Ghana",
    "Iran",
    "Japan",
    "Mexico",
    "Morocco",
    "Netherlands",
    "Poland",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Senegal",
    "South Korea",
    "Spain",
    "Switzerland",
    "Tunisia",
    "USA",
    "Uruguay",
    "Wales",
]
team_options = [t for t in WC2026 if t in all_teams] + [
    t for t in all_teams if t not in WC2026
]


# ══════════════════════════════════════════════════════════
# DEV FLAGS
# TEST_NEWS_MODE = True  → bypasses June 11 gate so you can
#   preview the news/sentiment section right now.
#   Flip back to False before going live.
# ══════════════════════════════════════════════════════════
TEST_NEWS_MODE = True # ← set False for production

# ══════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "show_pulse_panel" not in st.session_state:
    st.session_state.show_pulse_panel = False


# ══════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<div style="background:#0D0D12;border-bottom:0.5px solid #2A2A3A;padding:14px 32px;
            display:flex;align-items:center;justify-content:space-between;
            font-family:'Hanken Grotesk',sans-serif;position:sticky;top:0;z-index:100">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="width:36px;height:36px;background:linear-gradient(135deg,#7C3AED,#A855F7);
                border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:18px">⚽</div>
    <div>
      <div style="display:flex;align-items:baseline;gap:8px">
        <span style="font-size:17px;font-weight:800;color:#E2E2F0;letter-spacing:-0.6px;line-height:1">FootballPulse</span>
        <span style="font-size:11px;font-weight:500;color:#A855F7;letter-spacing:0.2px;background:rgba(168,85,247,0.12);
                     border:0.5px solid rgba(168,85,247,0.3);border-radius:4px;padding:1px 6px">AI</span>
      </div>
      <div style="font-size:10px;color:#3A3A55;margin-top:3px;letter-spacing:0.3px">
        World Cup 2026 &nbsp;·&nbsp; 92 yrs of data &nbsp;·&nbsp; Match Intelligence
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:16px">
    <div style="font-size:11px;color:#3A3A55;letter-spacing:0.3px;display:none">
      <span style="color:#5A5A7A">Model acc.</span> <span style="color:#7A7A9A;font-weight:600">~57%</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#4ADE80;font-weight:600;
                background:rgba(74,222,128,0.06);border:0.5px solid rgba(74,222,128,0.2);
                border-radius:20px;padding:5px 12px">
      <div style="width:6px;height:6px;background:#4ADE80;border-radius:50%;animation:pulse 2s infinite"></div>
      Live
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════
# MAIN LAYOUT — two-panel grid
# ══════════════════════════════════════════════════════════
left_col, right_col = st.columns([2, 1], gap="small")


# ──────────────────────────────────────────────────────────
# LEFT PANEL — Tabs: Match Predictor | Head-to-Head | Pulse
# ──────────────────────────────────────────────────────────
with left_col:
    st.markdown("<div style='padding:24px 24px 0 32px'>", unsafe_allow_html=True)

    tab_pred, tab_h2h, tab_pulse = st.tabs(
        ["Match Predictor", "Head · to · Head", "World Cup Pulse"]
    )

    # ══ TAB 1 — MATCH PREDICTOR ══════════════════════════
    with tab_pred:
        st.session_state.show_pulse_panel = False
        st.markdown("<div style='padding-top:20px'>", unsafe_allow_html=True)

        # ── Team selectors ────────────────────────────────
        sel1, sel_vs, sel2 = st.columns([10, 1, 10])
        with sel1:
            team_a = st.selectbox(
                "Team A",
                team_options,
                index=team_options.index("Brazil") if "Brazil" in team_options else 0,
                label_visibility="collapsed",
            )
        with sel_vs:
            st.markdown(
                "<div style='text-align:center;padding-top:10px;font-size:11px;"
                "color:#3A3A55;font-weight:700'>VS</div>",
                unsafe_allow_html=True,
            )
        with sel2:
            team_b = st.selectbox(
                "Team B",
                team_options,
                index=team_options.index("France") if "France" in team_options else 1,
                label_visibility="collapsed",
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Team cards ────────────────────────────────────
        elo_a = round(elo_ratings.get(team_a, 1000))
        elo_b = round(elo_ratings.get(team_b, 1000))
        ca, _, cb = st.columns([10, 1, 10])
        with ca:
            st.markdown(
                f"""
            <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                        padding:20px;text-align:center;font-family:'Hanken Grotesk',sans-serif">
              <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.2px;text-transform:uppercase;
                          font-weight:600;margin-bottom:8px">Team A</div>
              <div style="font-size:26px;font-weight:800;color:#E2E2F0;letter-spacing:-0.5px">{team_a}</div>
              <div style="margin-top:8px;display:inline-block;background:rgba(168,85,247,0.1);
                          border:0.5px solid rgba(168,85,247,0.3);border-radius:6px;
                          padding:3px 10px;font-size:11px;color:#A855F7;font-weight:600">Elo {elo_a}</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with cb:
            st.markdown(
                f"""
            <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                        padding:20px;text-align:center;font-family:'Hanken Grotesk',sans-serif">
              <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.2px;text-transform:uppercase;
                          font-weight:600;margin-bottom:8px">Team B</div>
              <div style="font-size:26px;font-weight:800;color:#E2E2F0;letter-spacing:-0.5px">{team_b}</div>
              <div style="margin-top:8px;display:inline-block;background:rgba(74,222,128,0.1);
                          border:0.5px solid rgba(74,222,128,0.3);border-radius:6px;
                          padding:3px 10px;font-size:11px;color:#4ADE80;font-weight:600">Elo {elo_b}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Stage selector ────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;'
            'font-weight:600;margin-bottom:10px;font-family:Hanken Grotesk,sans-serif">Tournament Stage</div>',
            unsafe_allow_html=True,
        )
        stage = st.radio(
            "stage",
            list(STAGE_MAP.keys()),
            index=2,
            horizontal=True,
            label_visibility="collapsed",
        )
        stage_val = STAGE_MAP[stage]

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── Predict button ────────────────────────────────
        pb1, pb2, pb3 = st.columns([2, 4, 2])
        with pb2:
            predict_btn = st.button("Predict Match", use_container_width=True)

        if predict_btn:
            if team_a == team_b:
                st.warning("Please select two different teams.")
            else:
                p_a, p_draw, p_b = predict(team_a, team_b, stage_val)
                st.session_state.prediction = {
                    "team_a": team_a,
                    "team_b": team_b,
                    "stage": stage,
                    "p_a": p_a,
                    "p_draw": p_draw,
                    "p_b": p_b,
                }

        # ── Results ───────────────────────────────────────
        pred = st.session_state.prediction
        if pred and pred["p_a"] is not None:
            ta = pred["team_a"]
            tb = pred["team_b"]
            p_a = pred["p_a"]
            p_draw = pred["p_draw"]
            p_b = pred["p_b"]
            winner = ta if p_a > p_b else tb
            win_prob = max(p_a, p_b)
            pa_pct = int(p_a * 100)
            pd_pct = int(p_draw * 100)
            pb_pct = int(p_b * 100)

            st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
            st.markdown(
                "<div style='height:0.5px;background:#1E1E2E;margin-bottom:24px'></div>",
                unsafe_allow_html=True,
            )

            # Winner card
            st.markdown(
                f"""
            <div class="fade-in" style="background:linear-gradient(135deg,rgba(124,58,237,0.18),rgba(168,85,247,0.06));
                        border:0.5px solid rgba(168,85,247,0.4);border-radius:14px;
                        padding:28px 24px;text-align:center;margin-bottom:20px;
                        font-family:'Hanken Grotesk',sans-serif">
              <div style="font-size:10px;color:#7C3AED;letter-spacing:1.6px;text-transform:uppercase;
                          font-weight:600;margin-bottom:10px">Predicted Winner</div>
              <div style="font-size:40px;font-weight:800;color:#E2E2F0;letter-spacing:-1.5px;
                          margin-bottom:12px">{winner}</div>
              <div style="display:inline-flex;align-items:center;gap:10px;
                          background:rgba(168,85,247,0.15);border:0.5px solid rgba(168,85,247,0.35);
                          border-radius:20px;padding:6px 20px">
                <span style="font-size:16px;font-weight:800;color:#A855F7">{win_prob*100:.0f}%</span>
                <span style="font-size:11px;color:#7C3AED;font-weight:500">Win Probability</span>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )

            # Probability bars
            def prob_bar(label, pct, color):
                st.markdown(
                    f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;
                            font-family:'Hanken Grotesk',sans-serif">
                  <div style="font-size:12px;color:#7A7A9A;width:130px;flex-shrink:0">{label}</div>
                  <div style="flex:1;height:4px;background:#1E1E2E;border-radius:2px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:2px"></div>
                  </div>
                  <div style="font-size:12px;font-weight:700;color:#E2E2F0;width:34px;text-align:right">{pct}%</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            prob_bar(f"{ta} win", pa_pct, "#A855F7")
            prob_bar("Draw", pd_pct, "#3A3A5A")
            prob_bar(f"{tb} win", pb_pct, "#4ADE80")

        st.markdown("</div>", unsafe_allow_html=True)  # tab padding

    # ══ TAB 2 — HEAD TO HEAD ═════════════════════════════
    with tab_h2h:
        st.session_state.show_pulse_panel = False
        st.markdown("<div style='padding-top:20px'>", unsafe_allow_html=True)

        # Use same teams from predictor tab
        h2h_matches = get_h2h(team_a, team_b)
        total = len(h2h_matches)

        if total == 0:
            st.markdown(
                f"""
            <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                        padding:40px;text-align:center;font-family:'Hanken Grotesk',sans-serif">
              <div style="font-size:28px;margin-bottom:12px">❌</div>
              <div style="font-size:14px;font-weight:700;color:#E2E2F0;margin-bottom:6px">
                No World Cup meetings found</div>
              <div style="font-size:12px;color:#5A5A7A">
                {team_a} and {team_b} have never met at a World Cup in the dataset.</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            a_wins = len(h2h_matches[h2h_matches["Winner"] == team_a])
            b_wins = len(h2h_matches[h2h_matches["Winner"] == team_b])
            draws = total - a_wins - b_wins

            # Stats row
            s1, s2, s3 = st.columns(3)
            for col, label, val, color in [
                (s1, f"{team_a} wins", a_wins, "#A855F7"),
                (s2, "Draws", draws, "#5A5A7A"),
                (s3, f"{team_b} wins", b_wins, "#4ADE80"),
            ]:
                with col:
                    st.markdown(
                        f"""
                    <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                                padding:20px;text-align:center;font-family:'Hanken Grotesk',sans-serif">
                      <div style="font-size:32px;font-weight:800;color:{color};letter-spacing:-1px">{val}</div>
                      <div style="font-size:11px;color:#5A5A7A;margin-top:4px;font-weight:500">{label}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

            # Win distribution bar
            if total > 0:
                aw_pct = int(a_wins / total * 100)
                dr_pct = int(draws / total * 100)
                bw_pct = 100 - aw_pct - dr_pct
                st.markdown(
                    f"""
                <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                            padding:16px 20px;margin-bottom:16px;font-family:'Hanken Grotesk',sans-serif">
                  <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                              font-weight:600;margin-bottom:12px">Win Distribution</div>
                  <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;gap:2px">
                    <div style="width:{aw_pct}%;background:#A855F7;border-radius:4px 0 0 4px"></div>
                    <div style="width:{dr_pct}%;background:#3A3A5A"></div>
                    <div style="width:{bw_pct}%;background:#4ADE80;border-radius:0 4px 4px 0"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;margin-top:8px">
                    <span style="font-size:11px;color:#A855F7;font-weight:600">{team_a[:3].upper()} {aw_pct}%</span>
                    <span style="font-size:11px;color:#5A5A7A">{dr_pct}% Draw</span>
                    <span style="font-size:11px;color:#4ADE80;font-weight:600">{bw_pct}% {team_b[:3].upper()}</span>
                  </div>
                </div>""",
                    unsafe_allow_html=True,
                )

            # Match timeline
            st.markdown(
                """
            <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                        padding:16px 20px;font-family:'Hanken Grotesk',sans-serif">
              <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                          font-weight:600;margin-bottom:14px">World Cup Meetings</div>""",
                unsafe_allow_html=True,
            )

            for _, row in h2h_matches.iterrows():
                hteam = row["Home Team Name"]
                ateam = row["Away Team Name"]
                hg = int(row.get("Home Team Goals", 0))
                ag = int(row.get("Away Team Goals", 0))
                stage_name = row.get("Stage", "")
                year = int(row.get("Year", 0))
                match_winner = row.get("Winner", "Draw")
                if match_winner == team_a:
                    wcolor, wlabel = "#A855F7", team_a
                elif match_winner == team_b:
                    wcolor, wlabel = "#4ADE80", team_b
                else:
                    wcolor, wlabel = "#5A5A7A", "Draw"

                st.markdown(
                    f"""
                <div style="display:flex;align-items:center;gap:12px;padding:10px 0;
                            border-bottom:0.5px solid #1E1E2E;">
                  <div style="font-size:12px;font-weight:700;color:#5A5A7A;width:36px;flex-shrink:0">{year}</div>
                  <div style="font-size:12px;color:#7A7A9A;flex:1">{stage_name}</div>
                  <div style="font-family:'Hanken Grotesk',sans-serif;font-size:13px;font-weight:700;
                              color:#E2E2F0;flex-shrink:0">{hteam[:3].upper()} {hg}–{ag} {ateam[:3].upper()}</div>
                  <div style="background:rgba(0,0,0,0.3);border:0.5px solid {wcolor}33;border-radius:6px;
                              padding:3px 9px;font-size:10px;font-weight:700;color:{wcolor};flex-shrink:0">{wlabel}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # tab padding

    # ══ TAB 3 — WORLD CUP PULSE ══════════════════════════
    with tab_pulse:
        st.markdown("<div style='padding-top:20px'>", unsafe_allow_html=True)

        # ── Tab header with inline action ─────────────────
        hdr_col, btn_col = st.columns([3, 2])
        with hdr_col:
            st.markdown(
                """<div style="padding-top:6px;font-family:'Hanken Grotesk',sans-serif">
                  <div style="font-size:18px;font-weight:800;color:#E2E2F0;letter-spacing:-0.4px;line-height:1.2">
                    World Cup Pulse</div>
                  <div style="font-size:11px;color:#5A5A7A;margin-top:4px">Live headlines &amp; sentiment</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with btn_col:
            if st.button(
                "View Sentiment Dashboard",
                use_container_width=True,
                key="pulse_panel_btn",
            ):
                st.session_state.show_pulse_panel = True

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        if date.today() >= date(2026, 6, 11) or TEST_NEWS_MODE:
            import re as _re
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            @st.cache_data(ttl=1800, show_spinner=False)
            def fetch_wc_headlines():
                queries = ["FIFA World Cup 2026", "World Cup 2026 football"]
                seen, results = set(), []
                for query in queries:
                    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en&gl=US&ceid=US:en"
                    feed = feedparser.parse(url)
                    for e in feed.entries:
                        title = e.get("title", "")
                        if not title:
                            continue
                        key = title[:40].lower()
                        if key in seen:
                            continue
                        seen.add(key)
                        src = e.get("source", {})
                        src_name = (
                            (src.get("title") or src.get("name") or "Google News")
                            if isinstance(src, dict)
                            else (str(src) or "Google News")
                        )
                        results.append(
                            {
                                "title": title,
                                "description": e.get("summary", ""),
                                "source": {"name": src_name},
                            }
                        )
                        if len(results) >= 12:
                            break
                    if len(results) >= 12:
                        break
                return results

            wc_headlines = fetch_wc_headlines()
            from collections import Counter

            STOP = {
                "world",
                "cup",
                "what",
                "when",
                "how",
                "who",
                "the",
                "for",
                "and",
                "with",
                "from",
                "this",
                "that",
                "will",
                "have",
                "been",
                "their",
                "they",
                "about",
                "into",
                "after",
                "ahead",
                "says",
                "said",
                "over",
                "just",
                "more",
                "than",
                "some",
                "also",
                "first",
                "last",
                "next",
                "year",
                "time",
                "game",
                "match",
                "men",
                "women",
                "top",
                "full",
                "list",
                "key",
                "new",
                "2026",
                "2025",
                "fifa",
                "football",
                "soccer",
                "power",
                "here",
                "guide",
                "news",
            }
            counter = Counter()
            for art in wc_headlines:
                title = _re.sub(
                    r"\s*[-–|]\s*[^-–|]{3,40}$", "", art.get("title", "")
                ).strip()
                for w in _re.findall(r"\b[A-Za-z][a-z]{2,}\b", title):
                    if w.lower() not in STOP and len(w) > 3:
                        counter[w.lower()] += 1
            topics = []
            for word, count in counter.most_common(10):
                if len(topics) >= 5:
                    break
                if count >= 2:
                    topics.append((word.capitalize(), count, count >= 3))

            topic_rows = ""
            for i, (word, count, is_hot) in enumerate(topics):
                border = (
                    "border-bottom:0.5px solid #1E1E2E;" if i < len(topics) - 1 else ""
                )
                badge = (
                    '<span style="background:rgba(168,85,247,0.15);border:0.5px solid rgba(168,85,247,0.3);'
                    "color:#A855F7;font-size:10px;padding:2px 7px;border-radius:4px;margin-right:8px;"
                    'font-weight:700">HOT</span>'
                    if is_hot
                    else ""
                )
                nc = "#E2E2F0" if is_hot else "#7A7A9A"
                cnt_lbl = f"{count} mentions" if count != 1 else "1 mention"
                topic_rows += f"""<div style="display:flex;justify-content:space-between;align-items:center;
                    padding:9px 0;{border}">{badge}<span style="font-size:12px;color:{nc};font-weight:500">#{word}</span>
                    <span style="font-size:11px;color:#3A3A5A">{cnt_lbl}</span></div>"""

            if date.today() >= date(2026, 6, 11):
                if not topic_rows:
                    topic_rows = '<div style="font-size:12px;color:#3A3A5A;padding:12px 0;text-align:center">Topics loading — check back soon</div>'
                st.markdown(
                    f"""
                <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                            padding:16px 20px;margin-bottom:16px;font-family:'Hanken Grotesk',sans-serif">
                  <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                              font-weight:600;margin-bottom:12px">Trending Topics</div>
                  {topic_rows}
                </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '''<div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                        padding:16px 20px;margin-bottom:16px;font-family:Hanken Grotesk,sans-serif">
                      <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                                  font-weight:600;margin-bottom:10px">Trending Topics</div>
                      <div style="display:flex;align-items:center;gap:10px;padding:12px 0">
                        <div style="width:7px;height:7px;border-radius:50%;background:#F97316;opacity:0.4"></div>
                        <span style="font-size:12px;color:#3A3A5A">Live trending unlocks June 11 · Opening match day</span>
                      </div>
                    </div>''',
                    unsafe_allow_html=True,
                )

            az = SentimentIntensityAnalyzer()
            st.markdown(
                """
            <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                        padding:16px 20px;font-family:'Hanken Grotesk',sans-serif">
              <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                          font-weight:600;margin-bottom:12px">Latest Headlines</div>""",
                unsafe_allow_html=True,
            )

            if wc_headlines:
                for art in wc_headlines[:6]:
                    raw = art.get("title") or ""
                    clean = _re.sub(r"\s*[-–|]\s*[^-–|]{3,40}$", "", raw).strip()[:90]
                    if not clean:
                        continue
                    source = (art.get("source", {}).get("name") or "News")[:18]
                    score = az.polarity_scores(clean)["compound"]
                    if score > 0.05:
                        bg, fg, br, label = (
                            "rgba(168,85,247,0.08)",
                            "#A855F7",
                            "rgba(168,85,247,0.3)",
                            "positive",
                        )
                    elif score < -0.05:
                        bg, fg, br, label = (
                            "rgba(239,68,68,0.08)",
                            "#ef4444",
                            "rgba(239,68,68,0.3)",
                            "negative",
                        )
                    else:
                        bg, fg, br, label = "#1E1E2E", "#5A5A7A", "#2A2A3A", "neutral"
                    st.markdown(
                        f"""
                    <div style="background:#1A1A24;border-radius:8px;padding:10px 12px;margin-bottom:6px;
                                display:flex;gap:10px;align-items:flex-start">
                      <div style="font-size:10px;color:#5A5A7A;min-width:62px;margin-top:1px;line-height:1.4">{source}</div>
                      <div style="font-size:12px;color:#9090A8;flex:1;line-height:1.6">{clean}</div>
                      <span style="background:{bg};color:{fg};font-size:10px;padding:3px 8px;
                                   border-radius:6px;border:0.5px solid {br};flex-shrink:0;font-weight:600">{label}</span>
                    </div>""",
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div style="font-size:12px;color:#3A3A5A;text-align:center;padding:16px 0">'
                    "No headlines found — check back soon.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        else:
            # Lock screen
            st.markdown(
                """
            <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:14px;
                        padding:56px 24px;text-align:center;font-family:'Hanken Grotesk',sans-serif">
              <div style="width:56px;height:56px;background:#1A1A24;border:0.5px solid #2A2A3A;
                          border-radius:14px;display:inline-flex;align-items:center;justify-content:center;
                          font-size:24px;margin-bottom:20px">🔒</div>
              <div style="font-size:16px;font-weight:800;color:#E2E2F0;margin-bottom:8px;letter-spacing:-0.3px">
                Live coverage unlocks June 11</div>
              <div style="font-size:12px;color:#5A5A7A;line-height:1.7;max-width:340px;margin:0 auto 24px">
                Trending headlines, social sentiment, and AI-driven match topics will appear here
                automatically once the World Cup kicks off.</div>
              <div style="display:inline-flex;align-items:center;gap:8px;
                          background:rgba(168,85,247,0.1);border:0.5px solid rgba(168,85,247,0.3);
                          border-radius:8px;padding:8px 20px;font-size:12px;color:#A855F7;font-weight:600;
                          margin-bottom:24px">
                 &nbsp;Opening match · June 11, 2026</div>
              <div style="display:flex;justify-content:center;gap:28px;font-size:11px;color:#5A5A7A">
                <span>Data Pipeline: <strong style="color:#4ADE80">Active</strong></span>
                <span>Sources: <strong style="color:#9090A8">1,420+ Global Feeds</strong></span>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)  # tab padding

    st.markdown("</div>", unsafe_allow_html=True)  # left panel padding


# ──────────────────────────────────────────────────────────
# RIGHT PANEL — Key Analytical Factors (always visible)
# Shows placeholder before first prediction, populates after.
# ──────────────────────────────────────────────────────────
with right_col:
    st.markdown(
        """
    <div style="padding:24px 32px 0 8px;font-family:'Hanken Grotesk',sans-serif;">""",
        unsafe_allow_html=True,
    )

    pred = st.session_state.prediction

    if st.session_state.get("show_pulse_panel", False):

        if date.today() >= date(2026, 6, 11) or TEST_NEWS_MODE:

            try:
                wc_headlines_right = fetch_wc_headlines()
            except Exception:
                wc_headlines_right = []

            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            from collections import Counter

            az2 = SentimentIntensityAnalyzer()

            pos = 0
            neu = 0
            neg = 0

            for art in wc_headlines_right:

                sentiment = az2.polarity_scores(art.get("title", ""))["compound"]

                if sentiment > 0.05:
                    pos += 1

                elif sentiment < -0.05:
                    neg += 1

                else:
                    neu += 1

            total = pos + neu + neg

            if total == 0:
                total = 1

            pos_pct = round(pos / total * 100)
            neu_pct = round(neu / total * 100)
            neg_pct = 100 - pos_pct - neu_pct

            # Determine mood by dominant bucket first; use score only to break ties
            overall = (pos - neg) / total

            if neu >= pos and neu >= neg:
                # Neutral is the majority bucket — check if sentiment strongly pulls either way
                if overall > 0.4:
                    mood = "Positive"
                    mood_color = "#4ADE80"
                    mood_icon = "📈"
                elif overall < -0.4:
                    mood = "Negative"
                    mood_color = "#EF4444"
                    mood_icon = "📉"
                else:
                    mood = "Neutral"
                    mood_color = "#A855F7"
                    mood_icon = "➖"
            elif pos >= neg:
                mood = "Positive"
                mood_color = "#4ADE80"
                mood_icon = "📈"
            else:
                mood = "Negative"
                mood_color = "#EF4444"
                mood_icon = "📉"

            WC_TEAMS = [
                "Brazil",
                "Argentina",
                "England",
                "France",
                "Germany",
                "Spain",
                "Portugal",
                "Netherlands",
                "USA",
                "Mexico",
                "Morocco",
                "Japan",
                "Croatia",
                "Belgium",
                "Denmark",
                "Switzerland",
                "South Korea",
                "Uruguay",
            ]

            team_counts = Counter(
                team
                for art in wc_headlines_right
                for team in WC_TEAMS
                if team.lower() in art.get("title", "").lower()
            )

            top_teams = [
                (team, count)
                for team, count in team_counts.most_common(5)
                if count >= 2
            ]

            if top_teams:

                team_rows = "".join([f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        align-items:center;
                        padding:6px 0;
                        border-bottom:0.5px solid #1E1E2E;
                    ">
                        <span style="
                            font-size:12px;
                            color:#E2E2F0;
                        ">
                            {team}
                        </span>

                        <span style="
                            font-size:11px;
                            color:#A855F7;
                            font-weight:700;
                        ">
                            {count}×
                        </span>
                    </div>
                    """ for team, count in top_teams])

            else:

             team_rows = """
<div style="
    font-size:11px;
    color:#8B8BA7;
    padding:8px 0;
    line-height:1.5;
">
    No dominant team discussions in current headlines
</div>
"""

            # ---------- SENTIMENT DASHBOARD CARD ----------

            # Build visual bar widths (min 2px shown so zero values still render as thin line)
            pos_bar = max(pos_pct, 0)
            neu_bar = max(neu_pct, 0)
            neg_bar = max(neg_pct, 0)

            # Mood ring colour used in the big circle
            ring_color = mood_color
            ring_bg = f"rgba({','.join(str(int(mood_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)"

            st.markdown(
            f"""
<div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:14px;
            padding:18px 16px 14px;margin-bottom:16px;font-family:'Hanken Grotesk',sans-serif">

  <!-- Header label -->
  <div style="font-size:9px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
              font-weight:600;margin-bottom:14px">Sentiment Dashboard</div>

  <!-- Mood hero row -->
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px">
    <div style="width:52px;height:52px;border-radius:50%;
                background:{ring_bg};border:2px solid {ring_color};
                display:flex;align-items:center;justify-content:center;
                font-size:22px;flex-shrink:0">{mood_icon}</div>
    <div>
      <div style="font-size:9px;color:#5A5A7A;letter-spacing:1px;text-transform:uppercase;
                  font-weight:600;margin-bottom:3px">Overall Media Mood</div>
      <div style="font-size:22px;font-weight:800;color:{mood_color};letter-spacing:-0.5px;line-height:1">{mood}</div>
      <div style="font-size:10px;color:#4A4A6A;margin-top:3px">based on {len(wc_headlines_right)} headlines</div>
    </div>
  </div>

  <!-- Sentiment stacked bar -->
  <div style="margin-bottom:14px">
    <div style="font-size:9px;color:#5A5A7A;letter-spacing:1.2px;text-transform:uppercase;
                font-weight:600;margin-bottom:8px">Sentiment Breakdown</div>
    <div style="display:flex;height:6px;border-radius:3px;overflow:hidden;gap:1px;margin-bottom:10px">
      <div style="width:{pos_bar}%;background:#4ADE80;border-radius:3px 0 0 3px;min-width:{2 if pos_bar>0 else 0}px"></div>
      <div style="width:{neu_bar}%;background:#A855F7"></div>
      <div style="width:{neg_bar}%;background:#EF4444;border-radius:0 3px 3px 0;min-width:{2 if neg_bar>0 else 0}px"></div>
    </div>
    <!-- Row items -->
    <div style="display:flex;flex-direction:column;gap:5px">
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center;gap:7px">
          <div style="width:8px;height:8px;border-radius:2px;background:#4ADE80;flex-shrink:0"></div>
          <span style="font-size:11px;color:#9090A8">Positive</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:60px;height:3px;background:#1E1E2E;border-radius:2px;overflow:hidden">
            <div style="width:{pos_bar}%;height:100%;background:#4ADE80;border-radius:2px"></div>
          </div>
          <span style="font-size:11px;font-weight:700;color:#4ADE80;width:28px;text-align:right">{pos_pct}%</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center;gap:7px">
          <div style="width:8px;height:8px;border-radius:2px;background:#A855F7;flex-shrink:0"></div>
          <span style="font-size:11px;color:#9090A8">Neutral</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:60px;height:3px;background:#1E1E2E;border-radius:2px;overflow:hidden">
            <div style="width:{neu_bar}%;height:100%;background:#A855F7;border-radius:2px"></div>
          </div>
          <span style="font-size:11px;font-weight:700;color:#A855F7;width:28px;text-align:right">{neu_pct}%</span>
        </div>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center;gap:7px">
          <div style="width:8px;height:8px;border-radius:2px;background:#EF4444;flex-shrink:0"></div>
          <span style="font-size:11px;color:#9090A8">Negative</span>
        </div>
        <div style="display:flex;align-items:center;gap:8px">
          <div style="width:60px;height:3px;background:#1E1E2E;border-radius:2px;overflow:hidden">
            <div style="width:{neg_bar}%;height:100%;background:#EF4444;border-radius:2px"></div>
          </div>
          <span style="font-size:11px;font-weight:700;color:#EF4444;width:28px;text-align:right">{neg_pct}%</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Divider -->
  <div style="height:0.5px;background:#1E1E2E;margin:14px 0"></div>

  <!-- Most mentioned teams -->
  <div style="font-size:9px;color:#5A5A7A;letter-spacing:1.2px;text-transform:uppercase;
              font-weight:600;margin-bottom:10px">Most Mentioned Teams</div>
  {team_rows}

</div>""",
    unsafe_allow_html=True,
)

        # ── KEY ANALYTICAL FACTORS (Predictor / H2H) ─────────
    elif pred and pred.get("p_a") is not None:
        st.markdown(
            """
        <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                    font-weight:600;margin-bottom:16px">Key Analytical Factors</div>""",
            unsafe_allow_html=True,
        )

        ta = pred["team_a"]
        tb = pred["team_b"]
        la = ta[:3].upper()
        lb = tb[:3].upper()
        elo_a_v = round(elo_ratings.get(ta, 1000))
        elo_b_v = round(elo_ratings.get(tb, 1000))
        atk_a = attack(ta)
        atk_b = attack(tb)
        def_a = defense(ta)
        def_b = defense(tb)
        frm_a = recent_form(ta)
        frm_b = recent_form(tb)
        wr_a = win_rate(ta)
        wr_b = win_rate(tb)
        stage_label = pred["stage"]

        def winner_pill(va, vb, la, lb, lower_is_better=False):
            if lower_is_better:
                a_better = va < vb
            else:
                a_better = va > vb
            if va == vb:
                return '<span style="background:#1E1E2E;color:#5A5A7A;font-size:10px;padding:3px 9px;border-radius:6px;font-weight:600">Equal</span>'
            winner = la if a_better else lb
            color = "#A855F7" if a_better else "#4ADE80"
            bg = "rgba(168,85,247,0.12)" if a_better else "rgba(74,222,128,0.1)"
            diff = abs(round(va - vb, 2))
            return f'<span style="background:{bg};color:{color};font-size:10px;padding:3px 9px;border-radius:6px;font-weight:700">{winner} +{diff}</span>'

        def factor_card(
            title, val_a, val_b, la, lb, lower_is_better=False, is_stage=False
        ):
            if is_stage:
                return f"""<div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:10px;
                            padding:14px;margin-bottom:8px">
                  <div style="font-size:11px;color:#5A5A7A;font-weight:500;margin-bottom:10px">{title}</div>
                  <div style="display:flex;align-items:center;gap:8px">
                    <span style="font-size:13px;color:#9090B0;font-weight:600">{val_a}</span>
                    <span style="background:#1E1E2E;color:#5A5A7A;font-size:10px;padding:3px 9px;border-radius:6px">Neutral</span>
                  </div></div>"""
            pill = winner_pill(val_a, val_b, la, lb, lower_is_better)
            return f"""<div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:10px;
                        padding:14px;margin-bottom:8px">
              <div style="font-size:11px;color:#5A5A7A;font-weight:500;margin-bottom:10px">{title}</div>
              <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span style="font-size:14px;font-weight:800;color:#A855F7">{val_a}</span>
                <span style="font-size:10px;color:#3A3A5A">vs</span>
                <span style="font-size:14px;font-weight:800;color:#4ADE80">{val_b}</span>
                {pill}
              </div></div>"""

        st.markdown(
            factor_card("Elo Rating", elo_a_v, elo_b_v, la, lb)
            + factor_card("Attack Strength", atk_a, atk_b, la, lb)
            + factor_card(
                "Defensive Record", def_a, def_b, la, lb, lower_is_better=True
            )
            + factor_card("Recent Form", frm_a, frm_b, la, lb)
            + factor_card("Win Rate (WC)", wr_a, wr_b, la, lb)
            + factor_card("Stage Pressure", stage_label, None, la, lb, is_stage=True),
            unsafe_allow_html=True,
        )

        p_a = pred["p_a"]
        p_b = pred["p_b"]
        winner = ta if p_a > p_b else tb
        form_team = la if frm_a > frm_b else lb
        insight = (
            f"{winner}'s Elo advantage (+{abs(elo_a_v - elo_b_v)}) and World Cup win rate "
            f"are the primary drivers. {form_team} holds the recent form edge. "
            f"Prediction based on 92 years of tournament data."
        )
        st.markdown(
            f"""
        <div style="background:rgba(124,58,237,0.08);border:0.5px solid rgba(168,85,247,0.2);
                    border-radius:10px;padding:14px;margin-top:4px">
          <div style="font-size:11px;color:#7C3AED;font-weight:700;margin-bottom:6px">AI Insight</div>
          <div style="font-size:11px;color:#9090A8;line-height:1.65">{insight}</div>
          <div style="margin-top:10px;display:inline-flex;align-items:center;gap:6px;
                      background:rgba(74,222,128,0.08);border:0.5px solid rgba(74,222,128,0.2);
                      border-radius:6px;padding:3px 10px;font-size:10px;color:#4ADE80;font-weight:600">
            ✦ Confidence: High
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

        # ── PLACEHOLDER (nothing predicted yet) ──────────────
    else:
        st.markdown(
            """
        <div style="font-size:10px;color:#5A5A7A;letter-spacing:1.4px;text-transform:uppercase;
                    font-weight:600;margin-bottom:16px">Key Analytical Factors</div>
        <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:12px;
                    padding:32px 20px;text-align:center;margin-bottom:12px">
          <div style="font-size:28px;margin-bottom:14px">⚽</div>
          <div style="font-size:13px;font-weight:700;color:#7A7A9A;margin-bottom:8px">No prediction yet</div>
          <div style="font-size:11px;color:#5A5A7A;line-height:1.7">
            Select two teams on the<br>Match Predictor tab and<br>
            hit <strong style="color:#A855F7">Predict Match</strong> to see<br>analytical factors here.
          </div>
        </div>
        <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:10px;
                    padding:14px;margin-bottom:8px;opacity:0.35">
          <div style="font-size:11px;color:#5A5A7A;margin-bottom:8px">Elo Rating</div>
          <div style="height:8px;background:#1E1E2E;border-radius:4px"></div>
        </div>
        <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:10px;
                    padding:14px;margin-bottom:8px;opacity:0.25">
          <div style="font-size:11px;color:#5A5A7A;margin-bottom:8px">Attack Strength</div>
          <div style="height:8px;background:#1E1E2E;border-radius:4px"></div>
        </div>
        <div style="background:#13131A;border:0.5px solid #2A2A3A;border-radius:10px;
                    padding:14px;margin-bottom:8px;opacity:0.15">
          <div style="font-size:11px;color:#5A5A7A;margin-bottom:8px">Defensive Record</div>
          <div style="height:8px;background:#1E1E2E;border-radius:4px"></div>
        </div>""",
            unsafe_allow_html=True,
        )

st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<div style="text-align:center;padding:24px 32px;border-top:0.5px solid #1E1E2E;margin-top:16px;
            font-family:'Hanken Grotesk',sans-serif">
  <div style="font-size:11px;color:#3A3A5A">
    FootballPulse AI &nbsp;·&nbsp; Built on 92 years of World Cup data &nbsp;·&nbsp;
    <div style="margin-top:10px">
    <a href="https://github.com/SHalima8/FootballPulse-AI"
       target="_blank"
       style="
            color:#8FA8FF;
            font-family:'Hanken Grotesk',sans-serif"
            text-decoration:none;
            font-size:11px;
            letter-spacing:0.5px;
            text-transform:uppercase;
            font-weight:600;">
        ↗ Explore the Codebase
    </a>
</div>
  
  <div style="font-size:11px;color:#2A2A3A;margin-top:4px">
    Predictions are probabilistic — football is beautifully unpredictable
  </div>
</div>
""",
    unsafe_allow_html=True,
)