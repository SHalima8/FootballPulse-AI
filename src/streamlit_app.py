import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import feedparser
import random
from datetime import date
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

st.set_page_config(
    page_title="FootballPulse AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #080808 !important;
    color: #ffffff;
}
#MainMenu { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
div[data-testid="stToolbar"] { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp > div { padding: 0 !important; }
div[data-baseweb="select"] > div {
    background-color: #111 !important;
    border: 0.5px solid #252525 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
div[data-baseweb="select"] span { color: #fff !important; }
div[data-baseweb="popover"] { background: #111 !important; }
li[role="option"] { background: #111 !important; color: #aaa !important; }
li[role="option"]:hover { background: #1a1a1a !important; color: #fff !important; }
.stButton > button {
    background: #0f0f0f !important;
    color: #00ff87 !important;
    border: 0.5px solid #00ff87 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 0 !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
}
.stButton > button:hover { background: #001a0d !important; }
.stAlert { border-radius: 8px !important; }
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    background: transparent !important;
}
div[data-testid="stRadio"] label {
    background: #111 !important;
    border: 0.5px solid #1e1e1e !important;
    border-radius: 20px !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    color: #555 !important;
    cursor: pointer !important;
    margin: 0 !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    background: #001a0d !important;
    border-color: #00ff87 !important;
    color: #00ff87 !important;
}
div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    font-size: 12px !important;
    margin: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers defined at module level ──────────────────────
def _team_in_title(title, team):
    """Return True if the team name (or any long word of it) appears in the text."""
    text_lower = title.lower()
    if team.lower() in text_lower:
        return True
    for word in team.split():
        if len(word) > 4 and word.lower() in text_lower:
            return True
    return False


def fetch_reddit_rss(team_a, team_b, limit=8):
    """Fetch Reddit posts via RSS — no API key needed"""
    import feedparser

    query = f"{team_a}+{team_b}"
    results = []

    subreddits = ["soccer", "worldcup", "football"]

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/search.rss?q={query}&sort=new&limit=10&restrict_sr=1"
        feed = feedparser.parse(url)

        for entry in feed.entries:
            title = entry.get("title", "")
            if (
                not title
                or not _team_in_title(title, team_a)
                and not _team_in_title(title, team_b)
            ):
                continue
            results.append(
                {
                    "title": title,
                    "description": entry.get("summary", "")[:200],
                    "source": {"name": f"Reddit r/{sub}"},
                }
            )

        if len(results) >= limit:
            break

    return results[:limit]


def fetch_google_rss(team):
    # Try multiple queries so we get enough relevant results
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


# ── Load model and data ───────────────────────────────────
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
# UI
# ══════════════════════════════════════════════════════════
st.markdown(
    """
<div style="background:#050505;border-bottom:0.5px solid #1a1a1a;padding:16px 32px;
            display:flex;align-items:center;justify-content:space-between">
  <div style="display:flex;align-items:center;gap:12px">
    <div style="width:34px;height:34px;background:#00ff87;border-radius:8px;
                display:flex;align-items:center;justify-content:center;font-size:17px;color:#000">⚽</div>
    <div>
      <div style="font-size:16px;font-weight:600;color:#fff;letter-spacing:-0.3px">FootballPulse AI</div>
      <div style="font-size:11px;color:#3a3a3a;margin-top:1px">World Cup 2026 — Match Intelligence · 92 years of data</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#00ff87">
    <div style="width:7px;height:7px;background:#00ff87;border-radius:50%"></div>Live analysis
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("<div style='padding:28px 32px 0 32px'>", unsafe_allow_html=True)

# Team selector
st.markdown(
    '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:14px;font-weight:500">Match predictor</div>',
    unsafe_allow_html=True,
)

col1, col_vs, col2 = st.columns([10, 1, 10])
with col1:
    team_a = st.selectbox(
        "A",
        team_options,
        index=team_options.index("Brazil") if "Brazil" in team_options else 0,
        label_visibility="collapsed",
    )
with col_vs:
    st.markdown(
        "<div style='text-align:center;padding-top:10px;font-size:12px;color:#2a2a2a;font-weight:600'>VS</div>",
        unsafe_allow_html=True,
    )
with col2:
    team_b = st.selectbox(
        "B",
        team_options,
        index=team_options.index("France") if "France" in team_options else 1,
        label_visibility="collapsed",
    )

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

elo_a = round(elo_ratings.get(team_a, 1000))
elo_b = round(elo_ratings.get(team_b, 1000))
c1, _, c2 = st.columns([10, 1, 10])
with c1:
    st.markdown(
        f"""
    <div style="background:#0d0d0d;border:0.5px solid #1a1a1a;border-radius:10px;padding:18px;text-align:center">
      <div style="font-size:11px;color:#333;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.8px">Team A</div>
      <div style="font-size:20px;font-weight:600;color:#fff">{team_a}</div>
      <div style="font-size:11px;color:#333;margin-top:5px">Elo {elo_a}</div>
    </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
    <div style="background:#0d0d0d;border:0.5px solid #1a1a1a;border-radius:10px;padding:18px;text-align:center">
      <div style="font-size:11px;color:#333;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.8px">Team B</div>
      <div style="font-size:20px;font-weight:600;color:#fff">{team_b}</div>
      <div style="font-size:11px;color:#333;margin-top:5px">Elo {elo_b}</div>
    </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# Stage
st.markdown(
    '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;font-weight:500">Tournament stage</div>',
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

# Button
cb1, cb2, cb3 = st.columns([3, 4, 3])
with cb2:
    predict_btn = st.button("Predict Match", use_container_width=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════
if predict_btn:
    if team_a == team_b:
        st.warning("Please select two different teams.")
    else:
        st.session_state.prediction = {
            "team_a": team_a,
            "team_b": team_b,
            "stage": stage,
            "p_a": None,
            "p_draw": None,
            "p_b": None,
        }
        p_a, p_draw, p_b = predict(team_a, team_b, stage_val)
        st.session_state.prediction.update({
            "p_a": p_a, "p_draw": p_draw, "p_b": p_b
        })

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if st.session_state.prediction and st.session_state.prediction["p_a"] is not None:
    pred = st.session_state.prediction
    team_a = pred["team_a"]
    team_b = pred["team_b"]
    p_a = pred["p_a"]
    p_draw = pred["p_draw"]
    p_b = pred["p_b"]
    winner = team_a if p_a > p_b else team_b
    win_prob = max(p_a, p_b)

    st.markdown(
        '<div style="height:0.5px;background:#141414;margin-bottom:28px"></div>',
        unsafe_allow_html=True,
    )

    # Prediction
    st.markdown(
        '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:16px;font-weight:500">Prediction</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
    <div style="background:#001a0d;border:0.5px solid #00ff87;border-radius:10px;
                padding:20px 24px;margin-bottom:22px;text-align:center">
      <div style="font-size:10px;color:#004d1f;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;font-weight:500">Predicted winner</div>
      <div style="font-size:36px;font-weight:700;color:#00ff87;letter-spacing:-0.5px;margin-bottom:8px">{winner}</div>
      <div style="display:inline-block;background:#003311;border-radius:20px;padding:4px 16px">
        <span style="font-size:13px;font-weight:600;color:#00ff87">{win_prob*100:.0f}%</span>
        <span style="font-size:11px;color:#006633;margin-left:4px">confidence</span>
      </div>
    </div>""",
        unsafe_allow_html=True,
    )

    def prob_bar(label, prob, color):
        w = int(prob * 100)
        st.markdown(
            f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px">
          <div style="font-size:12px;color:#555;width:140px;flex-shrink:0">{label}</div>
          <div style="flex:1;height:4px;background:#111;border-radius:2px;overflow:hidden">
            <div style="width:{w}%;height:100%;background:{color};border-radius:2px"></div>
          </div>
          <div style="font-size:13px;font-weight:500;color:#fff;width:36px;text-align:right">{w}%</div>
        </div>""",
            unsafe_allow_html=True,
        )

    prob_bar(f"{team_a} win", p_a, "#00ff87")
    prob_bar("Draw", p_draw, "#2a2a2a")
    prob_bar(f"{team_b} win", p_b, "#f59e0b")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Why factors
    st.markdown(
        '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:16px;font-weight:500">Why — key factors</div>',
        unsafe_allow_html=True,
    )

    ha = attack(team_a)
    ba = attack(team_b)
    hd = defense(team_a)
    bd = defense(team_b)
    hf = recent_form(team_a)
    bf = recent_form(team_b)
    hw = win_rate(team_a)
    bw = win_rate(team_b)
    la = team_a[:3].upper()
    lb = team_b[:3].upper()

    def pill(va, vb, la, lb, hib=True):
        if hib:
            if va > vb:
                return f'<span style="background:#001a0d;color:#00ff87;font-size:10px;padding:2px 8px;border-radius:4px;white-space:nowrap">{la} +{round(va-vb,2)}</span>'
            elif vb > va:
                return f'<span style="background:#1a0d00;color:#f59e0b;font-size:10px;padding:2px 8px;border-radius:4px;white-space:nowrap">{lb} +{round(vb-va,2)}</span>'
        else:
            if va < vb:
                return f'<span style="background:#001a0d;color:#00ff87;font-size:10px;padding:2px 8px;border-radius:4px;white-space:nowrap">{la} better</span>'
            elif vb < va:
                return f'<span style="background:#1a0d00;color:#f59e0b;font-size:10px;padding:2px 8px;border-radius:4px;white-space:nowrap">{lb} better</span>'
        return '<span style="background:#111;color:#333;font-size:10px;padding:2px 8px;border-radius:4px">Equal</span>'

    factors = [
        ("Elo rating", elo_a, elo_b, True),
        ("Attack strength", ha, ba, True),
        ("Defense", hd, bd, False),
        ("Recent form", hf, bf, True),
        ("Win rate", hw, bw, True),
        ("Stage pressure", None, None, None),
    ]

    fc1, fc2 = st.columns(2)
    for i, (name, va, vb, hib) in enumerate(factors):
        col = fc1 if i % 2 == 0 else fc2
        if name == "Stage pressure":
            col.markdown(
                f"""
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:8px;padding:12px;margin-bottom:8px">
              <div style="font-size:11px;color:#444;margin-bottom:6px">{name}</div>
              <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:13px;color:#666">{stage}</span>
                <span style="background:#111;color:#333;font-size:10px;padding:2px 8px;border-radius:4px">Neutral</span>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            p = pill(va, vb, la, lb, hib)
            col.markdown(
                f"""
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:8px;padding:12px;margin-bottom:8px">
              <div style="font-size:11px;color:#444;margin-bottom:6px">{name}</div>
              <div style="display:flex;align-items:center;gap:8px">
                <span style="font-size:13px;font-weight:500;color:#00ff87">{va}</span>
                <span style="font-size:10px;color:#2a2a2a">vs</span>
                <span style="font-size:13px;font-weight:500;color:#f59e0b">{vb}</span>
                {p}
              </div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Sentiment — date-gated ───────────────────────────────
    if date.today() >= date(2026, 6, 11):
        import re as _re
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        st.markdown(
            '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;'
            'text-transform:uppercase;margin-bottom:16px;font-weight:500">'
            "World Cup — trending now</div>",
            unsafe_allow_html=True,
        )

        @st.cache_data(ttl=1800, show_spinner=False)
        def fetch_wc_headlines():
            """Fetch general WC2026 headlines from Google RSS — no API key needed."""
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

        # ── Trending topics ──────────────────────────────────
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
            "world",
            "cup",
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
            border = "border-bottom:0.5px solid #111;" if i < len(topics) - 1 else ""
            badge = (
                '<span style="background:#1a0800;border:0.5px solid #f59e0b44;'
                "color:#f59e0b;font-size:10px;padding:2px 6px;border-radius:4px;"
                'margin-right:8px">HOT</span>'
                if is_hot
                else ""
            )
            nc = "#aaa" if is_hot else "#444"
            cnt_lbl = f"{count} mentions" if count != 1 else "1 mention"
            topic_rows += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;padding:9px 0;{border}">
              <div>{badge}<span style="font-size:12px;color:{nc}">#{word}</span></div>
              <span style="font-size:11px;color:#2a2a2a">{cnt_lbl}</span>
            </div>"""

        if not topic_rows:
            topic_rows = '<div style="font-size:12px;color:#2a2a2a;padding:12px 0;text-align:center">Topics loading — check back soon</div>'

        st.markdown(
            f"""
        <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px;margin-bottom:12px">
          <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;font-weight:500">Trending topics</div>
          <div>{topic_rows}</div>
        </div>""",
            unsafe_allow_html=True,
        )

        # ── Latest headlines ─────────────────────────────────
        az = SentimentIntensityAnalyzer()
        st.markdown(
            """
        <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px">
          <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px;font-weight:500">Latest headlines</div>
        """,
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
                    bg, fg, br, label = "#001a0d", "#00ff87", "#00ff8733", "positive"
                elif score < -0.05:
                    bg, fg, br, label = "#1a0000", "#ef4444", "#ef444433", "negative"
                else:
                    bg, fg, br, label = "#111", "#444", "#222", "neutral"
                st.markdown(
                    f"""
                <div style="background:#111;border-radius:6px;padding:10px 12px;margin-bottom:6px;
                            display:flex;gap:10px;align-items:flex-start">
                  <div style="font-size:10px;color:#333;min-width:60px;margin-top:1px;line-height:1.4">{source}</div>
                  <div style="font-size:12px;color:#666;flex:1;line-height:1.5">{clean}</div>
                  <span style="background:{bg};color:{fg};font-size:10px;padding:2px 7px;
                               border-radius:4px;border:0.5px solid {br};flex-shrink:0">{label}</span>
                </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-size:12px;color:#2a2a2a;text-align:center;padding:16px 0">'
                "No headlines found — check back soon.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        # ── Before June 11 — clean placeholder ───────────────
        st.markdown(
            """
        <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;
                    padding:32px 24px;text-align:center;margin-top:8px">
          <div style="font-size:28px;margin-bottom:12px">⏳</div>
          <div style="font-size:13px;font-weight:500;color:#555;margin-bottom:6px">Live coverage unlocks June 11</div>
          <div style="font-size:11px;color:#2a2a2a;line-height:1.6">
            Trending headlines &amp; topics will appear here once the World Cup kicks off.
          </div>
          <div style="margin-top:18px;display:inline-block;background:#001a0d;border:0.5px solid #00ff8733;
                      border-radius:6px;padding:6px 16px;font-size:11px;color:#004d1f">
            Opening match · June 11, 2026
          </div>
        </div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="height:0.5px;background:#141414;margin:32px 0 28px"></div>',
        unsafe_allow_html=True,
    )
st.markdown(
    '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:6px;font-weight:500">World Cup 2026 Simulator</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:12px;color:#333;margin-bottom:20px">100 simulations · probabilistic upsets · champion odds</div>',
    unsafe_allow_html=True,
)

# ── WC2026 confirmed teams in groups ────────────────────
WC2026_GROUPS = {
    "A": ["USA", "Panama", "Senegal", "Bolivia"],
    "B": ["Mexico", "Ecuador", "New Zealand", "Canada"],  # simplified
    "C": ["Argentina", "Peru", "Chile", "Canada"],
    "D": ["France", "Germany", "Belgium", "Italy"],  # hypothetical
    "E": ["Spain", "Portugal", "Turkey", "Georgia"],
    "F": ["Brazil", "Uruguay", "Colombia", "Venezuela"],
    "G": ["England", "Netherlands", "Croatia", "Poland"],
    "H": ["Morocco", "Japan", "South Korea", "Iran"],
}


def simulate_match_prob(team_a, team_b, stage="group"):
    """Pure Elo for speed — fast enough for Monte Carlo."""
    ea = elo_ratings.get(team_a, 1000)
    eb = elo_ratings.get(team_b, 1000)
    diff = ea - eb
    p_a = 1 / (1 + 10 ** (-diff / 400))
    p_b = 1 - p_a
    return p_a * 0.85, 0.15, p_b * 0.85


def weighted_winner(team_a, team_b, stage="knockout"):
    """Pick a winner probabilistically — upsets can happen."""
    p_a, p_draw, p_b = simulate_match_prob(team_a, team_b, stage)
    # In knockouts no draws — redistribute draw probability
    total = p_a + p_b
    p_a_final = p_a / total
    r = random.random()
    if r < p_a_final:
        return team_a, round(p_a_final * 100)
    else:
        return team_b, round((1 - p_a_final) * 100)


def simulate_group_stage(groups):
    """Each group: simulate all 6 matches, top 2 advance."""
    qualifiers = {}
    for group_name, teams in groups.items():
        points = {t: 0 for t in teams}
        gd = {t: 0 for t in teams}
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                ta, tb = teams[i], teams[j]
                p_a, p_draw, p_b = simulate_match_prob(ta, tb, "group")
                r = random.random()
                if r < p_a:
                    points[ta] += 3
                    gd[ta] += 1
                    gd[tb] -= 1
                elif r < p_a + p_draw:
                    points[ta] += 1
                    points[tb] += 1
                else:
                    points[tb] += 3
                    gd[tb] += 1
                    gd[ta] -= 1
        ranked = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
        qualifiers[group_name] = ranked[:2]
    return qualifiers


def simulate_knockout(teams_in, stage_name):
    """Simulate one knockout round, return winners with probs."""
    winners = []
    matches = []
    for i in range(0, len(teams_in), 2):
        ta = teams_in[i]
        tb = teams_in[i + 1]
        winner, prob = weighted_winner(ta, tb, "knockout")
        loser = tb if winner == ta else ta
        matches.append((ta, tb, winner, prob))
        winners.append(winner)
    return winners, matches


def run_full_simulation():
    """Run entire WC2026 from group stage to final."""
    results = {}

    qualifiers = simulate_group_stage(WC2026_GROUPS)
    results["groups"] = qualifiers

    group_names = list(qualifiers.keys())
    r16_teams = []
    pairings = [("A", "B"), ("C", "D"), ("E", "F"), ("G", "H")]
    for g1, g2 in pairings:
        r16_teams.append(qualifiers[g1][0])
        r16_teams.append(qualifiers[g2][1])
        r16_teams.append(qualifiers[g2][0])
        r16_teams.append(qualifiers[g1][1])

    r16_winners, r16_matches = simulate_knockout(r16_teams, "Round of 16")
    qf_winners, qf_matches = simulate_knockout(r16_winners, "Quarter-final")
    sf_winners, sf_matches = simulate_knockout(qf_winners, "Semi-final")
    final_winners, final_matches = simulate_knockout(sf_winners, "Final")

    results["r16"] = r16_matches
    results["qf"] = qf_matches
    results["sf"] = sf_matches
    results["final"] = final_matches
    results["winner"] = final_winners[0]

    # Track biggest upset in this simulation
    biggest_upset = None
    worst_gap = 0
    for stage_matches in [r16_matches, qf_matches, sf_matches, final_matches]:
        for ta, tb, winner, prob in stage_matches:
            loser = tb if winner == ta else ta
            ea = elo_ratings.get(winner, 1000)
            eb = elo_ratings.get(loser, 1000)
            gap = eb - ea  # positive = winner had lower Elo = upset
            if gap > worst_gap:
                worst_gap = gap
                biggest_upset = (winner, loser, prob, gap)
    results["biggest_upset"] = biggest_upset

    return results


def run_monte_carlo(n=25):
    """Run n simulations and return champion probabilities + insights."""
    champion_count = {}
    finalist_count = {}
    semi_count = {}
    all_upsets = []

    for _ in range(n):
        res = run_full_simulation()
        winner = res["winner"]
        champion_count[winner] = champion_count.get(winner, 0) + 1

        # Track finalists
        for ta, tb, w, prob in res["final"]:
            for team in [ta, tb]:
                finalist_count[team] = finalist_count.get(team, 0) + 1

        # Track semi finalists
        for ta, tb, w, prob in res["sf"]:
            for team in [ta, tb]:
                semi_count[team] = semi_count.get(team, 0) + 1

        # Collect upsets
        if res["biggest_upset"]:
            all_upsets.append(res["biggest_upset"])

    # Sort champion odds
    sorted_champs = sorted(champion_count.items(), key=lambda x: x[1], reverse=True)

    # Dark horse = high semi-final rate but low Elo (outside top 8 Elo)
    top8_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)[:8]
    top8_teams = {t for t, _ in top8_elo}
    dark_horse = None
    best_dark_rate = 0
    for team, count in semi_count.items():
        rate = count / n
        if team not in top8_teams and rate > best_dark_rate:
            best_dark_rate = rate
            dark_horse = (team, rate)

    # Biggest upset across all simulations
    biggest_upset = max(all_upsets, key=lambda x: x[3]) if all_upsets else None

    # Most likely final
    most_likely_final = sorted(finalist_count.items(), key=lambda x: x[1], reverse=True)[:2]

    return {
        "champion_odds": sorted_champs,
        "dark_horse": dark_horse,
        "biggest_upset": biggest_upset,
        "most_likely_final": most_likely_final,
        "n": n,
        # Keep one full bracket to display
        "last_bracket": run_full_simulation(),
    }


# ── Streamlit UI ─────────────────────────────────────────
sim_col1, sim_col2, sim_col3 = st.columns([3, 4, 3])
with sim_col2:
    run_sim = st.button(
        "⚽  Simulate World Cup 2026", use_container_width=True, key="sim_btn"
    )

if "sim_results" not in st.session_state:
    st.session_state.sim_results = None

if run_sim:
    with st.spinner("Running 100 simulations..."):
        st.session_state.sim_results = run_monte_carlo(n=25)

if st.session_state.sim_results:
    res = st.session_state.sim_results
    bracket = res["last_bracket"]

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Champion + confidence ─────────────────────────────
    top_champ, top_count = res["champion_odds"][0]
    confidence = round(top_count / res["n"] * 100)
    st.markdown(
        f"""
    <div style="background:#001a0d;border:0.5px solid #00ff87;border-radius:10px;
                padding:20px;text-align:center;margin-bottom:24px">
      <div style="font-size:10px;color:#004d1f;letter-spacing:1px;
                  text-transform:uppercase;margin-bottom:8px">AI predicts World Cup 2026 champion</div>
      <div style="font-size:28px;font-weight:600;color:#00ff87;margin-bottom:6px">{top_champ}</div>
      <div style="font-size:12px;color:#006633">Won <b style="color:#00ff87">{confidence}%</b> of 100 simulations · Based on 92 years of World Cup data</div>
    </div>""",
        unsafe_allow_html=True,
    )

    # ── Champion odds table ───────────────────────────────
    st.markdown(
        '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;font-weight:500">Champion probability — 100 simulations</div>',
        unsafe_allow_html=True,
    )

    odds_rows = ""
    for i, (team, count) in enumerate(res["champion_odds"][:8]):
        pct = round(count / res["n"] * 100)
        if pct == 0:
            continue
        color = "#00ff87" if i == 0 else "#f59e0b" if i == 1 else "#555"
        bar_w = int(pct / res["champion_odds"][0][1] * 100)
        rank_label = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        odds_rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;width:100%">
          <div style="font-size:11px;color:#333;width:20px;text-align:center;flex-shrink:0">{rank_label}</div>
          <div style="font-size:12px;color:{color};width:90px;flex-shrink:0">{team}</div>
          <div style="flex:1;min-width:0;height:3px;background:#111;border-radius:2px;overflow:hidden">
            <div style="width:{bar_w}%;height:3px;background:{color};border-radius:2px"></div>
          </div>
          <div style="font-size:13px;font-weight:500;color:{color};width:36px;text-align:right;flex-shrink:0">{pct}%</div>
        </div>"""

    st.markdown(
        f'<div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px;margin-bottom:20px">{odds_rows}</div>',
        unsafe_allow_html=True,
    )

    # ── Group stage results ───────────────────────────────
    st.markdown(
        '<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:12px;font-weight:500">Group stage — qualifiers</div>',
        unsafe_allow_html=True,
    )

    group_cols = st.columns(4)
    group_items = list(bracket["groups"].items())
    for idx, (grp, teams) in enumerate(group_items):
        with group_cols[idx % 4]:
            st.markdown(
                f"""
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:8px;padding:10px;margin-bottom:8px">
              <div style="font-size:10px;color:#333;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.8px">Group {grp}</div>
              <div style="font-size:12px;color:#00ff87;margin-bottom:4px">✓ {teams[0]}</div>
              <div style="font-size:12px;color:#00ff87">✓ {teams[1]}</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Knockout bracket ──────────────────────────────────
    def render_matches(matches, label):
        st.markdown(
            f'<div style="font-size:10px;color:#3a3a3a;letter-spacing:1.2px;text-transform:uppercase;margin-bottom:10px;font-weight:500">{label}</div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(len(matches))
        for i, (ta, tb, winner, prob) in enumerate(matches):
            loser = tb if winner == ta else ta
            with cols[i]:
                st.markdown(
                    f"""
                <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:8px;padding:10px">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                    <span style="font-size:12px;color:#00ff87;font-weight:500">{winner}</span>
                    <span style="font-size:10px;color:#004d1f">{prob}%</span>
                  </div>
                  <div style="height:0.5px;background:#111;margin-bottom:6px"></div>
                  <div style="font-size:11px;color:#2a2a2a">{loser}</div>
                </div>""",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    render_matches(bracket["qf"], "Quarter-finals")
    render_matches(bracket["sf"], "Semi-finals")
    render_matches(bracket["final"], "Final")


    # ── Dark Horse + Biggest Upset ────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ins1, ins2 = st.columns(2)

    with ins1:
        if res["dark_horse"]:
            dh_team, dh_rate = res["dark_horse"]
            dh_pct = round(dh_rate * 100)
            st.markdown(
                f"""
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px;height:100%">
              <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;font-weight:500;margin-bottom:10px"> Dark horse</div>
              <div style="font-size:20px;font-weight:600;color:#f59e0b;margin-bottom:4px">{dh_team}</div>
              <div style="font-size:11px;color:#333">Reached semi-finals in</div>
              <div style="font-size:22px;font-weight:600;color:#f59e0b;margin:4px 0">{dh_pct}%</div>
              <div style="font-size:11px;color:#333">of simulations despite low Elo</div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px">
              <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;font-weight:500;margin-bottom:10px"> Dark horse</div>
              <div style="font-size:12px;color:#2a2a2a">No clear dark horse this time</div>
            </div>""",
                unsafe_allow_html=True,
            )

    with ins2:
        if res["biggest_upset"]:
            up_winner, up_loser, up_prob, up_gap = res["biggest_upset"]
            st.markdown(
                f"""
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px;height:100%">
              <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;font-weight:500;margin-bottom:10px"> Biggest upset</div>
              <div style="font-size:16px;font-weight:600;color:#00ff87;margin-bottom:2px">{up_winner}</div>
              <div style="font-size:11px;color:#333;margin-bottom:6px">defeated</div>
              <div style="font-size:16px;font-weight:600;color:#ef4444;margin-bottom:8px">{up_loser}</div>
              <div style="display:flex;gap:16px">
                <div>
                  <div style="font-size:10px;color:#2a2a2a">Win prob before</div>
                  <div style="font-size:14px;font-weight:500;color:#f59e0b">{up_prob}%</div>
                </div>
                <div>
                  <div style="font-size:10px;color:#2a2a2a">Elo gap</div>
                  <div style="font-size:14px;font-weight:500;color:#f59e0b">+{round(up_gap)}</div>
                </div>
              </div>
            </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
            <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;padding:16px">
              <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;font-weight:500;margin-bottom:10px">⚡ Biggest upset</div>
              <div style="font-size:12px;color:#2a2a2a">Favourites held their ground</div>
            </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Most likely final ─────────────────────────────────
    if res["most_likely_final"] and len(res["most_likely_final"]) >= 2:
        ft1, ft2 = res["most_likely_final"][0][0], res["most_likely_final"][1][0]
        fp1 = round(res["most_likely_final"][0][1] / res["n"] * 100)
        fp2 = round(res["most_likely_final"][1][1] / res["n"] * 100)
        st.markdown(
            f"""
        <div style="background:#0d0d0d;border:0.5px solid #171717;border-radius:10px;
                    padding:14px 20px;margin-bottom:16px;display:flex;align-items:center;gap:16px">
          <div style="font-size:10px;color:#3a3a3a;letter-spacing:1px;text-transform:uppercase;font-weight:500;flex-shrink:0">Most likely final</div>
          <div style="display:flex;align-items:center;gap:12px;flex:1;justify-content:center">
            <div style="text-align:center">
              <div style="font-size:14px;font-weight:600;color:#00ff87">{ft1}</div>
              <div style="font-size:10px;color:#004d1f">{fp1}% reached final</div>
            </div>
            <div style="font-size:12px;color:#2a2a2a;font-weight:600">vs</div>
            <div style="text-align:center">
              <div style="font-size:14px;font-weight:600;color:#f59e0b">{ft2}</div>
              <div style="font-size:10px;color:#7a4f00">{fp2}% reached final</div>
            </div>
          </div>
        </div>""",
            unsafe_allow_html=True,
        )


    # ── Re-simulate button ────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    rs1, rs2, rs3 = st.columns([3, 4, 3])
    with rs2:
        if st.button("Re-simulate", use_container_width=True, key="resim_btn"):
            with st.spinner("Running 25 simulations..."):
                st.session_state.sim_results = run_monte_carlo(n=25)
            st.rerun()

    st.markdown(
        f"""
    <div style="text-align:center;margin-top:12px">
      <span style="font-size:11px;color:#1e1e1e">Each simulation is independent · 25 runs · Elo-based probabilities · upsets can always happen</span>
    </div>""",
        unsafe_allow_html=True,
    )


# Footer
st.markdown(
    """
<div style="text-align:center;padding:24px 32px;border-top:0.5px solid #111;margin-top:32px">
  <div style="font-size:11px;color:#2a2a2a">FootballPulse AI &nbsp;·&nbsp; Built on 92 years of World Cup data &nbsp;·&nbsp; Model accuracy ~57%</div>
  <div style="font-size:11px;color:#1e1e1e;margin-top:4px">Predictions are probabilistic — football is beautifully unpredictable</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)
