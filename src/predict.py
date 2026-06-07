import pandas as pd
import numpy as np
import joblib

# ── Load saved model and feature list ─────────────────────
model    = joblib.load('../models/model.pkl')
features = joblib.load('../models/feature_columns.pkl')

# ── Load clean data ───────────────────────────────────────
df = pd.read_csv('../data/processed/clean_matches.csv')
df = df.sort_values('Year').reset_index(drop=True)

# ── Rebuild Elo from full match history ───────────────────
BASE_ELO = 1000
K_FACTOR = 20
elo_ratings = {}

def get_elo(team):
    return elo_ratings.get(team, BASE_ELO)

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(winner, loser, draw=False):
    elo_w  = get_elo(winner)
    elo_l  = get_elo(loser)
    exp_w  = expected_score(elo_w, elo_l)
    actual = 0.5 if draw else 1.0
    elo_ratings[winner] = elo_w + K_FACTOR * (actual - exp_w)
    elo_ratings[loser]  = elo_l + K_FACTOR * ((1 - actual) - (1 - exp_w))

for _, row in df.iterrows():
    ht = row['Home Team Name']
    at = row['Away Team Name']
    if row['Winner'] == ht:
        update_elo(ht, at)
    elif row['Winner'] == at:
        update_elo(at, ht)
    else:
        update_elo(ht, at, draw=True)

# ── Helper functions (full history, no leakage needed for prediction) ──
def get_attack(team):
    home = df[df['Home Team Name'] == team]['Home Team Goals']
    away = df[df['Away Team Name'] == team]['Away Team Goals']
    all_goals = pd.concat([home, away])
    return all_goals.mean() if len(all_goals) >= 3 else 1.0

def get_defense(team):
    home = df[df['Home Team Name'] == team]['Away Team Goals']
    away = df[df['Away Team Name'] == team]['Home Team Goals']
    all_conceded = pd.concat([home, away])
    return all_conceded.mean() if len(all_conceded) >= 3 else 1.0

def get_win_rate(team):
    home  = df[df['Home Team Name'] == team]
    away  = df[df['Away Team Name'] == team]
    total = len(home) + len(away)
    if total == 0:
        return 0.5
    wins = len(df[df['Winner'] == team])
    return wins / total

def get_recent_form(team, last_n=5):
    matches = df[
        (df['Home Team Name'] == team) |
        (df['Away Team Name'] == team)
    ].tail(last_n)
    if len(matches) == 0:
        return 0.5
    weights = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    weights = weights[-len(matches):]
    weights = weights / weights.sum()
    scores  = []
    for _, m in matches.iterrows():
        if m['Winner'] == team:
            scores.append(1.0)
        elif m['Winner'] == 'Draw':
            scores.append(0.5)
        else:
            scores.append(0.0)
    return float(np.dot(scores, weights))

# ── Stage mapping ─────────────────────────────────────────
stage_mapping = {
    'group': 1, 'round of 16': 2,
    'quarter': 3, 'semi': 4,
    'third': 5, 'final': 6
}

# ── Build feature dict for one team pair direction ────────
def get_features(home, away, stage_val):
    h_elo = get_elo(home)
    a_elo = get_elo(away)
    h_att = get_attack(home)
    a_att = get_attack(away)
    h_def = get_defense(home)
    a_def = get_defense(away)
    h_gd  = h_att - h_def
    a_gd  = a_att - a_def
    h_form = get_recent_form(home)
    a_form = get_recent_form(away)
    h_wr   = get_win_rate(home)
    a_wr   = get_win_rate(away)

    return {
        'home_elo':          h_elo,
        'away_elo':          a_elo,
        'elo_diff':          h_elo - a_elo,
        'elo_ratio':         h_elo / max(a_elo, 1),
        'home_attack':       h_att,
        'away_attack':       a_att,
        'home_defense':      h_def,
        'away_defense':      a_def,
        'home_goal_diff':    h_gd,
        'away_goal_diff':    a_gd,
        'attack_diff':       h_att - a_att,
        'defense_diff':      a_def - h_def,
        'goal_diff_diff':    h_gd  - a_gd,
        'home_recent_form':  h_form,
        'away_recent_form':  a_form,
        'home_win_rate':     h_wr,
        'away_win_rate':     a_wr,
        'form_diff':         h_form - a_form,
        'stage_encoded':     stage_val,
    }

# ── Neutral prediction — average both directions ──────────
def predict_match(team_a, team_b, stage='group'):
    stage_val = stage_mapping.get(stage.lower(), 1)

    feat_ab = pd.DataFrame([get_features(team_a, team_b, stage_val)])
    feat_ba = pd.DataFrame([get_features(team_b, team_a, stage_val)])

    # Keep only columns the model was trained on
    feat_ab = feat_ab.reindex(columns=features, fill_value=0)
    feat_ba = feat_ba.reindex(columns=features, fill_value=0)

    proba_ab = model.predict_proba(feat_ab)[0]
    proba_ba = model.predict_proba(feat_ba)[0]

    classes = list(model.classes_)

    # AB direction
    draw_ab  = proba_ab[classes.index(0)]
    a_win_ab = proba_ab[classes.index(1)]
    b_win_ab = proba_ab[classes.index(2)]

    # BA direction — roles flipped
    draw_ba  = proba_ba[classes.index(0)]
    b_win_ba = proba_ba[classes.index(1)]
    a_win_ba = proba_ba[classes.index(2)]

    # Average and normalize
    prob_draw  = (draw_ab  + draw_ba)  / 2
    prob_a_win = (a_win_ab + a_win_ba) / 2
    prob_b_win = (b_win_ab + b_win_ba) / 2
    total      = prob_draw + prob_a_win + prob_b_win

    return {
        'Draw':          round(prob_draw  / total, 4),
        f'{team_a} Win': round(prob_a_win / total, 4),
        f'{team_b} Win': round(prob_b_win / total, 4),
    }

# ── Test predictions ──────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*45)
    print("FootballPulse-AI — Match Predictor")
    print("="*45)

    test_matches = [
        ("Brazil",    "Argentina",  "semi"),
        ("France",    "Germany",    "final"),
        ("Spain",     "England",    "quarter"),
        ("Brazil",    "France",     "group"),
        ("Brazil",    "San Marino", "group"),
        ("France",    "Qatar",      "group"),
        ("Argentina", "Panama",     "group"),
        ("Germany",   "New Zealand","group"),
    ]

    for team_a, team_b, stage in test_matches:
        result = predict_match(team_a, team_b, stage)
        print(f"\n{team_a} vs {team_b} ({stage.title()})")
        print("-" * 38)
        for outcome, prob in sorted(
            result.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(prob * 30)
            print(f"  {outcome:<22} {prob:.1%}  {bar}")