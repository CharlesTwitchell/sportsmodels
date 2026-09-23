#!/usr/bin/env python3
"""
Append newly-completed games' model predictions to data/results_log.json.

This is the one and only place predictions get logged. Once a row is
written for a (model_id, game_id) pair, this script never touches it
again — even if the model file is later retrained with new weights. That's
what makes the log a real track record instead of a live re-simulation:
`dashboard.html` just reads this file, it never recomputes history.

Usage:
    python3 scripts/log_results.py [--model data/model_nfl_logreg.json]
                                    [--log data/results_log.json]

No third-party dependencies — stdlib only, so this runs on a bare
GitHub Actions runner with no pip install step.
"""
import argparse
import csv
import io
import json
import math
import sys
import urllib.request
from datetime import datetime, timezone

SCHEDULE_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}


def normalize_team(code):
    return TEAM_ALIASES.get(code, code)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def load_model(path):
    with open(path) as f:
        model = json.load(f)
    model["_teams_by_abbr"] = {t["team"]: t for t in model["teams"]}
    return model


def predict_home_win_prob(model, home_abbr, away_abbr):
    teams = model["_teams_by_abbr"]
    home = teams.get(normalize_team(home_abbr))
    away = teams.get(normalize_team(away_abbr))
    if home is None or away is None:
        return None
    logit = model["log_reg_intercept"]
    for i, feat in enumerate(model["features"]):
        diff = home[feat] - away[feat]
        standardized = (diff - model["scaler_mean"][i]) / model["scaler_scale"][i]
        logit += model["log_reg_coef"][i] * standardized
    return sigmoid(logit)


def fetch_schedule():
    with urllib.request.urlopen(SCHEDULE_URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def is_played(row):
    return row.get("home_score", "") not in ("", None) and row.get("away_score", "") not in ("", None)


def load_log(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def already_logged_keys(log_rows):
    return {(r["model_id"], r["game_id"]) for r in log_rows}


def build_new_rows(model, schedule, target_season, existing_keys, now_iso):
    new_rows = []
    for row in schedule:
        if row.get("game_type") != "REG":
            continue
        try:
            season = int(row["season"])
        except (TypeError, ValueError):
            continue
        if season != target_season:
            continue
        if not is_played(row):
            continue

        game_id = row["game_id"]
        key = (model["model_id"], game_id)
        if key in existing_keys:
            continue

        home_abbr, away_abbr = row["home_team"], row["away_team"]
        prob = predict_home_win_prob(model, home_abbr, away_abbr)
        if prob is None:
            print(f"  skip {game_id}: no team stats for {home_abbr} or {away_abbr}", file=sys.stderr)
            continue

        home_score, away_score = float(row["home_score"]), float(row["away_score"])
        predicted_home = prob >= 0.5
        predicted_winner = home_abbr if predicted_home else away_abbr

        if home_score == away_score:
            actual_winner = "TIE"
            correct = None
        else:
            actual_home_win = home_score > away_score
            actual_winner = home_abbr if actual_home_win else away_abbr
            correct = predicted_home == actual_home_win

        new_rows.append({
            "model_id": model["model_id"],
            "sport": model["sport"],
            "model_version": model["model_version"],
            "season": season,
            "week": int(row["week"]),
            "game_id": game_id,
            "gameday": row["gameday"],
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_score": home_score,
            "away_score": away_score,
            "home_win_prob": round(prob, 4),
            "predicted_winner": predicted_winner,
            "actual_winner": actual_winner,
            "correct": correct,
            "logged_at": now_iso,
        })
    return new_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/model_nfl_logreg.json")
    parser.add_argument("--log", default="data/results_log.json")
    args = parser.parse_args()

    model = load_model(args.model)
    target_season = model["metrics"]["stats_season"] + 1
    print(f"Model {model['model_id']} ({model['sport']}) targets season {target_season}")

    log_rows = load_log(args.log)
    existing_keys = already_logged_keys(log_rows)
    print(f"Existing log: {len(log_rows)} rows")

    schedule = fetch_schedule()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    new_rows = build_new_rows(model, schedule, target_season, existing_keys, now_iso)

    if not new_rows:
        print("No new completed games to log.")
        return

    log_rows.extend(new_rows)
    log_rows.sort(key=lambda r: (r["season"], r["week"], r["gameday"], r["game_id"], r["model_id"]))

    with open(args.log, "w") as f:
        json.dump(log_rows, f, indent=2)
        f.write("\n")

    print(f"Logged {len(new_rows)} new game(s):")
    for r in new_rows:
        mark = "?" if r["correct"] is None else ("hit" if r["correct"] else "miss")
        print(f"  season {r['season']} wk {r['week']}: {r['away_team']} @ {r['home_team']} "
              f"-> picked {r['predicted_winner']} ({r['home_win_prob']:.0%} home) [{mark}]")


if __name__ == "__main__":
    main()
