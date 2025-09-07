from flask import Flask, jsonify, request, render_template
from recommender import RecommendationEngine
import os

app = Flask(__name__)

# Path to the trained model
MODEL_PATH = os.environ.get("MODEL_PATH", "models/recommender.joblib")

if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Run: python build_index.py --csv data/your_file.csv")

# Load the recommendation engine
engine = RecommendationEngine().load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html", numeric_cols=engine.numeric_cols)


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])
    titles = engine.df_meta["title"].astype(str).fillna("")
    mask = titles.str.lower().str.contains(q.lower())
    results = []
    for i in titles[mask].head(10).index:
        row = engine.df_meta.iloc[int(i)]
        results.append({
            "title": row.get("title", ""),
            "artist": row.get("artist", ""),
            "popularity": row.get("popularity", None)
        })
    if not results:
        from difflib import get_close_matches
        matches = get_close_matches(q, titles.tolist(), n=5, cutoff=0.6)
        for m in matches:
            j = titles[titles == m].index[0]
            row = engine.df_meta.iloc[int(j)]
            results.append({
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "popularity": row.get("popularity", None)
            })
    return jsonify(results)


@app.get("/api/recommend/by-track")
def api_reco_by_track():
    track = request.args.get("track", "").strip()
    k = int(request.args.get("k", 10))
    if not track:
        return jsonify({"error": "missing 'track'"}), 400
    recs = engine.recommend_by_track(track, k=k)
    return jsonify(recs)


@app.get("/api/recommend/by-mood")
def api_reco_by_mood():
    k = int(request.args.get("k", 10))
    mood = {}
    for c in engine.numeric_cols:
        if c in request.args:
            try:
                mood[c] = float(request.args.get(c))
            except:
                pass
    # 👇 Debug print to check slider values sent from frontend
    print("Mood request:", mood)
    recs = engine.recommend_by_mood(k=k, **mood)
    return jsonify(recs)


if __name__ == "__main__":
    app.run(debug=True)
