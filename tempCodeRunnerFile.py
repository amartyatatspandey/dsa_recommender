import os
import time
import math
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from supabase import create_client, Client

# ----------------------------------------------------------
# SUPABASE SETUP
# ----------------------------------------------------------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ----------------------------------------------------------
# FLASK APP
# ----------------------------------------------------------
app = Flask(__name__, template_folder="templates")
CORS(app)


# ----------------------------------------------------------
# HELPER: TOPIC EXTRACTOR
# ----------------------------------------------------------
def extract_topic(title):
    if " - " in title:
        return title.split(" - ")[0].strip()
    return title.strip()


# ----------------------------------------------------------
# SIGMOID FOR ELO PREDICTION
# ----------------------------------------------------------
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


# ----------------------------------------------------------
# ELO RECOMMENDER — SQL BASED
# ----------------------------------------------------------
class ScalableRecommender:

    # -----------------------
    # Load user from SQL
    # -----------------------
    def load_user(self, user_id):
        res = supabase.table("users").select("*").eq("user_id", user_id).execute()
        if len(res.data) == 0:
            return None
        return res.data[0]

    # -----------------------
    # Load problem
    # -----------------------
    def load_problem(self, pid):
        res = supabase.table("problems").select("*").eq("problem_id", pid).execute()
        if len(res.data) == 0:
            return None
        return res.data[0]

    # -----------------------
    # Load ALL problems
    # -----------------------
    def load_all_problems(self):
        res = supabase.table("problems").select("*").execute()
        return res.data

    # -----------------------
    # Predict probability
    # -----------------------
    def predict_prob(self, user, problem):
        skill = user["skill"]
        diff = problem["difficulty"]
        return sigmoid(skill - diff)

    # -----------------------
    # Recommend — scalable SQL version
    # -----------------------
    def recommend(self, user_id, seen_ids, last_topic):
        user = self.load_user(user_id)
        if user is None:
            return None

        problems = self.load_all_problems()

        # Build candidate list
        candidates = []
        for p in problems:

            topic = extract_topic(p["title"])
            prob = self.predict_prob(user, p)
            info = prob * (1 - prob)

            candidates.append({
                "pid": p["problem_id"],
                "prob": prob,
                "info": info,
                "topic": topic,
                "problem": p
            })

        # Sort by information gain
        candidates.sort(key=lambda x: x["info"], reverse=True)

        # Unseen first
        unseen = [c for c in candidates if c["pid"] not in seen_ids]

        if unseen:
            filtered = [c for c in unseen if c["topic"] != last_topic]
            if not filtered:
                filtered = unseen
            return filtered[0]

        # if all seen → allow repetition
        filtered = [c for c in candidates if c["topic"] != last_topic]
        if not filtered:
            filtered = candidates

        return filtered[0]

    # -----------------------
    # Update user after submission
    # -----------------------
    def update_user(self, user, problem, correct, time_taken, predicted_prob):
        # Compute error
        error = (1 if correct else 0) - predicted_prob

        # ELO learning rates
        lr_user = 0.6
        lr_item = 0.2

        # Update user skill
        user["skill"] += lr_user * error
        user["skill"] = max(-4, min(4, user["skill"]))

        # Update XP
        xp_gain = 10 if correct else 2
        if correct and time_taken:
            if time_taken < 20: xp_gain += 5
            elif time_taken < 60: xp_gain += 2

        user["xp"] += xp_gain
        user["streak"] = user["streak"] + 1 if correct else 0
        user["level"] = 1 + int(math.sqrt(user["xp"] // 50))

        badges = set(user["badges"] or [])
        if user["streak"] >= 3: badges.add("3-Streak")
        if user["streak"] >= 7: badges.add("7-Streak")
        if user["level"] >= 5: badges.add("Rising Star")
        if user["xp"] >= 100: badges.add("Committed Learner")
        user["badges"] = list(badges)

        # Write back to SQL
        supabase.table("users").update({
            "skill": user["skill"],
            "xp": user["xp"],
            "level": user["level"],
            "streak": user["streak"],
            "badges": user["badges"]
        }).eq("user_id", user["user_id"]).execute()

        # Update problem difficulty
        new_difficulty = problem["difficulty"] - lr_item * error

        supabase.table("problems").update({
            "difficulty": new_difficulty
        }).eq("problem_id", problem["problem_id"]).execute()

        return user


recommender = ScalableRecommender()


# ----------------------------------------------------------
# ROUTES
# ----------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------------------
# CREATE USER
# -----------------------------------------------
@app.route("/api/create_user", methods=["POST"])
def create_user():
    data = request.json
    username = data.get("username")

    user_id = str(int(time.time() * 1000))

    # Insert into SQL
    supabase.table("users").insert({
        "user_id": user_id,
        "username": username,
        "skill": 0,
        "xp": 0,
        "level": 1,
        "streak": 0,
        "badges": []
    }).execute()

    return jsonify({"user_id": user_id})


# -----------------------------------------------
# GET USER
# -----------------------------------------------
@app.route("/api/get_user/<user_id>")
def get_user(user_id):
    res = supabase.table("users").select("*").eq("user_id", user_id).execute()
    if len(res.data) == 0:
        return jsonify({"error": "user not found"}), 404
    return jsonify(res.data[0])


# -----------------------------------------------
# RECOMMENDATION API
# -----------------------------------------------
@app.route("/api/recommend/<user_id>")
def recommend(user_id):
    seen_raw = request.args.get("seen", "")
    seen = set([int(x) for x in seen_raw.split(",") if x.strip().isdigit()])

    last_topic = request.args.get("last_topic", None)
    if last_topic == "":
        last_topic = None

    rec = recommender.recommend(user_id, seen, last_topic)

    if rec is None:
        return jsonify({"recs": []})

    p = rec["problem"]

    return jsonify({
        "recs": [{
            "problem_id": p["problem_id"],
            "title": p["title"],
            "prompt": p["prompt"],
            "choices": p["choices"],
            "difficulty": p["difficulty"],
            "topic": rec["topic"],
            "estimated_prob_correct": round(rec["prob"], 4),
            "information": round(rec["info"], 4)
        }]
    })


# -----------------------------------------------
# SUBMIT ANSWER
# -----------------------------------------------
@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.json

    user_id = data["user_id"]
    problem_id = data["problem_id"]
    selected = data["selected_index"]
    time_taken = data.get("time_taken")

    # Load from DB
    user = recommender.load_user(user_id)
    problem = recommender.load_problem(problem_id)

    predicted_prob = recommender.predict_prob(user, problem)
    correct = (selected == problem["answer"])

    # Insert attempt
    supabase.table("attempts").insert({
        "user_id": user_id,
        "problem_id": problem_id,
        "selected": selected,
        "correct": correct,
        "predicted_prob": float(predicted_prob),
        "time_taken": time_taken
    }).execute()

    # Update ELO model
    updated_user = recommender.update_user(
        user=user,
        problem=problem,
        correct=correct,
        time_taken=time_taken,
        predicted_prob=predicted_prob
    )

    return jsonify({
        "correct": correct,
        "updated_user": updated_user
    })


# -----------------------------------------------
# LEADERBOARD
# -----------------------------------------------
@app.route("/api/leaderboard")
def leaderboard():
    res = supabase.table("users").select("*").order("xp", desc=True).limit(20).execute()
    return jsonify({"leaderboard": res.data})


# -----------------------------------------------
# GET PROBLEM BY ID
# -----------------------------------------------
@app.route("/api/problems/<int:pid>")
def get_problem(pid):
    res = supabase.table("problems").select("*").eq("problem_id", pid).execute()
    if len(res.data) == 0:
        return jsonify({"error": "not found"}), 404
    return jsonify(res.data[0])


# ----------------------------------------------------------
# START
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
