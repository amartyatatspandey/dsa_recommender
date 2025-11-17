'''
import os
import json
import math
import time
from flask import Flask, jsonify, request  
from flask_cors import CORS

# ========== PATHS ==========
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
PROBLEMS_FILE = os.path.join(DATA_DIR, "problems.json")
INTERACTIONS_FILE = os.path.join(DATA_DIR, "interactions.json")


# ========== LOAD/SAVE HELPERS ==========
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ========== TOPIC EXTRACTOR ==========
def extract_topic(title):
    """
    Extract topic from titles like:
    'Binary Search - First True in Boolean Array'
    """
    if " - " in title:
        return title.split(" - ")[0].strip()
    return title.strip()


# ========== SIGMOID ==========
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


# ========== MODEL ==========
class SimpleEloRecommender:
    def __init__(self, users, problems):
        self.users = users
        self.problems = {p["id"]: p for p in problems}

    def predict_prob(self, user_id, problem_id):
        uid = str(user_id)
        user = self.users.get(uid, {})
        skill = user.get("skill", 0.0)
        diff = float(self.problems[problem_id].get("difficulty", 0.0))
        return sigmoid(skill - diff)

    def recommend(self, user_id, exclude_ids, last_topic):
        """
        NEW RULES:
        --------------------------------------------------------
        1. Never repeat a problem (IDs in exclude_ids)
        2. Never show the same topic consecutively (last_topic)
        3. Avoid topic repetition inside the recommendation list
        --------------------------------------------------------
        """

        uid = str(user_id)
        if uid not in self.users:
            return []

        candidates = []
        for pid, p in self.problems.items():

            # RULE 1: avoid same problem
            if pid in exclude_ids:
                continue

            topic = extract_topic(p["title"])
            prob = self.predict_prob(user_id, pid)
            info = prob * (1 - prob)

            candidates.append({
                "pid": pid,
                "prob": prob,
                "info": info,
                "topic": topic,
            })

        # Sort by highest information gain
        candidates.sort(key=lambda x: x["info"], reverse=True)

        # ============================
        # RULE 2: Avoid same last_topic
        # ============================
        filtered = [c for c in candidates if c["topic"] != last_topic]

        # If filtering becomes EMPTY → fallback to all candidates (still avoiding repeats)
        if not filtered:
            filtered = candidates

        # Pick best one
        best = filtered[0] if filtered else None

        return best
# ========== UPDATE MODEL (skill update) ==========
    def update(self, user_id, problem_id, correct, time_taken=None):
        uid = str(user_id)
        user = self.users.get(uid)

        if user is None:
            user = {"username": uid, "skill": 0.0, "xp": 0, "level": 1, "streak": 0, "badges": []}

        prev_prob = self.predict_prob(user_id, problem_id)
        error = (1 if correct else 0) - prev_prob

        # learning rates
        lr_user = 0.6
        lr_item = 0.2

        # update user skill
        user["skill"] += lr_user * error
        user["skill"] = max(-4, min(4, user["skill"]))

        # update problem difficulty
        prob = self.problems[problem_id]
        prob["difficulty"] -= lr_item * error

        # XP system
        xp_gain = 10 if correct else 2
        if correct and time_taken:
            if time_taken < 20: xp_gain += 5
            elif time_taken < 60: xp_gain += 2
        user["xp"] = user.get("xp", 0) + xp_gain

        # streak
        user["streak"] = user.get("streak", 0) + 1 if correct else 0

        # level
        user["level"] = 1 + int(math.sqrt(user["xp"] // 50))

        # badges
        badges = set(user.get("badges", []))
        if user["streak"] >= 3: badges.add("3-Streak")
        if user["streak"] >= 7: badges.add("7-Streak")
        if user["level"] >= 5: badges.add("Rising Star")
        if user["xp"] >= 100: badges.add("Committed Learner")
        user["badges"] = list(badges)

        self.users[uid] = user
        return prev_prob, user


# ========== INIT DATA ==========
users = load_json(USERS_FILE, {})
problems = load_json(PROBLEMS_FILE, [])
interactions = load_json(INTERACTIONS_FILE, [])

model = SimpleEloRecommender(users, problems)


# ========== FLASK APP ==========
app = Flask(__name__)
CORS(app)
from flask import send_from_directory

@app.route("/")
def index():
    return send_from_directory(".", "index.html")



# ===============================
# CREATE USER
# ===============================
@app.route("/api/create_user", methods=["POST"])
def create_user():
    data = request.json or {}
    username = data.get("username", f"user{len(users)+1}")
    uid = str(int(time.time()*1000) % 1000000)

    users[uid] = {
        "username": username,
        "skill": 0.0,
        "xp": 0,
        "level": 1,
        "streak": 0,
        "badges": []
    }

    save_json(USERS_FILE, users)
    return jsonify({"user_id": uid, "user": users[uid]})


# ===============================
# GET USER
# ===============================
@app.route("/api/get_user/<user_id>", methods=["GET"])
def get_user(user_id):
    u = users.get(str(user_id))
    if not u:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"user_id": user_id, "user": u})


# ===============================
# RECOMMENDATION ENDPOINT
# NOW ENFORCES NO REPEAT + NO SAME TOPIC BACK-TO-BACK
# ===============================
@app.route("/api/recommend/<user_id>", methods=["GET"])
def recommend(user_id):
    # seen list
    seen_ids = request.args.get("seen", "")
    seen_set = set()
    if seen_ids:
        for x in seen_ids.split(","):
            x = x.strip()
            if x.isdigit():
                seen_set.add(int(x))

    # last topic
    last_topic = request.args.get("last_topic", None)
    if last_topic == "":
        last_topic = None

    rec = model.recommend(
        user_id=user_id,
        exclude_ids=seen_set,
        last_topic=last_topic
    )

    if not rec:
        return jsonify({"recs": []})

    pid = rec["pid"]
    p = model.problems[pid]

    return jsonify({
        "recs": [{
            "problem_id": pid,
            "title": p["title"],
            "prompt": p["prompt"],
            "choices": p["choices"],
            "difficulty": p["difficulty"],
            "topic": rec["topic"],
            "estimated_prob_correct": round(rec["prob"], 4),
            "information": round(rec["info"], 4)
        }]
    })


# ===============================
# SUBMIT ANSWER
# ===============================
@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.json or {}

    user_id = str(data["user_id"])
    pid = int(data["problem_id"])
    selected = int(data["selected_index"])
    time_taken = data.get("time_taken")

    correct = (selected == model.problems[pid]["answer"])
    prev_prob, user = model.update(user_id, pid, correct, time_taken)

    interactions.append({
        "timestamp": int(time.time()),
        "user_id": user_id,
        "problem_id": pid,
        "selected": selected,
        "correct": correct,
        "predicted_prob": round(prev_prob, 3),
        "time_taken": time_taken
    })

    save_json(USERS_FILE, users)
    save_json(PROBLEMS_FILE, list(model.problems.values()))
    save_json(INTERACTIONS_FILE, interactions)

    return jsonify({
        "correct": correct,
        "updated_user": user
    })


# ===============================
# LEADERBOARD
# ===============================
@app.route("/api/leaderboard", methods=["GET"])
def leaderboard():
    arr = []
    for uid, u in users.items():
        arr.append({
            "user_id": uid,
            "username": u["username"],
            "xp": u["xp"],
            "level": u["level"]
        })

    arr.sort(key=lambda x: (-x["xp"], x["username"]))
    return jsonify({"leaderboard": arr[:20]})


# ===============================
# GET PROBLEM BY ID
# ===============================
@app.route("/api/problems/<int:pid>", methods=["GET"])
def get_problem(pid):
    p = model.problems.get(pid)
    if not p:
        return jsonify({"error": "not found"}), 404
    return jsonify(p)
# ===============================
# AUTO-GENERATED FRONTEND (index.html)
# Updated to send last_topic + avoid repeated topics
# ===============================

# -----------------------
# Frontend handling (user-controlled)
# -----------------------
# NOTE: app will serve index.html from the same directory as app.py.
# Do NOT auto-generate or overwrite index.html — allow the developer
# to edit and replace the frontend freely.
index_path = os.path.join(APP_DIR, "index.html")
if not os.path.exists(index_path):
    print("⚠️ index.html not found in project folder. Please add your custom 'index.html' in the project root.")


# ===============================
# START SERVER
# ===============================
if __name__ == "__main__":
    save_json(USERS_FILE, users)
    save_json(PROBLEMS_FILE, problems)
    save_json(INTERACTIONS_FILE, interactions)

    print("🔥 AI DSA Recommender running at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
'''
import os
import json
import math
import time
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# ========== PATHS ==========
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
PROBLEMS_FILE = os.path.join(DATA_DIR, "problems.json")
INTERACTIONS_FILE = os.path.join(DATA_DIR, "interactions.json")


# ========== LOAD/SAVE HELPERS ==========
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# ========== TOPIC EXTRACTOR ==========
def extract_topic(title):
    if " - " in title:
        return title.split(" - ")[0].strip()
    return title.strip()


# ========== SIGMOID ==========
def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


# ========== MODEL ==========
class SimpleEloRecommender:
    def __init__(self, users, problems):
        self.users = users
        self.problems = {p["id"]: p for p in problems}

    def predict_prob(self, user_id, problem_id):
        uid = str(user_id)
        user = self.users.get(uid, {})
        skill = user.get("skill", 0.0)
        diff = float(self.problems[problem_id].get("difficulty", 0.0))
        return sigmoid(skill - diff)

    def recommend(self, user_id, exclude_ids, last_topic):

        uid = str(user_id)
        if uid not in self.users:
            return []

        candidates = []
        for pid, p in self.problems.items():
            if pid in exclude_ids:
                continue

            topic = extract_topic(p["title"])
            prob = self.predict_prob(user_id, pid)
            info = prob * (1 - prob)

            candidates.append({
                "pid": pid,
                "prob": prob,
                "info": info,
                "topic": topic,
            })

        candidates.sort(key=lambda x: x["info"], reverse=True)

        filtered = [c for c in candidates if c["topic"] != last_topic]

        if not filtered:
            filtered = candidates

        best = filtered[0] if filtered else None
        return best

    # ========== UPDATE MODEL ==========
    def update(self, user_id, problem_id, correct, time_taken=None):
        uid = str(user_id)
        user = self.users.get(uid)

        if user is None:
            user = {
                "username": uid,
                "skill": 0.0,
                "xp": 0,
                "level": 1,
                "streak": 0,
                "badges": []
            }

        prev_prob = self.predict_prob(user_id, problem_id)
        error = (1 if correct else 0) - prev_prob

        lr_user = 0.6
        lr_item = 0.2

        user["skill"] += lr_user * error
        user["skill"] = max(-4, min(4, user["skill"]))

        prob = self.problems[problem_id]
        prob["difficulty"] -= lr_item * error

        # XP
        xp_gain = 10 if correct else 2
        if correct and time_taken:
            if time_taken < 20: xp_gain += 5
            elif time_taken < 60: xp_gain += 2
        user["xp"] = user.get("xp", 0) + xp_gain

        user["streak"] = user.get("streak", 0) + 1 if correct else 0

        user["level"] = 1 + int(math.sqrt(user["xp"] // 50))

        badges = set(user.get("badges", []))
        if user["streak"] >= 3: badges.add("3-Streak")
        if user["streak"] >= 7: badges.add("7-Streak")
        if user["level"] >= 5: badges.add("Rising Star")
        if user["xp"] >= 100: badges.add("Committed Learner")
        user["badges"] = list(badges)

        self.users[uid] = user
        return prev_prob, user


# ========== INIT DATA ==========
users = load_json(USERS_FILE, {})
problems = load_json(PROBLEMS_FILE, [])
interactions = load_json(INTERACTIONS_FILE, [])

model = SimpleEloRecommender(users, problems)


# ========== FLASK APP ==========
app = Flask(__name__, template_folder="templates")
CORS(app)


# ===============================
# SERVE FRONTEND
# ===============================
@app.route("/")
def home():
    return render_template("index.html")


# ===============================
# CREATE USER
# ===============================
@app.route("/api/create_user", methods=["POST"])
def create_user():
    data = request.json or {}
    username = data.get("username", f"user{len(users)+1}")

    uid = str(int(time.time()*1000) % 1000000)

    users[uid] = {
        "username": username,
        "skill": 0.0,
        "xp": 0,
        "level": 1,
        "streak": 0,
        "badges": []
    }

    save_json(USERS_FILE, users)
    return jsonify({"user_id": uid, "user": users[uid]})


# ===============================
# GET USER
# ===============================
@app.route("/api/get_user/<user_id>", methods=["GET"])
def get_user(user_id):
    u = users.get(str(user_id))
    if not u:
        return jsonify({"error": "user not found"}), 404
    return jsonify({"user_id": user_id, "user": u})


# ===============================
# RECOMMENDATION ENDPOINT
# ===============================
@app.route("/api/recommend/<user_id>", methods=["GET"])
def recommend(user_id):

    seen_ids = request.args.get("seen", "")
    seen_set = set()
    if seen_ids:
        for x in seen_ids.split(","):
            x = x.strip()
            if x.isdigit():
                seen_set.add(int(x))

    last_topic = request.args.get("last_topic", None)
    if last_topic == "":
        last_topic = None

    rec = model.recommend(
        user_id=user_id,
        exclude_ids=seen_set,
        last_topic=last_topic
    )

    if not rec:
        return jsonify({"recs": []})

    pid = rec["pid"]
    p = model.problems[pid]

    return jsonify({
        "recs": [{
            "problem_id": pid,
            "title": p["title"],
            "prompt": p["prompt"],
            "choices": p["choices"],
            "difficulty": p["difficulty"],
            "topic": rec["topic"],
            "estimated_prob_correct": round(rec["prob"], 4),
            "information": round(rec["info"], 4)
        }]
    })


# ===============================
# SUBMIT ANSWER
# ===============================
@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.json or {}

    user_id = str(data["user_id"])
    pid = int(data["problem_id"])
    selected = int(data["selected_index"])
    time_taken = data.get("time_taken")

    correct = (selected == model.problems[pid]["answer"])
    prev_prob, user = model.update(user_id, pid, correct, time_taken)

    interactions.append({
        "timestamp": int(time.time()),
        "user_id": user_id,
        "problem_id": pid,
        "selected": selected,
        "correct": correct,
        "predicted_prob": round(prev_prob, 3),
        "time_taken": time_taken
    })

    save_json(USERS_FILE, users)
    save_json(PROBLEMS_FILE, list(model.problems.values()))
    save_json(INTERACTIONS_FILE, interactions)

    return jsonify({
        "correct": correct,
        "updated_user": user
    })


# ===============================
# LEADERBOARD
# ===============================
@app.route("/api/leaderboard", methods=["GET"])
def leaderboard():
    arr = []
    for uid, u in users.items():
        arr.append({
            "user_id": uid,
            "username": u["username"],
            "xp": u["xp"],
            "level": u["level"]
        })

    arr.sort(key=lambda x: (-x["xp"], x["username"]))
    return jsonify({"leaderboard": arr[:20]})


# ===============================
# GET PROBLEM BY ID
# ===============================
@app.route("/api/problems/<int:pid>", methods=["GET"])
def get_problem(pid):
    p = model.problems.get(pid)
    if not p:
        return jsonify({"error": "not found"}), 404
    return jsonify(p)


# ===============================
# START SERVER (LOCAL ONLY)
# ===============================
if __name__ == "__main__":
    save_json(USERS_FILE, users)
    save_json(PROBLEMS_FILE, problems)
    save_json(INTERACTIONS_FILE, interactions)

    print("🔥 AI DSA Recommender running at http://127.0.0.1:5000")
    app.run(debug=False, port=5000)
