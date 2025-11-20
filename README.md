# SORTIFY — DSA Practice Recommender

SORTIFY is a lightweight, minimalist DSA practice platform that recommends problems based on user performance. It includes a simple user system, XP and level progression, streak tracking, and a leaderboard.

---

## Features
- Automatic DSA problem recommendations  
- Local username login  
- XP, level, and streak system  
- Leaderboard with ranks  
- Simple and clean UI with animated sorting background  

---

## Tech Stack
- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python (Flask)  
- **Database:** Supabase  
- **Deployment:** Render (Gunicorn + persistent disk)  

---

## Run Locally
```bash
git clone <your-repo-url>
cd <your-folder>

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python3 app.py
