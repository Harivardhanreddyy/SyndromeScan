# SyndromeScan

SyndromeScan is a Flask web application with:
- user registration/login (SQLite),
- image upload UI,
- optional ML inference pipeline (`vnl_net`) when ML dependencies are installed.

## Current status

✅ The repository is now host-ready in **web mode** (Flask + SQLite) without external MySQL.  
✅ If ML packages/artifacts are missing, the app still starts and shows a model warning instead of crashing.  
ℹ️ To enable real predictions, install optional ML dependencies.

---

## Run locally

### 1) Web mode (lightweight, recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

### 2) Web + ML inference mode (optional)

```bash
pip install -r requirements-ml.txt
```

If the ML stack and model files load successfully, `/algorithm` can run inference.

---

## Host / Deploy

### Option A: Procfile platforms (Render/Railway/Heroku-style)

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT app:app`

### Option B: Docker

```bash
docker build -t syndrome-scan .
docker run -p 5000:5000 syndrome-scan
```



### Option C: Render Blueprint

This repo includes `render.yaml`. To deploy:
1. Push this repo to GitHub.
2. In Render, choose **New + > Blueprint** and select the repo.
3. Render will auto-create the web service from `render.yaml`.

After deploy, your public URL will look like:
`https://syndrome-scan.onrender.com`

---

## Notes

- Database file is auto-created at `syndrome.db`.
- Uploaded images are saved to `static/uploads/`.
- For production security, replace plain-text password storage with hashed passwords before public launch.
