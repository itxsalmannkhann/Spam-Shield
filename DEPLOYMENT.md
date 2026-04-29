# SpamShield — Deployment Guide

Full deployment: GitHub → Render (backend) → Vercel (frontend)

---

## Project Structure

```
spam-detector/
├── backend/
│   ├── main.py            ← FastAPI app
│   ├── requirements.txt   ← Python dependencies
│   ├── render.yaml        ← Render deployment config
│   ├── MNB.pkl            ← Trained model
│   └── vectorizer.pkl     ← Trained vectorizer
│
├── frontend/
│   ├── index.html         ← Full UI (deploy to Vercel)
│   └── vercel.json        ← Vercel routing config
│
├── .gitignore
└── DEPLOYMENT.md          ← This file
```

---

## STEP 1 — Push to GitHub

### 1.1 Create a GitHub repository

1. Go to [github.com](https://github.com) and log in
2. Click the **+** icon (top right) → **New repository**
3. Name it: `spam-detector`
4. Set it to **Public**
5. Do NOT initialize with README (you already have files)
6. Click **Create repository**

### 1.2 Push your local files

Open your terminal in the `spam-detector/` folder:

```bash
git init
git add .
git commit -m "Initial commit: Spam Detector project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/spam-detector.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your actual GitHub username.

After this, your code is live on GitHub.

---

## STEP 2 — Deploy Backend on Render (Free)

Render is a cloud platform that can run your FastAPI backend for free.

### 2.1 Create a Render account

Go to [render.com](https://render.com) and sign up (use your GitHub account to make it easier).

### 2.2 Create a new Web Service

1. On the Render dashboard, click **New +** → **Web Service**
2. Click **Connect GitHub** and authorize Render to access your repo
3. Find your `spam-detector` repo and click **Connect**

### 2.3 Configure the service

Fill in these settings:

| Setting | Value |
|---------|-------|
| **Name** | `spam-detector-api` |
| **Region** | Choose closest to you |
| **Branch** | `main` |
| **Root Directory** | `backend` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | Free |

### 2.4 Deploy

Click **Create Web Service**. Render will:
1. Pull your code from GitHub
2. Install Python dependencies
3. Start the FastAPI server

Wait 2-3 minutes. Once it says **Live**, you'll get a URL like:
```
https://spam-detector-api.onrender.com
```

### 2.5 Test the backend

Open your browser and visit:
```
https://spam-detector-api.onrender.com/docs
```

You should see the FastAPI automatic documentation page. Try the `/predict` endpoint there.

> **Note:** On Render's free tier, the server "sleeps" after 15 minutes of inactivity.
> The first request after sleeping takes ~30 seconds to wake up. This is normal.

---

## STEP 3 — Connect Frontend to Backend

Before deploying the frontend, you must update the API URL in `frontend/index.html`.

### 3.1 Edit index.html

Open `frontend/index.html` and find this line (around line 430):

```javascript
const API_URL = "https://YOUR-APP-NAME.onrender.com";
```

Replace it with your actual Render URL:

```javascript
const API_URL = "https://spam-detector-api.onrender.com";
```

### 3.2 Commit the change

```bash
git add frontend/index.html
git commit -m "Set production API URL"
git push
```

---

## STEP 4 — Deploy Frontend on Vercel

### 4.1 Create a Vercel account

Go to [vercel.com](https://vercel.com) and sign up with GitHub.

### 4.2 Import your project

1. On the Vercel dashboard, click **Add New** → **Project**
2. Find your `spam-detector` GitHub repo and click **Import**

### 4.3 Configure

| Setting | Value |
|---------|-------|
| **Framework Preset** | `Other` |
| **Root Directory** | `frontend` |

Leave everything else as default.

### 4.4 Deploy

Click **Deploy**. Vercel builds in about 30 seconds.

You'll get a live URL like:
```
https://spam-detector.vercel.app
```

That's your public UI — share it with anyone!

---

## STEP 5 — Update CORS (Optional but Recommended)

Once you have your Vercel URL, update the CORS setting in `backend/main.py` to only allow your frontend:

```python
# Find this line in main.py:
allow_origins=["*"],

# Replace with your actual Vercel URL:
allow_origins=["https://spam-detector.vercel.app"],
```

Then commit and push — Render will auto-redeploy.

---

## Quick Reference — All URLs

| Service | URL |
|---------|-----|
| GitHub repo | `https://github.com/YOUR-USERNAME/spam-detector` |
| Backend API | `https://spam-detector-api.onrender.com` |
| API docs | `https://spam-detector-api.onrender.com/docs` |
| Frontend UI | `https://spam-detector.vercel.app` |

---

## How the System Works Together

```
User types message
       ↓
  Vercel (Frontend)
  index.html sends POST request
       ↓
  Render (Backend)
  FastAPI receives message
  → preprocesses text (regex + stemming)
  → loads MNB.pkl + vectorizer.pkl
  → returns JSON: { label, is_spam, spam_probability, ham_probability }
       ↓
  Vercel (Frontend)
  Displays result with animations
```

---

## Testing the API Directly

You can test your backend without the UI using curl or Postman:

**Using curl:**
```bash
curl -X POST "https://spam-detector-api.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"message": "Congratulations! You have won a FREE iPhone!"}'
```

**Expected response:**
```json
{
  "original_message": "Congratulations! You have won a FREE iPhone!",
  "cleaned_text": "congratul won free iphon",
  "label": "SPAM",
  "is_spam": true,
  "spam_probability": 99.94,
  "ham_probability": 0.06,
  "confidence": "High"
}
```

---

## Common Issues & Fixes

**Backend not waking up (first request is slow)**
This is normal on Render free tier. First request after inactivity takes 20-30 seconds. Subsequent requests are fast.

**CORS error in browser**
Make sure the API_URL in index.html matches your exact Render URL (no trailing slash).

**Model not found error on Render**
Make sure `MNB.pkl` and `vectorizer.pkl` are inside the `backend/` folder and were committed to GitHub.

**Vercel shows blank page**
Make sure the Root Directory is set to `frontend` (not the repo root) in Vercel project settings.
