# CI-CD Pipeline for ENSO Prediction Streaming Application 

This repository contains the configuration files and scripts necessary to set up a Continuous Integration and Continuous Deployment (CI-CD) pipeline for the ENSO Prediction Streaming Application. The pipeline automates the process of building, testing, and deploying the application to ensure rapid and reliable updates.


## Overview

The ENSO Prediction Streaming Application is designed to provide real-time predictions of the El Niño-Southern Oscillation (ENSO) phenomenon. This CI-CD pipeline ensures that any changes made to the application code are automatically tested and deployed, minimizing downtime and ensuring that users always have access to the latest features and improvements.

## Code Repository

The ENSO Prediction Streaming Application code is hosted in this repository.
- enso_streaming/: Contains the source code for the ENSO Prediction Streaming Application.
- enso_streaming.app.py: Main application file for the ENSO Prediction Streaming Application.
- requirements.txt: Lists the dependencies required to run the application.
- stream_infer.py: Script for streaming inference through Streamlit UI interface.

## How to run the application locally:

1.  Create and activate a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
2.  Install the required dependencies:
    ``` 
    pip install -r requirements.txt
    ```
3.  Run the backend application:
    ``` 
    mkdir  out
    ```
     

    ```
    python -m enso_streaming.main `
  --model .\data\linear_lag.joblib `
  --sst_path .\data\sst.mon.mean.trefadj.anom.1880to2018.nc `
  --enso_path .\data\nino34.long.anom.data.txt `
  --start 2007-01-01 `
  --end 2017-12-31 `
  --lead 1 `
  --max_lag 15 `
  --interval 10 `
  --out_csv .\out\live_predictions.csv `
  --show_features
    ```
4.  In a new terminal, activate the virtual environment and run the Streamlit UI:
    ```
    streamlit run stream_infer.py -- --csv_path .\out\live_predictions.csv
    ```
# Github Application Packaging

This guide takes you from **installing Git** → using it inside **VS Code** → **auth/login to GitHub** → **push/pull** → **branches/merge** → **clone in a new window**.

Assumptions (edit if different):
- You use **Windows + PowerShell**
- Your project folder is:
  `C:\Users\Alberto\Desktop\IA_Corso_Lab\Application_Packaging`
- Your GitHub repo is:
  `https://github.com/Al-Moccardi/prova`

---

## 0) Install Git on Windows (one-time)

### 0.1 Download & install
1. Download **Git for Windows** from the official website.
2. Install it (default options are usually fine).
3. During install, make sure Git is added to PATH (often the default).

or 
Open **VS Code** → *Terminal → New Terminal* (PowerShell) and run:
1 . Open **PowerShell** (Start → PowerShell).
2. Install Git via **Chocolatey** (if you have it):
    ```powershell 
    choco install git -y
    ```
    or via **Winget**:
    ```powershell
    winget install --id Git.Git -e --source winget
    ```

    or via **Scoop**:
    ```powershell
    scoop install git
    ```   


### 0.2 Verify Git works

```powershell
git --version
```

If you get a version (e.g., `git version 2.x`), you’re good.

> If `git` is “not recognized”, restart VS Code (or Windows), and ensure Git is installed.
---

## 1) Make Git work nicely with VS Code (recommended)

### 1.1 VS Code Git UI
VS Code has a built-in Git interface:
- Left sidebar → **Source Control** (branch icon)
- It will detect your repo automatically after `git init` or `git clone`.

### 1.2 Optional: set your identity (one-time)
```powershell
git config --global user.name "Alberto Moccardi"
git config --global user.email "alberto.moccardi@unina.it"
```

Confirm:
```powershell
git config --global --get user.name
git config --global --get user.email
```

---

## 2) GitHub authentication (“login”) — choose ONE method

Git itself does not have a “login command” like a website.  
Authentication happens when you **push/pull** to GitHub.

### Option A (recommended): GitHub CLI (`gh`) login
1. Install GitHub CLI (`gh`) for Windows.
2. In VS Code terminal:

```powershell
gh --version
gh auth login
```

Follow prompts:
- GitHub.com
- HTTPS
- Login via browser

Then Git operations will work smoothly.

---

## 3) Decide your starting scenario (IMPORTANT)

### Scenario 1 — You have a local folder and want to connect it to a NEW empty GitHub repo
➡️ Use **git init** (Section 4)

### Scenario 2 — You already have a GitHub repo and want a fresh copy locally
➡️ Use **git clone** (Section 9)

> Don’t do both for the same folder. Usually it’s **either init OR clone**.

---

## 4) Connect your local folder to GitHub (Scenario 1: init → connect → push)

### 4.1 Open the folder in VS Code
- VS Code → **File → Open Folder…**
- Choose:
  `C:\Users\Alberto\Desktop\IA_Corso_Lab\Application_Packaging`

Open terminal:
- **Terminal → New Terminal** (PowerShell)

### 4.2 Go to the folder (confirm path)
```powershell
cd "C:\Users\Alberto\Desktop\IA_Corso_Lab\Application_Packaging"
```

### 4.3 Initialize Git (creates a `.git` folder)
```powershell
git init
```

Check status:
```powershell
git status
```

### 4.4 Create a README (recommended)
Overwrite/create:
```powershell
echo "# prova" > README.md
```

Append instead:
```powershell
echo "# prova" >> README.md
```

### 4.5 Stage files (choose one)
Stage only README:
```powershell
git add README.md
```

Stage everything:
```powershell
git add .
```

### 4.6 Commit (save a snapshot)
```powershell
git commit -m "first commit"
```

### 4.7 Set your main branch name
```powershell
git branch -M main
```

### 4.8 Add the remote `origin` and verify
```powershell
git remote add origin https://github.com/Al-Moccardi/prova.git
git remote -v
```

### 4.9 Push to GitHub (first push)
```powershell
git push -u origin main
```

After this, your local repo is connected and GitHub will show your files.

---

## 5) Pulling (VERY important)

### What is `pull`?
`git pull` = **download** updates from GitHub (remote) + merge them into your current branch.

### When to use it
- Before you start working (to avoid conflicts)
- Before you merge branches
- When someone else pushed changes

### Command
```powershell
git pull
```

If you want to pull from a specific branch:
```powershell
git pull origin main
```

---

## 6) Everyday workflow (edit → add → commit → push)

Typical cycle:

```powershell
git status
git add .
git commit -m "Describe your change"
git push
```

Check history:
```powershell
git log --oneline --graph --decorate --all
```

---

## 7) Branching (create, switch, merge)

### 7.1 Create a new branch and switch to it
```powershell
git checkout -b <branch_name>
```

Modern alternative:
```powershell
git switch -c <branch_name>
```

### 7.2 Switch between branches
```powershell
git checkout <branch_name>
```

Or:
```powershell
git switch <branch_name>
```

### 7.3 List branches
```powershell
git branch
git branch -a
```

### 7.4 Push a new branch to GitHub
After committing on your branch:

```powershell
git push -u origin <branch_name>
```

---

## 8) Merge branches (clean flow + conflict notes)

### 8.1 Merge a feature branch into `main`
```powershell
git checkout main
git pull
git merge <branch_name>
git push
```

### 8.2 If you get merge conflicts
1. Git tells you which files have conflicts
2. Open them and fix conflict markers
3. Then:

```powershell
git add .
git commit
git push
```
### 8.3 Delete a branch after merging (optional)

```powershell   
   git branch -d <branch_name>
```
---

## 9) Cloning (Scenario 2: clone → open in VS Code → check origin)

Cloning is used when you want a **fresh copy** of an existing GitHub repo on your machine.

### 9.1 Open a NEW window (PowerShell or VS Code)
- Open **PowerShell** (Start → PowerShell), OR
- Open a **new VS Code window** and terminal

### 9.2 Choose where to place the clone (example: Desktop)
```powershell
cd "$HOME\Desktop"
```

### 9.3 Clone
```powershell
git clone https://github.com/Al-Moccardi/prova.git
```

### 9.4 Enter the folder
```powershell
cd .\prova
```

### 9.5 Check the remote origin
```powershell
git remote -v
git remote show origin
```

### 9.6 Open in VS Code
From inside the cloned folder:

```powershell
code .
```

---

## 10) Remote management (origin)

### Check remotes
```powershell
git remote -v
```

### Change remote URL (recommended fix)
```powershell
git remote set-url origin https://github.com/Al-Moccardi/prova.git
git remote -v
```

### Remove remote
```powershell
git remote remove origin
```

---

## 12) Quick reference (most-used commands)

```powershell
git status
git add .
git commit -m "message"
git push
git pull
git checkout -b feature-x
git checkout main
git merge feature-x
git log --oneline --graph --decorate --all
```

# Docker Containerization

The application is containerized using Docker to ensure consistency across different environments. The Dockerfile in the repository defines the steps to build the Docker image for the ENSO Prediction Streaming Application.

## How to build and run the Applications with Docker container:


> **Mental model:**  
> **Image** = packaged app (blueprint).  
> **Container** = running instance of an image.

---

### 0) Quick checks

```bash
docker --version
docker version
docker info
docker compose version
```

---

### 1) Build an image

> Run this **from the folder that contains the `Dockerfile`**.

```powershell
cd C:\Users\Alberto\Desktop\IA_Course
docker build -t enso-backend .
```

---

## 2) Run a container (with bind mounts)

### PowerShell (Windows)

```powershell
docker run -d --restart unless-stopped `
  -v ${PWD}\data:/app/data `
  -v ${PWD}\out:/app/out `
  --name enso-backend `
  enso-backend
```

### Bash (macOS/Linux/Git Bash)

```bash
docker run -d --restart unless-stopped \
  -v "$PWD/data:/app/data" \
  -v "$PWD/out:/app/out" \
  --name enso-backend \
  enso-backend
```

**What this does**
- `-d` runs in the background.
- `--restart unless-stopped` auto-restarts after reboot/crash (until you stop it).
- `-v ...:/app/data` mounts local `data` into the container.
- `-v ...:/app/out` mounts local `out` so outputs persist on host.
- `--name enso-backend` sets a stable name (useful for logs/exec).

---

## 3) Health / debug commands

```bash
docker ps
docker ps -a
docker logs enso-backend --tail 200
docker logs -f enso-backend
docker stats enso-backend
docker top enso-backend
```

### Inspect container state

```bash
docker inspect -f '{{.State.Status}}' enso-backend
docker inspect -f '{{.RestartCount}}' enso-backend
```

### Healthcheck status (only if your image defines `HEALTHCHECK`)

```bash
docker inspect -f '{{.State.Health.Status}}' enso-backend
```

---

## 4) Push to Docker Hub (optional)

```bash
docker login
docker tag enso-backend:latest aiberto/enso_backend_final:latest
docker push aiberto/enso_backend_final:latest
```

---

## 5) Pull and run from Docker Hub (optional)

### Pull an existing image

```bash
docker pull aiberto/enso-backend:1.0
```

### Run it (mounting data read-only is safer)

```powershell
docker run -d --restart unless-stopped `
  --name enso-backend `
  -v ${PWD}\data:/app/data:ro `
  -v ${PWD}\out:/app/out `
  aiberto/enso-backend:1.0
```


## 6) Docker Compose workflow (optional)

### Lifecycle + rebuild

```bash
docker compose down
docker compose build --no-cache
docker compose up -d --build
docker compose ps
docker compose logs -f streamer
```

### Validate mounts + outputs (inside the container)

> Replace `enso-streamer` with your actual container name if different.

```bash
# confirm the container sees the mounted files
docker exec -it enso-streamer ls -lh /app/data
docker exec -it enso-streamer ls -lh /app/out

# check the CSV is being written
docker exec -it enso-streamer tail -n 5 /app/out/live_predictions.csv
```

#   p r o v a  
 