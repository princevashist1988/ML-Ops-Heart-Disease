# Runbook - what is done, and the four steps left for you

Everything in the assignment has been built and executed: data, EDA, training with MLflow, packaging, tests,
Docker image, Kubernetes deployment (k3s), Prometheus monitoring, report and a pipeline video. Four things can
only be done from **your** accounts and machine. Budget about 60-90 minutes.

| # | Step | Why only you can do it | Time |
|---|---|---|---|
| 1 | Fill the placeholders in the report | your BITS ID and repo URL | 2 min |
| 2 | Push to GitHub, let Actions run, screenshot it | needs your GitHub account | 20 min |
| 3 | Re-run Docker + Kubernetes + Grafana on your laptop (recommended) | screenshots from your own environment | 30-40 min |
| 4 | Record a short narrated video (optional but strongly advised) | evaluators expect your voice/understanding | 15 min |

---

## 1. Report placeholders (highlighted yellow)
`docs/Assignment1_Report_HeartRisk_MLOps.docx` - page 1: **BITS ID**, **GitHub repository URL**; section 7:
the GitHub Actions screenshot box. Export to PDF from Word after editing (File -> Save As -> PDF).

## 2. GitHub + Actions
```bash
cd heart-risk-mlops
git init -b main && git add . && git commit -m "Heart disease risk MLOps pipeline"
gh repo create heart-risk-mlops --private --source . --push      # or create on github.com and `git push`
```
* Repo -> **Actions** tab -> workflow **CI/CD** starts automatically (5 jobs, ~10 min).
* Screenshot 1: the run graph with all five jobs green -> `screenshots/github_actions_run.png`.
* Screenshot 2: the **Summary** page (test table + model comparison table) -> `screenshots/github_actions_summary.png`.
* Screenshot 3 (optional): the Artifacts list at the bottom of the run.
* Paste screenshot 1 into the report placeholder. If a job fails, open its log - every step prints a clear error.
* Note: the `docker` job pushes to GHCR only on `main`; that needs no extra secrets (uses `GITHUB_TOKEN`).

## 3. Your own Docker / Kubernetes / Grafana screenshots
The `screenshots/` folder already holds evidence from the build environment (k3s, Prometheus, MLflow). Re-running
on your laptop takes ~30 min and makes the evidence unmistakably yours.

**Docker Desktop (Windows/Mac) - simplest path**
```bash
python -m venv .venv && .venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m heartrisk.train                                # optional: regenerates mlflow.db for the MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db        # screenshot runs + model registry at :5000
docker build -t heart-risk-api:1.0.0 .
docker compose up -d                                     # API :8000, Prometheus :9090, Grafana :3000 (admin/admin)
python scripts/generate_traffic.py --url http://localhost:8000 --minutes 3
```
Screenshots: `http://localhost:8000/docs` (try `/predict`), Grafana dashboard **Heart Disease Risk API**,
Prometheus **Status -> Targets**.

**Kubernetes on Docker Desktop**: Settings -> Kubernetes -> Enable. Then
```bash
docker compose down
kubectl apply -k deploy/k8s
kubectl -n heart-risk rollout status deploy/heart-risk-api
kubectl -n heart-risk get pods,svc,ingress,hpa            # screenshot
bash scripts/smoke_test.sh http://localhost:8000          # Git Bash/WSL on Windows; screenshot
kubectl apply -k deploy/monitoring                         # Prometheus :30090, Grafana :30300 (NodePorts)
```
Ingress on Docker Desktop needs an ingress controller:
`kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml`,
then add `127.0.0.1 heart-risk.local` to your hosts file.

**Minikube** instead: see README section 6.

## 4. Narrated video (3-5 min)
`docs/pipeline_walkthrough.mp4` (2 min 39 s) is a ready recording of the real pipeline runs. A short screen
recording in your own voice is safer for the academic-integrity review. Suggested script:

1. (20 s) Problem, dataset, architecture diagram (`docs/architecture.png`).
2. (40 s) `make data` and the EDA notebook: the 85%-of-max-heart-rate finding.
3. (60 s) `make train`; MLflow UI: parent/nested runs, compare view, registry `@champion`;
   why ROC-AUC on repeated CV, why the 0.31 threshold, why logistic regression.
4. (40 s) GitHub Actions run: five jobs, test summary, artefacts; quality gate.
5. (40 s) `docker build`/`docker run`, `/docs` -> execute `/predict`.
6. (40 s) `kubectl get pods,svc`, smoke test, Grafana dashboard with traffic running.
7. (20 s) Limitations: small 1988 cohort, external-site recall drop, next steps.

Windows: Win+G (Xbox Game Bar) or OBS Studio; macOS: Cmd+Shift+5.

## Questions an evaluator is likely to ask (know these cold)
* Why is the hold-out AUC (0.96) higher than CV (0.90)? -> 61 patients; CV is the reliable estimate.
* Why threshold 0.31 and not 0.5? -> F2 on out-of-fold training predictions; zero missed cases on hold-out.
* Why keep missing values instead of dropping 6 rows? -> imputation inside the pipeline, same at inference.
* What did the engineered features do? -> nothing measurable (ablation in the grid); kept off.
* What happens if the model file is missing in a pod? -> `/ready` returns 503, pod never receives traffic.
* How do you detect drift? -> live input means vs training means (`HeartRiskInputDrift` alert).
