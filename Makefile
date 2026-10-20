# One-word entry points for every pipeline stage.  `make help` lists them.
PY        ?= python
IMAGE     ?= heart-risk-api:1.0.0
BASE_URL  ?= http://localhost:8000
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: help setup data eda train test lint format notebooks serve mlflow-ui \
        docker-build docker-run smoke compose-up compose-down k8s-deploy k8s-status \
        k8s-delete monitoring-deploy traffic ci all clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

setup:           ## install all dependencies (+ the package, editable)
	$(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt

data:            ## download raw data (UCI, checksum-verified fallback) and clean it
	$(PY) -m heartrisk.data all

eda:             ## EDA figures -> reports/figures/eda, summary -> reports/eda_summary.json
	$(PY) -m heartrisk.eda

train:           ## tune 3 model families, log to MLflow, register champion, quality gate
	$(PY) -m heartrisk.train

test:            ## unit tests with coverage
	$(PY) -m pytest --cov --cov-report=term-missing

lint:            ## ruff lint + format check (what CI runs)
	ruff check src tests scripts && ruff format --check src tests scripts

format:          ## auto-format
	ruff format src tests scripts && ruff check --fix src tests scripts

notebooks:       ## execute the notebooks in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

serve:           ## run the API locally on :8000 (reload on change)
	uvicorn heartrisk.api.main:app --reload --port 8000

mlflow-ui:       ## MLflow UI on :5000 over the local tracking store
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

docker-build:    ## build the serving image
	docker build -t $(IMAGE) .

docker-run:      ## run the image on :8000
	docker run -d --rm --name heart-risk-api -p 8000:8000 $(IMAGE)

smoke:           ## smoke test a running API (BASE_URL=...)
	bash scripts/smoke_test.sh $(BASE_URL)

compose-up:      ## API + Prometheus + Grafana
	docker compose up -d --build

compose-down:
	docker compose down -v

k8s-deploy:      ## apply manifests to the current kube-context and wait
	kubectl apply -k deploy/k8s && kubectl -n heart-risk rollout status deploy/heart-risk-api --timeout=180s

k8s-status:
	kubectl -n heart-risk get deploy,pods,svc,ingress,hpa,pdb -o wide

k8s-delete:
	kubectl delete -k deploy/k8s --ignore-not-found

monitoring-deploy: ## Prometheus + Grafana inside the cluster
	kubectl apply -k deploy/monitoring

traffic:         ## synthetic traffic for dashboards (BASE_URL=...)
	$(PY) scripts/generate_traffic.py --url $(BASE_URL) --minutes 3

ci:              ## run the CI stages locally: lint -> test -> train -> docker -> smoke
	bash scripts/ci_local.sh

all: data eda train test   ## full ML pipeline from a clean checkout

clean:
	rm -rf mlflow.db mlartifacts .pytest_cache .coverage htmlcov reports/junit.xml reports/coverage.xml
