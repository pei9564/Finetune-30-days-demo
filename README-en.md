# 🧠 Finetune Platform — End-to-End LoRA Training & Experiment Management

> A production-grade LoRA fine-tuning and experiment governance platform
> built with **FastAPI**, **Celery**, **Redis**, **MLflow**, and **Kubernetes**.
>
> Supports the full lifecycle from **data validation → training → experiment tracking → model governance → deployment → monitoring**.

---

## ✨ Key Features

* 🚀 **Multi-Hardware Support** — Compatible with CPU, NVIDIA CUDA, and Apple MPS (M3 chips)
* 📊 **Data Management** — Validation, version tracking, and distribution analysis
* 🎯 **Experiment Tracking** — MLflow integration with automatic logging of parameters, metrics, and artifacts
* 📦 **Model Registry & Recommendation** — Auto-generated Model Cards with semantic search and recommendation APIs
* 🧾 **Model Governance** — Integrated MLflow Model Registry with lifecycle stages: *Staging / Production / Archived*
* ☸️ **Kubernetes + Helm Deployment** — Modular Helm charts with layered configuration (`values.yaml`, `values.prod.yaml`)
* 🧰 **CI/CD Automation** — GitHub Actions + Docker + Helm Dry-Run validation pipeline
* 📈 **Observability & Monitoring** — Prometheus Exporter and Grafana Dashboard for system visibility
* 🌐 **Web Interface** — Task submission, live progress tracking, and experiment browsing
* 🔄 **Asynchronous Task Queue** — Celery + Redis for distributed job processing
* 🔐 **Security & RBAC** — JWT-based authentication and role-based access control
* 🧾 **Audit Logging** — Centralized operation tracking for all API events
* 🧪 **Comprehensive Testing** — Unit and integration tests with error handling coverage
* 🏗️ **Modular Architecture** — Clear separation of concerns, easy to maintain and extend

---

## 🔄 System Interaction Flow

```
sequenceDiagram
    participant U as User
    participant UI as Web UI
    participant API as FastAPI
    participant C as Celery Worker
    participant T as Training Script
    participant M as MLflow
    participant R as Redis
    participant P as Prometheus

    U->>UI: Submit training parameters
    UI->>API: POST /train
    API->>C: Dispatch Celery job
    C->>R: Push task to queue
    API-->>UI: Return task_id

    loop Task polling
        UI->>API: GET /task/{task_id}
        API->>R: Query task status
        API-->>UI: Update progress
    end

    C->>T: Execute LoRA training
    T->>M: Log parameters / metrics / artifacts
    T->>Registry: Register model (ModelCard + Stage)
    API->>P: Export metrics (latency, queue length, success count)
    P-->>Grafana: Display dashboard
```

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph Training["Training Pipeline"]
        Train[train_lora_v2.py] --> MLflow[MLflow Tracking]
        MLflow --> Registry[MLflow Registry]
        Registry --> ModelCard[ModelCard JSON]
        Train --> Results[(results/)]
        Results --> Config[config.yaml]
        Results --> Model[final_model/]
    end

    subgraph Monitoring["Monitoring Stack"]
        Exporter[Prometheus Exporter] --> P[Prometheus Server]
        P --> G[Grafana Dashboard]
    end

    subgraph CI/CD["CI/CD Automation"]
        GH[GitHub Actions] --> Build[Docker Build]
        Build --> HelmDryRun[Helm Dry-run]
        Build --> Push[DockerHub Push (tag=day-*)]
    end
```

---

## 📦 Model Registry & Recommendation

Each completed training automatically generates a **Model Card (JSON)** stored in
`data/model_registry/`, containing:

* `base_model`, `language`, `task`, `description`, `metrics`, `tags`
* Optional `embedding` vector for semantic similarity search

### 🔍 Core APIs

| Endpoint             | Method | Description                                                                |
| -------------------- | ------ | -------------------------------------------------------------------------- |
| `/models/search`     | GET    | Search models by base model, language, task, or tags                       |
| `/models/recommend`  | POST   | Recommend models based on embedding similarity                             |
| `/models/transition` | POST   | Manage MLflow Registry stage transitions (Staging → Production → Archived) |

> Future work: **Natural-language model recommendation** using text-to-embedding queries.

---

## 🧾 Experiment Tracking & Model Governance

### **MLflow Tracking**

* Automatically logs `params`, `metrics`, and `artifacts`
* Visualize and compare runs directly via MLflow UI
* API endpoint: `/experiments/mlflow/{run_id}` for programmatic retrieval

### **MLflow Model Registry**

* Each model version mapped by unique `run_id`
* Lifecycle stages: `Staging`, `Production`, `Archived`
* Automatically archives older production versions
* ModelCard and Registry states remain synchronized

---

## ☸️ Helm Deployment

**Chart Structure**

```
charts/finetune-platform/
├── Chart.yaml
├── values.yaml
├── values.prod.yaml
└── templates/
    ├── api-deployment.yaml
    ├── worker-deployment.yaml
    ├── redis-statefulset.yaml
    ├── ui-deployment.yaml
    ├── secret.yaml
    ├── service.yaml
    └── _helpers.tpl
```

**Deployment Examples**

```bash
# Development
helm install finetune charts/finetune-platform -f values.yaml

# Production
helm upgrade finetune charts/finetune-platform -f values.yaml -f values.prod.yaml
```

---

## 🔄 CI/CD Pipeline (GitHub Actions)

| Branch / Tag    | Tasks Executed               | Description                   |
| --------------- | ---------------------------- | ----------------------------- |
| Any branch / PR | Lint + Test                  | Quality assurance             |
| `main` branch   | Lint + Test + Helm Dry-Run   | Validate deployment templates |
| `tag = day-*`   | Build + Push + Deploy (echo) | Simulated release workflow    |

### Highlights

* ✅ Lint + Test — Ensure code quality and full unit test coverage
* 🧱 Helm Dry-Run — Verify deployment manifests without cluster changes
* 📦 Tag Release — Auto-build Docker image and push to registry

---

## 📊 Observability — Prometheus & Grafana

The platform includes a **Prometheus Exporter** exposing `/metrics`,
scraped periodically by Grafana for system-wide dashboards.

### **Metrics Overview**

| Metric                                     | Description                                |
| ------------------------------------------ | ------------------------------------------ |
| `task_success_total`, `task_failure_total` | Cumulative task success and failure counts |
| `task_queue_length`                        | Number of pending tasks in the queue       |
| `task_duration_seconds`                    | Histogram of task execution times          |
| `system_cpu_percent`                       | API / Worker CPU usage (%)                 |
| `system_memory_usage_gigabytes`            | Memory consumption (GB)                    |

### **Grafana Dashboards**

| Chart                        | Query                                                                       | Purpose                      |
| ---------------------------- | --------------------------------------------------------------------------- | ---------------------------- |
| Task Success / Failure Count | `increase(task_success_total[5m])`                                          | Track job outcomes over time |
| Queue Length                 | `task_queue_length`                                                         | Identify system congestion   |
| Average Task Duration        | `rate(task_duration_seconds_sum[5m])/rate(task_duration_seconds_count[5m])` | Measure task latency         |
| CPU Usage                    | `max(system_cpu_percent)`                                                   | Monitor load levels          |
| Memory Usage                 | `max(system_memory_usage_gigabytes)`                                        | Monitor resource utilization |

---

## ⚙️ Configuration & Deployment Notes

* Environment variables (`.env`) configure Redis, API, and UI ports
* Recommended production setup: Helm + GitHub Actions CI/CD
* Extend monitoring stack with `values.monitoring.yaml`
* Ensure proper volume and port configuration for MLflow, Registry, and Exporter
* Always use **HTTPS** for production deployments
