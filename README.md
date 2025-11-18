# 🚀 Mini MLOps Demo: FastAPI + Docker + MLflow

This repository is a **minimal end-to-end MLOps workflow**, including model training, experiment tracking, API serving, and Docker containerization.

## 📌 Features

* Train a logistic regression classifier using scikit-learn
* Track metrics + parameters using **MLflow**
* Serve the trained model with **FastAPI**
* Build a production-ready inference container using **Docker**
* Provide a clean and reproducible project structure

---

# 🧱 Tech Stack

* **Python** (scikit-learn, NumPy, joblib)
* **FastAPI** + **Uvicorn** (model API)
* **MLflow** (experiment tracking & artifacts)
* **Docker** (containerized deployment)

---

# 📂 Project Structure

```
.
├── app.py                  # FastAPI inference service
├── train.py                # Simple training script
├── model.pkl               # Exported model (after training)
├── Dockerfile
├── requirements.txt
└── mlruns/                 # MLflow experiment logs
```

---

# 🧠 1. Train the Model

### Option A: Simple training

```bash
python train.py
```

This generates some `model.pkl`s in the archive folder.

### View MLflow UI

```bash
mlflow ui
```

Open in browser:

👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

Here you can compare runs, metrics, parameters, and artifacts.
Then select the best performance model, copy it as model.pkl to the root folder.
---

# ⚡ 2. Run the FastAPI Inference Server

Make sure `model.pkl` exists.

```bash
uvicorn app:app --reload
```

Service available at:

👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)
Swagger docs:

👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

# 🧪 3. Call the API

### ✔️ Option A: Python client

```python
import requests

data = {"features": [0.5, -1.2, 0.3, 0.9, -0.4]}
res = requests.post("http://127.0.0.1:8000/predict", json=data)

print(res.json())
```

### ✔️ Option B: cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d "{\"features\": [0.5, -1.2, 0.3, 0.9, -0.4]}"
```

### ✔️ Option C: Swagger UI

1. Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
2. Click **POST /predict**
3. Input JSON
4. Execute

---

# 🐳 4. Build & Run with Docker

### Build image

```bash
docker build -t mlops-demo .
```

### Run container

```bash
docker run -p 8000:8000 mlops-demo
```

The API is now live at:

👉 [http://127.0.0.1:8000](http://127.0.0.1:8000)
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

# 🎯 Summary

This mini-project demonstrates the core workflow of modern MLOps:

* training
* logging
* reproducibility
* API serving
* containerization
