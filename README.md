# 🚁 AI-Driven Drone Design, Simulation & Swarm Reliability Platform
> **HVN Labs — Hackathon Prototype**

A prototype AI platform that assists drone companies in three key areas:
- 🧪 **Early-stage drone design optimisation** using physics-informed machine learning
- 🤖 **Swarm failure prediction** based on pre-flight diagnostics
- 💰 **Cost analysis** demonstrating the financial benefit of simulation-first development

---

## 📁 Project Structure

```
AI-DRONE-DESIGN-SIMULATOR/
│
├── Frontend/                   # UI (HTML/CSS)
│   ├── MLModel.html            # Flight time & cost predictor UI
│   ├── DroneSimulator.html     # Drone simulator + Swarm predictor UI
│   ├── MLDataUpload.html       # Dataset upload UI
│   └── main.css                # Shared styles
│
├── src/
│   ├── design_pipeline.py      # 3-stage drone design simulation engine
│   └── swarm_prediction.py     # Swarm failure prediction ML model
│
├── app.py                      # Flask web server (links frontend ↔ backend)
├── drone_data_pipeline.py      # Synthetic drone dataset generator
├── train_model.py              # ML model training (Random Forest)
├── cli.py                      # Terminal CLI for testing without UI
├── requirements.txt            # Python dependencies
└── hvn_drone_dataset.csv       # Auto-generated dataset (1,000 configs)
```

---

## 🧠 Machine Learning Models

### 1. Flight Time Predictor (Regression)
- **Type:** Supervised Learning — Regression
- **Algorithm:** Random Forest Regressor (with GridSearchCV hyperparameter tuning)
- **Trained on:** 1,000 synthetic drone configurations
- **Target:** Predicted flight time in minutes

| Model | MAE | R² Score |
|---|---|---|
| Linear Regression (baseline) | 2.35 min | 91.0% |
| **Random Forest (tuned)** ✅ | **1.64 min** | **95.3%** |

### 2. Swarm Failure Predictor (Classification)
- **Type:** Supervised Learning — Binary Classification
- **Algorithm:** Random Forest Classifier (class-weighted to minimise false negatives)
- **Features:** Battery %, motor RPM, temperature, IMU drift, signal strength
- **Threshold:** Drones with >70% failure probability are flagged as high-risk

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+

### Install dependencies
```bash
pip install -r requirements.txt
pip install Flask
```

### Generate the dataset
```bash
python drone_data_pipeline.py
```

### Start the web app
```bash
python app.py
```
Then open **http://localhost:5000** in your browser.

---

## 🖥️ CLI Usage (no UI required)

```bash
# Run the 3-stage drone design simulation pipeline
python cli.py simulate --count 1000

# Run swarm failure prediction for 300 drones
python cli.py predict-swarm --drones 300

# Show cost analysis / financial savings breakdown
python cli.py cost-analysis
```

---

## 🌐 API Endpoints (Flask)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Predict flight time & GBP cost for a drone config |
| `POST` | `/simulate_drone` | Get simulation stats for a drone type |
| `GET` | `/predict_swarm` | Run swarm failure prediction on 50 drones |

---

## 👥 Team
Built with ❤️ by the HVN Labs team for the hackathon demonstration.
