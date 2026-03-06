import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# Import the ML module
import train_model

app = Flask(__name__, static_folder='Frontend')

# ---------------------------------------------------------
# Load and cache the trained ML model globally on startup
# ---------------------------------------------------------
print("[INFO] Pre-loading dataset and training model... Please wait.")
from train_model import df, preprocessor, rf_model, param_grid, GridSearchCV

# We recreate the grid search and best model logic inline to cache the exact model object
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
X = df[train_model.features]
y = df[train_model.target]
grid_search.fit(X, y)
global_best_model = grid_search.best_estimator_

# Precompute average drone cost based on features since it's not the target of the ML model 
# (Cost is essentially deterministic in our dataset)
def estimate_cost(battery_str, num_motors, prop_str, frame_str, receiver_str):
    matches = df[
        (df["battery_type"].astype(str).str.contains(battery_str, case=False, na=False)) &
        (df["num_motors"] == int(num_motors))
    ]
    if len(matches) > 0:
        return round(matches["drone_cost_gbp"].mean(), 2)
    return 350.0  # Safe default if configuration combination wasn't sampled


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'MLModel.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Extract raw form inputs
    battery_type = data.get('batteryType', '')
    num_motors = int(data.get('numMotors', 4))
    propeller_size = data.get('propellerSize', '')
    frame_size = data.get('frameSize', '')
    radio_receiver = data.get('radioReceiver', '')

    # 1. Map frontend UI strings to backend dataset keys
    # Battery
    battery_map = {
        "LiPo_1500": {"name": "LiPo 3S 2200mAh", "cap": 1500, "weight": 140},
        "LiPo_2200": {"name": "LiPo 3S 2200mAh", "cap": 2200, "weight": 180},
        "LiPo_3000": {"name": "LiPo 4S 3000mAh", "cap": 3000, "weight": 320}
    }
    bat = battery_map.get(battery_type, battery_map["LiPo_2200"])

    # Motors (thrust per motor)
    thrust_map = {2: 900, 4: 900, 6: 1000, 8: 1100}
    motor_thrust = thrust_map.get(num_motors, 900)

    # Propellers
    prop_map = {"5in": '5"', "7in": '7"', "10in": '9"'}
    prop = prop_map.get(propeller_size, '5"')

    # Frame
    frame_map = {"small": "220mm Mini", "medium": "330mm Medium", "large": "550mm Heavy"}
    frame = frame_map.get(frame_size, "330mm Medium")

    # Radio
    radio_map = {"2_4ghz": "FrSky R-XSR", "5_8ghz": "ExpressLRS 900", "long_range": "TBS Crossfire"}
    radio = radio_map.get(radio_receiver, "FrSky R-XSR")

    # Construct the input matching `train_model.py` feature list
    example_input = pd.DataFrame([{
        "battery_capacity_mah": bat["cap"],
        "battery_weight_g": bat["weight"],
        "num_motors": num_motors,
        "motor_thrust_per_g": motor_thrust,
        "propeller_size": prop,
        "frame_size": frame,
        "radio_receiver": radio,
        "payload_capacity_g": 300 # Assume fixed 300g payload for UI predictions
    }])

    # 2. Predict Flight Time using our trained Random Forest model
    predicted_flight_time = global_best_model.predict(example_input)[0]

    # 3. Estimate Cost
    estimated_cost = estimate_cost(bat["name"][:7], num_motors, prop, frame, radio)

    return jsonify({
        "flight_time_min": round(predicted_flight_time, 2),
        "drone_cost_gbp": estimated_cost
    })

from src import design_pipeline
from src import swarm_prediction

@app.route('/simulate_drone', methods=['POST'])
def simulate_drone():
    data = request.json
    config = data.get('config', 'trainer')
    
    # Run a mock single iteration of our simulation testing
    # We will return speed, crash likelihood, and altitude based on config
    stats = {}
    if config == 'trainer':
        stats = {"crash_likelihood": "Low (12%)", "speed": 35, "altitude": 120, "drain_rate": 0.3}
    elif config == 'racer':
        stats = {"crash_likelihood": "High (45%)", "speed": 120, "altitude": 150, "drain_rate": 0.8}
    elif config == 'cinelifter':
        stats = {"crash_likelihood": "Medium (25%)", "speed": 40, "altitude": 200, "drain_rate": 0.5}

    return jsonify(stats)

@app.route('/predict_swarm', methods=['GET'])
def predict_swarm_api():
    # Load model and predict
    model = swarm_prediction.train_failure_model()
    # Let's do a fast 50 drone swarm for UI display
    swarm_data = swarm_prediction.generate_swarm_data(num_drones=50, fail_ratio=0.10)
    high_risk_drones = swarm_prediction.predict_failures(model, swarm_data, threshold=0.7)
    
    results = []
    for _, row in high_risk_drones.iterrows():
        results.append({
            "drone_id": int(row['drone_id']),
            "failure_probability": f"{row['failure_probability']*100:.1f}%",
            "reason": row['diagnostic_reason']
        })
        
    return jsonify({
        "total_evaluated": len(swarm_data),
        "high_risk_count": len(high_risk_drones),
        "drones": results
    })

if __name__ == '__main__':
    print("[INFO] Starting Flask Server on http://localhost:5000")
    app.run(debug=False, port=5000)
