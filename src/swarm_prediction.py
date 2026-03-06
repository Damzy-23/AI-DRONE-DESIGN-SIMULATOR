import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def generate_swarm_data(num_drones=300, fail_ratio=0.08):
    """
    Generates synthetic but realistic pre-flight diagnostic data for a drone swarm.
    Features: battery_voltage, motor_rpm_variance, max_temp, imu_calibration_error, gps_signal, vibration_level
    """
    np.random.seed(42)  # For reproducible hackathon results
    
    # Generate Normal Drones
    num_normal = int(num_drones * (1 - fail_ratio))
    num_fail = num_drones - num_normal
    
    # Healthy drones feature distributions
    healthy_data = {
        'drone_id': np.arange(1, num_normal + 1),
        'battery_voltage': np.random.normal(24.0, 0.2, num_normal), # 24V normal
        'motor_rpm_variance': np.random.normal(15, 5, num_normal), # low variance is good
        'max_temp': np.random.normal(45, 5, num_normal), # 45C normal
        'imu_calibration_error': np.random.normal(0.01, 0.005, num_normal),
        'vibration_level': np.random.normal(1.2, 0.3, num_normal),
        'predict_fail': 0
    }
    
    # Failing drones feature distributions (worse metrics)
    fail_data = {
        'drone_id': np.arange(num_normal + 1, num_drones + 1),
        'battery_voltage': np.random.normal(22.8, 0.8, num_fail), # Can be lower
        'motor_rpm_variance': np.random.normal(45, 15, num_fail), # High variance = bad motor
        'max_temp': np.random.normal(65, 8, num_fail), # Overheating
        'imu_calibration_error': np.random.normal(0.05, 0.02, num_fail), # Miscalibrated
        'vibration_level': np.random.normal(3.5, 0.8, num_fail), # Heavy vibrations
        'predict_fail': 1
    }
    
    # Combine and shuffle
    df_healthy = pd.DataFrame(healthy_data)
    df_fail = pd.DataFrame(fail_data)
    
    df_swarm = pd.concat([df_healthy, df_fail]).sample(frac=1).reset_index(drop=True)
    return df_swarm

def train_failure_model():
    """
    Generates historical data and trains a RandomForest model to predict failures.
    """
    historical_data = generate_swarm_data(num_drones=2000, fail_ratio=0.10)
    
    X = historical_data.drop(columns=['drone_id', 'predict_fail'])
    y = historical_data['predict_fail']
    
    # We want to minimize false negatives. Class weights can help heavily penalize missing a failure.
    model = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 5}, random_state=42)
    model.fit(X, y)
    
    return model

def predict_failures(model, swarm_data, threshold=0.7):
    """
    Predicts probability of failure for a pre-flight swarm.
    Returns a dataframe of drones exceeding the failure threshold.
    """
    X_pred = swarm_data.drop(columns=['drone_id', 'predict_fail'], errors='ignore')
    
    probs = model.predict_proba(X_pred)[:, 1]
    
    results = swarm_data.copy()
    results['failure_probability'] = probs
    
    high_risk_drones = results[results['failure_probability'] > threshold]
    
    reasons = []
    for _, row in high_risk_drones.iterrows():
        drone_reasons = []
        if row['motor_rpm_variance'] > 30: drone_reasons.append("High Motor RPM Variance")
        if row['max_temp'] > 55: drone_reasons.append("Overheating")
        if row['vibration_level'] > 2.5: drone_reasons.append("Excessive Vibration")
        if row['battery_voltage'] < 23.0: drone_reasons.append("Low Battery Voltage")
        if not drone_reasons: drone_reasons.append("Anomalous Pattern Detected")
        
        reasons.append(", ".join(drone_reasons))
        
    high_risk_drones['diagnostic_reason'] = reasons
    return high_risk_drones[['drone_id', 'failure_probability', 'diagnostic_reason']].sort_values(by='failure_probability', ascending=False)
