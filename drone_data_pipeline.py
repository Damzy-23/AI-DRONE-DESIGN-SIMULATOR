"""
Generates 1,000 simulated drone configurations with realistic
physics-based relationships between input parts and output metrics.

Input Variables (what the engineer selects):
  - battery_type
  - num_motors
  - propeller_size
  - frame_size
  - radio_receiver

Derived / Simulation Output Metrics:
  - battery_capacity_mah
  - battery_weight_g
  - motor_thrust_g (per motor)
  - total_thrust_g
  - frame_weight_g
  - total_weight_g
  - payload_capacity_g
  - flight_time_min
  - power_consumption_w
  - stability_score (0-100)
  - max_speed_kmh
  - crash_probability (0-1)
  - drone_cost_usd
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_SAMPLES = 1000

# ─────────────────────────────────────────────
# 1. PART DEFINITIONS (lookup tables)
# ─────────────────────────────────────────────

BATTERY_TYPES = {
    "LiPo 3S 2200mAh": {
        "capacity_mah": 2200, "voltage": 11.1, "weight_g": 180,
        "cost": 25, "discharge_rate": 30
    },
    "LiPo 4S 3000mAh": {
        "capacity_mah": 3000, "voltage": 14.8, "weight_g": 320,
        "cost": 45, "discharge_rate": 40
    },
    "LiPo 4S 5000mAh": {
        "capacity_mah": 5000, "voltage": 14.8, "weight_g": 520,
        "cost": 72, "discharge_rate": 50
    },
    "LiPo 6S 4000mAh": {
        "capacity_mah": 4000, "voltage": 22.2, "weight_g": 680,
        "cost": 95, "discharge_rate": 60
    },
    "LiPo 6S 8000mAh": {
        "capacity_mah": 8000, "voltage": 22.2, "weight_g": 1050,
        "cost": 150, "discharge_rate": 50
    },
}

MOTOR_OPTIONS = {
    # num_motors -> (thrust_per_motor_g, weight_per_motor_g, cost_per_motor)
    3: {"thrust_g": 800,  "weight_g": 55,  "cost": 18},
    4: {"thrust_g": 900,  "weight_g": 60,  "cost": 22},
    6: {"thrust_g": 1000, "weight_g": 72,  "cost": 28},
    8: {"thrust_g": 1100, "weight_g": 85,  "cost": 35},
}

PROPELLER_SIZES = {
    '5"':  {"diameter_in": 5,  "efficiency": 0.70, "speed_factor": 1.25, "weight_g": 8,  "cost": 4},
    '7"':  {"diameter_in": 7,  "efficiency": 0.80, "speed_factor": 1.10, "weight_g": 12, "cost": 6},
    '9"':  {"diameter_in": 9,  "efficiency": 0.88, "speed_factor": 0.95, "weight_g": 18, "cost": 9},
    '12"': {"diameter_in": 12, "efficiency": 0.92, "speed_factor": 0.80, "weight_g": 28, "cost": 14},
}

FRAME_SIZES = {
    "220mm Mini": {
        "base_weight_g": 120, "max_payload_g": 200, "durability": 0.5,
        "cost": 35, "drag_coeff": 0.30
    },
    "330mm Medium": {
        "base_weight_g": 250, "max_payload_g": 500, "durability": 0.7,
        "cost": 65, "drag_coeff": 0.40
    },
    "450mm Standard": {
        "base_weight_g": 400, "max_payload_g": 1000, "durability": 0.85,
        "cost": 95, "drag_coeff": 0.50
    },
    "550mm Heavy": {
        "base_weight_g": 620, "max_payload_g": 2000, "durability": 0.95,
        "cost": 140, "drag_coeff": 0.65
    },
}

RADIO_RECEIVERS = {
    "FrSky R-XSR":     {"range_m": 1500, "latency_ms": 12, "weight_g": 10, "cost": 22},
    "TBS Crossfire":    {"range_m": 5000, "latency_ms": 8,  "weight_g": 15, "cost": 55},
    "ExpressLRS 900":   {"range_m": 8000, "latency_ms": 5,  "weight_g": 12, "cost": 35},
    "DJI FPV Receiver": {"range_m": 4000, "latency_ms": 18, "weight_g": 20, "cost": 65},
}

# ─────────────────────────────────────────────
# 2. SIMULATION FUNCTIONS
# ─────────────────────────────────────────────

def compute_total_weight(battery, motors_n, motor_spec, propeller, frame, receiver):
    """Sum all component weights."""
    w = (
        battery["weight_g"]
        + motors_n * motor_spec["weight_g"]
        + motors_n * propeller["weight_g"]
        + frame["base_weight_g"]
        + receiver["weight_g"]
    )
    return w


def compute_flight_time(battery, total_weight_g, motors_n, propeller, noise_std=1.5):
    """
    Simplified flight-time model:
      energy_wh = capacity * voltage / 1000
      avg_power_draw = f(weight, motors, efficiency)
      flight_time = energy / power * 60  (minutes)
    """
    energy_wh = (battery["capacity_mah"] * battery["voltage"]) / 1000.0
    # Power draw: realistic hover power scales with weight^1.5 and prop efficiency
    weight_kg = total_weight_g / 1000.0
    avg_power_w = (weight_kg ** 1.3) * 120.0 / propeller["efficiency"]
    # More motors = slightly more overhead from ESCs
    avg_power_w *= (1 + (motors_n - 4) * 0.04)
    avg_power_w = max(avg_power_w, 30)  # floor
    flight_time_min = (energy_wh / avg_power_w) * 60.0
    # Add realistic noise
    flight_time_min += np.random.normal(0, noise_std)
    return round(max(flight_time_min, 1.0), 2), round(avg_power_w, 2)


def compute_max_speed(battery, propeller, total_weight_g, frame):
    """
    Speed model: higher voltage + smaller props + lighter weight = faster.
    Drag coefficient from frame limits top speed.
    """
    voltage_factor = battery["voltage"] / 11.1  # normalised to 3S
    speed = (
        80.0
        * voltage_factor
        * propeller["speed_factor"]
        * (1.0 / (1.0 + frame["drag_coeff"]))
        * (1500.0 / (total_weight_g + 500))
    )
    speed += np.random.normal(0, 3)
    return round(max(speed, 10.0), 2)


def compute_stability_score(motors_n, propeller, frame, total_weight_g, total_thrust_g, receiver):
    """
    Stability 0-100: more motors, larger props, durable frame, low latency = higher.
    Thrust-to-weight ratio must be healthy (>1.5 ideal).
    """
    twr = total_thrust_g / max(total_weight_g, 1)
    twr_score = min(twr / 3.0, 1.0) * 30  # max 30 pts

    motor_score = {3: 5, 4: 15, 6: 22, 8: 25}.get(motors_n, 10)

    prop_score = propeller["efficiency"] * 20  # max ~18

    frame_score = frame["durability"] * 15  # max ~14

    latency_score = max(0, 12 - receiver["latency_ms"]) # max 12 if <=0ms, realistically ~7

    raw = twr_score + motor_score + prop_score + frame_score + latency_score
    raw += np.random.normal(0, 3)
    return round(np.clip(raw, 0, 100), 2)


def compute_crash_probability(stability_score, twr):
    """Inverse relationship with stability and TWR."""
    base = 1.0 - (stability_score / 120.0)
    if twr < 1.5:
        base += 0.15
    elif twr < 2.0:
        base += 0.05
    base += np.random.normal(0, 0.04)
    return round(np.clip(base, 0.01, 0.95), 4)


def compute_cost(battery, motors_n, motor_spec, propeller, frame, receiver):
    """Total build cost in USD with assembly overhead."""
    parts_cost = (
        battery["cost"]
        + motors_n * motor_spec["cost"]
        + motors_n * propeller["cost"]
        + frame["cost"]
        + receiver["cost"]
    )
    assembly_overhead = parts_cost * 0.12  # 12% assembly/misc
    return round(parts_cost + assembly_overhead, 2)


# ─────────────────────────────────────────────
# 3. GENERATE THE DATASET
# ─────────────────────────────────────────────

def generate_dataset(n=NUM_SAMPLES):
    rows = []

    battery_keys = list(BATTERY_TYPES.keys())
    motor_keys = list(MOTOR_OPTIONS.keys())
    prop_keys = list(PROPELLER_SIZES.keys())
    frame_keys = list(FRAME_SIZES.keys())
    radio_keys = list(RADIO_RECEIVERS.keys())

    for i in range(n):
        # Randomly select parts
        bat_name = np.random.choice(battery_keys)
        mot_count = int(np.random.choice(motor_keys))
        prop_name = np.random.choice(prop_keys)
        frame_name = np.random.choice(frame_keys)
        radio_name = np.random.choice(radio_keys)

        bat = BATTERY_TYPES[bat_name]
        mot = MOTOR_OPTIONS[mot_count]
        prop = PROPELLER_SIZES[prop_name]
        frm = FRAME_SIZES[frame_name]
        rcv = RADIO_RECEIVERS[radio_name]

        # --- Run simulation ---
        total_weight = compute_total_weight(bat, mot_count, mot, prop, frm, rcv)
        total_thrust = mot_count * mot["thrust_g"]
        twr = total_thrust / max(total_weight, 1)
        payload_capacity = max(0, total_thrust - total_weight)

        flight_time, power_consumption = compute_flight_time(bat, total_weight, mot_count, prop)
        max_speed = compute_max_speed(bat, prop, total_weight, frm)
        stability = compute_stability_score(mot_count, prop, frm, total_weight, total_thrust, rcv)
        crash_prob = compute_crash_probability(stability, twr)
        cost = compute_cost(bat, mot_count, mot, prop, frm, rcv)

        rows.append({
            # ── Input variables (engineer selections) ──
            "config_id": f"CFG-{i+1:04d}",
            "battery_type": bat_name,
            "num_motors": mot_count,
            "propeller_size": prop_name,
            "frame_size": frame_name,
            "radio_receiver": radio_name,

            # ── Derived component specs ──
            "battery_capacity_mah": bat["capacity_mah"],
            "battery_weight_g": bat["weight_g"],
            "motor_thrust_per_g": mot["thrust_g"],
            "total_thrust_g": total_thrust,

            # ── Simulation output metrics ──
            "total_weight_g": total_weight,
            "thrust_to_weight_ratio": round(twr, 3),
            "payload_capacity_g": round(payload_capacity, 2),
            "flight_time_min": flight_time,
            "power_consumption_w": power_consumption,
            "max_speed_kmh": max_speed,
            "stability_score": stability,
            "crash_probability": crash_prob,
            "drone_cost_usd": cost,
        })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────
# 4. MAIN — Run pipeline & export
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  HVN Labs — Drone Configuration Data Pipeline")
    print("=" * 60)

    print(f"\n[1/3] Generating {NUM_SAMPLES} simulated drone configurations...")
    df = generate_dataset(NUM_SAMPLES)

    print(f"[2/3] Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Summary stats
    print(f"\n{'─'*50}")
    print("  DATASET SUMMARY")
    print(f"{'─'*50}")
    print(f"  Configurations generated : {len(df)}")
    print(f"  Unique battery types     : {df['battery_type'].nunique()}")
    print(f"  Motor configs            : {sorted(df['num_motors'].unique())}")
    print(f"  Flight time range        : {df['flight_time_min'].min():.1f} – {df['flight_time_min'].max():.1f} min")
    print(f"  Cost range               : ${df['drone_cost_usd'].min():.0f} – ${df['drone_cost_usd'].max():.0f}")
    print(f"  Stability score range    : {df['stability_score'].min():.1f} – {df['stability_score'].max():.1f}")
    print(f"  Crash probability range  : {df['crash_probability'].min():.2%} – {df['crash_probability'].max():.2%}")
    print(f"{'─'*50}")

    # Export
    output_path = "hvn_drone_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[3/3] Dataset saved → {output_path}")
    print(f"\nFirst 5 rows:\n")
    print(df.head().to_string(index=False))
    print("\n✓ Pipeline complete.")
