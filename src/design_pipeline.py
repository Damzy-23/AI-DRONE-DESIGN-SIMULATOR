import pandas as pd
import numpy as np

def generate_design_space(num_designs=1000):
    """
    Generates random combinations of drone designs.
    """
    np.random.seed(42)  # For reproducibility
    
    data = {
        'design_id': [f"DSN-{i:04d}" for i in range(1, num_designs + 1)],
        'motor_count': np.random.choice([4, 6, 8], num_designs),
        'propeller_diameter_inch': np.random.uniform(5.0, 15.0, num_designs),
        'battery_capacity_mah': np.random.uniform(2000, 10000, num_designs),
        'frame_mass_kg': np.random.uniform(0.5, 3.0, num_designs),
        'payload_capacity_kg': np.random.uniform(0.1, 5.0, num_designs)
    }
    return pd.DataFrame(data)

def stage1_low_fidelity(df):
    """
    Fast filtering using simplified physics equations (approximations).
    Filters down to Top 50.
    """
    # Approximations for scoring
    # Thrust approximation: proportional to motor_count * propeller_diameter^2
    estimated_thrust = df['motor_count'] * (df['propeller_diameter_inch'] ** 2) * 0.05
    total_mass = df['frame_mass_kg'] + df['payload_capacity_kg'] + (df['battery_capacity_mah'] / 5000)
    
    thrust_to_weight = estimated_thrust / total_mass
    
    # Filter designs with poor thrust-to-weight (< 2.0 is usually bad for maneuvering)
    df_filtered = df[thrust_to_weight >= 2.0].copy()
    
    # Score based on a mix of payload, flight time approximation, and thrust
    df_filtered['stage1_score'] = thrust_to_weight * (df_filtered['battery_capacity_mah'] / total_mass) * df_filtered['payload_capacity_kg']
    
    # Return top 50
    return df_filtered.sort_values(by='stage1_score', ascending=False).head(50)

def stage2_medium_fidelity(df):
    """
    Mocking an intermediate simulation (e.g. PyBullet rigid-body).
    Filters Top 50 down to Top 10.
    """
    # Introduce some non-linear stability scoring based on prop size vs frame mass
    stability = 1.0 - abs((df['propeller_diameter_inch'] / df['frame_mass_kg']) - 4.0) / 4.0
    stability = stability.clip(lower=0.1)
    
    df_filtered = df.copy()
    df_filtered['stage2_score'] = df_filtered['stage1_score'] * stability * np.random.uniform(0.8, 1.2, len(df))
    
    return df_filtered.sort_values(by='stage2_score', ascending=False).head(10)

def stage3_high_fidelity(df):
    """
    Mocking high-fidelity aerodynamic & CFD simulation (e.g. Gazebo).
    Returns final Top 5 robust designs.
    """
    df_filtered = df.copy()
    
    # Mocking environmental disturbances (wind resistance score)
    wind_resistance = (df_filtered['frame_mass_kg'] / df_filtered['propeller_diameter_inch']) * df_filtered['motor_count']
    
    df_filtered['final_score'] = df_filtered['stage2_score'] * wind_resistance * np.random.uniform(0.9, 1.1, len(df))
    
    return df_filtered.sort_values(by='final_score', ascending=False).head(5)

def run_pipeline(num_designs=1000):
    """
    Executes the full 3-stage simulation funnel.
    """
    initial_designs = generate_design_space(num_designs)
    
    stage1_results = stage1_low_fidelity(initial_designs)
    stage2_results = stage2_medium_fidelity(stage1_results)
    final_results = stage3_high_fidelity(stage2_results)
    
    return {
        "initial_count": len(initial_designs),
        "stage1_count": len(stage1_results),
        "stage2_count": len(stage2_results),
        "final_count": len(final_results),
        "top_designs": final_results[['design_id', 'motor_count', 'propeller_diameter_inch', 'battery_capacity_mah', 'payload_capacity_kg', 'final_score']]
    }
