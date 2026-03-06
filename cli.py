import argparse
import os
import sys

# Ensure src modules are discoverable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src import design_pipeline
from src import swarm_prediction

try:
    from prettytable import PrettyTable
except ImportError:
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def simulate_command(args):
    print(f"\\n[INFO] Starting Drone Design Simulation Pipeline for {args.count} configurations...")
    results = design_pipeline.run_pipeline(args.count)
    
    print(f"\\n--- PIPELINE FUNNEL ---")
    print(f"Initial Designs : {results['initial_count']}")
    print(f"Stage 1 (Low-Fi) : {results['stage1_count']}")
    print(f"Stage 2 (Mid-Fi) : {results['stage2_count']}")
    print(f"Stage 3 (High-Fi): {results['final_count']}")
    print(f"-----------------------\\n")
    
    print("[SUCCESS] Top 5 Recommended Drone Designs for Physical Prototyping:\\n")
    
    table = PrettyTable()
    table.field_names = ["Design ID", "Score", "Motors", "Propeller (in)", "Battery (mAh)", "Payload (kg)"]
    
    for _, row in results['top_designs'].iterrows():
        table.add_row([
            row['design_id'],
            f"{row['final_score']:.2f}",
            int(row['motor_count']),
            f"{row['propeller_diameter_inch']:.1f}",
            f"{row['battery_capacity_mah']:.0f}",
            f"{row['payload_capacity_kg']:.2f}"
        ])
        
    print(table)


def predict_swarm_command(args):
    print(f"\\n[INFO] Generating Swarm Operational Data for {args.drones} units...")
    
    # Train model first
    model = swarm_prediction.train_failure_model()
    
    # Get pre-flight diagnostic data
    print(f"[INFO] Evaluating pre-flight diagnostic data...")
    swarm_data = swarm_prediction.generate_swarm_data(num_drones=args.drones, fail_ratio=0.05) # Test 5% fail rate
    
    # Predict high risk drones
    high_risk_drones = swarm_prediction.predict_failures(model, swarm_data, threshold=0.7)
    
    print(f"\\n--- SWARM RELIABILITY REPORT ---")
    print(f"Total Drones Evaluated : {len(swarm_data)}")
    print(f"High-Risk Drones Flags : {len(high_risk_drones)}")
    print(f"Action Required        : REMOVE THE FOLLOWING DRONES BEFORE SHOW\\n")
    
    if len(high_risk_drones) == 0:
        print("[SUCCESS] All drones are optimal for flight. No high-risk failures predicted.")
    else:
        table = PrettyTable()
        table.field_names = ["Drone ID", "Failure Prob", "Diagnostic Reasons for Removal"]
        table.align["Diagnostic Reasons for Removal"] = "l"
        
        for _, row in high_risk_drones.iterrows():
            prob_pct = f"{row['failure_probability']*100:.1f}%"
            table.add_row([f"Unit #{int(row['drone_id']):03d}", prob_pct, row['diagnostic_reason']])
            
        print(table)


def cost_analysis_command(args):
    print("\\n[COST ANALYSIS] Simulation-First Development Strategy")
    print("=" * 60)
    print(f"Traditional Prototyping Cost:   £50,000 per year")
    print(f"Expected physical builds:       15-20 units")
    print("-" * 60)
    print(f"Using our Simulation Pipeline, we validate 1000s of designs virtually.")
    print(f"We only build the Top 5 candidates physically.")
    print("-" * 60)
    print(f"New Prototyping Cost:           £15,000 - £20,000 per year")
    print(f"Total Savings:                  £30,000 - £35,000 per year")
    print(f"Cost Reduction Percentage:      ~60 - 70%")
    print("=" * 60)
    print("[INFO] NOTE: Residual physical testing remains essential for EMI, Battery Degradation, and Real Aerodynamics.")


def main():
    parser = argparse.ArgumentParser(description="AI-Driven Drone Design & Swarm Reliability CLI Platform")
    subparsers = parser.add_subparsers(dest="command", help="Available Commands")
    
    # Simulate Command
    parser_sim = subparsers.add_parser("simulate", help="Run the drone design simulation pipeline")
    parser_sim.add_argument("--count", type=int, default=1000, help="Number of designs to simulate (default: 1000)")
    
    # Predict Swarm Command
    parser_pred = subparsers.add_parser("predict-swarm", help="Predict failures in a drone swarm")
    parser_pred.add_argument("--drones", type=int, default=300, help="Number of drones in the swarm to evaluate (default: 300)")
    
    # Cost Analysis Command
    parser_cost = subparsers.add_parser("cost-analysis", help="Show cost savings of the simulation pipeline")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        simulate_command(args)
    elif args.command == "predict-swarm":
        predict_swarm_command(args)
    elif args.command == "cost-analysis":
        cost_analysis_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
