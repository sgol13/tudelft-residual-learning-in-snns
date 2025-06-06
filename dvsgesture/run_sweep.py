import yaml
import subprocess
import os
import pandas as pd
from tabulate import tabulate
import json
import sys

def check_dependencies():
    """Checks for required packages and provides installation instructions if missing."""
    try:
        import pandas
        import tabulate
        import yaml
        import wandb
    except ImportError as e:
        print(f"Missing dependency: {e.name}. Please install it.")
        print("You can usually install missing packages with pip:")
        if e.name == 'yaml':
            print("pip install PyYAML")
        else:
            print(f"pip install {e.name}")
        sys.exit(1)

def run_experiment(config, T, seed):
    """Constructs and runs a single experiment command."""
    fixed_params = config['fixed_parameters']
    cmd = [
        'python', 'train.py',
        '--model', fixed_params['model'],
        '--connect_f', fixed_params['connect_f'],
        '--data-path', config['data_path'],
        '--device', fixed_params['device'],
        '--batch-size', str(fixed_params['batch_size']),
        '--epochs', str(fixed_params['epochs']),
        '--workers', str(fixed_params['workers']),
        '--lr', str(fixed_params['lr']),
        '--momentum', str(fixed_params['momentum']),
        '--weight-decay', str(fixed_params['weight_decay']),
        '--lr-step-size', str(fixed_params['lr_step_size']),
        '--lr-gamma', str(fixed_params['lr_gamma']),
        '--output-dir', config['output_dir'],
        '--T', str(T),
        '--seed', str(seed),
    ]

    if fixed_params.get('amp'):
        cmd.append('--amp')
    if fixed_params.get('tb'):
        cmd.append('--tb')
    
    cmd.append('--wandb')
    cmd.extend(['--wandb_project', config['wandb_project']])
    if config.get('wandb_entity'):
        cmd.extend(['--wandb_entity', config['wandb_entity']])
    
    if fixed_params.get('early_stop'):
        cmd.append('--early-stop')

    print(f"\nRunning experiment with T={T}, seed={seed}")
    print(f"Command: {' '.join(cmd)}")
    
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
    
    if process.returncode != 0:
        print(f"Error running experiment for T={T}, seed={seed}")
        print("Stderr:")
        print(process.stderr)
        return None

    try:
        json_line = None
        for line in reversed(process.stdout.strip().split('\n')):
            if line.strip().startswith('{') and line.strip().endswith('}'):
                json_line = line
                break
        
        if json_line:
            metrics = json.loads(json_line)
            return metrics
        else:
            print(f"Could not find JSON metrics in output for T={T}, seed={seed}")
            print("Stdout:")
            print(process.stdout)
            return None

    except (IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing metrics for T={T}, seed={seed}: {e}")
        print("Stdout:")
        print(process.stdout)
        return None

def main():
    """Main function to run the experiment sweep."""
    check_dependencies()
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config['data_path'] == "/path/to/your/DVS128Gesture":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE `data_path` in `dvsgesture/config.yaml`  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    results = []
    
    T_values = config['parameters']['T']
    seeds = config['parameters']['seed']
    
    for T in T_values:
        for seed in seeds:
            metrics = run_experiment(config, T, seed)
            if metrics:
                results.append(metrics)

    if not results:
        print("\nNo results were collected. Exiting.")
        return

    df = pd.DataFrame(results)
    df = df[['T', 'seed', 'acc1', 'acc5', 'train_time_s', 'imgs_per_s']]
    
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 't_sweep_results.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\nResults saved to {csv_path}")

    print("\n--- Experiment Results ---")
    print(tabulate(df, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    main() 