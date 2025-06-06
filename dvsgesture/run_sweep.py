import yaml
import subprocess
import os
import pandas as pd
from tabulate import tabulate
import json
import sys
import argparse
import wandb
from train import parse_args as train_parse_args


def run_experiment(config, T, seed):
    """Constructs and runs a single experiment command."""
    fixed_params = config['fixed_parameters'].copy()
    fixed_params['T'] = T
    fixed_params['seed'] = seed

    cmd = [sys.executable, 'train.py']

    # Arguments in train.py that use '_' instead of the default '-'
    underscore_args = ['connect_f', 'T_train', 'wandb_project', 'wandb_entity']

    for key, value in fixed_params.items():
        arg_key = key.replace('_', '-')
        if key in underscore_args:
            arg_key = key

        if isinstance(value, bool):
            if value:
                cmd.append(f'--{arg_key}')
        elif value is not None:
            cmd.append(f'--{arg_key}')
            cmd.append(str(value))

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
    wandb.login()

    parser = argparse.ArgumentParser(description="Run a sweep of experiments.")
    parser.add_argument('--T', type=int, nargs='+', help="Override T values from config.")
    parser.add_argument('--seed', type=int, nargs='+', help="Override seed values from config.")
    args, unknown_args = parser.parse_known_args()
    
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'fixed_parameters' not in config:
        config['fixed_parameters'] = {}

    for key in ['data_path', 'output_dir', 'wandb_project', 'wandb_entity']:
        if key in config and key not in config['fixed_parameters']:
            config['fixed_parameters'][key] = config.pop(key)

    default_train_args = train_parse_args([])
    if 'fixed_parameters' not in config:
        config['fixed_parameters'] = {}

    # Update fixed_parameters with defaults
    for key, value in vars(default_train_args).items():
        if key not in config['fixed_parameters']:
            config['fixed_parameters'][key] = value

    # Override with command line arguments
    override_args = train_parse_args(unknown_args)
    for key, value in vars(override_args).items():
        # Check if the argument was actually passed by the user
        arg_with_hyphen = f'--{key.replace("_", "-")}'
        arg_with_underscore = f'--{key}'
        if arg_with_hyphen in unknown_args or arg_with_underscore in unknown_args:
            config['fixed_parameters'][key] = value

    if config['fixed_parameters']['data_path'] == "/path/to/your/DVS128Gesture":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE `data_path` in `dvsgesture/config.yaml`  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(1)

    results = []
    
    T_values = args.T if args.T else config['parameters']['T']
    seeds = args.seed if args.seed else config['parameters']['seed']
    
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