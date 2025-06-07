import yaml
import os
import pandas as pd
from tabulate import tabulate
import sys
import argparse
import wandb
import copy
from train import run_training, parse_args


def run_experiment(base_args, T, seed):
    """Constructs and runs a single experiment."""
    train_args = copy.deepcopy(base_args)
    
    # Set sweep parameters
    train_args.T = T
    train_args.seed = seed

    print(f"\nRunning experiment with T={T}, seed={seed}")
    print(f"Arguments: {vars(train_args)}")

    try:
        metrics = run_training(train_args)
        return metrics
    except Exception as e:
        print(f"Error running experiment for T={T}, seed={seed}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
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

    # Build base arguments
    # 1. Start with train.py defaults
    base_args = parse_args([])
    # 2. Apply fixed parameters from config
    for key, value in config['fixed_parameters'].items():
        if hasattr(base_args, key):
            setattr(base_args, key, value)
    # 3. Apply command-line overrides
    override_args = parse_args(unknown_args or [])
    for key, value in vars(override_args).items():
        arg_with_hyphen = f'--{key.replace("_", "-")}'
        arg_with_underscore = f'--{key}'
        if arg_with_hyphen in unknown_args or arg_with_underscore in unknown_args:
            setattr(base_args, key, value)

    if hasattr(base_args, 'data_path') and base_args.data_path == "data" and not os.path.exists("data"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! PLEASE UPDATE `data_path` in `dvsgesture/config.yaml`  !!!")
        print("!!! or download the dataset to the `data` directory.     !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # sys.exit(1) # Commenting out to allow running without dataset for testing

    results = []
    
    T_values = args.T if args.T else config['parameters']['T']
    seeds = args.seed if args.seed else config['parameters']['seed']
    
    try:
        for T in T_values:
            for seed in seeds:
                metrics = run_experiment(base_args, T, seed)
                if metrics:
                    results.append(metrics)
    except KeyboardInterrupt:
        print("\n\nUser interrupted the sweep. Processing results gathered so far.")

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