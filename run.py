import subprocess
from rbf_dataset_utility import Dataset

run_list = [
    # "vae/cvae_train.py",
    # "hyper_net/hyper_net_train.py",
    "rbf/rbf_train.py",
    "tools/plotting.py"
]

for file in run_list:
    print(f"\nRunning: {file}")
    result = subprocess.run(["python", file], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error occurred while running {file}:")
        print(result.stderr)
        break

ds = Dataset('poisson')
print(ds.parameters['diversity_statistics']['per_solution'][0].keys())