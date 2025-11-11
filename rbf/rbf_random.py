import os
import jax
import time
import optax
import pickle
import numpy as np
import jax.numpy as jnp
from config import Config
from model.gmm import GMM
from rbf_dataset_utility import Dataset
from rbf_dataset_utility.model import apply_projection

cfg = Config()


def replace_params(params, lambdas):
    params['mus'] = lambdas[:, 0:2]
    params['log_sigmas'] = lambdas[:, 2:4]
    params['angles'] = lambdas[:, 4]
    params['weights'] = lambdas[:, 5]
    return params


@jax.jit
def train_step(params, opt_state, eval_points, u_gt):
    def loss_fn(params):
        u_pred = model.apply(params, eval_points)
        return jnp.mean((u_pred - u_gt.reshape(-1)) ** 2)

    # Compute loss and gradients
    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(params)

    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def main():
    global model, optimizer

    key = jax.random.PRNGKey(42)

    test_results_dir = '../data/test_results'
    test_loss_history_r_dir = '../data/test_loss_history/rand_init'

    # Get ground truth
    ds = Dataset('poisson')
    gt = ds.get_ground_truth()

    # Get testing indices
    test_split = ds.get_split('test_solution_idx')
    test_idx_list = list(test_split.keys())

    # Clear the file folder
    for f in os.listdir(test_loss_history_r_dir):
        os.remove(os.path.join(test_loss_history_r_dir, f))

    for test_idx in test_idx_list:
        # Initialize model
        model = GMM(K=cfg.k)

        # Initialize parameters
        dummy_X = jnp.ones((1, 1))
        dummy_Y = jnp.ones((1, 1))
        params = model.init(key, (dummy_X, dummy_Y))

        # Initialize optimizer
        optimizer = optax.adam(learning_rate=cfg.lr_gmm)
        opt_state = optimizer.init(params)

        # Load predicted initialization params
        file_path = test_results_dir + "/idx_" + f"{test_idx}" + ".npy"
        if not os.path.exists(file_path):
            continue

        # Training loop
        losses = []
        epochs = []
        it_times = []
        print(f"Testing index {test_idx} is tested")
        for epoch in range(1, cfg.epochs_gmm + 1):
            key, subkey = jax.random.split(key, 2)

            epoch_start_time = time.time()

            # Get evaluate points
            X = gt['X']
            Y = gt['Y']
            eval_points = (X, Y)

            # Projection
            mus = params['params']['mus']
            log_sigmas = params['params']['log_sigmas']
            angles = params['params']['angles'][:, None]
            weights = params['params']['weights'][:, None]
            lambdas = jnp.concatenate([mus, log_sigmas, angles, weights], axis=1)
            lambdas = apply_projection(lambdas, eval_points)
            params['params'] = replace_params(params['params'], lambdas)

            # Perform training step
            params, opt_state, loss = train_step(
                params, opt_state, eval_points, gt['solutions'][test_idx]
            )

            epoch_end_time = time.time()

            # Track metrics
            losses.append(float(loss))
            epochs.append(epoch)

            # Track iteration time
            epoch_time = epoch_end_time - epoch_start_time
            it_times.append(epoch_time)

        # Save loss history
        np.save(test_loss_history_r_dir + f'/idx_{test_idx}.npy', losses)

        # Compute final L2 error
        u_pred = model.apply(params, eval_points)
        u_gt = gt['solutions'][test_idx]
        err = jnp.sqrt(jnp.mean((u_pred - u_gt.reshape(-1)) ** 2))
        tr = ds.trajectories
        gt_err = tr['metadata'][test_idx]['final_l2_error']
        print(f"Final L2 error is: {err:.8e}/{gt_err:.8e}")


if __name__ == "__main__":
    main()
