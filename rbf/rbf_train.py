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


def lambda_to_params(params, lambdas):
    params['mus'] = lambdas[:, 0:2]
    params['log_sigmas'] = lambdas[:, 2:4]
    params['angles'] = lambdas[:, 4]
    params['weights'] = lambdas[:, 5]
    return params


def params_to_lambda(params):
    mus = params['mus']
    log_sigmas = params['log_sigmas']
    angles = params['angles'][:, None]
    weights = params['weights'][:, None]
    lambdas = jnp.concatenate([mus, log_sigmas, angles, weights], axis=1)
    return lambdas


@jax.jit
def train_step(params, opt_state, eval_points, u_gt):
    def loss_fn(params):
        u_pred = model.apply(params, eval_points)
        u_loss = jnp.mean((u_pred - u_gt.reshape(-1)) ** 2)
        return u_loss

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
    test_loss_history_dir = '../data/test_loss_history/good_init'

    # Get ground truth
    ds = Dataset('poisson')
    gt = ds.get_ground_truth()

    # Get testing indices
    test_split = ds.get_split('test_solution_idx')
    test_idx_list = list(test_split.keys())

    # Clear the file folder
    for f in os.listdir(test_loss_history_dir):
        os.remove(os.path.join(test_loss_history_dir, f))

    for test_idx in test_idx_list:
        # Initialize model
        model = GMM(K=cfg.k * cfg.group_size)

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

        predict_init_params = np.load(file_path, allow_pickle=True).squeeze(axis=0)

        # Replace initialization params
        params['params'] = lambda_to_params(params['params'], predict_init_params)

        # Training loop
        losses = []
        epochs = []
        it_times = []
        best_loss = 100000
        print(f"Testing index {test_idx} is tested")
        for epoch in range(1, cfg.epochs_gmm + 1):
            key, subkey = jax.random.split(key, 2)

            epoch_start_time = time.time()

            # Get evaluate points
            X = gt['X']
            Y = gt['Y']
            eval_points = (X, Y)

            # Projection
            lambdas = params_to_lambda(params['params'])
            lambdas = apply_projection(lambdas, eval_points)
            params['params'] = lambda_to_params(params['params'], lambdas)

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

            # Save the best model
            if loss < best_loss:
                best_loss = loss
                with open('./weight/best_params.pkl', 'wb') as f:
                    pickle.dump({'params': params}, f)

        # Save loss history
        np.save(test_loss_history_dir + f'/idx_{test_idx}.npy', losses)

        # Compute final L2 error
        u_pred = model.apply(params, eval_points)
        u_gt = gt['solutions'][test_idx]
        err = jnp.sqrt(jnp.mean((u_pred - u_gt.reshape(-1)) ** 2))
        tr = ds.trajectories
        gt_err = tr['metadata'][test_idx]['final_l2_error']
        print(f"Final L2 error is: {err:.8e}/{gt_err:.8e}")

        # Plotting
        import matplotlib.pyplot as plt

        x_1d = np.asarray(X[0, :]) if X.ndim == 2 else np.asarray(X)
        y_1d = np.asarray(Y[:, 0]) if Y.ndim == 2 else np.asarray(Y)
        u_gt_grid = np.asarray(u_gt).reshape(X.shape)
        u_pred_grid = np.asarray(u_pred).reshape(X.shape)
        abs_error = np.abs(u_pred_grid - u_gt_grid)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # GT
        im0 = axes[0].pcolormesh(x_1d, y_1d, u_gt_grid, shading='auto', cmap='viridis')
        axes[0].set_title('Ground Truth')
        axes[0].set_xlabel('x'); axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0])

        # Prediction
        im1 = axes[1].pcolormesh(x_1d, y_1d, u_pred_grid, shading='auto', cmap='viridis')
        axes[1].set_title('Prediction')
        axes[1].set_xlabel('x'); axes[1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[1])

        # Absolute Error
        im2 = axes[2].pcolormesh(x_1d, y_1d, abs_error, shading='auto', cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
        plt.colorbar(im2, ax=axes[2])
        plt.tight_layout()

        # Save
        plot_dir = "../data/pics"
        os.makedirs(plot_dir, exist_ok=True)
        out_path_png = os.path.join(plot_dir, f"idx_{test_idx}.png")
        plt.savefig(out_path_png, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
