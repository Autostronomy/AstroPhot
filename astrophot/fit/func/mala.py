import numpy as np
from tqdm import tqdm


def mala(
    initial_state,  # (num_chains, D)
    log_prob,  # (num_chains, D) -> (num_chains,)
    log_prob_grad,  # (num_chains, D) -> (num_chains, D)
    num_samples,
    epsilon,
    mass_matrix,  # covariance
    progress=True,
    desc="MALA",
):
    x = np.array(initial_state, copy=True)
    C, D = x.shape

    # mass, inv_mass, L
    mass = np.array(mass_matrix, copy=False)  # (D, D)
    inv_mass = np.linalg.inv(mass)  # (D, D)
    L = np.linalg.cholesky(mass)  # (D, D)

    samples = np.zeros((num_samples, C, D), dtype=x.dtype)  # (N, C, D)
    acceptance_rate = np.zeros([0])  # (0,)
    logp = np.zeros((num_samples, C), dtype=x.dtype)  # (N, C)

    # Cache current state
    logp_cur = log_prob(x)  # (C,)
    grad_cur = log_prob_grad(x)  # (C, D)

    # Random number generator
    rng = np.random.default_rng(np.random.randint(1e10))

    it = range(num_samples)
    if progress:
        it = tqdm(it, desc=desc, position=0, leave=True)

    for t in it:
        # proposal using current grad
        mu_x = 0.5 * (epsilon**2) * (grad_cur @ mass)  # (C, D)
        noise = rng.standard_normal((C, D)) @ L.T  # (C, D)
        x_prop = x + mu_x + epsilon * noise  # (C, D)

        # Evaluate proposal
        logp_prop = log_prob(x_prop)  # (C,)
        grad_prop = log_prob_grad(x_prop)  # (C, D)

        mu_xprop = 0.5 * (epsilon**2) * (grad_prop @ mass)  # (C, D)

        # q(x|x') \propto \exp(-0.5|x - x' - mu(x')|^2 / \epsilon^2)
        d1 = x - x_prop - mu_xprop  # for q(x | x')
        d2 = x_prop - x - mu_x  # for q(x'| x)

        logq1 = -0.5 * np.einsum("bi,ij,bj->b", d1, inv_mass, d1) / epsilon**2  # (C,)
        logq2 = -0.5 * np.einsum("bi,ij,bj->b", d2, inv_mass, d2) / epsilon**2  # (C,)

        log_alpha = (logp_prop - logp_cur) + (logq1 - logq2)  # (C,)

        accept = np.log(rng.random(C)) < log_alpha  # (C,)
        acceptance_rate = np.concatenate([acceptance_rate, accept])

        # Update all three pieces in-place where accepted
        x[accept] = x_prop[accept]  # (C, D)
        logp_cur[accept] = logp_prop[accept]  # (C,)
        grad_cur[accept] = grad_prop[accept]  # (C, D)

        samples[t] = x.copy()
        logp[t] = logp_cur.copy()

        if progress:
            it.set_postfix(acc_rate=f"{acceptance_rate.mean():0.2f}")

    return samples, logp
