import numpy as np
from scipy.optimize import minimize


def objective_function(w, S_xa, S_xb, S_ii_list, lambda_list, theta_r):
    """
    Objective function to be minimized.

    Args:
        w: Concatenated weight vectors [wx1, wx2, wa, wb].
        S_xa: Covariance matrix between neuroimaging (dataset 1) and behavioral (dataset 1).
        S_xb: Covariance matrix between neuroimaging (dataset 2) and behavioral (dataset 2).
        S_ii_list: List of covariance matrices [S_x1x1, S_x2x2, S_aa, S_bb].
        lambda_list: List of Lagrange multipliers [lambda_x1, lambda_x2, lambda_a, lambda_b].
        theta_r: Hyperparameter controlling dissimilarity penalty.

    Returns:
        Value of the objective function.
    """

    # Extract weight vectors from the concatenated vector 'w'.
    dim_x1 = S_xa.shape[0]
    dim_x2 = S_xb.shape[0]
    dim_a = S_xa.shape[1]
    dim_b = S_xb.shape[1]

    wx1 = w[:dim_x1]
    wx2 = w[dim_x1 : dim_x1 + dim_x2]
    wa = w[dim_x1 + dim_x2 : dim_x1 + dim_x2 + dim_a]
    wb = w[dim_x1 + dim_x2 + dim_a :]

    # Calculate correlation terms.
    term1 = -np.dot(wx1.T, np.dot(S_xa, wa))
    term2 = -np.dot(wx2.T, np.dot(S_xb, wb))

    # Calculate regularization terms.
    term3 = 0
    for i, S_ii in enumerate(S_ii_list):
        term3 += (
            0.5
            * lambda_list[i]
            * (
                np.dot(
                    w[
                        sum([S_ii_list[j].shape[0] for j in range(i)]) : sum(
                            [S_ii_list[j].shape[0] for j in range(i + 1)]
                        )
                    ].T,
                    np.dot(
                        S_ii,
                        w[
                            sum([S_ii_list[j].shape[0] for j in range(i)]) : sum(
                                [S_ii_list[j].shape[0] for j in range(i + 1)]
                            )
                        ],
                    ),
                )
                - 1
            )
        )

    # Calculate dissimilarity penalty (example using L2 norm).
    delta_w = wx1 - wx2
    term4 = theta_r * 0.5 * np.linalg.norm(delta_w) ** 2

    return term1 + term2 + term3 + term4


def solve_generalized_cca(S_xa, S_xb, S_ii_list, lambda_list, theta_r):
    """
    Solves the generalized CCA problem.

    Args:
        S_xa: Covariance matrix between neuroimaging (dataset 1) and behavioral (dataset 1).
        S_xb: Covariance matrix between neuroimaging (dataset 2) and behavioral (dataset 2).
        S_ii_list: List of covariance matrices [S_x1x1, S_x2x2, S_aa, S_bb].
        lambda_list: List of Lagrange multipliers [lambda_x1, lambda_x2, lambda_a, lambda_b].
        theta_r: Hyperparameter controlling dissimilarity penalty.

    Returns:
        Optimization result containing the weight vectors.
    """
    # Initial guess for the weight vectors.
    w_initial = np.random.rand(sum([S_ii_list[i].shape[0] for i in range(4)]))

    # Minimize the objective function.
    result = minimize(
        objective_function,
        w_initial,
        args=(S_xa, S_xb, S_ii_list, lambda_list, theta_r),
        method="L-BFGS-B",
    )
    return result
