import autograd.numpy as a_np
import autograd.scipy.stats as a_ss
from filterpy.kalman import MerweScaledSigmaPoints
from SigmaMaker import a_MerweScaledSigmaPoints
from math import log, pi


def expectation_term_one(dist_mean_t, dist_cov_t):
    """
    E[ theta_t^T  theta_t | X]

    :param dist_mean_t: E[theta_t | X]
    :param dist_cov_t: Cov[theta_t | X]
    :return: expectation of the first term: E[ theta_t^T theta_t | X]
    """
    return a_np.trace(dist_cov_t) + a_np.matmul(dist_mean_t.T, dist_mean_t)


def expectation_term_two(params,
                         dist_mean_t, dist_mean_t_1,
                         dist_cov_t, dist_cov_t_1, dist_cross_t_t_1, u_quad_t):
    """
    E[theta_t^TW^{-1} Phi(theta_{t-1}) | X]

    :param params:              [G, G_in, b, w, v]
    :param dist_mean_t:         E[theta_t, theta_{t-1} | X]
    :param dist_mean_t_1:       E[theta_t, theta_{t-1} | X]
    :param dist_cov_t:          Cov[theta_t | X]
    :param dist_cov_t_1:        Cov[theta_{t-1} | X]
    :param dist_cross_t_t_1:    P_{t,t-1}^*
    :param u_quad_t:            u_t^2
    :return: expectation of the second term: E[theta_t^T Phi(theta_{t-1}) | X]
    """
    G, G_in, b, w, v = params
    # Linear transform
    L = dist_mean_t.shape[0]
    trans_mean_t_1 = a_np.matmul(G, dist_mean_t_1) + a_np.matmul(G_in, u_quad_t) + b  # (L, 1)
    trans_mean = a_np.concatenate((dist_mean_t, trans_mean_t_1), axis=0)  # (2L, 1)

    trans_cross_t_t_1 = a_np.matmul(dist_cross_t_t_1, G.T)  # (L, L)
    trans_cov_t_1 = a_np.matmul(G, a_np.matmul(dist_cov_t_1, G.T))  # (L, L)

    first_line = a_np.concatenate((dist_cov_t, trans_cross_t_t_1), axis=1)  # (L, 2L)
    second_line = a_np.concatenate((trans_cross_t_t_1.T, trans_cov_t_1), axis=1)  # (L, 2L)
    trans_cov = a_np.concatenate((first_line, second_line), axis=0)  # (2L, 2L)

    # sigmoid transform
    dim = 2 * L
    SigmaMaker = a_MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=0)
    sigmas = SigmaMaker.sigma_points(trans_mean, trans_cov)  # (4L+1, 2L)
    num_sigmas = SigmaMaker.num_sigmas()
    Wm, Wc = SigmaMaker.Wm, SigmaMaker.Wc  # (4L+1, ) (4L+1, )

    def quasi_sigmoid(merged_vec, separate_dim_L):
        """
        :param merged_vec: a merged vector (2L, 1)
        :param separate_dim_L: size of separate vector L
        :return: quasi sigmoid transformed vector (2L, 1)
        """
        upper = merged_vec[:separate_dim_L]
        lower = merged_vec[L:]
        transformed_lower = 1 / (1 + a_np.exp(lower))
        return a_np.concatenate((upper, transformed_lower), axis=0)

    final_mean = a_np.zeros([dim, ])  # (2L, 1)
    for i in range(num_sigmas):
        final_mean += Wm[i] * quasi_sigmoid(sigmas[i, :], L)

    final_cov = a_np.zeros([dim, dim])  # (2L, 2L)
    for i in range(num_sigmas):
        diff = quasi_sigmoid(sigmas[i, :], L) - final_mean
        final_cov += Wc[i] * a_np.matmul(diff, diff.T)

    final_mean_upper = final_mean[:L]
    final_mean_lower = final_mean[L:]
    final_cross_cov = final_cov[:L, L:]
    return a_np.trace(final_cross_cov.T) + a_np.matmul(final_mean_upper.T, final_mean_lower)


def expectation_term_three(params, dist_mean_t_1, dist_cov_t_1, u_quad_t):
    """
    E[ Phi(theta_{t-1})^T Phi(theta_{t-1}) | X]

    :param params: [G, G_in, b, w, v]
    :param dist_mean_t_1: E[theta_{t-1} | X]
    :param dist_cov_t_1: Cov[theta_{t-1} | X]
    :param u_quad_t: u_t^2
    :return: expectation of the third term: E[ Phi(theta_{t-1})^T Phi(theta_{t-1}) | X]
    """
    G, G_in, b, w, v = params

    # First make some sigma points
    dim = dist_mean_t_1.shape[0]
    SigmaMaker = MerweScaledSigmaPoints(dim, alpha=.1, beta=2., kappa=0)
    sigmas = SigmaMaker.sigma_points(dist_mean_t_1, dist_cov_t_1)  # (2L+1, L)
    num_sigmas = sigmas.shape[0]
    Wm, Wc = SigmaMaker.Wm, SigmaMaker.Wc  # (2L+1, )  (2L+1, )
    linear_transform = a_np.matmul(G, sigmas.T) + \
                        a_np.matmul(G_in, u_quad_t.reshape([-1, 1])) + \
                        b.reshape([-1, 1])  # (L, 2L+1)
    sigmoid_transform = 1 / (1 + a_np.exp(linear_transform))  # (L, 2L+1)
    trans_mean = a_np.sum(a_np.multiply(Wm, sigmoid_transform), axis=1)  # (L, )
    trans_cov = a_np.zeros([dim, dim])  # (L, L)
    for i in range(num_sigmas):
        diff = (sigmoid_transform[:, i] - trans_mean).reshape([-1, 1])
        delta = Wc[i] * a_np.matmul(diff, diff.T)
        trans_cov += delta

    return a_np.trace(trans_cov) + a_np.matmul(trans_mean.T, trans_mean)


def expectation_term_four(dist_mean_t, dist_cov_t, p_t, r_t, K_t, T_t):
    """
    This term doesn't involve the calculation of any parameter, but the numerical result will be used later
    E[Psi(theta_t) | X]

    :param dist_mean_t:
    :param dist_cov_t:
    :param p_t: current asset price
    :param r_t: current risk free interest rate
    :param K_t: strike price, 1d vector # (N, )
    :param T_t: maturity time, 1d vector # (N, )
    :return: expectation of the fourth term: E[Psi(theta_t) | X]
    """
    K_t = K_t.reshape([1, -1])  # (1, N)
    T_t = T_t.reshape([1, -1])  # (1, N)

    # First make some sigma points
    dim = dist_mean_t.shape[0]
    average_vector = a_np.ones([1, dim]) / dim
    mean_vol_std_t = a_np.average(dist_mean_t)
    var_vol_std_t = a_np.matmul(a_np.matmul(average_vector, dist_cov_t), average_vector.T)

    SigmaMaker = MerweScaledSigmaPoints(1, alpha=.1, beta=2., kappa=0)
    sigmas = SigmaMaker.sigma_points(mean_vol_std_t, var_vol_std_t)  # (3, 1)
    Wm, Wc = SigmaMaker.Wm, SigmaMaker.Wc  # (3, )  (3, )

    # Black Scholes parallel computing
    divider = a_np.sqrt(a_np.matmul(a_np.power(sigmas, 2), T_t))  # (3, topK)
    d_pls = (a_np.log(p_t / K_t) + a_np.matmul(r_t + a_np.power(sigmas, 2) / 2, T_t)) / divider  # (3, topK)
    d_mns = (a_np.log(p_t / K_t) + a_np.matmul(r_t - a_np.power(sigmas, 2) / 2, T_t)) / divider  # (3, topK)
    first_term = a_np.multiply(p_t, a_ss.norm.cdf(d_pls))  # (3, topK)
    second_term = a_np.multiply(a_np.multiply(K_t, a_np.exp(-r_t * T_t)), a_ss.norm.cdf(d_mns))  # (3, topK)
    BS_transform = first_term - second_term  # (3, topK)

    trans_mean = a_np.dot(Wm, BS_transform)  # (topK, )

    return trans_mean


def expectation_term_five(dist_mean_t, dist_cov_t, p_t, r_t, K_t, T_t):
    """
    This term doesn't involve the calculation of any parameter, but the numerical result will be used later
    E[Psi(theta_t)^2 | X]

    :param dist_mean_t:
    :param dist_cov_t:
    :param p_t: current asset price
    :param r_t: current risk free interest rate
    :param K_t: strike price, 1d vector # (N, )
    :param T_t: maturity time, 1d vector # (N, )
    :return: expectation of the fifth term: E[Psi(theta_t)^2 | X]
    """
    topK = K_t.shape[0]
    K_t = K_t.reshape([1, -1])  # (1, N)
    T_t = T_t.reshape([1, -1])  # (1, N)

    # First make some sigma points
    dim = dist_mean_t.shape[0]
    average_vector = a_np.ones([1, dim]) / dim
    mean_vol_std_t = a_np.average(dist_mean_t)
    var_vol_std_t = a_np.matmul(a_np.matmul(average_vector, dist_cov_t), average_vector.T)

    SigmaMaker = MerweScaledSigmaPoints(1, alpha=.1, beta=2., kappa=0)
    sigmas = SigmaMaker.sigma_points(mean_vol_std_t, var_vol_std_t)  # (3, 1)
    Wm, Wc = SigmaMaker.Wm, SigmaMaker.Wc  # (3, )  (3, )

    # Black Scholes parallel computing
    divider = a_np.sqrt(a_np.matmul(a_np.power(sigmas, 2), T_t))  # (3, topK)
    d_pls = (a_np.log(p_t / K_t) + a_np.matmul(r_t + a_np.power(sigmas, 2) / 2, T_t)) / divider  # (3, topK)
    d_mns = (a_np.log(p_t / K_t) + a_np.matmul(r_t - a_np.power(sigmas, 2) / 2, T_t)) / divider  # (3, topK)
    first_term = a_np.multiply(p_t, a_ss.norm.cdf(d_pls))  # (3, topK)
    second_term = a_np.multiply(a_np.multiply(K_t, a_np.exp(-r_t * T_t)), a_ss.norm.cdf(d_mns))  # (3, topK)
    BS_transform = first_term - second_term  # (3, topK)

    trans_mean = a_np.dot(Wm, BS_transform)  # (topK, )
    trans_cov = a_np.zeros([topK, topK])  # (topK, topK) diagonal
    for i in range(3):
        diff = (BS_transform[i, :] - trans_mean).reshape([-1, 1])
        delta = Wc[i] * a_np.matmul(diff, diff.T)
        trans_cov += delta

    return a_np.sum(a_np.power(trans_mean, 2)) + a_np.trace(trans_cov)


def em_one(params,
           dist_mean_t, dist_mean_t_1,  # Distributional Mean input
           dist_cov_t, dist_cov_t_1, dist_cross_t_t_1,  # Distributional Covariance input
           u_quad_t,  # Stock wise input
           p_t, r_t, K_t, T_t, y_t):  # Option wise input
    """
    EM algorithm in one time step

    :param params:              [G, G_in, b, w, v]
    :param dist_mean_t:         E[theta_t | X]
    :param dist_mean_t_1:       E[theta_{t-1} | X]
    :param dist_cov_t:          Cov[theta_t | X]
    :param dist_cov_t_1:        Cov[theta_{t-1} | X]
    :param dist_cross_t_t_1:    P_{t,t-1}^*
    :param u_quad_t:            u_t^2
    :param p_t:                 current asset price
    :param r_t:                 current risk free interest rate
    :param K_t:                 strike price, 1d vector  # (N, )
    :param T_t:                 maturity time, 1d vector # (N, )
    :param y_t:                 option prices, 1d vector # (N, )
    :return: EM loss generated in one time step
    """
    G, G_in, b, w, v = params
    k = G.shape[0]  # dimension of state
    n_t = K_t.shape[0]  # topK of option observations

    # First line: variance part
    variance_part = - k/2 * a_np.log(w) - n_t/2 * a_np.log(v)

    # Second Line: dynamic part
    # Inside bracket components:
    dyn_component_one = expectation_term_one(dist_mean_t, dist_cov_t)

    dyn_component_two = -2 * expectation_term_two(params, dist_mean_t, dist_mean_t_1,
                                                  dist_cov_t, dist_cov_t_1, dist_cross_t_t_1,
                                                  u_quad_t)

    dyn_component_three = expectation_term_three(params, dist_mean_t_1, dist_cov_t_1, u_quad_t)
    dynamic_part = -1/(2*w) * (dyn_component_one + dyn_component_two + dyn_component_three)

    # # Third Line: observation part                    ##### 别忘了
    # # Inside bracket components:
    # obs_component_one = a_np.sum(a_np.power(y_t, 2))
    # obs_component_two = -2 * a_np.sum(a_np.multiply(y_t, expectation_term_four(dist_mean_t, dist_cov_t,
    #                                                                            p_t, r_t, K_t, T_t)))
    # obs_component_three = expectation_term_five(dist_mean_t, dist_cov_t, p_t, r_t, K_t, T_t)
    # observation_part = -1 / (2 * v) * (obs_component_one + obs_component_two + obs_component_three)
    #
    # Fourth Line: constant part (can be neglected)
    constant_part = log(2*pi) * (k+n_t/2)

    return variance_part + dynamic_part + constant_part  # + observation_part


def loss(params,
         dist_mean_lis, dist_cov_lis, dist_cross_lis,
         u_quad_lis,
         p_lis, r_lis, K_lis, T_lis, y_lis,
         num_observation,
         reg=0.1):
    """

    :param params:          [G, G_in, b, w, v]
    :param dist_mean_lis:   E[theta_t | X] for all t
    :param dist_cov_lis:    Cov[theta_t | X] for all t
    :param dist_cross_lis:  P_{t,t-1}^* for all t
    :param u_quad_lis:      u_t^2
    :param p_lis:           current asset price
    :param r_lis:           current risk free interest rate
    :param K_lis:           strike price, 1d vector  # (N, )
    :param T_lis:           maturity time, 1d vector # (N, )
    :param y_lis:           option prices, 1d vector # (N, )
    :param num_observation: number of total observations. For normalization purposes
    :param reg:             regularization parameter for LASSO loss
    :return:
    """
    G, G_in, b, w, v = params

    # EM loss
    total_em_loss = 0
    n = len(p_lis)
    for i in range(n):
        one_step_em = em_one(params,
                             dist_mean_t=dist_mean_lis[i + 1], dist_mean_t_1=dist_mean_lis[i],
                             dist_cov_t=dist_cov_lis[i + 1], dist_cov_t_1=dist_cov_lis[i],
                             dist_cross_t_t_1=dist_cross_lis[i],
                             u_quad_t=u_quad_lis[i],
                             p_t=p_lis[i], r_t=r_lis[i], K_t=K_lis[i], T_t=T_lis[i], y_t=y_lis[i])
        total_em_loss += one_step_em
        
    # LASSO
    lasso_loss = reg * (a_np.sum(a_np.abs(G)) + a_np.sum(a_np.abs(G_in)))
    return -total_em_loss / num_observation + lasso_loss
