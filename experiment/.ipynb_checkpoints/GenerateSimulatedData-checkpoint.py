from pyesg import CoxIngersollRossProcess, GeometricBrownianMotion
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import warnings
warnings.simplefilter('error')


def make_simulated_data(n_scenarios, n_steps, true_vol_std, topK, init_price, u_dim, random_state, num_test=25, num_valid=2):
    bday_p_year = 252

    generator = CoxIngersollRossProcess(mu=0.15, sigma=0.04, theta=10)
    x0 = 0.15  # the start value of our process
    dt = 1 / bday_p_year  # the length of each time step in years
    random_state = 1234  # optional random_state for reproducibility

    # Generate risk free interest rate
    # rs
    r_generator = GeometricBrownianMotion(mu=-0.05, sigma=0.01)
    rs = r_generator.scenarios(0.22, dt, 1, n_steps, random_state)

    # Generate true volatility
    # true_vol_mean
    true_vol_mean = generator.scenarios(x0, dt, n_scenarios, n_steps, random_state)
    n_step, n_timesteps = true_vol_mean.shape
    delta = np.random.normal(0, true_vol_std, (n_scenarios, n_timesteps, topK))
    gen_vol_obs = np.repeat(true_vol_mean[:, :, np.newaxis], topK, axis=2) + delta

    # Generate observed return
    # us
    us = np.random.normal(0, true_vol_mean / np.sqrt(bday_p_year), (n_scenarios, n_timesteps))

    # Generate observed price
    # ps
    ps = np.ones([n_scenarios, 1]) * init_price
    for i in range(1, n_timesteps):
        new_p = ps[:, -1] * (1 + us[:, i])
        ps = np.concatenate((ps, new_p.reshape([-1, 1])), axis=1)

    # Generate some expiration Dates
    # Ts # to 2 years
    mesh = 0.5 / (topK + 1)
    Ts = np.array(range(1, topK + 1)) * mesh

    # Generate some Strike Price
    # ps
    Ks = ps[:, :, np.newaxis].repeat(topK, axis=2) + np.random.normal(0, 100 * gen_vol_obs * Ts,
                                                                      (n_scenarios, n_timesteps, topK) )

    # Generate some Option Prices
    # ys
    ps_repeat = ps[:, :, np.newaxis].repeat(topK, axis=2)
    rs_repeat = rs.repeat(n_scenarios, axis=0)[:, :, np.newaxis].repeat(topK, axis=2)
    Ts_repeat = Ts[np.newaxis, :].repeat(n_timesteps, axis=0)[np.newaxis, :, :].repeat(n_scenarios, axis=0)

    gen_vol_var = np.power(gen_vol_obs, 2)
    dividor = np.sqrt(gen_vol_var * Ts)
    d_pls = (np.log(ps_repeat / Ks) + (rs_repeat + gen_vol_var / 2) * Ts_repeat) / dividor
    d_mns = (np.log(ps_repeat / Ks) + (rs_repeat - gen_vol_var / 2) * Ts_repeat) / dividor
    ys = np.multiply(ps_repeat, norm.cdf(d_pls)) - np.multiply(np.multiply(Ks, np.exp(-rs_repeat * Ts_repeat)),
                                                               norm.cdf(d_mns))

    sns.set()
    figure = plt.figure(figsize=(12.5, 2.5))
    for _ in true_vol_mean:
        plt.plot(range(n_steps + 1), _, linewidth=1)
        plt.title("True Volatility Mean")
#     figure.show()

    figure = plt.figure(figsize=(12.5, 2.5))
    for _ in us:
        plt.plot(range(n_steps + 1), _, linewidth=0.5)
        plt.title("Observed returns")
#     figure.show()

    figure = plt.figure(figsize=(12.5, 2.5))
    for _ in ps:
        plt.plot(range(n_steps + 1), _, linewidth=1)
        plt.title("Observed Stock Price")
#     figure.show()

    print("Shape of true_vol_mean is {}".format(true_vol_mean.shape))
    print("Shape of gen_vol_obs is {}".format(gen_vol_obs.shape))
    print("Shape of us is {}".format(us.shape))
    print("Shape of ps is {}".format(ps.shape))
    print("Shape of rs is {}".format(rs.shape))
    print("Shape of Ts is {}".format(Ts.shape))
    print("Shape of Ks is {}".format(Ks.shape))
    print("Shape of ys is {}".format(ys.shape))
    print("smallest strike price is {:.2f}".format(np.min(Ks)))

    us_ccd = us[:, 0:u_dim]
    us_ccd = us_ccd[:, np.newaxis, :]
    for i in range(u_dim, n_timesteps):
        u_ccd = us[:, i-u_dim:i]
        u_ccd = u_ccd[:, np.newaxis, :]
        us_ccd = np.concatenate((us_ccd, u_ccd), axis=1)

    ps = ps[:, u_dim-1:]
    rs = rs[:, u_dim-1:]
    Ks = Ks[:, u_dim-1:, :]
    ys = ys[:, u_dim-1:, :]
    true_vol_mean = true_vol_mean[:, u_dim-1:]
    gen_vol_obs = gen_vol_obs[:, u_dim-1, :]

    print("[=======100%=======] After Truncation")
    print("Shape of true_vol_mean is {}".format(true_vol_mean.shape))
    print("Shape of gen_vol_obs is {}".format(gen_vol_obs.shape))
    print("Shape of us_ccd is {}".format(us_ccd.shape))
    print("Shape of ps is {}".format(ps.shape))
    print("Shape of rs is {}".format(rs.shape))
    print("Shape of Ts is {}".format(Ts.shape))
    print("Shape of Ks is {}".format(Ks.shape))
    print("Shape of ys is {}".format(ys.shape))

    n_scenarios, n_timesteps, topK = Ks.shape

    train = []
    valid = []
    test = []
    for idx in range(n_scenarios):
        train_data = {}
        valid_data = {}
        test_data = {}
        for i in range(n_timesteps-num_test-num_valid):
            train_data[i] = {
                            "return": us_ccd[idx, i],
                            "price": ps[idx, i],
                            "Strike": Ks[idx, i],
                            "Exercise Time": Ts_repeat[idx, i],
                            "risk ir": rs[0, i],
                            "option price": ys[idx, i],
                            "true vol": true_vol_mean[idx, i],
                            }

        for i in range(n_timesteps-num_test-num_valid, n_timesteps-num_test):
            valid_data[i] = {
                            "return": us_ccd[idx, i],
                            "price": ps[idx, i],
                            "Strike": Ks[idx, i],
                            "Exercise Time": Ts_repeat[idx, i],
                            "risk ir": rs[0, i],
                            "option price": ys[idx, i],
                            "true vol": true_vol_mean[idx, i],
                            }

        for i in range(n_timesteps-num_test, n_timesteps):
            test_data[i] = {
                            "return": us_ccd[idx, i],
                            "price": ps[idx, i],
                            "Strike": Ks[idx, i],
                            "Exercise Time": Ts_repeat[idx, i],
                            "risk ir": rs[0, i],
                            "option price": ys[idx, i],
                            "true vol": true_vol_mean[idx, i],
                            }
        train.append(train_data)
        valid.append(valid_data)
        test.append(test_data)

    return train, valid, test, true_vol_mean
