import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def background_fit_2(nu, sigma, tau, pi=np.pi):
    k1 = ((4 * sigma ** 2 * tau) /
          (1 + (2 * pi * nu * tau) ** 2 +
           (2 * pi * nu * tau) ** 4))
    return k1


def background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n):
    k1 = background_fit_2(nu, sigma_0, tau_0)
    k2 = background_fit_2(nu, sigma_1, tau_1)
    return P_n + k1 + k2


def scipy_optimizer(freq_filt, powerden_filt, z0, data_weights=None, plot_cb=None):
    def logbackground_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n):
        r = background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n)
        inval = r <= 0
        r[inval] = 1
        lr = np.log10(r)
        lr[inval] = -10000
        return lr

    popt, pcov = curve_fit(logbackground_fit, freq_filt,
                           np.log10(powerden_filt), p0=z0)
    print('curve_fit returned parameters:', display_params(popt))
    return popt


def display_params(params):
    if len(params) == 2:
        variable_names = [
            '\N{GREEK SMALL LETTER SIGMA}',
            '\N{GREEK SMALL LETTER TAU}',
        ]
    else:
        variable_names = [
            '\N{GREEK SMALL LETTER SIGMA}\N{SUBSCRIPT ZERO}',
            '\N{GREEK SMALL LETTER TAU}\N{SUBSCRIPT ZERO}',
            '\N{GREEK SMALL LETTER SIGMA}\N{SUBSCRIPT ONE}',
            '\N{GREEK SMALL LETTER TAU}\N{SUBSCRIPT ONE}',
            'P_n',
        ]
    kvs = zip(variable_names, params)
    return ' '.join('%s=%+.3e' % kv for kv in kvs)


def tensorflow_optimizer(freq_data, powerden_data, z0, data_weights=None, learning_rate=1e-4, epochs=2000, batch_size=None, plot_cb=None):
    if data_weights is None:
        data_weights = np.ones(len(powerden_data), dtype=np.float32)
    tau_limit = 1e-6
    with tf.Graph().as_default():
        freq = tf.placeholder(tf.float32, (None,), 'freq')
        powerden = tf.placeholder(tf.float32, (None,), 'powerden')
        weights = tf.placeholder(tf.float32, (None,), 'weights')
        total_weight = tf.reduce_sum(weights)
        sigma_0 = tf.Variable(tf.constant(z0[0], tf.float32))
        tau_0 = tf.Variable(tf.constant(z0[1], tf.float32))
        sigma_1 = tf.Variable(tf.constant(z0[2], tf.float32))
        tau_1 = tf.Variable(tf.constant(z0[3], tf.float32))
        # initial_P_n = np.exp(np.mean(np.log(powerden_data)))
        P_n = tf.Variable(tf.constant(z0[4], tf.float32))
        # Pass max(tau_limit/2, tau) into background_fit to avoid nan
        bgfit = background_fit(
            freq, sigma_0, tf.maximum(tau_limit/2, tau_0),
            sigma_1, tf.maximum(tau_limit/2, tau_1), P_n)
        # Note we use the natural log here on both data and model
        # (but this is just for minimization; plotting still uses log10).
        log_bgfit = tf.log(bgfit)
        log_powerden = tf.log(powerden)

        # Minimize weighted distance squared
        #error = tf.reduce_sum(weights *
        #                      (log_bgfit - log_powerden) ** 2) / total_weight
        # Minimize weighted absolute distance
        error = tf.reduce_sum(weights *
                              tf.abs(log_bgfit - log_powerden)) / total_weight

        # Regularization: Don't let tau be too close to 0.
        tau_penalty_factor = 1e6
        tau_penalty = (
            tau_penalty_factor *
            (tf.maximum(0.0, tau_limit - tau_0) +
             tf.maximum(0.0, tau_limit - tau_1)))

        minimization_goal = error + tau_penalty
        minimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = minimizer.minimize(minimization_goal)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            t1 = time.time()
            for epoch in range(epochs):
                if batch_size is None:
                    # Old-fashioned batched gradient descent
                    data = {freq: freq_data,
                            powerden: powerden_data,
                            weights: data_weights}
                    session.run(train_step, feed_dict=data)
                else:
                    # Mini-batch gradient descent
                    perm = np.random.permutation(len(freq_data))
                    freq_perm = freq_data[perm]
                    powerden_perm = powerden_data[perm]
                    weights_perm = data_weights[perm]
                    for i in range(0, len(freq_data), batch_size):
                        j = min(i + batch_size, len(freq_data))
                        data = {freq: freq_perm[i:j],
                                powerden: powerden_perm[i:j],
                                weights: weights_perm[i:j]}
                        session.run(train_step, feed_dict=data)
                data = {freq: freq_data,
                        powerden: powerden_data,
                        weights: data_weights}
                err = session.run(error, feed_dict=data)
                params = session.run([sigma_0, tau_0, sigma_1, tau_1, P_n])
                if plot_cb:
                    plot_cb(freq_data, powerden_data,
                            lambda x: session.run(bgfit, feed_dict={freq: x}),
                            epoch, err, params)
                t2 = time.time()
                time_per_epoch = (t2 - t1) / (epoch + 1)
                if epoch % 100 == 0 or epoch + 1 == epochs:
                    print('[%4d] t=%5.2f err=%.3e %s' %
                          (epoch, time_per_epoch, err, display_params(params)))
                if not np.all(np.isfinite(params)):
                    raise Exception("Non-finite parameter")
            return params


def running_median(freq, powerden, weights=None, bin_size=None, bins=None):
    if bin_size is not None and bins is not None:
        raise TypeError('cannot specify both bin_size and bins')
    freq = np.squeeze(freq)
    powerden = np.squeeze(powerden)
    n, = freq.shape
    n_, = powerden.shape
    assert n == n_

    if weights is None:
        weights = np.ones(n, dtype=np.float32)

    # Sort data by frequency
    sort_ind = np.argsort(freq)
    freq = freq[sort_ind]
    powerden = powerden[sort_ind]
    weights = weights[sort_ind]

    # Compute log of frequencies
    log_freq = np.log10(freq)
    # Compute bin_size
    if bin_size is None:
        if bins is None:
            bins = 10000
        df = np.diff(log_freq)
        d = np.median(df)
        close = df < 100*d
        span = np.sum(df[close])
        bin_size = span / bins
    bin_index = np.floor((log_freq - log_freq[0]) / bin_size)
    internal_boundary = (bin_index[1:] != bin_index[:-1]).nonzero()[0]
    boundary = [0] + internal_boundary.tolist() + [n]

    bin_freq = []
    bin_pden = []
    bin_weight = []
    for i, j in zip(boundary[:-1], boundary[1:]):
        bin_freq.append(np.mean(freq[i:j]))
        bin_pden.append(np.median(powerden[i:j]))
        bin_weight.append(np.sum(weights[i:j]))
    return np.array(bin_freq), np.array(bin_pden), np.array(bin_weight)


def plotter(freq, powerden, npoints=10000):
    fmin, fmax = np.min(freq), np.max(freq)
    xs = np.linspace(fmin, fmax, npoints)
    ys = np.zeros_like(xs) + np.min(powerden)
    ind = np.random.permutation(len(freq))[:npoints]

    plt.ion()
    fig = plt.figure()  # type: plt.Figure
    ax = fig.add_subplot(111)  # type: plt.Axes
    data_line, = ax.plot(freq[ind], np.log10(powerden[ind]), ',')  # type: plt.Line2D
    model_line, = ax.plot(xs, np.log10(ys))  # type: plt.Line2D
    plt.pause(1e-3)

    def plot_cb(freq, powerden, bgfit, epoch, e, params):
        if epoch % 100 == 0:
            model_line.set_ydata(np.log10(bgfit(xs)))
            plt.pause(1e-3)

    return plot_cb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nox', action='store_false')
    args = parser.parse_args()
    filename = 'powerdensity_q1_s4_k801.npz'
    print("Loading %s..." % filename)
    freq, powerden = np.load(filename)['data'].T
    log_freq_diff = np.diff(np.log(freq))
    weights = np.concatenate((log_freq_diff, [log_freq_diff[-1]]))

    params_filename = 'data.npz'
    print("Loading initial params from %s..." % params_filename)
    initial_params = np.load(params_filename)['arr_2']
    initial_params[0] *= 0.8
    initial_params[2] *= 1.2
    initial_params = np.concatenate(
        (initial_params, [0.2]))
    print('Shape of freq:', freq.shape)
    print('Shape of powerden:', powerden.shape)
    freq0, powerden0, weights0 = running_median(freq, powerden, weights,
                                                bins=100)
    freq1, powerden1, weights1 = running_median(freq, powerden, weights,
                                                bins=1000)
    freq2, powerden2, weights2 = running_median(freq, powerden, weights,
                                                bins=10000)
    print('Shape of freq:', freq1.shape)
    print('Shape of powerden:', powerden1.shape)
    print('Shape of weights:', weights1.shape)
    print('Initial parameters:', display_params(initial_params))
    popt = scipy_optimizer(freq1, powerden1, initial_params, weights1,
                           plot_cb=None)
    plt.plot(freq2, powerden2, ',')
    plt.plot(freq1, powerden1, '.')
    plt.plot(freq0, powerden0)
    plt.plot(freq1, background_fit(freq1, *popt))
    plt.loglog()
    plt.savefig('test.png')
    print('Shape of freq:', freq2.shape)
    print('Shape of powerden:', powerden2.shape)
    print('Shape of weights:', weights2.shape)
    popt2 = tensorflow_optimizer(freq2, powerden2, popt, weights2,
                                 plot_cb=plotter(freq, powerden) if args.nox else None)
    print(popt2)
    popt3 = tensorflow_optimizer(freq, powerden, popt2,
                                 plot_cb=plotter(freq, powerden) if args.nox else None)
    print(popt3)


if __name__ == '__main__':
    main()
