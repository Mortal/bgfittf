import numpy as np
import tensorflow as tf
from scipy.optimize import curve_fit


def background_fit_2(nu, sigma_0, tau_0):
    k1 = ((4 * sigma_0 ** 2 * tau_0) /
          (1 + (2 * np.pi * nu * tau_0) ** 2 +
           (2 * np.pi * nu * tau_0) ** 4))
    return k1


def background_fit(nu, sigma_0, tau_0, sigma_1, tau_1, P_n=0):
    k1 = background_fit_2(nu, sigma_0, tau_0)
    k2 = background_fit_2(nu, sigma_1, tau_1)
    return P_n + k1 + k2


def scipy_optimizer(freq_filt, powerden_filt, z0):
    def logbackground_fit(nu, sigma_0, tau_0, sigma_1, tau_1):
        return np.log10(background_fit(nu, sigma_0, tau_0, sigma_1, tau_1))

    popt, pcov = curve_fit(logbackground_fit, freq_filt,
                           np.log10(powerden_filt), p0=z0)
    print('curve_fit returned parameters:', popt)
    return popt


def tensorflow_optimizer(freq_filt, powerden_filt, z0, learning_rate=2e-4, epochs=1000, batch_size=1024):
    with tf.Graph().as_default():
        freq = tf.placeholder(tf.float32, (None,), 'freq')
        powerden = tf.placeholder(tf.float32, (None,), 'powerden')
        sigma_0 = tf.Variable(tf.constant(z0[0], tf.float32))
        tau_0 = tf.Variable(tf.constant(z0[1], tf.float32))
        sigma_1 = tf.Variable(tf.constant(z0[2], tf.float32))
        tau_1 = tf.Variable(tf.constant(z0[3], tf.float32))
        bgfit = background_fit(freq, sigma_0, tau_0, sigma_1, tau_1)
        log_bgfit = tf.log(bgfit)
        log_powerden = tf.log(powerden)

        # Minimize distance squared
        error = tf.reduce_mean((log_bgfit - log_powerden) ** 2)

        # Regularization: Don't let tau be too close to 0.
        tau_penalty_factor = 1e6
        tau_penalty_limit = 0.5e-3
        tau_penalty = (
            tau_penalty_factor *
            (tf.maximum(0.0, tau_penalty_limit - tau_0) +
             tf.maximum(0.0, tau_penalty_limit - tau_1)))

        minimization_goal = error + tau_penalty
        minimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = minimizer.minimize(minimization_goal)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                perm = np.random.permutation(len(freq_filt))
                freq_perm = freq_filt[perm]
                powerden_perm = powerden_filt[perm]
                for i in range(0, len(freq_filt), batch_size):
                    j = min(i + batch_size, len(freq_filt))
                    data = {freq: freq_perm[i:j],
                            powerden: powerden_perm[i:j]}
                    session.run(train_step, feed_dict=data)
                data = {freq: freq_filt,
                        powerden: powerden_filt}
                e = session.run(error, feed_dict=data)
                params = session.run([sigma_0, tau_0, sigma_1, tau_1])
                print('[%4d] err=%.3e params=%s' % (epoch, e, params))
                if not np.all(np.isfinite(params)):
                    raise Exception("Non-finite parameter")
            return params


def main():
    filename = 'data.npz'
    print("Loading %s..." % filename)
    data = np.load(filename)
    freq_filt = data['arr_0']
    powerden_filt = data['arr_1']
    initial_params = data['arr_2']
    initial_params[0] *= 0.8
    initial_params[2] *= 1.2
    print('Shape of freq:', freq_filt.shape)
    print('Shape of powerden:', powerden_filt.shape)
    print('Initial parameters:', initial_params)
    tensorflow_optimizer(freq_filt, powerden_filt, initial_params)


if __name__ == '__main__':
    main()
