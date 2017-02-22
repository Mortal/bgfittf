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
    print('The best parameters are', popt)


def tensorflow_optimizer(freq_filt, powerden_filt, z0, learning_rate=1e-4, epochs=100):
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
        error = tf.nn.l2_loss(log_bgfit - log_powerden)
        minimizer = tf.train.AdamOptimizer(learning_rate)
        train_step = minimizer.minimize(error)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                data = {freq: freq_filt,
                        powerden: powerden_filt}
                _, e = session.run([train_step, error], feed_dict=data)
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
    z0 = data['arr_2']
    print('Shape of freq:', freq_filt.shape)
    print('Shape of powerden:', powerden_filt.shape)
    print('Initial parameters:', z0)
    tensorflow_optimizer(freq_filt, powerden_filt, z0)


if __name__ == '__main__':
    main()
