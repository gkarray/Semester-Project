import numpy as np
import scipy

def iaf_encode(u, dt, alpha, d, dte=0.0, y=0.0, quad_method='trapz', full_output=False):
    """
    ASDM time encoding machine.

    Encode a finite length signal using an Asynchronous Sigma-Delta
    Modulator.

    Parameters
    ----------
    u : array_like of floats
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    k : float
        Encoder integration constant.
    dte : float
        Sampling resolution assumed by the encoder (s).
        This may not exceed `dt`.
    y : float
        Initial value of integrator.
    interval : float
        Time since last spike (in s).
    sgn : {+1, -1}
        Sign of integrator.
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).
    full_output : bool
        If set, the function returns the encoded data block followed
        by the given parameters (with updated values for `y`, `interval`, and
        `sgn`). This is useful when the function is called repeatedly to
        encode a long signal.

    Returns
    -------
    s : ndarray of floats
        If `full_output` == False, returns the signal encoded as an
        array of time intervals between spikes.
    s, dt, b, d, k, dte, y, interval, sgn, quad_method, full_output : tuple
        If `full_output` == True, returns the encoded signal
        followed by updated encoder parameters.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.
    """

    Nu = len(u)
    if Nu == 0:
        if full_output:
            return np.array((), np.float), dt, alpha, d, dte, y, \
               quad_method, full_output
        else:
            return np.array((), np.float)

    # Check whether the encoding resolution is finer than that of the
    # original sampled signal:
    if dte > dt:
        raise ValueError('encoding time resolution must not exceeed original signal resolution')
    if dte < 0:
        raise ValueError('encoding time resolution must be nonnegative')
    if dte != 0 and dte != dt:

        # Resample signal and adjust signal length accordingly:
        M = int(dt/dte)
        u = scipy.signal.resample(u, len(u)*M)
        Nu *= M
        dt = dte

    # Use a list rather than an array to save the spike intervals
    # because the number of spikes is not fixed:
    s = []

    # Choose integration method and set the number of points over
    # which to integrate the input (see note above). This allows the
    # use of one loop below to perform the integration regardless of
    # the method chosen:
    if quad_method == 'rect':
        compute_y = lambda y, i: y + dt*u[i]*np.exp(alpha*dt)
        last = Nu
    elif quad_method == 'trapz':
        compute_y = lambda y, i : y + dt*np.exp(alpha*dt)*(u[i]+u[i+1])/2.0
        last = Nu-1
    else:
        raise ValueError('unrecognized quadrature method')

    ys = []
    qs = []
        
    for i in range(last):
        y = compute_y(y, i)
        ys.append(y)
        if np.abs(y) >= d:
            s.append(i)
            qs.append(np.sign(y))
            y = 0
            

    if full_output:
        return np.array(s), dt, alpha, d, dte, y, \
               quad_method, full_output
    else:
        return np.array(s), np.array(ys), np.array(qs)