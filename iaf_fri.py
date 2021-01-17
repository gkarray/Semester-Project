import numpy as np
import scipy
import helpers

def IAF_encode(u, dt, alpha, theta, t,  dte=0.0, quad_method='rect'):
    """
    Integrate-And-Fire sampler.

    Encode a finite length signal using an Integrate-And-Fire sampler (FREICHTINGER-2009).

    Parameters
    ----------
    u : array_like of floats
        Signal to encode.
    dt : float
        Sampling resolution of input signal; the sampling frequency
        is 1/dt Hz.
    alpha : float
        Firing parameter.
    theta : float
        Threshold.
    dte : float
        Sampling resolution assumed by the encoder (s).
        This may not exceed `dt`.
    quad_method : {'rect', 'trapz'}
        Quadrature method to use (rectangular or trapezoidal).

    Returns
    -------
    s : ndarray of floats
        Indices of spiking times in the time array
    ys : ndarray of floats
        Integral signal.
    q_signs : ndarray of floats
        Signs of spikes.

    Notes
    -----
    When trapezoidal integration is used, the value of the integral
    will not be computed for the very last entry in `u`.
    """

    Nu = len(u)
    if Nu == 0:
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
    ys = []
    q_signs = []
    
    def quad(y, i, tj):
        return y + dt*u[i]
    
    def trapz(y, i):
        return y + dt*(u[i]+u[i+1])/2.0

    # Choose integration method and set the number of points over
    # which to integrate the input (see note above). This allows the
    # use of one loop below to perform the integration regardless of
    # the method chosen:
    if quad_method == 'rect':
        compute_y = quad
        last = Nu
    elif quad_method == 'trapz':
        compute_y = trapz
        last = Nu-1
    else:
        raise ValueError('unrecognized quadrature method')
      
    y = 0
    tj = 0
    for i in range(last):
        y = compute_y(y, i, tj)
        ys.append(y)
        if np.abs(y) >= theta:
            s.append(i)
            tj = t[i]
            q_signs.append(np.sign(y))
            y = 0
            
    return np.array(s), np.array(ys), np.array(q_signs)