import numpy as np
import scipy
import helpers

def IAF_encode(u, dt, alpha, theta, dte=0.0, quad_method='rect'):
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
      
    y = 0
    for i in range(last):
        y = compute_y(y, i)
        ys.append(y)
        if np.abs(y) >= theta:
            s.append(i)
            q_signs.append(np.sign(y))
            y = 0
            
    return np.array(s), np.array(ys), np.array(q_signs)
    
def IAF_decode(z, q_signs, t, alpha, theta, psi_kernel, phi_kernel=None):
    """
    Integrate-And-Fire decoder.

    Decode a signal encoded with a Integrate-And-Fire sampler.

    Parameters
    ----------
    z : array_like of floats
        Spike indices.
    q_signs : array_like of floats
        Signs of spikes.
    t : array_like of float
        Sampling times.
    alpha : float
        Firing parameter.
    theta : float
        Encoder threshold.
    psi_kernel : array_like of floats
        Signal of the PSI kernel used for decoding. Centered around zero (TO BE REVIEWED)

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.
    """
    scaled_qs = q_signs * theta

    # Wis
    # To be optimized
    ws = []
    ws.append(0)
    for idx, value in enumerate(scaled_qs[1:]):
        w = np.exp(alpha * (t[int(z[idx])] - t[int(z[idx+1])])) * ws[idx] + value
        ws.append(w)
    
    # Sk
    # To be optimized
    sk = []
    nb = 0
    for idx, value in enumerate(t):
        if(nb == 0):
            s = 0
        else:
            s = np.exp(alpha * (t[int(z[nb])] - value)) * ws[nb]

        if(nb+1 < z.shape[0]):
            if(idx == int(z[nb+1])):
                nb = nb+1

        sk.append(s)
    
    # Applying the right filter
    # To be optimized
    if(phi_kernel is None):
        dt = t[1] - t[0]
        N = len(t)
        phi_kernel = helpers.get_phi_from_psi(psi_kernel, N, dt, alpha)
    
    convolved = np.convolve(phi_kernel, sk)
    filtered = convolved[int(len(t)/2): -int(len(t)/2)+1]
    
    return filtered, sk
    