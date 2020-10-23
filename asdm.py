import numpy as np
import scipy

__pinv_rcond__ = 1e-8

def asdm_encode(u, dt, b, d, k=1.0, dte=0.0, y=0.0, interval=0.0,
                sgn=1, quad_method='trapz', full_output=False):
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
            return np.array((), np.float), dt, b, d, k, dte, y, interval, sgn, \
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
        compute_y = lambda y, sgn, i: y + dt*(sgn*b+u[i])/k
        last = Nu
    elif quad_method == 'trapz':
        compute_y = lambda y, sgn, i : y + dt*(sgn*b+(u[i]+u[i+1])/2.0)/k
        last = Nu-1
    else:
        raise ValueError('unrecognized quadrature method')

    ys = []
    zs = []
        
    for i in range(last):
        y = compute_y(y, sgn, i)
        interval += dt
        zs.append(sgn*b)
        ys.append(y)
        if np.abs(y) >= d:
            s.append(interval)
            interval = 0.0
            y = d*sgn
            sgn = -sgn

    if full_output:
        return np.array(s), dt, b, d, k, dte, y, interval, sgn, \
               quad_method, full_output
    else:
        return np.array(s), np.array(ys), np.array(zs)
    
def asdm_decode(s, dur, dt, bw, b, d, k=1.0, sgn=-1):
    """
    ASDM time decoding machine.

    Decode a signal encoded with an Asynchronous Sigma-Delta Modulator.

    Parameters
    ----------
    s : ndarray of floats
        Encoded signal. The values represent the time between spikes (in s).
    dur : float
        Duration of signal (in s).
    dt : float
        Sampling resolution of original signal; the sampling frequency
        is 1/dt Hz.
    bw : float
        Signal bandwidth (in rad/s).
    b : float
        Encoder bias.
    d : float
        Encoder threshold.
    k : float
        Encoder integrator constant.
    sgn : {-1, 1}
        Sign of first spike.

    Returns
    -------
    u_rec : ndarray of floats
        Recovered signal.
    """

    Ns = len(s)
    if Ns < 2:
        raise ValueError('s must contain at least 2 elements')

    # Cast s to an ndarray to permit ndarray operations:
    s = np.asarray(s)

    # Compute the spike times:
    ts = np.cumsum(s)

    # Compute the midpoints between spike times:
    tsh = (ts[0:-1]+ts[1:])/2
    Nsh = len(tsh)

    bwpi = bw/np.pi

    # Compute G matrix:
    G = np.empty((Nsh, Nsh), np.float)
    for j in range(Nsh):

        # Compute the values for all of the sincs so that they do not
        # need to each be recomputed when determining the integrals
        # between spike times:
        temp = scipy.special.sici(bw*(ts-tsh[j]))[0]/np.pi
        G[:, j] = temp[1:]-temp[:-1]
    G_inv = np.linalg.pinv(G, __pinv_rcond__)

    # Compute quanta:
    if sgn == -1:
        q = (-1)**np.arange(1, Nsh+1)*(2*k*d-b*s[1:])
    else:
        q = (-1)**np.arange(0, Nsh)*(2*k*d-b*s[1:])

    # Reconstruct signal by adding up the weighted sinc functions. The
    # weighted sinc functions are computed on the fly here to save
    # memory:
    t = np.arange(0, dur, dt)
    u_rec = np.zeros(len(t), np.float)
    c = np.dot(G_inv, q)
    for i in range(Nsh):
        u_rec += np.sinc(bwpi*(t-tsh[i]))*bwpi*c[i]
    return u_rec