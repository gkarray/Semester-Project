import pylab as p
import numpy as np

def plot_encoded(t, u, s, fig_title='', file_name=''):
    """
    Plot a time-encoded signal.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the original signal was sampled.
    u : ndarray of floats
        Signal samples.
    s : ndarray of floats
        Intervals between encoded signal spikes.
    fig_title : string
        Plot title.
    file_name : string
        File in which to save the plot.

    Notes
    -----
    The spike times (i.e., the cumulative sum of the interspike
    intervals) must all occur within the interval `t-min(t)`.

    """

    dt = t[1]-t[0]
    cs = np.cumsum(s)
    if cs[-1] >= max(t)-min(t):
        raise ValueError('some spike times occur outside of signal''s support')

    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.axes([0.125, 0.3, 0.775, 0.6])
    p.vlines(cs+min(t), np.zeros(len(cs)), u[np.asarray(cs/dt, int)], 'b')
    p.hlines(0, 0, max(t), 'r')
    p.plot(t, u)
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    a = p.axes([0.125, 0.1, 0.775, 0.1])
    p.plot(cs+min(t), np.zeros(len(s)), 'ro')
    a.set_yticklabels([])
    p.xlabel('%d spikes' % len(s))
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()

def plot_compare(t, u, v, fig_title='', file_name=''):
    """
    Compare two signals and plot the difference between them.

    Parameters
    ----------
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u, v : ndarrays of floats
        Signal samples.
    fig_title : string
        Plot title.
    file_name : string
        File in which to save the plot.

    """

    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.subplot(211)
    p.plot(t, u, 'b', t, v, 'r')
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    p.subplot(212)
    p.plot(t, 20*np.log10(abs(u-v)))
    p.xlabel('t (s)')
    p.ylabel('error (dB)')
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    
