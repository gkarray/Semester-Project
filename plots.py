import pylab as p
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

def plot_signal_and_fourier(t, u, dt):
    """
    Plot a signal and its fourier transform.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the original signal was sampled.
    u : ndarray of floats
        Signal samples.
    dt : ndarray of floats
        Sampling resolution.

    """
    y = u

    yf = fft(y)
    xf = fftfreq(len(t), dt)
    xf = fftshift(xf)

    yplot = fftshift(yf)

    plt.figure(figsize=(20,8))

    plt.subplot(1, 2, 1)
    plt.plot(t, y)
    plt.xlabel('t (s)')
    plt.ylabel('u(t)')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(xf, 1.0/len(t) * np.abs(yplot))
    plt.xlabel('f (Hz)')
    plt.ylabel('|U(f)|')
    plt.grid()

    plt.show()
    
    
def plot_integral_and_spikes(t, ys, zs, algo):
    """
    Plot the integral signal and the spike signal (z(t) from Lazar's paper, q(t) from Feichtinger's paper)
    in the same plot.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the original signal was sampled.
    ys : ndarray of floats
        Integral signal.
    zs : ndarray of floats
        Spike signal.

    """
    plt.figure(figsize=(15, 10))
    
    ylabel = 'y(t)'
    zlabel = ''
    
    if(algo == 'lazar'):
        zlabel = 'z(t)'
    elif (algo == 'feichtinger'):
        zlabel = 'q(t)'
        
    plt.plot(t, ys, 'r', label=ylabel)
    plt.plot(t, zs, 'b', label=zlabel)
    plt.legend()
    
    plt.show()

def plot_encoded(t, u, s, fig_title):
    """
    Plot a time-encoded signal.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the original signal was sampled.
    u : ndarray of floats
        Signal samples.
    s : ndarray of floats
        Spiking times.

    """
    dt = t[1]-t[0]
    if s[-1] > max(t)-min(t):
        raise ValueError('some spike times occur outside of signal''s support')
        
    p.figure(figsize=(15,10))
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.axes([0.125, 0.3, 0.775, 0.6])
    p.vlines(s+min(t), np.zeros(len(s)), u[np.asarray(s/dt, int)], 'b')
    p.hlines(0, 0, max(t), 'r')
    p.plot(t, u)
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    a = p.axes([0.125, 0.1, 0.775, 0.1])
    p.plot(s+min(t), np.zeros(len(s)), 'ro')
    a.set_yticklabels([])
    p.xlabel('%d spikes' % len(s))
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()

def plot_compare(t, u, v, fig_title):
    """
    Compare two signals and plot the difference between them.

    Parameters
    ----------
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u, v : ndarrays of floats
        Signal samples.

    """
    p.figure(figsize=(15,10))
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.subplot(211)
    p.plot(t, u, 'b', label='u(t)')
    p.plot(t, v, 'r', label='u_rec(t)')
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.legend()
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    p.subplot(212)
    p.plot(t, 20*np.log10(abs(u-v)))
    p.xlabel('t (s)')
    p.ylabel('error (dB)')
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    
