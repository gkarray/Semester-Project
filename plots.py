import pylab as p
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

def plotSignalAndFourier(t, u, dt, fig_title):
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
    plt.plot(xf, (2.0/len(t)) * np.abs(yplot))
    plt.xlabel('f (Hz)')
    plt.ylabel('|U(f)|')
    plt.grid()
    
    plt.title(fig_title)

    plt.show()
    
def plotAny(t, u, fig_title, spikes = False):
    plt.figure(figsize=(15, 10))
    
    if spikes:
        plt.stem(t, u, use_line_collection = True)
    else:
        plt.plot(t, u)
    plt.xlabel('t (s)')
    plt.ylabel('u(t)')
    plt.title(fig_title)
    plt.grid()
    
    plt.show()
    
def plotIntegralAndEncoderOutput(t, y_signal, enc_output, enc_output_label, fig_title, spikes = False):
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
        
    plt.plot(t, y_signal, 'r', label='y(t)')
    if spikes == True:
        plt.stem(t, enc_output, 'b', label=enc_output_label, use_line_collection = True)
    else:
        plt.plot(t, enc_output, 'b', label=enc_output_label)
    
    plt.title(fig_title)
    plt.legend()
    
    plt.show()

def plotSignalAndSpikes(t, u, s, fig_title):
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
        
    p.figure(figsize=(15, 10))
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

def plotSignalAndRecoveredSignal(t, u, v, fig_title, spikes = False, denoised = None):
    """
    Compare two signals and plot the difference between them.

    Parameters
    ----------
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u, v : ndarrays of floats
        Signal samples.

    """
    p.figure(figsize=(15, 10))
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
#     p.subplot(211)
    p.plot(t, u, 'b', label='u(t)')
    if(spikes):
        p.stem(t, v, 'r', label='u_rec(t)', use_line_collection = True)
    else:
        p.plot(t, v, 'r', label='u_rec(t)')
        
    if(denoised != None):
        p.plot(t, denoised, 'o', label='denoised u(t)')
    
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.legend()
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    
def plotSignalAndRecoveredSignalAndDenoised(t, u, v, denoised, fig_title):
    """
    Compare two signals and plot the difference between them.

    Parameters
    ----------
    t : ndarray of floats
        Times (s) at which the signal is defined.
    u, v : ndarrays of floats
        Signal samples.

    """
    p.figure(figsize=(15, 10))
    p.clf()
    p.gcf().canvas.set_window_title(fig_title)
    p.plot(t, u, 'b', label='u(t)')
    p.plot(t, v, 'r', label='u_rec(t)')
        
    p.plot(t, denoised, 'g', label='denoised u(t)')
    
    p.xlabel('t (s)')
    p.ylabel('u(t)')
    p.legend()
    p.title(fig_title)
    p.gca().set_xlim(min(t), max(t))
    p.draw_if_interactive()
    
