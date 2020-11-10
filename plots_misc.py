import pylab as p
import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt

def plot_signal_and_fourier_vert(t, u, dt):
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

    plt.figure(figsize=(10,8))

    plt.subplot(2, 1, 1)
    plt.plot(t, y)
    plt.xlabel('t (s)')
    plt.ylabel('u(t)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(xf, 1.0/len(t) * np.abs(yplot))
    plt.xlabel('f (Hz)')
    plt.ylabel('|U(f)|')
    plt.grid()

    
    plt.savefig('saved_images/viz1.png')
    plt.show()
    
def plot_four_viz_kernels(t, psi_kernel, phi_kernel, dt):
    """
    Plot a signal and its fourier transform.

    Parameters
    ----------
    t : ndarray of floats
        Times (in s) at which the original signal was sampled.
    psi_kernel : ndarray of floats
        Signal samples.
    phi_kernel : ndarray of floats
        Signal samples.
    dt : ndarray of floats
        Sampling resolution.

    """
    y1 = psi_kernel
    y2 = phi_kernel

    yf1 = fft(y1)
    yf2 = fft(y2)
    xf = fftfreq(len(t), dt)
    xf = fftshift(xf)

    yplot1 = fftshift(yf1)
    yplot2 = fftshift(yf2)

    
    plt.figure(figsize=(10,8))

    plt.subplot(2, 2, 1)
    plt.plot(t, y1)
    plt.xlabel('Time (s)')
    plt.ylabel('Psi(t)')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(t, y2)
    plt.xlabel('Time (s)')
    plt.ylabel('Phi(t)')
    plt.grid()
    

    plt.subplot(2, 2, 3)
    plt.plot(xf, 1.0/len(t) * np.abs(yplot1))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(xf, 1.0/len(t) * np.abs(yplot2))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()

    
    plt.savefig('saved_images/viz4.png')
    plt.show() 
    
def plot_against_each_other(t, u, u_rec, fig_title='test'):
    plt.figure(figsize=(10,8))
    
    plt.plot(t, u, label="u(t)")
    plt.plot(t, u_rec, 'r', label='u_rec(t)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.grid()
    
    plt.savefig('saved_images/' + fig_title + '.png')
    plt.show()
    
    
def plot_signal_and_integral(t, u, dt, ys, zs):
    plt.figure(figsize=(10,8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, u)
    plt.xlabel('t (s)')
    plt.ylabel('u(t)')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    ylabel = 'y(t)'
    zlabel = 'z(t)'
    plt.plot(t, ys, 'r', label=ylabel)
    plt.plot(t, zs, 'b', label=zlabel)
    plt.legend()
    plt.xlabel('t (s)')
    plt.ylim(-10, 10)
    plt.grid()
    
    plt.savefig('saved_images/viz3.png')
    plt.show()
    
def plot_both_signals(t, u, u2, dt):
    plt.figure(figsize=(10,8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, u)
    plt.xlabel('t (s)')
    plt.ylabel('u(t)')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(t, u2)
    plt.xlabel('t (s)')
    plt.ylabel('u(t)')
    plt.grid()
    
    plt.savefig('saved_images/viz5.png')
    plt.show()