import time
from numpy import array, ceil, complex, exp, pi, zeros
from numpy.random import rand, randint, randn, normal
from numpy.fft import irfft, fft, fftfreq, fftshift, ifft
from scipy.signal import firwin, lfilter
import numpy as np
import scipy

def getSpikesSignal(spikes_idx, spikes_sgns, spikes_amplitude, N):
    minus_ones = np.where(spikes_sgns == -1)
    ones = np.where(spikes_sgns == 1)
    
    minus_ones_idx = spikes_idx[minus_ones]
    ones_idx = spikes_idx[ones]
    
    minus_impulse = scipy.signal.unit_impulse(N, minus_ones_idx) * -1
    ones_impulse = scipy.signal.unit_impulse(N, ones_idx)
    
    impulses = minus_impulse + ones_impulse
    impulses = impulses * spikes_amplitude
    
    return impulses

def getESpline(t_0_centered, w_0, l):
    
    t_left_first_index = int(len(t_0_centered)/2) - l
    t_left_last_index = t_left_first_index + int(l/2)
    
    t_right_first_index = t_left_last_index
    t_right_last_index = t_left_first_index + l
    
    t_left = t_0_centered[t_left_first_index : t_left_last_index]
    t_right = t_0_centered[t_right_first_index : t_right_last_index]
    
    phi_left = (1/w_0) * np.sin(w_0 * (2 + t_left))
    phi_right = - (1/w_0) * np.sin(w_0 * t_right) 
    
    phi = np.zeros(len(t_0_centered))
    
    phi[t_left_first_index : t_left_last_index] = phi_left
    phi[t_right_first_index : t_right_last_index] = phi_right
    
    return phi

def rcosFilter(t, gamma, Ts):
    """
    To be optimized
    """
    return np.sinc(t/Ts) * np.cos(np.pi*gamma*t/Ts) / (1 - ((2*gamma*t/Ts) ** 2))

def derivativeRcosFilter(t, gamma, Ts):
    a = np.sinc(t/Ts)
    a_prime = (np.cos(np.pi * t / Ts) - np.sinc(t/Ts)) / t
    
    b = np.cos(np.pi * gamma * t / Ts)
    b_prime = - (np.pi * gamma / Ts) * np.sin(np.pi * gamma * t / Ts)
    
    c = 1 / (1 - (2*gamma*t/Ts) ** 2)
    c_prime = (8 * ((Ts * gamma) **2) * t) / ((Ts**2 - (2*gamma*t) ** 2) **2)
    
    return a * b * c_prime + a * b_prime * c + a_prime * b * c

def closedPhiFromRcos(t, gamma, Ts, alpha):
    """
    To be optimized
    """
    return (derivativeRcosFilter(t, gamma, Ts) + alpha * rcosFilter(t, gamma, Ts)) 

def getPhiFromPsi(psi_kernel, N, dt, alpha):
    """
    To be optimized
    """
    fft_psi = fft(psi_kernel)
    fft_xf = fftfreq(N, dt)
    
    fft_phi = (2 * np.pi * fft_xf * 1j + alpha) * fft_psi
    phi = ifft(fft_phi)
    
    return phi


def addNoise(u, var):
    noise = normal(0, var, u.shape)
    new_signal = u + noise
    
    return new_signal