from Sampler import Status, Sampler
from helpers import getSpikesSignal, rcosFilter, getPhiFromPsi, closedPhiFromRcos
from plots import plotSignalAndFourier, plotAny, plotSignalAndSpikes, plotSignalAndRecoveredSignal, plotIntegralAndEncoderOutput
import numpy as np
from scipy import special
from scipy.fft import fft, fftfreq, irfft

class IAF_KernelBased(Sampler):
    def __init__(self, alpha, theta, gamma, Ts):
        super().__init__()
        
        if alpha <= 0:
            raise ValueError('The alpha parameter must be strictly positive')
        
        self.alpha = alpha
        
        if theta <= 0:
            raise ValueError('The theta parameter must be strictly positive')
            
        self.theta = theta
        
        if gamma < 0 or gamma > 1:
            raise ValueError('The gamma parameter for the psi_kernel must be strictly positive')
            
        self.gamma = gamma
        
        if Ts <= 0:
            raise ValueError('The Ts parameter must be strictly positive')
            
        self.Ts = Ts
        
    def setSignal(self, t, u, dt):
        self.t = t
        self.u = u
        self.dt = dt
        
        self.t_phi_kernel = np.arange(-self.t[-1]/2, self.t[-1]/2 + self.dt, self.dt)
        if(self.t_phi_kernel.shape[0] > self.t.shape[0]):
            self.t_phi_kernel = self.t_phi_kernel[:-1]
        
        self.psi_kernel = rcosFilter(self.t_phi_kernel, self.gamma, self.Ts)
        self.phi_kernel = closedPhiFromRcos(self.t_phi_kernel, self.gamma, self.Ts, self.alpha)
        
        u_fourier = fft(u)
        ffreq = fftfreq(len(t), dt)
        mult = 2 * np.pi * ffreq * 1j + self.alpha
        v_fourier = u_fourier / mult
        self.v = irfft(v_fourier, len(t))
    
        self.w_0 = self.v[0]
        
        self.status = Status.HAS_SIGNAL
        
    def integrate(self, y, i):
        return y * (1 - self.alpha * self.dt) + self.u[i] * self.dt
    
    def encode(self, initial_y = 0):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of encode()")
        
        y = initial_y
        
        y_signal = [initial_y]
        spikes_idx = []
        q_signs = []
        
        for i in range(len(self.u)):
            y = self.integrate(y, i)
            y_signal.append(y)
            
            
            if np.abs(y) >= self.theta:
                y_signal.pop()
                y_signal.append(self.theta * np.sign(y))
                spikes_idx.append(i)
                tj = self.t[i]
                q_signs.append(np.sign(y))
                y = 0
                
        y_signal.pop()
        self.y_signal = np.array(y_signal)
        self.spikes_idx = np.array(spikes_idx)
        self.q_signs = np.array(q_signs)
        self.spikes_timings = self.t[self.spikes_idx]
        self.spikes_signal = getSpikesSignal(self.spikes_idx, self.q_signs, self.theta, self.t.shape[0])
        
        self.status = Status.ENCODED
        
        return self.spikes_signal

    def decode(self, gamma = None, Ts = None):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of decode()")
            
        if self.spikes_timings.shape[0] < 2:
            raise ValueError('Spikes signal contains less than 2 spikes')
            
        if gamma != None and Ts != None :
            self.gamma = gamma
            self.Ts = Ts
            
            self.t_phi_kernel = np.arange(-self.t[-1]/2, self.t[-1]/2 + self.dt, self.dt)
            if(self.t_phi_kernel.shape[0] > self.t.shape[0]):
                self.t_phi_kernel = self.t_phi_kernel[:-1]
            
            self.psi_kernel = rcosFilter(self.t_phi_kernel, self.gamma, self.Ts)
            self.phi_kernel = closedPhiFromRcos(self.t_phi_kernel, self.gamma, self.Ts, self.alpha)
            
        scaled_qs = self.q_signs * self.theta
        spikes_idx_with_t_0 = np.insert(self.spikes_idx, 0, 0)

        # Wi
        # To be optimized
        ws = []
        ws.append(self.w_0)
        for idx, value in enumerate(scaled_qs):
            tj = self.t[spikes_idx_with_t_0[idx]]
            tj1 = self.t[spikes_idx_with_t_0[idx+1]]
            wj = ws[idx]
            qj1 = value
            
            wj1 = np.exp(self.alpha * (tj - tj1)) * wj + qj1
            ws.append(wj1)
            
        self.ws = ws

        # Sk
        # To be optimized
        sk = []
        nb = 0
        
        for idx, value in enumerate(self.t):
            if(nb + 1< self.spikes_idx.shape[0]):
                if(idx == int(spikes_idx_with_t_0[nb+1])):
                    nb = nb+1
            
            s = np.exp(self.alpha * (self.t[spikes_idx_with_t_0[nb]] - value)) * ws[nb]

            sk.append(s)
            
        self.sk = np.array(sk)

        # Applying the right filter
        # To be optimized
        convolved = np.convolve(self.phi_kernel, sk, mode='same')
        self.u_rec = convolved
            
#         self.u_rec = filtered
        
        self.status = Status.DECODED
        
        return self.u_rec
    
    def summary(self):
        print(80 * f'=')
        print(80 * f'=')
        print(f'Integrate-and-Fire Sampler - KernelBased')
        print(f'From "APPROXIMATE RECONSTRUCTION OF BANDLIMITED FUNCTIONS FOR THE INTEGRATE AND FIRE SAMPLER"')
        print(f'Hans G. Feichtinger, 2009')
        print(f'Status: {self.status.name}')
        print(f'Parameters:')
        print(f"Firing parameter 'alpha': {self.alpha}")
        print(f"Threshold 'theta': {self.theta}")
        print(80 * f'=')
        print(80 * f'=')
        
    def plotSignal(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignal()")
        
        fig_title = 'Signal to be encoded with an Integrate-and-Fire - KernelBased sampler'
        
        plotSignalAndFourier(self.t, self.u, self.dt, fig_title)
        
    def plotSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSpikes()")
        
        fig_title = 'Spikes resulting from an Integrate-and-Fire - KernelBased sampler encoding'
        
        plotAny(self.t, self.spikes_signal, fig_title, spikes = True)
        
    def plotSignalAndSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndSpikes()")
        
        fig_title = 'Signal encoded with an Integrate-and-Fire - KernelBased sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u, self.spikes_timings, fig_title)
        
    def plotRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignal()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - KernelBased sampler'
        
        plotSignalAndFourier(self.t, self.u_rec, self.dt, fig_title)
        
    def plotRecoveredSignalAndSpikes(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignalAndSpikes()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - KernelBased sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u_rec, self.spikes_timings, fig_title)
        
    def plotSignalAndRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndRecoveredSignal()")
        
        fig_title = 'Comparaison of original signal and recovered signal encoded then decoded with an Integrate-and-Fire - KernelBased sampler (with spikes)'
        
        plotSignalAndRecoveredSignal(self.t, self.u, self.u_rec, fig_title)
        
    def computeError(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of computeError()")
        
        return self.u - self.u_rec
    
    def computeMeanSquaredError(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of computeMeanSquaredError()")
        
        return np.sum(self.computeError() ** 2)
    
    def plotError(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotError()")
        
        fig_title = 'Error between the reconstructed signal and the original signal'
        
        plotAny(self.t, self.computeError(), fig_title)
        
    def getNumberOfSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of getNumberOfSpikes()")
        
        return self.spikes_timings.shape[0]
    
    def plotIntegralAndEncoderOutput(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotIntegralAndEncoderOutput()")
            
        fig_title = 'Output of the integrator / Output of the encoder'    
        
        plotIntegralAndEncoderOutput(self.t, self.y_signal, self.spikes_signal, 'q(t)', fig_title, spikes = True)
        
    def plotPsiKernel(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotPhiKernel()")
            
        fig_title = 'Psi kernel'
        
        plotSignalAndFourier(self.t_phi_kernel, self.psi_kernel, self.dt, fig_title)
        
    def plotPhiKernel(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotPhiKernel()")
            
        fig_title = 'Phi kernel'
        
        plotSignalAndFourier(self.t_phi_kernel, self.phi_kernel, self.dt, fig_title)
        