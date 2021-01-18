from Generator import Generator
from numpy import ceil, zeros, array, exp, pi
from numpy.random import randint, rand, choice
from numpy.fft import irfft
import numpy as np


class NBL_Diracs(Generator):
    def __init__(self, dur, dt, L_support, B, K):
        super().__init__(dur, dt)
        
        if L_support <= 0:
            raise ValueError('The support of the E-spline must be strictly positive')
            
        if B <= 0:
            raise ValueError('The number of bursts must be strictly positive')
            
        if L_support >= ((dur - L_support) / B):
            raise ValueError(f"Signal duration is not high enough to hold {B} filtered bursts of spikes")
         
        
        self.L_support = L_support
        self.B = B
        
        if K <= 0:
            raise ValueError('The number of diracs per burst must be strictly positive')
            
        self.K = K
        
        
    def generate(self):
        N = int(ceil(self.dur/self.dt))
        
        N_E_spline = int(ceil(self.L_support/self.dt))
        half_N_E_spline = int(ceil(N_E_spline/2))
        
        signal = zeros(N)
        
        start = half_N_E_spline
        end = N - half_N_E_spline
        
        bursts = []
        signs = []

        s_wind_min = start
        s_wind_max = int((N - N_E_spline) / self.B) - half_N_E_spline

        for i in range(self.B):
            temp_ci = set()
            signs.append(choice([-1, 1]))
            
            while len(temp_ci) < self.K:
                temp = randint(s_wind_min, s_wind_max)
            
                
                union_temps = temp_ci.union({temp})
                
                while temp in temp_ci or (max(union_temps) - min(union_temps) > half_N_E_spline):
                    temp = randint(s_wind_min, s_wind_max)
                    union_temps = temp_ci.union({temp})
                
                temp_ci.add(temp)
            
            bursts.append(temp_ci)
            
            s_wind_min = s_wind_max + N_E_spline
            s_wind_max = s_wind_max + int((N - N_E_spline) / self.B)
            
            if s_wind_max > end:
                s_wind_max = end
        
        for idx, burst in enumerate(bursts):
            ci = array(list(burst))
            amps = signs[idx] * rand(self.K)
            
            signal[ci] = amps
        
        u = signal
        
        return self.t, u