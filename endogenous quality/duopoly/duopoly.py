
from numba import jit, vectorize, float64, int64, prange
from numba.experimental import jitclass
from numba.core import types
from numpy.random import Generator, PCG64
import numpy as np
from scipy.integrate import quad

from .funcs import *

spec = [
    ('t1', types.float64),            # Transport cost
    ('t2', types.float64),            # Transport cost
    ('p', types.float64),            # Price
    ('γ1', types.float64),           # Productivity of 1
    ('γ2', types.float64),           # Productivity of 2
    ('μ_λ', types.float64),          # Mean of the normal distribution of log(λ)
    ('V_λ', types.float64),          # Variance of the normal distribution of log(λ)
    ('k', types.float64),            # Scale parameter for joint distribution
    ('α', types.float64),            # Beta distribution parameter
    ('β', types.float64),            # Beta distribution parameter
    ('beta_ind', types.int64),       # Set to 1 for independent beta draws with parameters α and β
    ('N', types.int64),             # Number of Monte Carlo draws 
    ('σ_λ', types.float64),          # Standard deviation of the normal distribution of log(λ)
    ('E_λ', types.float64),          # Mean of the normal distribution of log(λ)
    ('Var_λ', types.float64),        # Variance of the normal distribution of log(λ)
    ('θ', types.float64[:,:]),       # Array for individual types
    ('λ', types.float64[:]),       # Array for individual types
    ('b', types.float64[:]),       # Array for individual types
    ('seed', int64),
    ('N', int64)              #parameter for draw method
]

#@jitclass(spec)
class Duopoly():
    def __init__(self,
                 t1=1,   # Transport cost to 1
                 t2=1,   # Transport cost to 2
                 p=10,    # Price
                 γ1=1.5,  # Productivity of 1
                 γ2=1,    # Productivity of 2
                 μ_λ=0,   # Mean of the normal distribution of log(λ)
                 V_λ=1,   # Variance of the normal distribution of log(λ)
                 k=1,     # Scale parameter for joint distribution
                 α=1,     # Beta distribution parameter
                 β=1,     # Beta distribution parameter
                 beta_ind=0, # Set to 1 for independent beta draws with parameters α and β
                 N = 100_000, # Number of Monte Carlo draws
                 seed=42  
                 ):
        # Setting original instance variables
        self.t1, self.t2, self.p, self.γ1, self.γ2 = t1, t2, p, γ1, γ2
        self.μ_λ, self.V_λ, self.k, self.N = μ_λ, V_λ, k, N

        # Calculate parameters of distributions
        σ_λ = np.sqrt(V_λ)
        E_λ = np.exp(μ_λ + V_λ/2)
        Var_λ = E_λ*(np.exp(V_λ) - 1)

        # Draw θ, but create seeds first
        np.random.seed(seed)
        seeds = np.random.randint(0, 2**32-1, N)
        θ = θ_draws(μ_λ, σ_λ, E_λ, k, beta_ind, α, β, N, seeds)

        # Setting new instance variables
        self.σ_λ, self.E_λ, self.Var_λ, self.θ, self.N = σ_λ, E_λ, Var_λ, θ, N
        self.λ = θ[:, 0]
        self.b = θ[:, 1]
    
    def c(self, i, x, λ):
        '''Cost function of firm i'''
        return c(i, x, λ, self.γ1, self.γ2)
    
    def mc(self, i, x, λ):
        '''Marginal cost of firm i'''
        return MC(i, x, λ, self.γ1, self.γ2)
    
    def λ_tilde(self, x1, x2, b):
        '''Return marginal type λ* as a function of b'''
        return λ_tilde(x1, x2, b, self.t1, self.t2)
    
    def u(self, x, λ):
        '''Utility of x'''
        return U(x, λ)

    def tc(self, i, b):
        '''Transport cost'''
        return tc(i, b, self.t)
    
    def d(self, i, x1, x2, λ, b):
        '''Demand indicator of firm i'''
        return D(i, x1, x2, λ, b, self.t)
    
    def q(self, i, x1, x2,):
        '''Quantity demanded from firm i'''
        θ, N, t1, t2 = self.θ, self.N, self.t1, self.t2
        return Q(i, x1, x2, θ, N, t1, t2)

    def E_TC(self, i, x1, x2):
        '''Total cost of firm i'''
        q = self.q(i, x1, x2)
        if q==0:
            return 0
        else:
            if i==1:
                F_θ = self.c(i, x1, self.λ)
            else:
                F_θ = self.c(i, x2, self.λ)
            return q * EXP_B(i, x1, x2, F_θ, self.θ, self.N, self.t1, self.t2)
    
    def Π(self, i, x1, x2):
        '''Expected profit of firm i'''
        q = self.q(i, x1, x2)
        return q*self.p - self.E_TC(i, x1, x2)
    

    def FOC(self, i, x1, x2):
        '''First order condition of firm i'''
        q = self.q(i, x1, x2)
        if i==1:
            F_θ = self.mc(1, x1, self.λ)
            def Z(b):
                λ_t = self.λ_tilde(x1, x2, b)
                c = self.c(1, x1, λ_t)
                F_b_λ = f_b_λ(b, λ_t, self.E_λ, self.k)
                F_λ = f_λ(λ_t, self.μ_λ, self.σ_λ)
                return (1/(x1-x2))*λ_t*(self.p-c)*F_b_λ*F_λ
        else:
            F_θ = self.mc(2, x2, self.λ)
            def Z(b):
                λ_t = self.λ_tilde(x1, x2, b)
                c = self.c(2, x2, λ_t)
                F_b_λ = f_b_λ(b, λ_t, self.E_λ, self.k)
                F_λ = f_λ(λ_t, self.μ_λ, self.σ_λ)
                return (1/(x1-x2))*λ_t*(self.p-c)*F_b_λ*F_λ
        Exp_B_mc = EXP_B(i, x1, x2, F_θ, self.θ, self.N, self.t1, self.t2)
        eps = np.finfo(float).eps
        b_bar = self.t2/(self.t1+self.t2) + eps
        if x1>x2:
            int_Z_M = quad(Z, b_bar, 1)[0]
        else:
            int_Z_M = quad(Z, 0, b_bar)[0]*(-1) 
        return -q*Exp_B_mc + int_Z_M
    
    

    
    


    
    

    
