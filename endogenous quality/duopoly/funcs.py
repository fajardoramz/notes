'''
Vectorized functions for Duopoly and ToyDuopoly classes
'''
import numpy as np
from numba import guvectorize, vectorize, float64, int64, njit, prange
from scipy.special import gamma , gammaln

@vectorize([float64(int64, *([float64]*4))], nopython=True)
def c(i, x, λ, γ1, γ2):
    '''Cost function of firm i'''
    if i == 1:
        return (1/γ1)*λ*x**2
    else:
        return (1/γ2)*λ*x**2
    
@vectorize([float64(int64, *([float64]*4))], nopython=True)
def MC(i, x, μ, γ1, γ2):
    '''Marginal cost of firm i'''
    if i == 1:
        return 2*μ*x/γ1
    else:
        return 2*μ*x/γ2

@vectorize([float64(*([float64]*4))], nopython=True)
def b_m(x1, x2, t, μ):        
        '''Marginal b for toy'''
        b = (1/(2*t))*(μ*(x1-x2)+t)
        b = np.maximum(0, b)
        b = np.minimum(1, b)
        return b

@vectorize([float64(float64, float64)], nopython=True)  
def U(x, λ):
    '''Utility of x for type μ'''
    ν = 0
    return λ*x + ν*(1-(1-x)**2)

@vectorize([float64(int64, float64, float64, float64)], nopython=True)
def tc(i, b, t1, t2):
    '''Transport cost'''
    if i == 1:
        return t1*b
    else:
        return t2*(1-b)

@vectorize([int64(int64, *([float64]*6))], nopython=True)    
def D(i, x1, x2, λ, b, t1, t2):
    '''Demand indicator of firm i'''
    v1 = U(x1, λ) - tc(1, b, t1, t2)
    v2 = U(x2, λ) - tc(2, b, t1, t2)
    if i == 1:
        return v1 >= v2
    else:
        return v2 > v1

@vectorize([float64(int64, *([float64]*4))], nopython=True)
def x_max(i, γ1, γ2, μ, p):
    '''Upper bound to quality of firm i'''
    if i==1:
        return np.sqrt((γ1*p)/μ)
    else:
        return np.sqrt((γ2*p)/μ)

@vectorize([float64(*([float64]*5))], nopython=True)
def λ_tilde(x1, x2, b, t1, t2):
    '''Switching margin λ_tilde'''
    if (b >= t2 / (t1+t2) and x1>=x2) or (b < t2 / (t1+t2) and x1<=x2):
        return (b*(t1+t2)-t2)/(x1-x2)
    else:
        return 0

@njit(parallel=True)
def θ_draws(μ_λ, σ_λ, E_λ, k, beta_ind, α, β, N, seeds):
    '''Return N draws of θ = (λ, b)'''
    θ_draws = np.empty((N, 2))
    for r in prange(N):
        np.random.seed(seeds[r])
        λ = np.exp(np.random.normal(μ_λ, σ_λ))
        if beta_ind==1:
            α, β = α, β
        else:
            α = .5+k*E_λ/λ # Beta parameter
            β = α
        b = np.random.beta(a=α, b=β)
        θ_draws[r, 0] = λ
        θ_draws[r, 1] = b
    return θ_draws

@njit(parallel=True)
def Q(i, x1, x2, θ, N, t1, t2):
    '''Return the quantity demanded from firm i'''
    total = 0
    for r in prange(N):
        λ, b = θ[r]
        total += D(i, x1, x2, λ, b, t1, t2)
    return total / N

@njit(parallel=True)
def EXP_B(i, x1, x2, F_θ, θ, N, t1, t2):
    '''Expected value of function F conditional on buying from i
        - F_θ: vector of F(θ) for each θ
        - θ: draws of θ
    '''
    q = Q(i, x1, x2, θ, N, t1, t2)
    if q == 0:
        return 0
    else:
        total = 0
        for r in prange(N):
            λ, b = θ[r,0], θ[r,1]
            total += F_θ[r]*D(i, x1, x2, λ, b, t1, t2)
        return total / (N*q)

@vectorize([float64(float64, float64, float64)], nopython=True)
def f_λ(λ, μ, σ):
    '''Density of λ'''
    return (1 / (λ * σ * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(λ) - μ) / σ)**2)

#@vectorize([float64(*([float64]*4))], nopython=True)
def f_b_λ(b, λ, E_λ, k):
    '''Density of b conditional on λ'''
    α = .5+k*E_λ/λ
    β = α
    #B = (gamma(α)*gamma(β))/gamma(α+β)
    # To address overflow of gamma function, use log-gamma
    B = np.exp(gammaln(α) + gammaln(β) - gammaln(α+β))
    if B < 1e-8:
        return 0
    else:
        return (1/B)*b**(α-1)*(1-b)**(β-1)


