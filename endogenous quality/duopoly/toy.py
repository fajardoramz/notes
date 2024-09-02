from .funcs_toy import *

class ToyDuopoly():
    def __init__(self,
                 t=1,    # Transport cost
                 μ=10,    # Mean expenditures
                 p=10,   # Variance of the normal distribution of log(λ)
                 γ1=1.5,  # Productivity of 1
                 γ2=1,    # Productivity of 2
                 ):
        # Setting original instance variables
        self.t, self.μ, self.p, self.γ1, self.γ2 = t, μ, p, γ1, γ2

    def c(self, i, x):
        '''Cost function of firm i'''
        return C(i, x, self.μ, self.γ1, self.γ2)
    
    def b_m(self, x1, x2):
        '''Marginal b'''
        return b_m(x1, x2, self.t, self.μ)
    
    def q(self, i, x1, x2):
        '''Return the quantity demanded from firm i'''
        return Q(i, x1, x2, self.t, self.μ)
    
    def u(self, x):
        '''Utility of x'''
        return U(x, self.μ)
    
    def tc(self, i, b):
        '''Transport cost'''
        return tc(i, b, self.t)
    
    def π(self, i, x1, x2):
        '''Return the profit of firm i'''
        return Π(i, x1, x2, self.t, self.μ, self.p, self.γ1, self.γ2)
        
    def foc(self, i, x1, x2):
        '''Return the first order condition of firm i'''
        return FOC(i, x1, x2, self.t, self.μ, self.p, self.γ1, self.γ2)

    def x_max(self, i):
        '''Return the upper bound value of x'''
        return x_max(i, self.γ1, self.γ2, self.μ, self.p)
    
    def cs(self, i, x1, x2):
        '''Consumer surplus of firm i buyers, i=0 for total CS'''
        if i == 1:
            return (self.u(x1) - self.p)*self.q(1, x1, x2)
        if i==2:
            return (self.u(x2) - self.p)*self.q(2, x1, x2)
        if i==0:
            U1, U2, Q1, Q2 = self.u(x1), self.u(x2), self.q(1, x1, x2), self.q(2, x1, x2)
            return U1*Q1 + U2*Q2 - self.p
    
    def w(self, x1, x2):
        '''Welfare'''
        return self.cs(0, x1, x2) + self.π(1, x1, x2) + self.π(2, x1, x2)
    
    def d(self, i, x1, x2, b, μ):
        '''Demand indicator of firm i'''
        return D(i, x1, x2, b, μ, self.t)
    
    def max_welf(self):
        '''Maximum welfare'''
        γ1, γ2, μ = self.γ1, self.γ2, self.μ
        ν = 0
        if γ1 >= γ2:
            x_max = (μ+2*ν) / (2*ν +(2*μ/γ1))
            w_max = self.u(x_max) - self.c(1, x_max)
            return np.array([x_max, w_max ])
        else:
            x_max = (μ+2*ν) / (2*ν +(2*μ/γ2))
            w_max = self.u(x_max) - self.c(2, x_max)
            return np.array([x_max, w_max])
        

