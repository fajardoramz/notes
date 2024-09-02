'''Solvers for Duopoly and ToyDuopoly classes'''
from scipy.optimize import root
import numpy as np

def solve_toy(toy, x1_0 = .9, x2_0 = .9, 
              method='hybr'):
    '''Solve for equilibrium quality and quantities for toy model
     - toy: instance of ToyDuopoly
    '''
    # Define function for solver
    def FOCS(eq):
        x1, x2 = eq
        foc_1 = toy.foc(1, x1, x2)
        foc_2 = toy.foc(2, x1, x2)
        if x1>toy.x_max(1) or x1<0 or toy.π(1, x1, x2)<0:
            foc_1 = 10e9
        if x2>toy.x_max(2) or x2<0 or toy.π(2, x1, x2)<0:
            foc_2 = 10e9
        return np.array([foc_1, foc_2])
   
    # Set initial conditions and solve
    eq0 = np.array([x1_0, x2_0])
    # Solve
    sol = root(FOCS, x0=eq0, method=method)
    if sol.success:
        x1_sol = sol.x[0]
        x2_sol = sol.x[1]
        q1 = toy.q(1, x1_sol, x2_sol)
        q2 = toy.q(2, x1_sol, x2_sol)
        q = q1 + q2
        π1 = toy.π(1, x1_sol, x2_sol)
        π2 = toy.π(2, x1_sol, x2_sol)
        foc_x1_sol = toy.foc(1, x1_sol, x2_sol)
        foc_x2_sol = toy.foc(2, x1_sol, x2_sol)
        return {'x1': x1_sol, 
                'x2': x2_sol, 
                'q1': q1, 
                'q2': q2,
                'q': q,
                'foc_x1': foc_x1_sol, 
                'foc_x2': foc_x2_sol,
                'π1': π1,
                'π2': π2}
    else:
        print('Optimization failed:', sol.message )
        return {'p': np.nan, 'x': np.nan, 'Q': np.nan, 'foc_p': np.nan, 'foc_x': np.nan}
    
def solve_duopoly(duopoly, x1_0 = .9, x2_0 = .7, method='broyden1', maxiter=1000):
    '''Solve for equilibrium quality and quantities for duopoly model
     - duopoly: instance of Duopoly
    '''
    # Define function for solver
    def FOCS(eq):
        x1, x2 = eq
        foc_1 = duopoly.FOC(1, x1, x2)
        foc_2 = duopoly.FOC(2, x1, x2)
        Π1 = duopoly.Π(1, x1, x2)
        Π2 = duopoly.Π(2, x1, x2)
        q1 = duopoly.q(1, x1, x2)
        q2 = duopoly.q(2, x1, x2)
        if x1<0 or Π1<0 or q1==0:
            foc_1 = 10e9
        if x2<0 or Π2<0 or q2==0:
            foc_2 = 10e9
        return np.array([foc_1, foc_2])
   
    # Set initial conditions and solve
    eq0 = np.array([x1_0, x2_0])
    # Solve
    sol = root(FOCS, x0=eq0, method=method, 
              options={'maxiter': maxiter})
    if sol.success:
        x1_sol = sol.x[0]
        x2_sol = sol.x[1]
        q1 = duopoly.q(1, x1_sol, x2_sol)
        q2 = duopoly.q(2, x1_sol, x2_sol)
        q = q1 + q2
        π1 = duopoly.Π(1, x1_sol, x2_sol)
        π2 = duopoly.Π(2, x1_sol, x2_sol)
        foc_x1_sol = duopoly.FOC(1, x1_sol, x2_sol)
        foc_x2_sol = duopoly.FOC(2, x1_sol, x2_sol)
        return {'x1': x1_sol, 
                'x2': x2_sol, 
                'q1': q1, 
                'q2': q2,
                'q': q,
                'foc_x1': foc_x1_sol, 
                'foc_x2': foc_x2_sol,
                'π1': π1,
                'π2': π2}
    else:
        print('Optimization failed:', sol.message)
        return {'x1': np.nan, 'x2': np.nan, 'q1': np.nan, 
                'q2': np.nan, 'q': np.nan, 'foc_x1': np.nan, 
                'foc_x2': np.nan, 'π1': np.nan, 'π2': np.nan}
    
