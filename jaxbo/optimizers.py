from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import optax


def minimize_lbfgs(objective, x0, bnds = None, callback = None):
    
    result = minimize(objective, x0, jac=True,
                      method='L-BFGS-B', bounds = bnds,
                      callback=callback)
    return result.x, result.fun

def minimize_de(objective,bnds = None, maxiter = 10000, popsize = 200, tol = 0.0001, seed = None):
    result = differential_evolution(objective, bnds, maxiter = maxiter, popsize= popsize, seed = seed)
    return result.x, result.fun

def minimize_optax(objective, x0, bnds = None, nit = 200):
    def update(params, opt_state):
        grads, loss = objective(params) #da modificare
        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    start_learning_rate = 1e-3
    opt = optax.adam(start_learning_rate)
    opt_state = opt.init(x0)

    for _ in range(nit):
        params, opt_state, loss = update(x0, opt_state)
    return params, loss