import jax
import time
import numpy as onp
import jax.numpy as np
#import pygmo as pg
from jax import vmap, jit, jvp, vjp, random
from jax.scipy.linalg import cholesky, solve_triangular
from jax.flatten_util import ravel_pytree
from jax.scipy.special import expit as sigmoid
#from jax.ops import index_update, index
from functools import partial

import jaxbo.kernels as kernels
import jaxbo.acquisitions as acquisitions
import jaxbo.initializers as initializers
import jaxbo.utils as utils
from jaxbo.optimizers import minimize_lbfgs, minimize_de, minimize_optax
from sklearn import mixture
from pyDOE import lhs

from jax.scipy.stats import norm
from jax._src.lax import linalg as lax_linalg
from jax.config import config
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", False)

from jax.lax import Precision
#onp.random.seed(1234)

# Define a general master class
class GPmodel():
    def __init__(self, options):
        # Initialize the class
        self.options = options
        self.input_prior = options['input_prior']
        if options['kernel'] == 'RBF':
            self.kernel = kernels.RBF
        elif options['kernel'] == 'Matern52':
            self.kernel = kernels.Matern52
        elif options['kernel'] == 'Matern32':
            self.kernel = kernels.Matern32
        elif options['kernel'] == 'RatQuad':
            self.kernel = kernels.RatQuad
        elif options['kernel'] == 'RBF_fractional':
            self.kernel = kernels.RBF_fractional
        elif options['kernel'] == None:
            self.kernel = kernels.RBF
        else:
            raise NotImplementedError


    @partial(jit, static_argnums=(0,))
    def likelihood(self, params, batch):
        y = batch['y']
        N = y.shape[0]

        # Compute Cholesky
        L = self.compute_cholesky(params, batch)
        # Compute negative log-marginal likelihood
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        NLML = 0.5 * np.matmul(np.transpose(y),alpha, precision=Precision.HIGHEST) + \
                np.sum(np.log(np.diag(L))) + 0.5 * N * np.log(2.0 * np.pi)

        return NLML

    @partial(jit, static_argnums=(0,))
    def likelihood_value_and_grad(self, params, batch):
        fun = lambda params: self.likelihood(params, batch)
        primals, f_vjp = vjp(fun, params)
        grads = f_vjp(np.ones_like(primals, dtype=np.float64))[0]
        grads = np.nan_to_num(grads, nan = -1)
        #temp = onp.asarray(grads)
        return primals, grads

    def fit_gmm(self, num_comp=2, N_samples=10000, **kwargs):
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        # load the seed
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]
        # Sample data across the entire domain

        # set the seed for sampling X
        onp.random.seed(rng_key[0])
        X = lb + (ub - lb) * lhs(dim, N_samples)

        y = self.predict(X, **kwargs)[0]
        # Sample data according to prior

        # set the seed for sampling X_samples
        rng_key = random.split(rng_key)[0]
        onp.random.seed(rng_key[0])

        X_samples = lb + (ub - lb) * lhs(dim, N_samples)
        y_samples = self.predict(X_samples, **kwargs)[0]

        # Compute p_x and p_y from samples across the entire domain
        p_x = self.input_prior.pdf(X)
        p_x_samples = self.input_prior.pdf(X_samples)

        p_y = utils.fit_kernel_density(y_samples, y, weights = p_x_samples)
        weights = p_x / p_y
        # Label each input data
        indices = np.arange(N_samples)
        # Scale inputs to [0, 1]^D
        X = (X - lb) / (ub - lb)
        # rescale weights as probability distribution
        weights = weights / np.sum(weights)
        # Sample from analytical w
        idx = onp.random.choice(indices,
                                N_samples,
                                p=weights.flatten())
        X_train = X[idx]
        # fit GMM
        clf = mixture.GaussianMixture(n_components=num_comp,
                                      covariance_type='full')
        clf.fit(X_train)
        return clf.weights_, clf.means_, clf.covariances_

    @partial(jit, static_argnums=(0,))
    def acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict(x, **kwargs)
        if self.options['criterion'] == 'LW-LCB':
            kappa = kwargs['kappa']
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCB(mean, std, weights, kappa)
        elif self.options['criterion'] == 'LCB':
            kappa = kwargs['kappa']
            return acquisitions.LCB(mean, std, kappa)
        elif self.options['criterion'] == 'EI':
            batch = kwargs['batch']
            best = np.min(batch['y'])
            return acquisitions.EI(mean, std, best)
        elif self.options['criterion'] == 'US':
            return acquisitions.US(std)
        elif self.options['criterion'] == 'TS':
            sample = self.draw_posterior_sample(x, **kwargs)
            return sample
        elif self.options['criterion'] == 'LW-US':
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_US(std, weights)
        elif self.options['criterion'] == 'CLSF':
            kappa = kwargs['kappa']
            # The following two lines for learning constraints
            norm_const = kwargs['norm_const']
            mean = mean * norm_const['sigma_y'] + norm_const['mu_y']
            std = std * norm_const['sigma_y']
            return acquisitions.CLSF(mean, std, kappa)
        elif self.options['criterion'] == 'LW_CLSF':
            kappa = kwargs['kappa']
            # The following two lines for learning constraints
            norm_const = kwargs['norm_const']
            mean = mean * norm_const['sigma_y'] + norm_const['mu_y']
            std = std * norm_const['sigma_y']
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_CLSF(mean, std, weights, kappa)
        elif self.options['criterion'] == 'IMSE':
            # See Eq.  (2.7), https://arxiv.org/pdf/2006.12394.pdf
            rng_key = kwargs['rng_key']
            bounds = kwargs['bounds']
            lb = bounds['lb']
            ub = bounds['ub']
            dim = lb.shape[0]
            xp = lb + (ub - lb) * random.uniform(rng_key, (10000,dim))    
            cov = self.posterior_covariance(x, xp, **kwargs)
            IMSE = np.mean(cov ** 2) / std ** 2
            return IMSE[0]
        elif self.options['criterion'] == 'IMSE_L':
            rng_key = kwargs['rng_key']
            bounds = kwargs['bounds']
            lb = bounds['lb']
            ub = bounds['ub']
            dim = lb.shape[0]
            _, std = self.predict_L(x, **kwargs)
            xp = lb + (ub - lb) * random.uniform(rng_key, (10000,dim)) 
            cov = self.posterior_covariance_L(x, xp, **kwargs)
            IMSE = np.mean(cov ** 2) / std ** 2
            return IMSE[0]
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def compute_next_point_lbfgs(self, num_restarts=10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        a_min = np.min(acq)
        x_new = loc[idx_best:idx_best + 1,:]
        return x_new, a_min, loc

    def compute_next_point_de(self, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.acq_value_and_grad(x, **kwargs)
            out = (onp.array(value))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))

        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        pos, val = minimize_de(objective, bnds = dom_bounds, maxiter = 1000, popsize = 50, seed = None)

        #x_new = device_put(pos)
        x_new = np.reshape(np.asarray(pos), (1, -1))
        ei = np.asarray(val)
        return x_new, val

    def compute_next_point_gs(self, X_cand, **kwargs):
        fun = lambda x: self.acquisition(x, **kwargs)
        acq = vmap(fun)(X_cand)
        idx_best = np.argmin(acq)
        x_new = X_cand[idx_best:idx_best + 1,:]
        return x_new

#def compute_next_point_pygmo(gp, **kwargs):
#        loc = []
#        acq = []
#        bounds = kwargs['bounds']
#        lb = bounds['lb']
#        ub = bounds['ub']
#        #fun = lambda x: self.acquisition(x, **kwargs)
#        class acquisition_prob(GP):
#            def __init__(self, gp):
#                self.__dict__ = gp.__dict__.copy()

#            def fitness(self, x):
#                value, grads = self.acq_value_and_grad(x, **kwargs)
#                print("VALUE = ", value)
#                out = onp.array(value)
#                print("OUT ARRAY = ", out)
#                out.tolist()
#                print("OUT LIST = ", out)
#                out = onp.reshape(out, (1, -1))
#                out = out[0]
#                return out

#            def get_bounds(self):
#                return (lb, ub)
       
#        p = pg.problem(acquisition_prob(acquisition_prob(gp)))
#        print(p)
#        algo_sade = pg.algorithm(pg.sade(2000))
#        pop = pg.population(prob = p , size = int(70))
#        isl = pg.island(algo = algo_sade, pop = pop)
#        print(isl)

#        isl.evolve(1)
#        isl.wait_check()
#        x_new = pop.champion_x
#        #archi = pg.archipelago(n=6, algo = algo_sade, prob = p, pop_size =
#        70)
#        #archi.evolve()
#        #archi.wait_check()
#        #champs_f = onp.array(archi.get_champions_f())
#        #champs_x = onp.array(archi.get_champions_x())
#        #idx_best = champs_f.argmax()
#        #x_new = champs_x[idx_best]
#        x_new = np.reshape(np.asarray(x_new), (1, -1))

#        return x_new

# A minimal Gaussian process regression class (inherits from GPmodel)
class GP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        X = batch['X']
        N, D = X.shape
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N) * (sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts=10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            #print(onp.asarray(grads))
            return out
        # Optimize with random restarts
        start_time = time.time()
        params = []
        likelihood = []
        dim = batch['X'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            #init = 1e-3*initializers.random_init_GP(rng_key[i], dim)
            init = initializers.random_init_GP(rng_key[i], dim)
            p, val = minimize_lbfgs(objective, init)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)
        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best,:]
        end_time = time.time()
        duration = end_time-start_time
        return best_params
#######################################################################################################
############################################ O P T A X ################################################
#######################################################################################################
    def train_optax(self, batch, rng_key, num_restarts=10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            return grads, value
        # Optimize with random restarts
        start_time = time.time()

        params = []
        likelihood = []
        dim = batch['X'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            init = 1e-3*initializers.random_init_GP(rng_key[i], dim)
            p, val = minimize_optax(objective, init)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)
        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best,:]
        end_time = time.time()
        duration = end_time-start_time
        return best_params


    @partial(jit, static_argnums=(0,))
    def predict(self, X_star,  **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def posterior_covariance(self, x, xp,  **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        x = (x - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        xp = (xp - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        # Compute kernels
        k_pp = self.kernel(x, xp, theta)
        k_pX = self.kernel(x, X, theta)
        k_Xp = self.kernel(X, xp, theta)
        L = self.compute_cholesky(params, batch)
        # Compute covariance
        beta = solve_triangular(L.T,solve_triangular(L, k_Xp, lower=True))        
        cov = k_pp - np.matmul(k_pX, beta)
        return cov

    @partial(jit, static_argnums=(0,))
    def draw_posterior_sample(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        rng_key = kwargs['rng_key']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        sample = random.multivariate_normal(rng_key, mu, cov)
        return sample


# A minimal Gaussian process regression class (inherits from GPmodel)
class MultipleIndependentOutputsGP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums = (0,))
    def _check_symmetry(self, x: np.ndarray) -> bool:
      """Check if the array is symmetric."""
      m, n = x.shape
      eps = np.finfo(x.dtype).eps
      tol = 50.0 * eps
      is_hermitian = False
      if m == n:
        if np.linalg.norm(x - x.T.conj()) / np.linalg.norm(x) < tol:
          is_hermitian = True
      return is_hermitian

    @partial(jit, static_argnums = (0,))
    def compute_cholesky(self, params, batch):
        X = batch['X']
        N, D = X.shape
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N) * (sigma_n + 1e-8)
        K = (K+ np.conj(K)) / 2.
        L = lax_linalg.cholesky(K, symmetrize_input=False)
        #L = cholesky(K, lower=True)
        return L

    def train(self, batch_list, rng_key, num_restarts=10):
        best_params = []
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, grads = self.likelihood_value_and_grad(params, batch)
                out = (onp.array(value), onp.array(grads))
                print(onp.asarray(grads))
                return out
            # Optimize with random restarts
            params = []
            likelihood = []
            dim = batch['X'].shape[1]
            rng_keys = random.split(rng_key, num_restarts)
            lista = dict()
            res = []
            niter = 0
            def callbackF(param):
                global niter 
                lista[niter] = []
                lista[niter] = [param, objective(param)] 
                niter += 1

            for i in range(num_restarts):
                init = 1e-3*initializers.random_init_GP(rng_keys[i], dim)
                p, val = minimize_lbfgs(objective, init, callback = None)
                #res = np.diff(np.array(list(lista.values())))
                #print(res[0])
                params.append(p)
                likelihood.append(val)

            params = np.vstack(params)
            likelihood = np.vstack(likelihood)
            #### find the best likelihood besides nan ####
            #print("likelihood", likelihood)
            bestlikelihood = np.nanmin(likelihood)
            idx_best = np.where(likelihood == bestlikelihood)
            idx_best = idx_best[0][0]
            best_params.append(params[idx_best,:])
            print("best_params", best_params)
        return best_params

    def train_de(self, batch_list):
        best_params = []
        dim = 22
        lb = -10.0 * onp.ones((dim + 2,))
        lb[-1] = -14
        ub = 0.0 * onp.ones((dim + 2,))
        ub[0] = 3.912
        ub[-1] = -4
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, _ = self.likelihood_value_and_grad(params, batch)
                out = onp.array(value)
                return out
            # Optimize with random restarts
            params = []
            likelihood = []
            dim = batch['X'].shape[1]
            #rng_keys = random.split(rng_key)[0]
            
            best_params, val = minimize_de(objective, bnds = dom_bounds)
            
        return best_params

    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star, **kwargs):
        mu_list = []
        std_list = []
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize to [0,1] (We should do this for once instead of iteratively
        # doing so in the for loop)
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])

        for k, (params, batch, norm_const) in enumerate(zipped_args):
            # Fetch normalized training data
            X, y = batch['X'], batch['y']
            # Fetch params
            sigma_n = np.exp(params[-1])
            theta = np.exp(params[:-1])
            # Compute kernels
            k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
            k_pX = self.kernel(X_star, X, theta)
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean, std
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            if k > 0:
                mu = mu * norm_const['sigma_y'] + norm_const['mu_y']
                std = std * norm_const['sigma_y']
            mu_list.append(mu)
            std_list.append(std)
        return np.array(mu_list), np.array(std_list)

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star,  **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = np.exp(params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std

    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict_all(x, **kwargs)
        if self.options['constrained_criterion'] == 'EIC':
            batch_list = kwargs['batch']
            best = np.min(batch_list[0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif self.options['constrained_criterion'] == 'LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 2*sigma
            #norm_const = kwargs['norm_const'][0]
            #mean[0,:] = (mean[0,:] - norm_const['mu_y']) /
            #norm_const['sigma_y'] - 3 * norm_const['sigma_y']
            #std[0,:] = std[0,:] / norm_const['sigma_y']
            #####
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options['constrained_criterion'] == 'LW_LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 3*sigma
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts=10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        best_val = np.min(acq)
        id_prova = onp.argmin(acq)
        best_prova = onp.min(acq)
        x_new = loc[idx_best:idx_best + 1,:]
        return x_new, acq, loc



    def fit_gmm(self, num_comp=2, N_samples=10000, **kwargs):
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']

        # load the seed
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]
        # Sample data across the entire domain
        X = lb + (ub - lb) * lhs(dim, N_samples)

        # set the seed for sampling X
        onp.random.seed(rng_key[0])
        X = lb + (ub - lb) * lhs(dim, N_samples)

        # We only keep the first row that correspond to the objective
        # prediction and same for y_samples
        prova = self.predict_all(X, **kwargs)
        y = self.predict_all(X, **kwargs)[0][0,:] #era: predict

        # Prediction of the constraints
        mu, std = self.predict_all(X, **kwargs)
        mu_c, std_c = mu[1:,:], std[1:,:]

        #print('mu_c', 'std_c', mu_c.shape, std_c.shape)
        constraint_w = np.ones((std_c.shape[1],1)).flatten()
        for k in range(std_c.shape[0]):
            constraint_w_temp = norm.cdf(mu_c[k,:] / std_c[k,:])
            if np.sum(constraint_w_temp) > 1e-8:
                constraint_w = constraint_w * constraint_w_temp
        #print("constraint_w", constraint_w.shape)

        # set the seed for sampling X_samples
        rng_key = random.split(rng_key)[0]
        onp.random.seed(rng_key[0])

        X_samples = lb + (ub - lb) * lhs(dim, N_samples)
        y_samples = self.predict_all(X_samples, **kwargs)[0][0,:]


        # Compute p_x and p_y from samples across the entire domain
        p_x = self.input_prior.pdf(X)
        p_x_samples = self.input_prior.pdf(X_samples)

        p_y = utils.fit_kernel_density(y_samples, y, weights = p_x_samples)

        #print("constraint_w", constraint_w.shape, "p_x", p_x.shape)
        weights = p_x / p_y * constraint_w
        # Label each input data
        indices = np.arange(N_samples)
        # Scale inputs to [0, 1]^D
        X = (X - lb) / (ub - lb)
        # rescale weights as probability distribution
        weights = weights / np.sum(weights)
        # Sample from analytical w
        idx = onp.random.choice(indices,
                                N_samples,
                                p=weights.flatten())
        X_train = X[idx]
        # fit GMM
        clf = mixture.GaussianMixture(n_components=num_comp,
                                      covariance_type='full')
        clf.fit(X_train)
        return clf.weights_, clf.means_, clf.covariances_



    @partial(jit, static_argnums=(0,))
    def draw_posterior_sample(self, X_star, **kwargs):
        sample_list = []
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        rng_key_list = kwargs['rng_key']
        zipped_args = zip(params_list, batch_list, norm_const_list, rng_key_list)
        for i, (params, batch, norm_const, rng_key) in enumerate(zipped_args):
            # Normalize to [0,1]
            X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
            # Fetch normalized training data
            X, y = batch['X'], batch['y']
            # Fetch params
            sigma_n = np.exp(params[-1])
            theta = np.exp(params[:-1])
            # Compute kernels
            k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
            k_pX = self.kernel(X_star, X, theta)
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            sample = random.multivariate_normal(rng_key, mu, cov)
            sample_list.append(sample)
        return np.array(sample)





# A minimal ManifoldGP regression class (inherits from GPmodel)
class ManifoldGP(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = utils.init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_GP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, batch['X'])
        N = X.shape[0]
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N) * (sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts=10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch['X'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            key1, key2 = random.split(rng_key[i])
            gp_params = initializers.random_init_GP(key1, dim)
            nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
            init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
            p, val = minimize_lbfgs(objective, init_params)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)
        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best,:]

        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, X)
        X_star = self.net_apply(nn_params, X_star)
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std

# A minimal ManifoldGP with Multiple Outputs regression class (inherits from
# GPmodel)
class ManifoldGP_MultiOutputs(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = utils.init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_GP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, batch['X'])
        N = X.shape[0]
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N) * (sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    def train(self, batch_list, rng_key, num_restarts=10):
        best_params = []
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, grads = self.likelihood_value_and_grad(params, batch)
                out = (onp.array(value), onp.array(grads))
                return out
            # Optimize with random restarts
            params = []
            likelihood = []
            dim = batch['X'].shape[1]
            rng_keys = random.split(rng_key, num_restarts)
            for i in range(num_restarts):
                key1, key2 = random.split(rng_keys[i])
                gp_params = initializers.random_init_GP(key1, dim)
                nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
                init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
                p, val = minimize_lbfgs(objective, init_params)
                params.append(p)
                likelihood.append(val)
            params = np.vstack(params)
            likelihood = np.vstack(likelihood)
            #### find the best likelihood besides nan ####
            bestlikelihood = np.nanmin(likelihood)
            idx_best = np.where(likelihood == bestlikelihood)
            idx_best = idx_best[0][0]
            best_params.append(params[idx_best,:])

        return best_params
    
    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star, **kwargs):
        mu_list = []
        std_list = []
        
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        
        for k, (params, batch, norm_const) in enumerate(zipped_args):
            # Fetch normalized training data
            X, y = batch['X'], batch['y']
            # Warp inputs
            gp_params = params[self.gp_params_ids]
            nn_params = self.unravel(params[self.nn_params_ids])
            X = self.net_apply(nn_params, X)
            X_star_nn = self.net_apply(nn_params, X_star)
            # Fetch params
            sigma_n = np.exp(gp_params[-1])
            theta = np.exp(gp_params[:-1])
            # Compute kernels
            k_pp = self.kernel(X_star_nn, X_star_nn, theta) + np.eye(X_star_nn.shape[0]) * (sigma_n + 1e-8)
            k_pX = self.kernel(X_star_nn, X, theta)
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean, std
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            
            mu_list.append(mu)
            std_list.append(std)
            
        return np.array(mu_list), np.array(std_list)
    
    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        
        # Fetch normalized training data
        X, y = batch['X'], batch['y']
        # Warp inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        X = self.net_apply(nn_params, X)
        X_star_nn = self.net_apply(nn_params, X_star)
        # Fetch params
        sigma_n = np.exp(gp_params[-1])
        theta = np.exp(gp_params[:-1])
        # Compute kernels
        k_pp = self.kernel(X_star_nn, X_star_nn, theta) + np.eye(X_star_nn.shape[0]) * (sigma_n + 1e-8)
        k_pX = self.kernel(X_star_nn, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict_all(x, **kwargs)
        if self.options['constrained_criterion'] == 'EIC':
            batch_list = kwargs['batch']
            best = np.min(batch_list[0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif self.options['constrained_criterion'] == 'LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 2*sigma
            #norm_const = kwargs['norm_const'][0]
            #mean[0,:] = (mean[0,:] - norm_const['mu_y']) /
            #norm_const['sigma_y'] - 3 * norm_const['sigma_y']
            #std[0,:] = std[0,:] / norm_const['sigma_y']
            #####
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options['constrained_criterion'] == 'LW_LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 3*sigma
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts=10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best + 1,:]
        return x_new, acq , loc
    
    

# A minimal MultifidelityGP regression class (inherits from GPmodel)
class MultifidelityGP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * (sigma_n_L + 1e-8)
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = rho ** 2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH) * (sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts=10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch['XH'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            init = initializers.random_init_MultifidelityGP(rng_key[i], dim)
            p, val = minimize_lbfgs(objective, init)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)
        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best,:]

        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
        psi1 = rho * self.kernel(X_star, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def predict_L(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta_L) + \
               np.eye(X_star.shape[0]) * (sigma_n_L + 1e-8)
        psi1 = self.kernel(X_star, XL, theta_L)
        psi1 = rho * self.kernel(X_star, XH, theta_L)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def posterior_covariance_L(self, x, xp, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        x = (x - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        xp = (xp - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        k_pp = self.kernel(x, xp, theta_L)
        psi1 = self.kernel(x, XL, theta_L)
        psi2 = rho * self.kernel(x, XH, theta_L)
        k_pX = np.hstack((psi1,psi2))
        psi1 = self.kernel(XL, xp, theta_L)
        psi2 = rho * self.kernel(XH, xp, theta_L)
        k_Xp = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        # Compute predictive mean, std
        beta = solve_triangular(L.T,solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov
    
    @partial(jit, static_argnums=(0,))
    def posterior_covariance_H(self, x, xp, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        x = (x - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        xp = (xp - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(x, xp, theta_L) + \
                        self.kernel(x, xp, theta_H)
        psi1 = rho * self.kernel(x, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(x, XH, theta_L) + \
                        self.kernel(x, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        psi1 = rho * self.kernel(XL, xp, theta_L)
        psi2 = rho ** 2 * self.kernel(XH, xp, theta_L) + \
                        self.kernel(XH, xp, theta_H)
        k_Xp = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        # Compute predictive mean, std
        beta = solve_triangular(L.T,solve_triangular(L, k_Xp, lower=True))
        cov = k_pp - np.matmul(k_pX, beta)
        return cov
    
    

# A minimal MultifidelityGP regression class (inherits from GPmodel)
class DeepMultifidelityGP(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        if options['net_arch'] == 'MLP':
            self.net_init, self.net_apply = utils.init_NN(layers)
            nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        if options['net_arch'] == 'ResNet':
            self.net_init, self.net_apply = utils.init_ResNet(layers,options['depth'],options['is_spect'])
            nn_params = self.net_init(random.PRNGKey(0))
        if options['net_arch'] == 'MomentumResNet':
            self.net_init, self.net_apply = utils.init_MomentumResNet(layers,options['depth'],options['vel_zeros'],options['gamma'])
            nn_params = self.net_init(random.PRNGKey(0))
        # Determine parameter IDs
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_MultifidelityGP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params
        
    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        # Warp inputs
        XL = self.net_apply(nn_params, XL)
        XH = self.net_apply(nn_params, XH)
        D = XH.shape[1]
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = np.exp(gp_params[:D + 1])
        theta_H = np.exp(gp_params[D + 1:-3])
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * (sigma_n_L + 1e-8)
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = rho ** 2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH) * (sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts=10, verbose=False, maxfun=15000):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = self.layers[-1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            key1, key2 = random.split(rng_key[i])
            gp_params = initializers.random_init_MultifidelityGP(key1, dim)
            if self.options['net_arch'] == 'MLP':
                nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
            if self.options['net_arch'] == 'ResNet':
                nn_params = self.net_init(key2)
            if self.options['net_arch'] == 'MomentumResNet':
                nn_params = self.net_init(key2)
            init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
            p, val = minimize_lbfgs(objective, init_params, verbose, maxfun)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)
        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best,:]

        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        y = batch['y']
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        # Warp inputs
        XL = self.net_apply(nn_params, XL)
        XH = self.net_apply(nn_params, XH)
        X_star = self.net_apply(nn_params, X_star)
        D = XH.shape[1]
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = np.exp(gp_params[:D + 1])
        theta_H = np.exp(gp_params[D + 1:-3])
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
        psi1 = rho * self.kernel(X_star, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std

# A minimal DeepMultifidelityGP with Multiple Outputs regression class
# (inherits from GPmodel)
class DeepMultifidelityGP_MultiOutputs(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        if options['net_arch'] == 'MLP':
            self.net_init, self.net_apply = utils.init_NN(layers)
            nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        if options['net_arch'] == 'ResNet':
            self.net_init, self.net_apply = utils.init_ResNet(layers,options['depth'],options['is_spect'])
            nn_params = self.net_init(random.PRNGKey(0))
        if options['net_arch'] == 'MomentumResNet':
            self.net_init, self.net_apply = utils.init_MomentumResNet(layers,options['depth'],options['vel_zeros'],options['gamma'])
            nn_params = self.net_init(random.PRNGKey(0))
        # Determine parameter IDs
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_MultifidelityGP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        # Warp inputs
        XL = self.net_apply(nn_params, XL)
        XH = self.net_apply(nn_params, XH)
        D = XH.shape[1]
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = np.exp(gp_params[:D + 1])
        theta_H = np.exp(gp_params[D + 1:-3])
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * (sigma_n_L + 1e-8)
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = rho ** 2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH) * (sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch_list, rng_key, num_restarts=10, verbose=False, maxfun=15000):
        best_params = []
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, grads = self.likelihood_value_and_grad(params, batch)
                out = (onp.array(value), onp.array(grads))
                return out
            # Optimize with random restarts
            params = []
            likelihood = []
            dim = self.layers[-1]
            rng_keys = random.split(rng_key, num_restarts)
            for i in range(num_restarts):
                key1, key2 = random.split(rng_keys[i])
                gp_params = initializers.random_init_MultifidelityGP(key1, dim)
                if self.options['net_arch'] == 'MLP':
                    nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
                if self.options['net_arch'] == 'ResNet':
                    nn_params = self.net_init(key2)
                if self.options['net_arch'] == 'MomentumResNet':
                    nn_params = self.net_init(key2)
                init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
                p, val = minimize_lbfgs(objective, init_params, verbose, maxfun)
                params.append(p)
                likelihood.append(val)
            params = np.vstack(params)
            likelihood = np.vstack(likelihood)
            #### find the best likelihood besides nan ####
            bestlikelihood = np.nanmin(likelihood)
            idx_best = np.where(likelihood == bestlikelihood)
            idx_best = idx_best[0][0]
            best_params.append(params[idx_best,:])

        return best_params

    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star, **kwargs):
        mu_list = []
        std_list = []
        
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        
        for k, (params, batch, norm_const) in enumerate(zipped_args):
            # Fetch normalized training data
            XL, XH = batch['XL'], batch['XH']
            y = batch['y']
            gp_params = params[self.gp_params_ids]
            nn_params = self.unravel(params[self.nn_params_ids])
            # Warp inputs
            XL = self.net_apply(nn_params, XL)
            XH = self.net_apply(nn_params, XH)
            X_star_nn = self.net_apply(nn_params, X_star)
            D = XH.shape[1]
            # Fetch params
            rho = gp_params[-3]
            sigma_n_L = np.exp(gp_params[-2])
            sigma_n_H = np.exp(gp_params[-1])
            theta_L = np.exp(gp_params[:D + 1])
            theta_H = np.exp(gp_params[D + 1:-3])
            # Compute kernels
            k_pp = rho ** 2 * self.kernel(X_star_nn, X_star_nn, theta_L) + \
                            self.kernel(X_star_nn, X_star_nn, theta_H) + \
                            np.eye(X_star_nn.shape[0]) * (sigma_n_H + 1e-8)
            psi1 = rho * self.kernel(X_star_nn, XL, theta_L)
            psi2 = rho ** 2 * self.kernel(X_star_nn, XH, theta_L) + \
                            self.kernel(X_star_nn, XH, theta_H)
            k_pX = np.hstack((psi1,psi2))
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean, std
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            
            mu_list.append(mu)
            std_list.append(std)
            
        return np.array(mu_list), np.array(std_list)
    
    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        y = batch['y']
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        # Warp inputs
        XL = self.net_apply(nn_params, XL)
        XH = self.net_apply(nn_params, XH)
        X_star_nn = self.net_apply(nn_params, X_star)
        D = XH.shape[1]
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = np.exp(gp_params[:D + 1])
        theta_H = np.exp(gp_params[D + 1:-3])
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(X_star_nn, X_star_nn, theta_L) + \
                        self.kernel(X_star_nn, X_star_nn, theta_H) + \
                        np.eye(X_star_nn.shape[0]) * (sigma_n_H + 1e-8)
        psi1 = rho * self.kernel(X_star_nn, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star_nn, XH, theta_L) + \
                        self.kernel(X_star_nn, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            
        return mu, std
    
    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict_all(x, **kwargs)
        if self.options['constrained_criterion'] == 'EIC':
            batch_list = kwargs['batch']
            best = np.min(batch_list[0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif self.options['constrained_criterion'] == 'LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 2*sigma
            #norm_const = kwargs['norm_const'][0]
            #mean[0,:] = (mean[0,:] - norm_const['mu_y']) /
            #norm_const['sigma_y'] - 3 * norm_const['sigma_y']
            #std[0,:] = std[0,:] / norm_const['sigma_y']
            #####
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options['constrained_criterion'] == 'LW_LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 3*sigma
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts=10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best + 1,:]
        return x_new, acq, loc
    
# A minimal GradientGP regression class (inherits from GPmodel)
class GradientGP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def k_dx2(self, x1, x2, params):
        fun = lambda x2: self.kernel(x1, x2, params)
        g = jvp(fun, (x2,), (np.ones_like(x2),))[1]
        return g

    @partial(jit, static_argnums=(0,))
    def k_dx1dx2(self, x1, x2, params):
        fun = lambda x1: self.k_dx2(x1, x2, params)
        g = jvp(fun, (x1,), (np.ones_like(x1),))[1]
        return g

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XF, XG = batch['XF'], batch['XG']
        NF, NG = XF.shape[0], XG.shape[0]
        # Fetch params
        sigma_n_F = np.exp(params[-2])
        sigma_n_G = np.exp(params[-1])
        theta = np.exp(params[:-2])
        # Compute kernels
        K_FF = self.kernel(XF, XF, theta) + np.eye(NF) * (sigma_n_F + 1e-8)
        K_FG = self.k_dx2(XF, XG, theta)
        K_GG = self.k_dx1dx2(XG, XG, theta) + np.eye(NG) * (sigma_n_G + 1e-8)
        K = np.vstack((np.hstack((K_FF,K_FG)),
                       np.hstack((K_FG.T,K_GG))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts=10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch['XF'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            init = initializers.random_init_GradientGP(rng_key[i], dim)
            p, val = minimize_lbfgs(objective, init)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)

        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
        best_params = params[idx_best,:]
        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        norm_const = kwargs['norm_const']
        # (do not Normalize!)
        # X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        # Fetch training data
        XF, XG = batch['XF'], batch['XG']
        y = batch['y']
        # Fetch params
        sigma_n_F = np.exp(params[-2])
        sigma_n_G = np.exp(params[-1])
        theta = np.exp(params[:-2])
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0]) * (sigma_n_F + 1e-8)
        psi1 = self.kernel(X_star, XF, theta)
        psi2 = self.k_dx2(X_star, XG, theta)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std



# A minimal Gaussian process regression class (inherits from GPmodel)
class MultipleIndependentMFGP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * (sigma_n_L + 1e-8)
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = rho ** 2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH) * (sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch_list, rng_key, num_restarts=10):
        best_params = []
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, grads = self.likelihood_value_and_grad(params, batch)
                out = (onp.array(value), onp.array(grads))
                return out
            # Optimize with random restarts
            params = []
            likelihood = []
            dim = batch['XH'].shape[1]
            rng_keys = random.split(rng_key, num_restarts)
            for i in range(num_restarts):
                init = initializers.random_init_MultifidelityGP(rng_keys[i], dim)
                p, val = minimize_lbfgs(objective, init)
                params.append(p)
                likelihood.append(val)
            params = np.vstack(params)
            likelihood = np.vstack(likelihood)
            #### find the best likelihood besides nan ####
            #print("likelihood", likelihood)
            bestlikelihood = np.nanmin(likelihood)
            idx_best = np.where(likelihood == bestlikelihood)
            idx_best = idx_best[0][0]
            best_params.append(params[idx_best,:])
            #print("best_params", best_params)
        return best_params

    # Predict all high fidelity prediction (objective + constraints)
    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star, **kwargs):
        mu_list = []
        std_list = []
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize to [0,1] (We should do this for once instead of iteratively
        # doing so in the for loop)
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])

        for k, (params, batch, norm_const) in enumerate(zipped_args):
            # Fetch normalized training data
            XL, XH = batch['XL'], batch['XH']
            D = batch['XH'].shape[1]
            y = batch['y']
            # Fetch params
            rho = params[-3]
            sigma_n_L = np.exp(params[-2])
            sigma_n_H = np.exp(params[-1])
            theta_L = np.exp(params[:D + 1])
            theta_H = np.exp(params[D + 1:-3])
            # Compute kernels
            k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                            self.kernel(X_star, X_star, theta_H) + \
                            np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
            psi1 = rho * self.kernel(X_star, XL, theta_L)
            psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                            self.kernel(X_star, XH, theta_H)
            k_pX = np.hstack((psi1,psi2))
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean, std
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            if k > 0:
                mu = mu * norm_const['sigma_y'] + norm_const['mu_y']
                std = std * norm_const['sigma_y']
            mu_list.append(mu)
            std_list.append(std)
        return np.array(mu_list), np.array(std_list)

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Fetch params
        rho = params[-3]
        sigma_n_L = np.exp(params[-2])
        sigma_n_H = np.exp(params[-1])
        theta_L = np.exp(params[:D + 1])
        theta_H = np.exp(params[D + 1:-3])
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
        psi1 = rho * self.kernel(X_star, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std


    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict_all(x, **kwargs)
        if self.options['constrained_criterion'] == 'EIC':
            batch_list = kwargs['batch']
            best = np.min(batch_list[0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif self.options['constrained_criterion'] == 'LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 2*sigma
            #norm_const = kwargs['norm_const'][0]
            #mean[0,:] = (mean[0,:] - norm_const['mu_y']) /
            #norm_const['sigma_y'] - 3 * norm_const['sigma_y']
            #std[0,:] = std[0,:] / norm_const['sigma_y']
            #####
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options['constrained_criterion'] == 'LW_LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 3*sigma
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts=10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best + 1,:]
        return x_new, acq, loc



    def fit_gmm(self, num_comp=2, N_samples=10000, **kwargs):
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']

        # load the seed
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]
        # Sample data across the entire domain
        X = lb + (ub - lb) * lhs(dim, N_samples)

        # set the seed for sampling X
        onp.random.seed(rng_key[0])
        X = lb + (ub - lb) * lhs(dim, N_samples)

        # We only keep the first row that correspond to the objective
        # prediction and same for y_samples
        y = self.predict_all(X, **kwargs)[0][0,:]

        # Prediction of the constraints
        mu, std = self.predict_all(X, **kwargs)
        mu_c, std_c = mu[1:,:], std[1:,:]

        #print('mu_c', 'std_c', mu_c.shape, std_c.shape)
        constraint_w = np.ones((std_c.shape[1],1)).flatten()
        for k in range(std_c.shape[0]):
            constraint_w_temp = norm.cdf(mu_c[k,:] / std_c[k,:])
            if np.sum(constraint_w_temp) > 1e-8:
                constraint_w = constraint_w * constraint_w_temp
        #print("constraint_w", constraint_w.shape)

        # set the seed for sampling X_samples
        rng_key = random.split(rng_key)[0]
        onp.random.seed(rng_key[0])

        X_samples = lb + (ub - lb) * lhs(dim, N_samples)
        y_samples = self.predict_all(X_samples, **kwargs)[0][0,:]


        # Compute p_x and p_y from samples across the entire domain
        p_x = self.input_prior.pdf(X)
        p_x_samples = self.input_prior.pdf(X_samples)

        p_y = utils.fit_kernel_density(y_samples, y, weights = p_x_samples)

        #print("constraint_w", constraint_w.shape, "p_x", p_x.shape)
        weights = p_x / p_y * constraint_w
        # Label each input data
        indices = np.arange(N_samples)
        # Scale inputs to [0, 1]^D
        X = (X - lb) / (ub - lb)
        # rescale weights as probability distribution
        weights = weights / np.sum(weights)
        # Sample from analytical w
        idx = onp.random.choice(indices,
                                N_samples,
                                p=weights.flatten())
        X_train = X[idx]
        # fit GMM
        clf = mixture.GaussianMixture(n_components=num_comp,
                                      covariance_type='full')
        clf.fit(X_train)
        return clf.weights_, clf.means_, clf.covariances_



# A minimal MultifidelityGP regression class for heterogeneous inputs (inherits
# from GPmodel)
class HeterogeneousMultifidelityGP(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = utils.init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_MultifidelityGP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Warp low-fidelity inputs to [0,1]^D_H
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[:D + 1]
        theta_H = gp_params[D + 1:-3]
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * (sigma_n_L + 1e-8)
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = rho ** 2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH) * (sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch, rng_key, num_restarts=10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch['XH'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            key1, key2 = random.split(rng_key[i])
            gp_params = initializers.random_init_MultifidelityGP(key1, dim)
            nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
            init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
            p, val = minimize_lbfgs(objective, init_params)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)

        #### find the best likelihood besides nan ####
        bestlikelihood = np.nanmin(likelihood)
        idx_best = np.where(likelihood == bestlikelihood)
        idx_best = idx_best[0][0]
#        idx_best = np.argmin(likelihood)
        #print("best index: {}".format(idx_best))
        best_params = params[idx_best,:]
        #print("shape of parameters: {}".format(best_params.shape))

        #### modify it according to the jupyter notebook
        #### Maybe jax will not stop when it see NaNs or Infs
        #print('Likelihoods for current iteration:
        #{}'.format(likelihood.flatten()))
        #print('Best likelihood for current iteration:
        #{}'.format(likelihood[idx_best]))
        return best_params

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Warp low-fidelity inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[:D + 1]
        theta_H = gp_params[D + 1:-3]
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
        psi1 = rho * self.kernel(X_star, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std




# A minimal Gaussian process regression class (inherits from GPmodel)
class MultipleIndependentHeterogeneousMFGP(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers
        self.net_init, self.net_apply = utils.init_NN(layers)
        # Determine parameter IDs
        nn_params = self.net_init(random.PRNGKey(0), (-1, layers[0]))[1]
        nn_params_flat, self.unravel = ravel_pytree(nn_params)
        num_nn_params = len(nn_params_flat)
        num_gp_params = initializers.random_init_MultifidelityGP(random.PRNGKey(0), layers[-1]).shape[0]
        self.gp_params_ids = np.arange(num_gp_params)
        self.nn_params_ids = np.arange(num_nn_params) + num_gp_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # Warp low-fidelity inputs to [0,1]^D_H
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[:D + 1]
        theta_H = gp_params[D + 1:-3]
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL) * (sigma_n_L + 1e-8)
        K_LH = rho * self.kernel(XL, XH, theta_L)
        K_HH = rho ** 2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH) * (sigma_n_H + 1e-8)
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        return L

    def train(self, batch_list, rng_key, num_restarts=10):
        best_params = []
        for _, batch in enumerate(batch_list):
            # Define objective that returns NumPy arrays
            def objective(params):
                value, grads = self.likelihood_value_and_grad(params, batch)
                out = (onp.array(value), onp.array(grads))
                return out
            # Optimize with random restarts
            params = []
            likelihood = []
            dim = batch['XH'].shape[1]
            rng_keys = random.split(rng_key, num_restarts)
            for i in range(num_restarts):
                key1, key2 = random.split(rng_keys[i])
                gp_params = initializers.random_init_MultifidelityGP(key1, dim)
                nn_params = self.net_init(key2,  (-1, self.layers[0]))[1]
                init_params = np.concatenate([gp_params, ravel_pytree(nn_params)[0]])
                p, val = minimize_lbfgs(objective, init_params)
                params.append(p)
                likelihood.append(val)
            params = np.vstack(params)
            likelihood = np.vstack(likelihood)
            #### find the best likelihood besides nan ####
            #print("likelihood", likelihood)
            bestlikelihood = np.nanmin(likelihood)
            idx_best = np.where(likelihood == bestlikelihood)
            idx_best = idx_best[0][0]
            best_params.append(params[idx_best,:])
            #print("best_params", best_params)
        return best_params

    # Predict all high fidelity prediction (objective + constraints)
    @partial(jit, static_argnums=(0,))
    def predict_all(self, X_star, **kwargs):
        mu_list = []
        std_list = []
        params_list = kwargs['params']
        batch_list = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const_list = kwargs['norm_const']
        zipped_args = zip(params_list, batch_list, norm_const_list)
        # Normalize to [0,1] (We should do this for once instead of iteratively
        # doing so in the for loop)
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])

        for k, (params, batch, norm_const) in enumerate(zipped_args):
            # Fetch normalized training data
            XL, XH = batch['XL'], batch['XH']
            D = batch['XH'].shape[1]
            y = batch['y']
            # Warp low-fidelity inputs
            gp_params = params[self.gp_params_ids]
            nn_params = self.unravel(params[self.nn_params_ids])
            XL = sigmoid(self.net_apply(nn_params, XL))
            # Fetch params
            rho = gp_params[-3]
            sigma_n_L = np.exp(gp_params[-2])
            sigma_n_H = np.exp(gp_params[-1])
            theta_L = gp_params[:D + 1]
            theta_H = gp_params[D + 1:-3]
            # Compute kernels
            k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                            self.kernel(X_star, X_star, theta_H) + \
                            np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
            psi1 = rho * self.kernel(X_star, XL, theta_L)
            psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                            self.kernel(X_star, XH, theta_H)
            k_pX = np.hstack((psi1,psi2))
            L = self.compute_cholesky(params, batch)
            alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
            beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
            # Compute predictive mean, std
            mu = np.matmul(k_pX, alpha)
            cov = k_pp - np.matmul(k_pX, beta)
            std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
            if k > 0:
                mu = mu * norm_const['sigma_y'] + norm_const['mu_y']
                std = std * norm_const['sigma_y']
            mu_list.append(mu)
            std_list.append(std)
        return np.array(mu_list), np.array(std_list)

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        bounds = kwargs['bounds']
        norm_const = kwargs['norm_const']
        # Normalize to [0,1]
        X_star = (X_star - bounds['lb']) / (bounds['ub'] - bounds['lb'])
        # Fetch normalized training data
        XL, XH = batch['XL'], batch['XH']
        D = batch['XH'].shape[1]
        y = batch['y']
        # Warp low-fidelity inputs
        gp_params = params[self.gp_params_ids]
        nn_params = self.unravel(params[self.nn_params_ids])
        XL = sigmoid(self.net_apply(nn_params, XL))
        # Fetch params
        rho = gp_params[-3]
        sigma_n_L = np.exp(gp_params[-2])
        sigma_n_H = np.exp(gp_params[-1])
        theta_L = gp_params[:D + 1]
        theta_H = gp_params[D + 1:-3]
        # Compute kernels
        k_pp = rho ** 2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0]) * (sigma_n_H + 1e-8)
        psi1 = rho * self.kernel(X_star, XL, theta_L)
        psi2 = rho ** 2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))

        return mu, std


    @partial(jit, static_argnums=(0,))
    def constrained_acquisition(self, x, **kwargs):
        x = x[None,:]
        mean, std = self.predict_all(x, **kwargs)
        if self.options['constrained_criterion'] == 'EIC':
            batch_list = kwargs['batch']
            best = np.min(batch_list[0]['y'])
            return acquisitions.EIC(mean, std, best)
        elif self.options['constrained_criterion'] == 'LCBC':
            kappa = kwargs['kappa']
            return acquisitions.LCBC(mean, std, kappa)
        elif self.options['constrained_criterion'] == 'LW_LCBC':
            kappa = kwargs['kappa']
            ##### normalize the mean and std again and subtract the mean by
            ##### 3*sigma
            weights = utils.compute_w_gmm(x, **kwargs)
            return acquisitions.LW_LCBC(mean, std, weights, kappa)
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def constrained_acq_value_and_grad(self, x, **kwargs):
        fun = lambda x: self.constrained_acquisition(x, **kwargs)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def constrained_compute_next_point_lbfgs(self, num_restarts=10, **kwargs):
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.constrained_acq_value_and_grad(x, **kwargs)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]

        onp.random.seed(rng_key[0])
        x0 = lb + (ub - lb) * lhs(dim, num_restarts)
        #print("x0 for bfgs", x0)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = minimize_lbfgs(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best + 1,:]
        return x_new, acq, loc


    def fit_gmm(self, num_comp=2, N_samples=10000, **kwargs):
        bounds = kwargs['bounds']
        lb = bounds['lb']
        ub = bounds['ub']

        # load the seed
        rng_key = kwargs['rng_key']
        dim = lb.shape[0]
        # Sample data across the entire domain
        X = lb + (ub - lb) * lhs(dim, N_samples)

        # set the seed for sampling X
        onp.random.seed(rng_key[0])
        X = lb + (ub - lb) * lhs(dim, N_samples)

        # We only keep the first row that correspond to the objective
        # prediction and same for y_samples
        y = self.predict_all(X, **kwargs)[0][0,:]

        # Prediction of the constraints
        mu, std = self.predict_all(X, **kwargs)
        mu_c, std_c = mu[1:,:], std[1:,:]

        #print('mu_c', 'std_c', mu_c.shape, std_c.shape)
        constraint_w = np.ones((std_c.shape[1],1)).flatten()
        for k in range(std_c.shape[0]):
            constraint_w_temp = norm.cdf(mu_c[k,:] / std_c[k,:])
            if np.sum(constraint_w_temp) > 1e-8:
                constraint_w = constraint_w * constraint_w_temp
        #print("constraint_w", constraint_w.shape)

        # set the seed for sampling X_samples
        rng_key = random.split(rng_key)[0]
        onp.random.seed(rng_key[0])

        X_samples = lb + (ub - lb) * lhs(dim, N_samples)
        y_samples = self.predict_all(X_samples, **kwargs)[0][0,:]


        # Compute p_x and p_y from samples across the entire domain
        p_x = self.input_prior.pdf(X)
        p_x_samples = self.input_prior.pdf(X_samples)

        p_y = utils.fit_kernel_density(y_samples, y, weights = p_x_samples)

        #print("constraint_w", constraint_w.shape, "p_x", p_x.shape)
        weights = p_x / p_y * constraint_w
        # Label each input data
        indices = np.arange(N_samples)
        # Scale inputs to [0, 1]^D
        X = (X - lb) / (ub - lb)
        # rescale weights as probability distribution
        weights = weights / np.sum(weights)
        # Sample from analytical w
        idx = onp.random.choice(indices,
                                N_samples,
                                p=weights.flatten())
        X_train = X[idx]
        # fit GMM
        clf = mixture.GaussianMixture(n_components=num_comp,
                                      covariance_type='full')
        clf.fit(X_train)
        return clf.weights_, clf.means_, clf.covariances_



