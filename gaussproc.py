
import numpy as np

from typing import Union, Dict, Callable, Optional, Tuple
import jax
import jaxopt
import jax.numpy as jnp
import jax.scipy as jsc
from jax import vmap, jit
jax.config.update("jax_enable_x64", True)

import numpyro
import numpyro.distributions as dist
from numpyro.infer import (
    MCMC,
    NUTS,
    init_to_feasible,
    init_to_median,
    init_to_sample,
    init_to_uniform,
    init_to_value,
)
from numpyro.handlers import seed, trace
numpyro.util.enable_x64()

if jax.__version__ < '0.2.26':
    clear_cache = jax.interpreters.xla._xla_callable.cache_clear
else:
    clear_cache = jax._src.dispatch._xla_callable.cache_clear


def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)

def square_scaled_distance(X, Z,lengthscale = 1.):
    """
    Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Z2 = (scaled_Z ** 2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)
def _sqrt(x, eps=1e-12):
    return jnp.sqrt(x + eps)

def square_scaled_distance(X, Z,lengthscale = 1.):
    """
    Computes a square of scaled distance, :math:`\|\frac{X-Z}{l}\|^2`,
    between X and Z are vectors with :math:`n x num_features` dimensions
    """
    scaled_X = X / lengthscale
    scaled_Z = Z / lengthscale
    X2 = (scaled_X ** 2).sum(1, keepdims=True)
    Z2 = (scaled_Z ** 2).sum(1, keepdims=True)
    XZ = jnp.matmul(scaled_X, scaled_Z.T)
    r2 = X2 - 2 * XZ + Z2.T
    return r2.clip(0)

def kernel_RBF(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    r2 = square_scaled_distance(X, Z, params["k_length"])
    k = params["k_scale"] * jnp.exp(-0.5 * r2)
    if X.shape == Z.shape:
        k +=  (noise + jitter) * jnp.eye(X.shape[0])
    return k

def kernel_Matern12(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    Matern nu=1/2 kernel; exponentiel decay
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    k = params["k_scale"] * jnp.exp(-r)
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

########### Kernels #############

def kernel_Matern32(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    Matern nu=3/2 kernel
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    sqrt3_r = 3**0.5 * r
    k = params["k_scale"] * (1.0 + sqrt3_r) * jnp.exp(-sqrt3_r)
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

def kernel_Matern52(X: jnp.ndarray, 
               Z: jnp.ndarray,  
               params: Dict[str, jnp.ndarray],
               noise: float =0.0, jitter: float=1.0e-6)-> jnp.ndarray:
    """
    Matern nu=5/2 kernel
    """
    r2 = square_scaled_distance(X, Z, params["k_length"])
    r = _sqrt(r2)
    sqrt5_r = 5**0.5 * r
    k = params["k_scale"] * (1.0 + sqrt5_r + sqrt5_r**2 /3.0) * jnp.exp(-sqrt5_r)
    if X.shape == Z.shape:
        k += (noise + jitter) * jnp.eye(X.shape[0])
    return k

########### Class #############
class GaussProc:
    """
    Gaussian process class
        Using C. E. Rasmussen & C. K. I. Williams, 
        Gaussian Processes for Machine Learning, the MIT Press, 2006 
        Alg. 2.1

    Args:
        kernel: GP kernel 
        mean_fn: optional deterministic mean function (use 'mean_fn_priors' to make it probabilistic)
        kernel_prior: optional custom priors over kernel hyperparameters 
        mean_fn_prior: optional priors over mean function parameters
        noise_prior: optional custom prior for observation noise
    """

    def __init__(self, kernel: Callable[[jnp.ndarray, 
                                         jnp.ndarray, 
                                         Dict[str, jnp.ndarray], 
                                         float, float],jnp.ndarray],
                 mean_fn: Optional[Callable[[jnp.ndarray, Dict[str, jnp.ndarray]], jnp.ndarray]] = None,
                 kernel_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 mean_fn_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None,
                 noise_prior: Optional[Callable[[], Dict[str, jnp.ndarray]]] = None
                 ) -> None:
        clear_cache()
        self.kernel = kernel
        self.mean_fn = mean_fn
        self.kernel_prior = kernel_prior
        self.mean_fn_prior = mean_fn_prior
        self.noise_prior = noise_prior
        self.mcmc = None

    def model(self, X:jnp.ndarray , y: jnp.ndarray):
        """GP probabilistic model"""
        # Initialize mean function at zeros
        f_loc = jnp.zeros(X.shape[0])
        # Sample kernel parameters
        kernel_params = self.kernel_prior()
        noise = self.noise_prior()
        # Add mean function (if any)
        if self.mean_fn is not None:
            args = [X]
            if self.mean_fn_prior is not None:
                args += [self.mean_fn_prior()]
            f_loc += self.mean_fn(*args).squeeze()
        # compute GP K(X,X)
        K = self.kernel(X, X, kernel_params,noise)
        # sample y according to the standard Gaussian process formula
        numpyro.sample(
            "y",
            dist.MultivariateNormal(loc=f_loc, covariance_matrix=K),
            obs=y,
        )
        
    def fit(self, rng_key:jnp.array, X_train: jnp.ndarray, y_train: jnp.ndarray, 
                num_warmup: int = 1_000, 
                num_samples: int = 1_000,
                num_chains: int = 1, 
                chain_method: str = 'vectorized',
                dense_mass: bool = True,
                progress_bar: bool = False, 
                print_summary: bool = True
                ) -> None:
        
        """
        Fit GP Kernel parameters using MCMC NUTS
        
        Args:
            rng_key: random number generator key
            X: 2D 'feature vector' with :math:`n x num_features`
            y: 1D 'target vector' with :math:`(n,)` dimensions
            num_warmup: number of MCMC warmup states
            num_samples: number of MCMC samples
            num_chains: number of MCMC chains
            chain_method: 'sequential', 'parallel' or 'vectorized'
            dense_mass: diagonal HMC mass matrix or full dense (optimized during warmup)
            progress_bar: show progress bar
            print_summary: print summary at the end of sampling
        """

        init_strategy = init_to_median(num_samples=100)

        kernel_nuts = NUTS(self.model, init_strategy=init_strategy, dense_mass=dense_mass)
        self.mcmc = MCMC(
            kernel_nuts,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method = chain_method,
            progress_bar=progress_bar
        )
        self.mcmc.run(rng_key, X_train, y_train)
        if print_summary:
            self.mcmc.print_summary()
            

    
    
    def get_samples(self, chain_dim: bool = False) -> Dict[str, jnp.ndarray]:
        """Get posterior samples (after running the MCMC chains)"""
        return self.mcmc.get_samples(group_by_chain=chain_dim)

    
    def get_marginal_logprob(self, X_train: jnp.ndarray, 
                        y_train: jnp.ndarray,
                        noise: float = None):
        
        log2pi = jnp.log(2.*jnp.pi)
        n_train = y_train.shape[0]
        mlp = - n_train/2 * log2pi
        
        samples = self.get_samples(chain_dim=False)
        params = {name:np.mean(value) for name, value in samples.items()}
        
        y_residual = y_train
        if self.mean_fn is not None:
            args = [X_train, params] if self.mean_fn_prior else [X_train]
            y_residual -= self.mean_fn(*args).squeeze()

        noise = noise if noise is not None else params["noise"]
        
        k_XX = self.kernel(X_train, X_train, params, noise)
        chol_XX = jsc.linalg.cholesky(k_XX, lower=True)
        v  = jsc.linalg.solve_triangular(chol_XX, y_residual, lower=True)
        mlp -= 0.5 * (jnp.dot(v.T,v)+jnp.sum(jnp.log(jnp.diag(chol_XX))))
        
        return mlp
        
        
        
    
    def get_mvn_posterior_cholesky(self,
                rng_key:jnp.array, 
                X_train: jnp.ndarray, y_train: jnp.ndarray, 
                X_new: jnp.ndarray, 
                params: Dict[str, jnp.ndarray],
                noise: float = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns parameters (mean and mean+srd) of multivariate normal posterior
        for a single sample of GP hyperparameters
        Version with the use of Cholesky decompostion
        """
        
        y_residual = y_train
        if self.mean_fn is not None:
            args = [X_train, params] if self.mean_fn_prior else [X_train]
            y_residual -= self.mean_fn(*args).squeeze()
        # compute kernel matrices for train and test data
        k_pp = self.kernel(X_new, X_new, params, jitter=0.0)    # was with "noise)" but certainly a bug
        k_pX = self.kernel(X_new, X_train, params, jitter=0.0)
        k_XX = self.kernel(X_train, X_train, params, noise)
        # compute the predictive covariance and mean
        chol_XX = jsc.linalg.cholesky(k_XX, lower=True)
        kinv_XX_y = jsc.linalg.solve_triangular(
            chol_XX.T, jsc.linalg.solve_triangular(chol_XX, y_residual, lower=True))

        mean = jnp.matmul(k_pX, kinv_XX_y)   # nb. K_pX = K(X_new, X_train) sometimes the transpose is used
        if  self.mean_fn is not None:
            args = [X_new, params] if self.mean_fn_prior else [X_new]
            mean += self.mean_fn(*args).squeeze()
            
        v = jsc.linalg.solve_triangular(chol_XX, k_pX.T, lower=True)
        cov = k_pp - jnp.dot(v.T, v)
        
        sigma = jnp.sqrt(jnp.clip(jnp.diag(cov), a_min=0.0)) 
        #####*\jax.random.normal(rng_key, X_new.shape[:1])

            
        return mean, sigma

    
    def predict(self, rng_key: jnp.ndarray, 
                X_train: jnp.ndarray, y_train: jnp.ndarray, 
                X_new: jnp.ndarray,
                samples: Optional[Dict[str, jnp.ndarray]] = None,
                noise: float = None
                ) -> Tuple[jnp.ndarray, jnp.ndarray]:

        X_new = X_new if X_new.ndim > 1 else X_new[:, None]
    
        if samples is None:
            samples = self.get_samples(chain_dim=False)
            
        num_samples=samples[list(samples.keys())[0]].shape[0]
        
        
        # do prediction
        
        vmap_args = (
            jax.random.split(rng_key, num_samples),
            samples,
            jnp.array([noise]*num_samples) if noise is not None else samples["noise"]
        )
        means, predictions = vmap(
            lambda rng_key, samples, noise: self.get_mvn_posterior_cholesky(
                rng_key, X_train, y_train, X_new, samples, noise)
            )(*vmap_args)
        
        return means, predictions 