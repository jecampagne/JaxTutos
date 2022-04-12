# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: JaxTutos
#     language: python
#     name: jaxtutos
# ---

# +
from pathinit import *

import numpy as np

# Simple Gaussian process class & utils
from gaussproc import *

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 18
mpl.rcParams["font.family"] = "Times New Roman"
import matplotlib.patches as mpatches
import corner
import arviz as az

import GPy


# -

def plot_params_kde(samples,hdi_probs=[0.393, 0.865, 0.989], 
                    patName=None, fname=None, pcut=None,
                   var_names=None, point_estimate="median", figsize=(8,8)):
        
    samples = {n:samples[n] for n in var_names}
    if pcut is not None:
        low = pcut[0]
        up  = pcut[1] 
        #keep only data in the [low, up] percentiles ex. 0.5, 99.5
        samples={name:value[(value>np.percentile(value,low)) &  (value<np.percentile(value,up))] \
          for name, value in samples.items()}
        len_min = np.min([len(value) for name, value in samples.items()])
        len_max = np.max([len(value) for name, value in samples.items()])
        if (len_max-len_min)>0.01*len_max:
            print(f"Warning: pcut leads to min/max spls size = {len_min}/{len_max}")
            samples = {name:value[:len_min] for name, value in samples.items()}
    
    axs= az.plot_pair(
            samples,
            var_names=var_names,
            kind="kde",
            figsize=figsize,
    #        marginal_kwargs={"plot_kwargs": {"lw": 3, "c": "b"}},
            kde_kwargs={
#                "hdi_probs": [0.68, 0.9],  # Plot 68% and 90% HDI contours
                "hdi_probs":hdi_probs,  # 1, 2 and 3 sigma contours
                "contour_kwargs":{"colors":('r', 'green', 'blue'), "linewidths":3},
                "contourf_kwargs":{"alpha":0},
            },
            point_estimate_kwargs={"lw": 3, "c": "b"},
            marginals=True, textsize=20, point_estimate=point_estimate
        );
    
    plt.tight_layout()
    
    if patName is not None:
        patName_patch = mpatches.Patch(color='b', label=patName)
        axs[0,0].legend(handles=[patName_patch], fontsize=40, bbox_to_anchor=(1, 0.7));
    if fname is not None:
        plt.savefig(fname)
        plt.close()


def mean_fn(x, params):
    """Parametrisation avant et apres RT (t=0) """
    R0 = params["R0"]
    v  = params["v"]
    k  = params["k"]
    tau =  params["tau"]
    return jnp.piecewise(
        x, [x < 0, x >= 0],
        [lambda x: R0 + v*x, 
         lambda x: R0 + v*x - k*(1.-jnp.exp(-x/tau))
        ])


rng_key = jax.random.PRNGKey(42)
rng_key, rng_key0, rng_key1, rng_key2 = jax.random.split(rng_key, 4)

tMes = jax.random.uniform(rng_key0,minval=-5.,maxval=5.0,shape=(20,))
tMes=jnp.append(tMes,0.0)
tMes=jnp.sort(tMes)

par_true={"R0":35.0, "v":2.20, "k":15.5, "tau": 1.0}
sigma_obs=1.0

RMes = mean_fn(tMes,par_true) + sigma_obs * jax.random.normal(rng_key1,shape=tMes.shape)



plt.errorbar(tMes,RMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
plt.xlabel("t")
plt.ylabel("R")
plt.legend()
plt.grid();


# +
def kernel_prior():
    length = numpyro.sample("k_length", numpyro.distributions.Uniform(0.1, 10))
    scale = numpyro.sample("k_scale", numpyro.distributions.LogNormal(0, 1))
    return {"k_length": length, "k_scale": scale}

def noise_prior():
    noise = numpyro.deterministic("noise",sigma_obs**2)
    return noise

gp = GaussProc(kernel=kernel_RBF, kernel_prior=kernel_prior, noise_prior=noise_prior)
# -

gp.fit(rng_key2,X_train=tMes[:,jnp.newaxis],y_train=RMes, num_warmup=1_000, num_samples=5_000, progress_bar=True)

gp.get_marginal_logprob(X_train=tMes[:,jnp.newaxis],y_train=RMes)

samples = gp.get_samples()

samples.keys()

az.ess(samples, relative=True)

plot_params_kde(samples,pcut=[0,99.9],var_names=['k_length','k_scale'], figsize=(8,8))

rng_key, rng_key_new = jax.random.split(rng_key)

t_val = np.linspace(-5,5,200)

Rtrue_val = mean_fn(t_val,par_true)

means,  stds= gp.predict(rng_key_new, X_train=tMes[:,jnp.newaxis],y_train=RMes, X_new=t_val[:,jnp.newaxis])

means.shape

Rmean_val = jnp.mean(means,axis=0)

# +
#percentiles = jnp.percentile(means_std, jnp.array([4.55, 31.73, 68.27, 95.45]), axis=0) 
# -

std = jnp.mean(stds,axis=0)

# +
fig=plt.figure(figsize=(10,8))
plt.errorbar(tMes,RMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
plt.plot(t_val,Rtrue_val,c='k',label="true")

plt.fill_between(t_val,Rmean_val-2*std,Rmean_val+2*std, color="lightblue", label=r"$\pm$ 2 std. dev.")
plt.fill_between(t_val, Rmean_val-std,Rmean_val+std, color="lightgray",label=r"$1-\sigma$")
# plot mean prediction
plt.plot(t_val, Rmean_val, "blue", ls="--", lw=2.0, label="mean")


plt.xlabel("t")
plt.ylabel("R")
plt.legend()
plt.grid();
# -
#

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


kernel = ConstantKernel(constant_value=100., constant_value_bounds=(10., 1000.)) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 10.0))\
        + WhiteKernel(noise_level=sigma_obs**2, noise_level_bounds="fixed")

gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(tMes[:,np.newaxis], RMes)

print(f"Kernel parameters before fit:\n{kernel})")
print(
    f"Kernel parameters after fit: \n{gpr.kernel_} \n"
    f"Log-likelihood: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

R_mean_test, R_std_test = gpr.predict(t_val[:,jnp.newaxis], return_std=True)

# +
fig=plt.figure(figsize=(10,8))
plt.errorbar(tMes,RMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
plt.plot(t_val,Rtrue_val,c='k',label="true")

plt.fill_between(t_val, R_mean_test-R_std_test, R_mean_test+R_std_test, color="lightgray",label=r"$1-\sigma$")
# plot mean prediction
plt.plot(t_val, R_mean_test, "blue", ls="--", lw=2.0, label="mean")


plt.xlabel("t")
plt.ylabel("R")
plt.legend()
plt.grid();
# -

# # D'où vient la différence de largeur de la bande à 1-$\sigma$ ?

kernel = GPy.kern.RBF(input_dim=1, variance=100., lengthscale=1.0)

m = GPy.models.GPRegression(tMes[:,np.newaxis], RMes[:,np.newaxis],kernel)

m.parameter_names()

m['rbf.variance']

m['Gaussian_noise.variance'].constrain_fixed(sigma_obs**2)

m['Gaussian_noise.variance']

m.optimize(messages=True)

y_mean_test, y_var_test = m.predict(t_val[:,np.newaxis])
y_mean_test_nn, y_var_test_nn = m.predict_noiseless(t_val[:,np.newaxis])

# +
y_mean_test  = y_mean_test.squeeze()
y_std_test = np.sqrt(y_var_test)
y_std_test  = y_std_test.squeeze()

y_mean_test_nn  = y_mean_test_nn.squeeze()
y_std_test_nn = np.sqrt(y_var_test_nn)
y_std_test_nn  = y_std_test_nn.squeeze()


# +
fig1,axs = plt.subplots(1,2,figsize=(20,8))

axs[0].errorbar(tMes,RMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
axs[0].plot(t_val,Rtrue_val,c='k',label="true")

axs[0].plot(t_val, y_mean_test, color="blue", ls='--', label="mean")
axs[0].fill_between(t_val,y_mean_test - 2*y_std_test,y_mean_test + 2*y_std_test,
        color="lightblue", label=r"$\pm$ 2 std. dev.")

axs[0].fill_between(t_val,y_mean_test - y_std_test,y_mean_test + y_std_test,
        color="lightgray", label=r"$\pm$ 1 std. dev." )
axs[0].set_xlabel("t")
axs[0].set_ylabel("R")
axs[0].legend()
axs[0].grid()
axs[0].set_title("Predict w/ noise");

axs[1].errorbar(tMes,RMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
axs[1].plot(t_val,Rtrue_val,c='k',label="true")

axs[1].plot(t_val, y_mean_test_nn, color="blue", ls='--', label="mean")
axs[1].fill_between(t_val,y_mean_test_nn - 2*y_std_test_nn,y_mean_test_nn + 2*y_std_test_nn,
        color="lightblue", label=r"$\pm$ 2 std. dev.")

axs[1].fill_between(t_val,y_mean_test_nn - y_std_test_nn,y_mean_test_nn + y_std_test_nn,
        color="lightgray", label=r"$\pm$ 1 std. dev." )
axs[1].set_xlabel("t")
axs[1].set_ylabel("R")
axs[1].legend()
axs[1].grid()
axs[1].set_title("Predict w/o noise");

# -

# # Q: si on change le range de [-5,5] à [-20,20] que se passe-t'il ? Pourquoi ?

# # Gaussian Process avec $m(x)\neq 0$

# +
def mean_fn(x, params):
    """Power-law behavior before and after the transition"""
    return jnp.piecewise(
        x, [x < 0, x >= 0],
        [lambda x: params["R0"]+params["v"]*x, 
         lambda x: params["R0"]+params["v"]*x - params["k"]*(1.-jnp.exp(-x/params["tau"]))
        ])

def mean_fn_prior():
    # Sample model parameters
    R0 = numpyro.sample("R0", numpyro.distributions.HalfCauchy(2.))
    v = numpyro.sample("v", numpyro.distributions.HalfCauchy(2.))
    k = numpyro.sample("k", numpyro.distributions.HalfCauchy(2.))
    tau = numpyro.sample("tau", numpyro.distributions.HalfCauchy(2.))
    # Return sampled parameters as a dictionary
    return { "R0": R0, "v":v, "k":k, "tau":tau}


def kernel_prior():
    length = numpyro.sample("k_length", numpyro.distributions.HalfCauchy(2.))
    scale = numpyro.sample("k_scale", numpyro.distributions.HalfCauchy(2.))
    return {"k_length": length, "k_scale": scale}

def noise_prior():
    noise = numpyro.deterministic("noise",sigma_obs**2)
    return noise



# -

gp = GaussProc(kernel=kernel_RBF, 
                kernel_prior=kernel_prior, 
                noise_prior=noise_prior,
                mean_fn=mean_fn,
                mean_fn_prior=mean_fn_prior)

rng_key = jax.random.PRNGKey(42)
rng_key, rng_key0, rng_key1, rng_key2 = jax.random.split(rng_key, 4)

gp.fit(rng_key2,X_train=tMes[:,jnp.newaxis],y_train=RMes, num_warmup=1_000, num_samples=5_000, progress_bar=True)

samples = gp.get_samples()

az.ess(samples,relative=True)

plot_params_kde(samples,pcut=[5,95],var_names=['R0','v','k','tau','k_length','k_scale'],figsize=(10,10))

means,  stds= gp.predict(rng_key_new, X_train=tMes[:,jnp.newaxis],y_train=RMes, X_new=t_val[:,jnp.newaxis])

Rmean_val = jnp.mean(means,axis=0)
std = jnp.mean(stds,axis=0)

# +
fig=plt.figure(figsize=(10,8))
plt.errorbar(tMes,RMes,yerr=sigma_obs,fmt='o', linewidth=2, capsize=0, c='k', label="data")
plt.plot(t_val,Rtrue_val,c='k',label="true")

plt.fill_between(t_val,Rmean_val-2*std,Rmean_val+2*std, color="lightblue", label=r"$\pm$ 2 std. dev.")
plt.fill_between(t_val, Rmean_val-std,Rmean_val+std, color="lightgray",label=r"$1-\sigma$")
# plot mean prediction
plt.plot(t_val, Rmean_val, "blue", ls="--", lw=2.0, label="mean")


plt.xlabel("t")
plt.ylabel("R")
plt.legend()
plt.grid();
# -


