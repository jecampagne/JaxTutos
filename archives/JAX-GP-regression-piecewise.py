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
import matplotlib.patches as mpatches
import corner
import arviz as az

# Librairies "non JAX" de GP
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
# -

import matplotlib as mpl



mpl.__version__


# # Thème: usage simple des Gaussian processes (codage maison avec Numpyro)
#
# On se sert du use-case du notebook `JAX-Optim-regression`.

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
    """Parametrisation avant et apres (t=0) """
    R0 = params["R0"]
    v  = params["v"]
    k  = params["k"]
    tau =  params["tau"]
    return jnp.where(x < 0, R0 + v*x, R0 + v*x - k*(1.-jnp.exp(-x/tau)))


rng_key = jax.random.PRNGKey(42)
rng_key, rng_key0, rng_key1, rng_key2 = jax.random.split(rng_key, 4)

n_dataset = 20
tMes = jax.random.uniform(rng_key0,minval=-5.,maxval=5.0,shape=(n_dataset,))
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


# # On va se servir d'une implémentation JAX/Numpyro d'un GP
#
# ![image.png](attachment:58042b6f-a12e-47c7-87ea-2adc786b77fa.png)
#
# ![image.png](attachment:68a5429c-8cb8-4aa1-ba99-4c90034c6d97.png)
#
# ![image.png](attachment:eb2d2c67-9945-4297-b866-3aab14a1b14e.png)
#
# ![image.png](attachment:e006e4ec-f2fd-4fdb-8e32-2a560685e1f2.png)
#
# ## Ouvrons ensemble le fichier `gaussproc.py` et vous allez vous rendre compte que vous `comprenez le code` (incroyable! Bravo!)

# # Les priors...
# Connaissant le sigma sur les données, on utilise `numpyro.deterministic`

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

# # Le fit des hyperparamètres
# ## Ici j'ai utilisé NUTS, quelle autre méthode aurai-je pu utiliser?

gp.fit(rng_key2,X_train=tMes[:,jnp.newaxis],y_train=RMes, 
       num_warmup=1_000, num_samples=5_000, progress_bar=True)

gp.get_marginal_logprob(X_train=tMes[:,jnp.newaxis],y_train=RMes)

samples = gp.get_samples()

samples.keys()

az.ess(samples, relative=True) # nb. "noise" étant `deterministe` il faut oublier le calcul

plot_params_kde(samples,pcut=[0,99.9],var_names=['k_length','k_scale'], figsize=(8,8))

rng_key, rng_key_new = jax.random.split(rng_key)

t_val = np.linspace(-5,5,200)

Rtrue_val = mean_fn(t_val,par_true)

means,  stds= gp.predict(rng_key_new, X_train=tMes[:,jnp.newaxis],y_train=RMes, 
                         X_new=t_val[:,jnp.newaxis])

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
# ## Commentez la figure en comparant par exemple avec celle de `Jax-Optim-regression`?

# # Voyons ce que donne Sklean


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

# # D'où vient la différence de largeur de la bande à 1-$\sigma$ en comparant avec le résultat obtenu avec la classe `gaussproc.py` ou les versions `Jax-Optim` `Jax-NUTS`?
#
# # Voyons ce que donne la librairie `GPy`

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

# ##1 Donc la différence des bandes à 1-sigma (et 2-sigma aussi) vient  de ce qu'on appelle une prédiction avec un modèle de bruit des données. La version `predict` de `Sklearn` (et `GPy`) considère qu'en un nouveau point $t_{new}$ on doit donner à l'utilisateur l'idée de ce qu'il aurait eu s'il avait une mesure en ce point; tandis que `predict_noiseless` de `GPy` et `predict` de `gaussproc.py` est conforme la l'algorithme de *C. E. Rasmussen & C. K. I. Williams* qui présente la prédiction en $t_{new}$ du modèle à la manière que fait un filtre de Kalman. Le problème devient aigü quand dans Sklearn on ajoute un WhiteKernel.
#
# ![image.png](attachment:b4f1fa2e-ca56-4879-b2e2-6af481670481.png)
#
# Vous pouvez lire cette issue que j'ai ouverte [ici](https://github.com/scikit-learn/scikit-learn/issues/22945) sur le Github de Sklearn.

# # Q: si on change le range de [-5,5] à [-20,20] que se passe-t'il ? Pourquoi ?

# # Gaussian Process avec $m(x)\neq 0$

# +
def mean_fn_prior(
    R0_min=10.,v_min = 0.5,k_min = 1., tau_min=0.1,
    R0_max=50.,v_max = 3.5,k_max = 20., tau_max=5.0):
    
    R0 = numpyro.sample("R0", dist.Uniform(R0_min,R0_max))
    v  = numpyro.sample("v", dist.Uniform(v_min,v_max))
    k  = numpyro.sample("k", dist.Uniform(k_min,k_max))
    tau= numpyro.sample("tau", dist.Uniform(tau_min,tau_max))
    
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

# # Dans un premier temps mettre `k_max=5` par exemple, et voir ce qu'il se passe, puis augmenter cette valeur (`k_max = 20` devrait aller). Regarder si les autres variables sont ok.

gp.fit(rng_key2,X_train=tMes[:,jnp.newaxis],y_train=RMes, 
       num_chains=1,
       chain_method='vectorized',
       num_warmup=1_000, num_samples=5_000,
       progress_bar=True)

samples = gp.get_samples()

az.ess(samples,relative=True)

plot_params_kde(samples,pcut=[5,95],var_names=['R0','v','k','tau','k_length','k_scale'],figsize=(10,10))

means,  stds= gp.predict(rng_key_new, X_train=tMes[:,jnp.newaxis],
                         y_train=RMes, X_new=t_val[:,jnp.newaxis])

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
# ## Commentez la figure. 


# # Exercices:
# - Posez vous la question de savoir comment obtenir l'hiistogramme  sur `tmin` (voir la fin de `JAX-Optim-regression`)? 
# - Dans le code de GaussProc.fit, j'ai utilisé un méthode MCMC NUTS pour ajuster les hyperparamètres du GP. Essayer de coder une méthode basée sur `jaxopt`(nb `JAX-Optim-regression`).
#
# 1-2-3 à vous de coder...

# # Takeaway message
# - Dans ce nb, nous n'avons pas appris de nouvelles fonctionalités de JAX car `vous en savez déjà beaucoup pour vous lancer`. Bravo!
# - Concernant les GP, on peut tout à fait se faire la main en codant soit même une petite librairie qui a des fonctionalités minimales. 
# - Ce faisant on découvre que `la notion de bruit des données  en lien avec la notion de prédiction` est plus délicate qu'il n'y parait et il faut faire très attention à ce que délivre finalement les librairies et on ne peut totalemen les utilsier coemme des `boites-noires`.


