# JaxTutos
This repository provides some notebooks to learn JAX (basics and advenced) and use some libraries such as JaxOptim/Numpyro/...

Ce repository fournit quelques notebooks pour apprendre JAX (simple & complex) et utliser quelques librairies telles que JaxOptim/Numpyro/...

# Echanges: 
- Pour discuter vous pouvez utiliser le champ/`Discussions`
- To discuss you can use :  `Discussions`

- Pour suggérer des modifications de code et/ou rapporter des bugs: utiliser les traditionels `Issues` & `Pull requests`
- To suggest code changes and/or report bugs use Issues & PR.

# Here the list of Tutos:
- [JAX_get_started.ipynb](./JAX_get_started.ipynb) : get a flavour of the coding and exemple of auto-diff
- [JAX-AutoDiff-UserCode.ipynb](./JAX-AutoDiff-UserCode.ipynb) : more on usage of auto diff in  real user-case "integration methods"  
- [JIT_fractals.ipynb](./JIT_fractals.ipynb) : **(GPU better)** with some fractal images production discover some control flow jax.lax functions and nested vmap
- [JAX_control_flow.ipynb](./JAX_control_flow.ipynb): jax.lax control flow (fori_loop/scan/while_loop, cond). Some "crashes" are analysed: **"always scan when you can!"**
- [JAX-MultiGPus.ipynb](./JAX-MultiGPus.ipynb) : **(4 GPUs)*** (ok on Jean Zay nb. plateform) use the "data sharding module" to distribute arrays and perform parallelization (2D image productions: simple 2d function and revisit of Julia set from `JIT_fractals.ipynb`.
# More advanced topics:
Designed for people with OO thinking (C++/Python), and/or with in mind  to existing code to transform into JAX. Based on real use case I experienced. This is more advenced and technical but with with "crashes" analysed
- [JAX-JIT_in_class.ipynb](./JAX-JIT_in_class.ipynb): how to use JIT for class methods (as opposed to JIT for an isolated function). 
- [JAX_PyTree_initialisation.ipynb](./JAX_PyTree_initialisation.ipynb): how to perform variable initilisation in a class
- [JAX_static_traced_var_func.ipynb](./JAX_static_traced_var_func.ipynb): why and when one needs to use pure Numpy function to make JIT ok
- `JAX_MC_Sampling.ipynb`: more on coding style/method in a use case of Monte Carlo Sampler implementation from scratch
# Using JAX & some thrid party libraries for real job
- [MC_Sampling.ipynb](./MC_Sampling.ipynb): pedagogical nb for Monte Carlo Sampling using different techniques. Beyond the math, one experiences the random number generation in JAX which by itself can be a subject. I implement a simple HMC MCMC both in Python and JAX to see the difference.
- [Numpyro_MC_Sampling.ipynb](./Numpyro_MC_Sampling.ipynb): here we give some simple examples using the Numpyro Probabilistic Programming Language
- [JAX_jaxopt_optax.ipynb](./JAX_jaxopt_optax.ipynb): some use of JaxOptim & Optax libraries


# Not yet ported to Collab.

- [JAX-GP-regression-piecewise.ipynb](./JAX-GP-regression-piecewise.ipynb): a home made Gaussian Processes lib to see differences with Sklearn et GPy.


## Installation
Most of the nbs are running on Colab. (JAX 0.4.4) 

If you want an environement Conda `JaxTutos` (but this is not garanteed to work due to the local & specific cuda library to be used for the GPU-based nb)
```
conda create -n JaxTutos python=3.8
conda activate JaxTutos
pip install --upgrade "jax[cuda]==0.4.4" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpyro==0.10.1
pip install jaxopt==0.6
pip install optax==0.1.4
pip install corner==2.2.1
pip install arviz==0.11.4
#pip install GPy==1.10.0
#pip install scikit-learn==1.0.2
pip install matplotlib_inline
pip install seaborn==0.12.2
```
# Docs
- JAX: https://jax.readthedocs.io
- numpy : https://numpy.org/doc/stable/reference/index.html
- Numpyro : https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro
- arviz : https://arviz-devs.github.io/arviz/index.html
- matplotlib : https://matplotlib.org/stable/index.html
- JaxOpt: https://jaxopt.github.io/stable/
- scikit-learn = https://scikit-learn.org/stable/index.html

- Anaconda : https://docs.anaconda.com/anaconda/install/index.html
- environement anaconda: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands
- PIP : https://docs.python.org/fr/3.8/installing/index.html#basic-usage

# Qq (autres) librairies JAX: 
- voir le site https://project-awesome.org/n2cholas/awesome-jax 
