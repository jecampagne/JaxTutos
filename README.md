# JaxTutos
This repository provides some notebooks to learn JAX (basics and advenced) and use some libraries such as JaxOptim/Numpyro/...

Ce repository fournit quelques notebooks pour apprendre JAX (simple & complex) et utliser quelques librairies telles que JaxOptim/Numpyro/...

This work was partily performed using resources from GENCI–IDRIS (Grant 2024-[AD010413957R1]).

# Questions:  
- Why JAX?: You need Auto-differention first and want a code accelerated ready on CPU/GPU/TPU devices, you probably already know a bit of Python.
- Does my code will be scalable? Gemini (ie. the Google ChatGPT alter-ego) is coded in JAX so I guess you will find good framework to get your use-case working nicely.

# Echanges/Exchanges: 
- To discuss you can use the `Discussions` menu
- To suggest new notebooks, code changes and/or report bugs use `Issues`.

# Here the list of Tutos in this repo:
## A tour of some basics
- [JAX_get_started.ipynb](./JAX_get_started.ipynb) : get a flavour of the coding and exemple of auto-diff
- [JAX_AutoDiff_UserCode.ipynb](./JAX_AutoDiff_UserCode.ipynb) : more on usage of auto diff in  real user-case "integration methods"  
- [JIT_fractals.ipynb](./JIT_fractals.ipynb) : **(GPU better)** with some fractal images production discover some control flow jax.lax functions and nested vmap
- [JAX_control_flow.ipynb](./JAX_control_flow.ipynb): jax.lax control flow (fori_loop/scan/while_loop, cond). Some "crashes" are analysed.
- [JAX_exo_sum_image_patches.ipynb](./JAX_exo_sum_image_patches.ipynb): Exercice: sum patches of identical size from a 2D image. Experience the compilation/execution times differences of different methods on CPU and GPU (if possible).
- [JAX-MultiGPus.ipynb](./JAX-MultiGPus.ipynb) : **(4 GPUs)*** (eg. on Jean Zay jupytyterHub plateform) use the "data sharding module" to distribute arrays and perform parallelization (2D image productions: simple 2d function and revisit of Julia set from `JIT_fractals.ipynb`.
## More advanced topics:
Designed for people with OO thinking (C++/Python), and/or with in mind  to existing code to transform into JAX. Based on real use case I experienced. This is more advenced and technical but with with "crashes" analysed
- [JAX_JIT_in_class.ipynb](./JAX_JIT_in_class.ipynb): how to use JIT for class methods (as opposed to JIT for an isolated function). 
- [JAX_static_traced_var_func.ipynb](./JAX_static_traced_var_func.ipynb): why and when one needs to use pure Numpy function to make JIT ok
- [JAX_PyTree_initialisation.ipynb](./JAX_PyTree_initialisation.ipynb): how to perform variable initilisation in a class
## Using JAX & some thrid party libraries for real job
- [JAX_jaxopt_optax.ipynb](./JAX_jaxopt_optax.ipynb): some use of JaxOptim & Optax libraries
- [JAX_MC_Sampling.ipynb](./JAX_MC_Sampling.ipynb): pedagogical nb for Monte Carlo Sampling using different techniques. Beyond the math, one experiences the random number generation in JAX which by itself can be a subject. I implement a simple HMC MCMC both in Python and JAX to see the difference.
- [Numpyro_MC_Sampling.ipynb](./Numpyro_MC_Sampling.ipynb): here we give some simple examples using the Numpyro Probabilistic Programming Language
- [JAX-GP-regression-piecewise.ipynb](./JAX-GP-regression-piecewise.ipynb): (**Not ready for Calob**) my implementation of Gaussian Processe library to see differences with Sklearn et GPy.

## Other TUTOs (absolutly not exhaustive...)
- [JAX readthedocs Tutos](https://jax.readthedocs.io/en/latest/tutorials.html): at least up-to-date
- [Kaggle TF_JAX Tutos (23 Dec. 2021)](https://www.kaggle.com/code/aakashnain/tf-jax-tutorials-part1): Ok, but pb. JAX  v0.2.26
- [Keras 3 JAX Backend guide](https://keras.io/guides/): jax==0.4.20 

# Other librairies JAX: 
- Have a look at  [awesome-jax](https://project-awesome.org/n2cholas/awesome-jax)
- More Cosmo-centred:
   - [JaxPM](https://github.com/DifferentiableUniverseInitiative/JaxPM): JAX-powered Cosmological Particle-Mesh N-body Solver
   - [S2FFT](http://www.jasonmcewen.org/project/s2fft/): JAX package for computing Fourier transforms on the sphere and rotation group
   - [JAX-Cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo): a differentiable cosmology library in JAX
   - [JAX-GalSim](https://github.com/GalSim-developers/JAX-GalSim): JAX version (paper in draft version) of the C++ Galsim code (GalSim is open-source software for simulating images of astronomical objects (stars, galaxies) in a variety of ways)
   - [CosmoPower-JAX](https://github.com/dpiras/cosmopower-jax): example of cosmological power spectra emulator in a differentiable way
   - DISCO-DJ I: a differentiable Einstein-Boltzmann solver for cosmology ([here](https://arxiv.org/abs/2311.03291)): not yet public repo.
- and many others concerning for instance Simulation Based Inference...


## Installation (it depends on your local environment)
Most of the nbs are running on Colab. (JAX 0.4.2x) 

If you want an environement Conda `JaxTutos` (but this is not garanteed to work due to the local & specific cuda library to be used for the GPU-based nb)
```
conda create -n JaxTutos python=3.8
conda activate JaxTutos
pip install --upgrade "jax[cuda]==<XYZ>" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpyro==0.10.1
pip install jaxopt==0.6
pip install optax==0.1.4
pip install corner==2.2.1
pip install arviz==0.11.4
pip install matplotlib_inline
pip install seaborn==0.12.2
```
# Some Docs
- JAX: https://jax.readthedocs.io
- numpy : https://numpy.org/doc/stable/reference/index.html
- Numpyro : https://num.pyro.ai/en/stable/getting_started.html#what-is-numpyro
- JaxOpt: https://jaxopt.github.io/stable/
