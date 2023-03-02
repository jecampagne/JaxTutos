# JaxTutos
This repository provides some notebooks to learn JAX (basics and advenced) and use some libraries such as JaxOptim/Numpyro/...

Ce repository fournit quelques notebooks pour apprendre JAX (simple & complex) et utliser quelques librairies telles que JaxOptim/Numpyro/...

# Echanges: 
- Pour discuter vous pouvez utiliser le champ/`Discussions`
- To discuss you can use :  `Discussions`

- Pour suggérer des modifications de code et/ou rapporter des bugs: utiliser les traditionels `Issues` & `Pull requests`
- To suggest code changes and/or report bugs use Issues & PR.

# Here a list of Tutos:
- `JAX_get_started.ipynb` : get a flavour of the coding and exemple of auto-diff
- `JAX-AutoDiff-UserCode.ipynb` : more on usage of auto diff in real user integration methods  
- `JAX_fractals.ipynb` : **(GPU better)** throw some fractal images production discover some control flow jax.lax functions
- `JAX_MC_Sampling.ipynb`: more on coding style/method in a use case of Monte Carlo Sampler implementation from scratch
- `Numpyro_MC_Sampling`: here we give some simple examples using Numpyro PPL
- `JAX_jaxopt_optax.ipynb`: some use of JaxOptim & Optax lib.
- `JAX_control_flow.ipynb`: jax.lax control flow (fori_loop/scan/while_loop, cond) with crashes analysed: **"always scan when you can!"**
- `JAX-JIT_in_class.ipynb`: (advenced, technical nb with crashes analysed) how to use JIT for class methods (as opposed to JIT for an isolated function).
- `JAX_PyTree_initialisation.ipynb`: (advenced, technical nb, crash analysed) how to perform variable initilisation wrt the use  


# Not yet ported to Collab.

- `JAX-GP-regression-piecewise.ipynb`: a home made Gaussian Processes lib to see differences with Sklearn et GPy.


## Installation
The nbs are running on Colab. (JAX 0.4.4 recently) 


If you want an environement Conda `JaxTutos`
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
