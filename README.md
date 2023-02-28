# JaxTutos
Ce repository fournit quelques notebooks pour apprendre JAX et utliser quelques librairies telles que JaxOptim/Numpyro/...

# Echanges: 
- Pour discuter vous pouvez utiliser le champ `Dscissions`
- Pour suggérer des modifications de code et/ou rapporter des bugs: utiliser les traditionels `Issues` & `Pull requests`

# Menu des Notebooks dans l'ordre des Tutos:
1) `JAX-first-grad-vmap.ipynb` : prise de contact avec l'auto-diff sur un exemple simple, puis quelques illustrations de JAX (vmap/jit) sur un exemple de minimisation avec la méthode de Gradient-Descent, et de Newton
2) `JAX-Julia_set.ipynb` : à travers l'exmple des fractales de Julia on aborde quelques fonctions JAX/LAX (condition, while-loop) en mirroir d'un code Numpy basique
3) `JAX-AutoDiff-UserCode.ipynb` : on commence par l'usage d'une autre fonction de JAX/LAX (fori_loop); puis on va commencer à implémenter un algorithme d'intégration de Simpson et jouer avec pour montrer la vectorization et l'auto-diff sur un code user. Ensuite en utilisant une methode d'integration plus efficace (Clenshaw-Curtis) on jouera avec une Classe et des Fonctions en lien avec les distances cosmologiques pour voir un aspect d'extension des structures qui sont accessibles à une vectorization et l'auto-diff.  
4) `JAX-JIT_in_class.ipynb`: comment utilsier JIT pour des méthodes de classe (par opposition à JIT pour une fonction isolée).
5) `JAX-Optim-regression-piecewise.ipynb`: avec ce notebook on explore quelques fonctionalités de la librairie JaxOpt d'optimisation (equivalent de ScipyMinimize). On utilisera un exemple de dataset 1D qui nous sert dans les deux autres nbs.
6) `JAX-MC-Sampling.ipynb`: passage en revue de méthodes classiques de génération Monte carlo. On le fait à travers notament le calcul d'un intégrale 1D (et aussi la détermination de paramètres en 3D): échantillonnage selon $dx$, `Importance Sampling`, `Metropolis-Hastings`, `HMC` (NUTS en exo extra car vu avec le nb. `JAX-NUTS...`)
7) `JAX-NUTS-regression-piecewise.ipynb`: en reprenant le dataset 1D du (4) on utilise la librairie Numpyro MCMC NUTS pour sampler une distribution posterior et obtenir les coutours de parametres et des predictions
8) `JAX-GP-regression-piecewise.ipynb`: dans la même philosophie que le (6) on va étudier les Gaussian Processes avec un librairie 'maison' et voir des différences avec Sklearn et GPy.


## Installation

La majorité des nbs tournent sur Colab. (jax 0.3.25 au passge) 


Selon l'ordre suivant vous allez procéder à l'installation de l'environement Conda `JaxTutos`
```
conda create -n JaxTutos python=3.8
conda activate JaxTutos
pip install --upgrade "jax[cuda]==0.3.5" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpyro==0.9.1
pip install jaxopt==0.3.1
pip install optax==0.0.1
pip install corner==2.2.1
pip install arviz==0.11.4
pip install GPy==1.10.0
pip install scikit-learn==1.0.2
pip install matplotlib_inline
pip install seaborn==0.11.1
```

Un petit test de la version de `jaxlib` ...
```python
python -c "import jaxlib; print(jaxlib.__version__)"
0.3.5
```

# Docs de packages
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
