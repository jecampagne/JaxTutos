# JaxTutos
Ce repository fournit quelques notebooks pour apprendre JAX et utliser quelques librairies telles que JaxOptim/Numpyro/...

# Menu des Notebooks dans l'ordre des Tutos:
1) `JAX-first-grad-vmap.ipynb` : prise de contact avec l'auto-diff sur un exemple simple, puis quelques illustrations de JAX (vmap/jit) sur un exemple de minimisation avec la méthode de Gradient-Descent, et de Newton
2) `JAX-Julia_set.ipynb` : à travers l'exmple des fractales de Julia on aborde quelques fonctions JAX/LAX (condition, while-loop) en mirroir d'un code Numpy basique
3) `JAX-AutoDiff-UserCode.ipynb` : on commence par l'usage d'une autre fonction de JAX/LAX (fori_loop); puis on va commencer à implémenter un algorithme d'intégration de Simpson et jouer avec pour montrer la vectorization et l'auto-diff sur un code user. Ensuite en utilisant une methode d'integration plus efficace (Clenshaw-Curtis) on jouera avec une Classe et des Fonctions en lien avec les distances cosmologiques pour voir un aspect d'extension des structures qui sont accessibles à une vectorization et l'auto-diff.  
4) `JAX-Optim-regression-piecewise.ipynb`: avec ce notebook on explore quelques fonctionalités de la librairie JaxOpt d'optimisation (equivalent de ScipyMinimize). On utilisera un exemple de dataset 1D qui nous sert dans les deux autres nbs.
5) `JAX-MC-Sampling.ipynb`: passage en revue de méthodes classiques de génération Monte carlo. On le fait à travers notament le calcul d'un intégrale 1D (et aussi la détermination de paramètres en 3D): échantillonnage selon $dx$, `Importance Sampling`, `Metropolis-Hastings`, `HMC` (NUTS en exo extra car vu avec le nb. `JAX-NUTS...`)
6) `JAX-NUTS-regression-piecewise.ipynb`: en reprenant le dataset 1D du (4) on utilise la librairie Numpyro MCMC NUTS pour sampler une distribution posterior et obtenir les coutours de parametres et des predictions
7) `JAX-GP-regression-piecewise.ipynb`: dans la même philosophie que le (6) on va étudier les Gaussian Processes avec un librairie 'maison' et voir des différences avec Sklearn et GPy.




# Installation @CCIN2P3

*Attention: l'installation demande du soin et d'aller jusqu'au bout de ce fichier en particulier pour celle(s) et ceux qui auraient l'habitude d'installer des kernels, il y a une procédure ajoutée qui consiste à spécifier l'ordre des PATHs.*


## Git clone du repository
Dans votre espace de travail quotidien faire
```
git clone https://github.com/jecampagne/JaxTutos.git
```

Dans le directory `JaxTutos` il y a des notebooks et 2 fichiers pour configurer le `kernel` Jupyter spécifique pour activer l'environement conda que vous allez installer de suite.

Je vous conseille de créer un **lien symbolique** vers ce directory (`JaxTutos`) à partir de votre *home directory* afin de faciliter la procédure de login sur la plateforme des notebooks du CC.


## Environement Conda `JaxTutos`

Elle se base sur Anaconda (v 4.11+) mais peut peut-être fonctionner avec une autre version.


> Nb. la version d'Anaconda spécifique LSST ne convient pas, donc il faut se faire sa propre install

```
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -p miniconda
```

Voir sa version via
> `conda --version`

Si besoin, mettre à jour la variable `PATH` pour pointer vers `/chemin/miniconda/bin`.


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

## Kernel Jupyter `JaxTutos` pour les notebooks
Maintenant il nous faut procéder à l'édition/création de quelques fichiers afin de pouvoir activer l'environement `JaxTutos` sur la plateforme des notebooks du CC.

Dans votre espace HOME du CC, je suppose que vous avez le directory `~/.local/share/jupyter` sinon il faudra voir comment le créer. Ensuite,
```
cd  ~/.local/share/jupyter
mkdir kernels      # s'il n'existe pas déjà
mkdir JaxTutos
cd JaxTutos
```

Copiez dans directory  `~/.local/share/jupyter/kernels/JaxTutos` les deux fichiers qui sont venus avec le `git clone`
```
jupyter-helper.sh
kernel.json
```
Il faut maintenant éditer ces 2 fichiers afin de tuner les chaines **"A REMPLACER"**.
Commençons par `jupyter-helper.sh` où il faut renseigner où se trouve le script `conda.sh`:
```bash
#!/bin/bash

#echo "jupyter init..."
source "A REMPLACER"/etc/profile.d/conda.sh

conda activate JaxTutos

export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec python -m ipykernel_launcher "$@"
```

Enchainons avec `kernel.json` où il faut renseigner le path complet de
`~/.local/share/jupyter/kernels/JaxTutos/jupyter-helper.sh` (**ne pas utilser $HOME ou d'autres variables d'environement**:
```json
{
  "display_name": "JaxTutos",
  "language":     "python",
  "argv": [
      "A REMPLACER (ex. /pbs/home/... path explicit)"/.local/share/jupyter/kernels/JaxTutos/jupyter-helper.sh",
      "-f",
      "{connection_file}"
  ]
}
```
Maintenant, nous allons procéder au login sur la plateforme des notebooks du CC, et voir si l'installation précédente est correcte. Il nous faudra cependant effectuer un dernier ajustement des PATH Python avant d'utilser les notebooks.

# Login sur la plateforme des notebooks au CC
Pour se connecter à la plateforme des notebooks du CC il faut entrer le lien `https://notebook.cc.in2p3.fr/` dans son navigateur (j'ai par défault `FireFox` mais j'ai l'impression qu'avec `Safari` cela marche aussi).

Dans un premier temps vous devriez vous trouver devant ce panel si vous n'êtes pas déjà authentifié:

![image](https://user-images.githubusercontent.com/20539759/162919846-a8218c05-6d50-4eb7-b1b7-ae964c132b34.png)

En principe il faut opter pour **EDUGAIN** qui mène à ce paneau dans lequel il faut taper **CNRS**
![image](https://user-images.githubusercontent.com/20539759/162919922-737e4b01-8f8d-4e96-b2e9-935498552993.png)

ou un équivalent en langue française. Ensuite, il se peut que selon votre navigateur et ou si c'est la première fois on vous demande un **mot de pass Janus**. A signaler si vous avez un problème pour retrouver ce mot de pass...


Ensuite vous devriez vous retrouver devant ce panneau qui vous indique le choix d'opter pour un plateforme CPU ou GPU. Pour le moment sélectionner la version **CPU**.

![image](https://user-images.githubusercontent.com/20539759/162920000-4c787b99-e46e-4068-9171-9b7dee2aa5d9.png)

Vous devriez vous retrouver avec ce paneau où apparait à gauche une liste de fichiers/directories de votre home directory. Alors vous pouvez clicker sur **JaxTutos** (*le lien symbolique créé à la section sur le clonage du repository*).

![image](https://user-images.githubusercontent.com/20539759/162964730-0ff447fe-54d4-4e1f-9a65-a6ff712d4a53.png)

Maintenant nous allons procédé à la vérification de l'installation en activant le notebook `ATestInstall.ipynb` en double-cliquant dessus dans la liste à gauche

![image](https://user-images.githubusercontent.com/20539759/163125536-0cca0592-c118-4a96-b692-f1d28ae39ca0.png)

# Avant de lancer le kernel

Avant de lancer le kernel `JaxTutos` qui apparaitra en haut à droite à la place de `NoKernel` nous devons faire un dernier réglage nécessaire pour bypasser l'installation par défault des paths Python du CC.  Pour cela il nous faut double cliquer sur `pathinit_to_TUNE_and_Rename.py` qui doit vous donner cela

![image](https://user-images.githubusercontent.com/20539759/162968185-dda54cac-db44-4a65-bc2a-9d1b4ca8f45d.png)

Ce sont les lignes suivantes qu'il vous faut changer: la première pour donner le path du repository cloné de JaxTutos, la seconde pour donner le path de l'environement Conda JaxTutos
```python
rootPath_to_JaxTutos_localrepository = '/sps/lsst/users/campagne/'
rootPath_to_JaxTutos_conda_env = '/sps/lsst/users/campagne/anaconda3/envs/'
```

Une fois mis à jour **renomer** `pathinit_to_TUNE_and_Rename.py`
```
mv pathinit_to_TUNE_and_Rename.py pathinit.py
```

# Lancement (activation) du kernel `JaxTutos`

Ensuite, nous allons activer le kernel "JaxTutos" dans le menu déroulant qui s'ouvre après avoir cliqué sur `NoKernel`

![image](https://user-images.githubusercontent.com/20539759/162968728-1a4625a1-85f4-4b5e-b0a3-5ed38c59876c.png)


![image](https://user-images.githubusercontent.com/20539759/162924541-8a69641e-b85c-4e37-976e-d8cac5cf9a3b.png)

Une fois fait en principe le système cherche à activer le kernel en exécutant le script `jupyter-helper.sh` et le fichier `kernel.json` que vous avez installés plus haut. Si tout se passe bien au niveau de l'installation vous devriez vous retrouver avec dans le bandeau du bas de la fenêtre la situation suivante où après avoir été en mode `Connecting` **le kernel JaxTutos est en `Idle`** c'est-à-dire en attente.

![image](https://user-images.githubusercontent.com/20539759/162969669-848dfcf4-d462-455b-8660-da1b09fdfd62.png)

En activant la première cellule avec les imports Python, si tout se passe bien vous ne devriez pas avoir de message d'erreur mais simplement un warning sur le fait que la librairie `libcudart.so.11.0` n'est pas là. Ce n'est pas grave car elle n'est activée que dans le cadre de l'usage des GPUs (NVidia).

En poursuivant l'activation des deux cellules suivantes s'il n'y a pas d'erreur alors à ce stade on doit pouvoir conclure que l'on peut commencer par ... ce prendre un café ! Bravo.
S'il y a un package missing ou une différence de version alors vous aurez une levée d'exception qu'il faudra mentionner.


# Interactions avec le Kernel

Les procédures de Start/reStart/Arrêt/Reconnection... se font via le panel suivant

![image](https://user-images.githubusercontent.com/20539759/162920945-08b645f2-d028-43b0-9faa-d199b8fac1ba.png)

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
