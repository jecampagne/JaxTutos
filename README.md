# JaxTutos
Ce repository fournit quelques notebooks pour apprendre JAX et utliser quelques librairies telles que JaxOptim/Numpyro/...

# Installation @CCIN2P3

## Git clone du repository
Dans votre espace de travail quotidien faire
```
git clone git@github.com:jecampagne/JaxTutos.git
```

Dans le directory `JaxTutos` il y a des notebooks et 2 fichiers pour configurer le `kernel` Jupyter spécifique pour activer l'environement conda que vous allez installer de suite.

Je vous conseille de créer un **lien symbolique** vers ce directory (`JaxTutos`) à partir de votre *home directory* afin de faciliter la procédure de login sur la plateforme des notebooks du CC. 


## Environement Conda `JaxTutos`

Elle se base sur Anaconda (v 4.12.0) mais peut peut-être fonctionner avec une autre version. 
Voir sa version via 
> `conda --version`

Selon l'ordre suivant vous allez procéder à l'installation de l'environement Conda `JaxTutos`
```
conda create -n JaxTutos python=3.8
conda activate JaxTutos
pip install --upgrade "jax[cuda]>=0.3.5" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpyro==0.9.1
pip install jaxopt==0.3.1
pip install optax==0.0.1
pip install corner==2.2.1
pip install arviz==0.11.4
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
source "A REMPLACER"/anaconda3/etc/profile.d/conda.sh

conda activate JaxTutos

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

Ensuite, pour la première fois nous utiliserons le nb **JaxTutos/JAX-first-grad-vmap.ipynb**, en double cliquant dessus vous devriez le voir apparaitre dans la partie droite de l'application et voir

![image](https://user-images.githubusercontent.com/20539759/162966336-fd65e5b6-d4e9-4f41-8df9-288668c1709d.png)

Avant de lancer le kernel `JaxTutos` qui apparaitra en haut à droite à la place de `NoKernel` nous devons faire un dernier réglage nécessaire pour bypasser l'installation par défault des paths Python du CC.  Pour cela il nous faut double cliquer sur `pathinit.py` qui doit vous donner cela 

![image](https://user-images.githubusercontent.com/20539759/162968185-dda54cac-db44-4a65-bc2a-9d1b4ca8f45d.png)

Ce sont les lignes suivantes qu'il vous faut changer: la première pour donner le path du repository cloné de JaxTutos, la seconde pour donner le path de l'environement Conda JaxTutos
```python
rootPath_to_JaxTutos_localrepository = '/sps/lsst/users/campagne/'
rootPath_to_JaxTutos_conda_env = '/sps/lsst/users/campagne/anaconda3/envs/'
```

Ensuite, nous allons activer le kernel "JaxTutos" dans le menu déroulant qui s'ouvre après avoir cliqué sur `NoKernel`

![image](https://user-images.githubusercontent.com/20539759/162968728-1a4625a1-85f4-4b5e-b0a3-5ed38c59876c.png)


![image](https://user-images.githubusercontent.com/20539759/162924541-8a69641e-b85c-4e37-976e-d8cac5cf9a3b.png)

Une fois fait en principe le système cherche à activer le kernel en exécutant le script `jupyter-helper.sh` et le fichier `kernel.json` que vous avez installés plus haut. Si tout se passe bien au niveau de l'installation vous devriez vous retrouver avec dans le bandeau du bas de la fenêtre la situation suivante où après avoir été en mode `Connecting` **le kernel JaxTutos est en `Idle`** c'est-à-dire en attente.

![image](https://user-images.githubusercontent.com/20539759/162969669-848dfcf4-d462-455b-8660-da1b09fdfd62.png)

En activant la première cellule avec les imports Python, si tout se passe bien vous ne devriez pas avoir de message d'erreur mais simplement un warning sur le fait que la librairie `libcudart.so.11.0` n'est pas là. Ce n'est pas grave car elle n'est activée que dans le cadre de l'usage des GPUs (NVidia).

Bon... à ce stade on doit pouvoir conclure que l'on peut commencer par ... ce prendre un café ! Bravo.

# Interactions avec le Kernel

Les procédures de Start/reStart/Arrêt/Reconnection... se font via le panel suivant

![image](https://user-images.githubusercontent.com/20539759/162920945-08b645f2-d028-43b0-9faa-d199b8fac1ba.png)





