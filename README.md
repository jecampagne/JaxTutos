# JaxTutos
Ce repository fournit quelques notebooks pour apprendre JAX et utliser quelques librairies telles que JaxOptim/Numpyro/...

# Installation @CCIN2P3

## Git clone du repository
Dans votre espace de travail quotidien faire
```
git clone git@github.com:jecampagne/JaxTutos.git
```
Dans le directory `JaxTutos` il y a des notebooks et 2 fichiers pour configurer le `kernel` Jupyter spécifique pour activer l'environement conda que vous allez installer de suite.

## Environement Conda `JaxTutos`

Elle se base sur Anaconda (v 4.12.0) mais peut peut-être fonctionner avec une autre version. 
Voir sa version via 
> `conda --version`

Selon l'ordre suivant vous allez procéder à l'installation de l'environement Conda `JaxTutos`
```
conda create -n JaxTutos python=3.8
conda activate JaxTutos
pip install --upgrade "jax[cuda]>=0.3.5" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpyro==0.8.0
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
![image](https://user-images.githubusercontent.com/20539759/162919652-c788af2a-0698-4d74-8bd0-154918bd6e1e.png)

![image](https://user-images.githubusercontent.com/20539759/162919846-a8218c05-6d50-4eb7-b1b7-ae964c132b34.png)


![image](https://user-images.githubusercontent.com/20539759/162919922-737e4b01-8f8d-4e96-b2e9-935498552993.png)

![image](https://user-images.githubusercontent.com/20539759/162920000-4c787b99-e46e-4068-9171-9b7dee2aa5d9.png)


![image](https://user-images.githubusercontent.com/20539759/162924541-8a69641e-b85c-4e37-976e-d8cac5cf9a3b.png)


![image](https://user-images.githubusercontent.com/20539759/162920945-08b645f2-d028-43b0-9faa-d199b8fac1ba.png)





