# JaxTutos
Ce repository fournit quelques notebooks pour apprendre JAX et utliser quelques librairies telles que JaxOptim/Numpyro/...
# Installation @CCIN2P3
Elle se base sur Anaconda (v 4.12.0) mais peut peut-être fonctionner avec une autre version. 
Voir sa version via 
> `conda --version`

Donc procédez selon l'ordre suivant à l'installation de l'environement Conda `jaxTutos`
```
conda create -n JaxTutos python=3.8
conda activate JaxTutos
pip --upgrade "jax[cuda]>=0.3.5" -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install numpyro==0.8.0
pip install jaxopt==0.3.1
pip install optax==0.0.1
pip install corner==2.2.1
pip install arviz==0.11.4
```

test jaxlib...



Maintenant il nous faut procéder à l'édition/création de quelques fichiers afin de pouvoir activer l'environement `jaxTutos` sur la plateforme des notebooks du CC.



# Login sur la plateforme des notebooks au CC
![image](https://user-images.githubusercontent.com/20539759/162919652-c788af2a-0698-4d74-8bd0-154918bd6e1e.png)

![image](https://user-images.githubusercontent.com/20539759/162919846-a8218c05-6d50-4eb7-b1b7-ae964c132b34.png)


![image](https://user-images.githubusercontent.com/20539759/162919922-737e4b01-8f8d-4e96-b2e9-935498552993.png)

![image](https://user-images.githubusercontent.com/20539759/162920000-4c787b99-e46e-4068-9171-9b7dee2aa5d9.png)


![image](https://user-images.githubusercontent.com/20539759/162924541-8a69641e-b85c-4e37-976e-d8cac5cf9a3b.png)


![image](https://user-images.githubusercontent.com/20539759/162920945-08b645f2-d028-43b0-9faa-d199b8fac1ba.png)





