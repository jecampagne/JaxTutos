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

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import jacfwd, jacrev, hessian
#from jax.ops import index, index_update
jax.config.update("jax_enable_x64", True)

from functools import partial


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('image', cmap='jet')
mpl.rcParams['font.size'] = 18


# -

# ## Attention: Il y a (au moins) 2 pratiques en vigueur: 
# - soit (numpy -> np, jax.numpy -> jnp) 
# - soit (numpy -> onp, jax.numpy -> np)

# # Example de l'usage de l'auto-diff et la vectorisation

def f(x):
    return jnp.exp(-x*0.5)*jnp.sin(x)


# +
# Expressions Exactes de f'(x) et f''(x)
def fp(x):
    return jnp.exp(-x*0.5)*(2.*jnp.cos(x)-jnp.sin(x))/2.
def fpp(x):
    return -jnp.exp(-x*0.5)*(4.*jnp.cos(x)+3.*jnp.sin(x))/4.

# Expressions numériques centrées approchées de f'(x) et f''(x)
def fpnum(x,h=0.1):
    return (f(x+h)-f(x-h))/(2*h)
def fppnum(x,h=0.1):
    return (f(x+h)-2*f(x)+f(x-h))/h**2


# -

# # Différéntiations successives

jfp  = grad(f)             # grad pour gradient
jfpp = grad(grad(f))

# ## Voir que fait JAX quand il produit le code...

jax.make_jaxpr(jfp)(0.)

# ### prenez un papier et un crayon et vérifier que ce code est correcte :) 

# ## La vectorisation est utile pour appeler la fonction avec un array (nb. et d'autres structures)

jfp  = vmap(grad(f))
jfpp = vmap(grad(grad(f)))
#jfjacobien = vmap(jacfwd(f))
#jfhessien  = vmap(hessian(f))

fig,_ = plt.subplots(figsize=(8,9))
x=jnp.arange(0,10,0.1)
plt.plot(x,f(x), label="$f(x)$")
plt.scatter(x,fpnum(x), label="$f^{\prime}(x)$ (Num.)")
plt.plot(x,jfp(x), lw=3,label="$f^{\prime}(x)$ (JAX)")
plt.plot(x,fp(x),ls="--",c='k', label="$f^{\ \prime}(x)$ (True)")
#plt.plot(x,jfjacobien(x),ls='-.', lw=3, label='jacobien')
plt.scatter(x,fppnum(x), label="$f^{\prime\prime}(x)$ (Num.)")
plt.plot(x,jfpp(x), lw=3,label="$f^{\prime\prime}(x)$ (JAX)")
plt.plot(x,fpp(x),ls="--", c='purple',label="$f^{\ \prime\prime}(x)$ (True)")
#plt.plot(x,jfhessien(x), ls='-.', lw=3, label='hessian')
plt.grid()
plt.xlabel("$x$")
plt.legend(loc="upper right");

# ### Zoom: => Q: à quoi sert l'auto-différentiation? 

x=jnp.arange(0,10,0.1)
plt.scatter(x,fppnum(x)-fpp(x),s=10, c="r", label="Num.-True")
plt.plot(x,jfpp(x)-fpp(x),lw=3,c="g",  label="Jax-True")
plt.grid()
plt.xlabel("$x$")
plt.title("$f^{\prime\prime}(x)$")
plt.legend(loc="upper right");

x=jnp.arange(0,10,0.1)
jnp.max(jnp.abs(jfpp(x)-fpp(x)))


# vous pouvez essayer d'utiliser le *jacobien* et le *hessien* ...

# ## La différentiation est par défaut sur le 1er argument de la fonction mais cela n'est pas exclusif

# +
def func(x, y):
    return 2 * x * y

print(grad(func)(3., 4.))  # 8.                 # la fonction evaluee en (3,4)
print(grad(func, argnums=0)(3., 4.))  # 8.      # la derivee par rapport a x evaluee en (3,4)
print(grad(func, argnums=1)(3., 4.))  # 6       # la derivee par rapport a ye valuee en (3,4)
print(grad(func, argnums=(0, 1))(3., 4.))  # (8., 6.)  # ici on fait en mm temps la derivée par rapport a x et a y...
# -

grad(func)(3., 4.)


# essayer avec une autre fonction...

# # Exemple via la minimisation de la Mean Squared Error (MSE)

# +
def model(p, x):
    return jnp.exp(-x*p[0])*jnp.sin(x*p[1])

def loss_fun(p, xi, yi):
    yhat = model(p, xi)
    return jnp.mean( (yhat - yi)**2 )


# -

# jnp.exp, jnp.sin ... sont vectorisees donc `model` et `loss_fun` n'ont pas besoin d'un vmap
#

# +
# dataset
ptrue = jnp.array([0.5,1])

xin = jnp.arange(0,10,1.)
yerr = 0.05
# jax.random.PRNGKey(seed) gestion des randoms numbers
y_true =  model(ptrue, xin)
yin = y_true + yerr*jax.random.normal(jax.random.PRNGKey(42),shape=xin.shape)
# -

xin

# on peut directement utiliser les DeviceArray de JAX dans matplotlib

plt.errorbar(xin,yin,yerr=yerr, fmt='o', linewidth=2, capsize=0, c='k', label="data");
plt.plot(xin,y_true, label='True (noiseless)')
plt.legend();


# ## Descente de Gradient

@jit
def gradient_descent_step(p, xi, yi, lr=0.1):
    return p - lr * jax.grad(loss_fun)(p, xi, yi)


# nb. nous verrons le pourquoi du `@jit` un peu plus loin

# Un minizer basique (version plus JAX/LAX-like à la fin)
def minimzer(loss_fun, x_data, y_data, par_init, method, verbose=True):
    p_cur = par_init
    new_loss=jnp.inf
    
    for t in range(5000):
   
        if (t % 100 == 0) and verbose:
            print(t, p_cur,new_loss)

        old_loss = new_loss
        new_p = method(p_cur, x_data,y_data)
        new_loss = loss_fun(new_p, x_data,y_data)

        if jnp.abs(new_loss-old_loss) < 1e-9:
            print(f"Converged after {t} epochs: p = {new_p}, loss = {new_loss}")
            break

        p_cur = new_p

    return p_cur


par_mini_GD = minimzer(loss_fun, x_data=xin, y_data=yin, par_init=jnp.array([0., 0.5]),
                    method=gradient_descent_step, verbose=True)

# ## Recommencez en decommentant "@jit" dans la celleule du `gradient_descent_step` : qu'observe-t'on?

plt.errorbar(xin,yin,yerr=yerr, fmt='o', linewidth=2, capsize=0, c='k', label="data")
x_val = jnp.linspace(0.,10,50)
plt.plot(x_val,model(ptrue, x_val),label="true noiseless")
plt.plot(x_val,model(par_mini_GD, x_val) ,lw=3,label="GD fit")
plt.legend();

# ## Méthod  de Newton => usage du hessien (cf. méthode d'ordre 2, alors que GD : méthode d'ordre 1)

# +
gLoss = lambda p,xi,yi: jacfwd(loss_fun)(p,xi,yi)
hLoss = lambda p,xi,yi: jax.hessian(loss_fun)(p,xi,yi)

@jit
def oneStepNewton(p,xi,yi,lr=0.1):
    return p - lr*jnp.linalg.inv(hLoss(p,xi,yi)) @ gLoss(p,xi,yi)


# -

par_mini_Newton = minimzer(loss_fun, x_data=xin, y_data=yin, par_init=jnp.array([0., 0.5]),
                    method=oneStepNewton, verbose=True)

plt.errorbar(xin,yin,yerr=yerr, fmt='o', linewidth=2, capsize=0, c='k', label="data")
x_val = jnp.linspace(0.,10,50)
plt.plot(x_val,model(ptrue, x_val),label="true noiseless")
plt.plot(x_val,model(par_mini_GD, x_val) ,lw=3,label="GD fit")
plt.plot(x_val,model(par_mini_Newton, x_val),ls='--',lw=3,label="Newton fit")
plt.legend();


# # Gradient/Vmap... par rapport à: tuples, lists, and dicts

def f(p,x):
    return p["a"]**2 + p["b"]*x


# ## Vectorization partielle: `in-axes`

print(grad(f)({"a":3.,"b":1.},10.))

print(grad(f, argnums=1)({"a":3.,"b":1.},10.))

print(vmap(f, in_axes=({"a": None, "b": 0},None))({"a":1.,"b":jnp.array([1.,2.,3.])},10.))

vmap(f, in_axes=({"a":0, "b": None},None))({"a":jnp.array([1.,2.,3.]),"b":2.0},10)


# # Un peu plus de `in_axis` dans vmap (equivalent Numpy)

def func(a,b,x):
    return a**2 + b*x


vfunc = vmap(func, in_axes=(0,None,None))

vfunc(jnp.array([1.,2.,3.]), 1.,10.)

# map sur 2 axis
vvfunc = vmap(vmap(func, in_axes=(0,None,None)),in_axes=(None,0,None))

vvfunc(jnp.array([1.,2.,3.]),jnp.array([0.,1.,4.,5.]),10.)
# a:(1,2,3) b: 0
# a:(1,2,3) b: 1
# a:(1,2,3) b: 4
# a:(1,2,3) b: 5


# equivalent Numpy
[[func(ai,bj,x=10.) for ai in [1,2,3]] for bj in [0.,1.,4.,5.]]

vmap(func,in_axes=(0,0,None))(jnp.array([1.,2.,3.]),jnp.array([0.,1.,4.]),10.)
# ici map a,b de meme dimension car
# a:1 b:0
# a:2 b:1
# a:3 b:4

# equivalent Numpy
[func(ai,bj,x=10.) for ai,bj in zip([1,2,3],[0.,1.,4.,5.])]

# ## On peut vouloir specifier les axes de `a` et `b` qui sont mis ensemble 

vmap(func,in_axes=(0,1,None))(jnp.array([1.,2.,3.]),jnp.array([0.,1.,4.])[jnp.newaxis,:],10.)

#idem
vmap(func,in_axes=(0,1,None))(jnp.array([1.,2.,3.]),jnp.array([[0.,1.,4.]]),10.)

a = np.array([1.,2.,3.])
b = np.array([[0.,1.,4.],
               [1.,2.,8.]])
vmap(func,in_axes=(0,1,None))(a,b,10.)

a.shape, b.shape

# +
#a[0]:1  b[0,0]:0 => 1, a[0]:1, b[1,0]:1 => 11 
#a[1]:2  b[0,1]:1 => 14, a[1]:2, b[1,1]:2 => 24
#a[2]:3  b[0,2]:4 => 49 , a[2]:3  b[1,2]:8 => 89 

# equivalent Numpy
[[func(a[i],b[j,i],x=10.) for i in range(a.shape[0])] for j in range(b.shape[0])]
# -

# # PyTree: une structure user pour que JAX puisse faire du grad/vmap/jit

# +
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Params:
    def __init__(self, a, b):
        self._a = a
        self._b = b

    def __repr__(self):
        return f"Params(x={self._a}, y={self._b})"

    @property
    def a(self):
        return self._a
    
    @property
    def b(self):
        return self._b

    def tree_flatten(self):
        children = (self._a, self._b)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# -

my_params = Params(3.,1.)
my_params

my_params.a, my_params.b


def g(p,x):
    return (p.a)**2 + (p.b)*x


print(grad(g)(my_params,10))

print(vmap(g, in_axes=(Params(None, 0), None))(Params(3., jnp.array([1.,2.,3.])), 10))

from jax.tree_util import tree_flatten, tree_unflatten
def show_example(structured):
    flat, tree = tree_flatten(structured)
    unflattened = tree_unflatten(tree, flat)
    print("structured={}\n  flat={}\n  tree={}\n  unflattened={}".format(
          structured, flat, tree, unflattened))


show_example(my_params)


# # Takeaway message:
# - JAX ne nécessite pas l'apprentissage d'un nouveau langage de programmation 
# - Avec `jax.numpy` il est (presque) aussi simple d'utiliser les JAX DeviceArray en lieu et place des Numpy array
# - L'autofifferentiation: c'est pas compliqué (`grad`, `jacfwd`, `hessian`)
# - `jit` permet d'accélérer le code (compilation au vol)
# - `vmap` permet de vectoriser l'appel de fonctions (nb. il existe `pmap` pour la paraléllisation sur plusieurs devices)
# - vectorization partielle `in_axis`
# - On peut utiliser (grad/vmap...) ses structures autres que les arrays: tuple, liste, dico et user PyTree

# # Exercice:
# - reprendre la vectorization sur 2 axes avec Dictionaire.
# - Reprener les méthodes `model(p, x)`, `loss_fun(p, xi, yi)`,  `oneStepNewton(p,xi,yi,lr=0.1)` en changeant la sturture des paramètres `p` (le default est `jnp.array`): ex utiliser un `dictionnary` et/ou une classe sur le modèle de `Params`. 
#
#

# # EXTRA qd on aura pratiqué l'exemple de la fratale de Julia

# Exercice: Réécrire la fonction `minimize` (notée `minimize_bis`) en utilisant le pattern `cond_fun/body/jax.lax.while_loop`

def minimzer_bis(loss_fun, x_data, y_data, par_init, method, maxiter=5000, loss_diff=1e-9):
    cond_fun = True #...
    @jit
    def body(val):
        #...
        return #...
    val = [par_init, 0.,0.,jnp.inf] 
    #...
    return val


# + tags=[] jupyter={"source_hidden": true}
def minimzer_bis(loss_fun, x_data, y_data, par_init, method, maxiter=5000, loss_diff=1e-9):

    cond_fun = lambda val: (val[1] < maxiter) & (jnp.abs(val[2]-val[3]) > loss_diff)
    
    @jit
    def body(val):
        p_cur    = val[0]
        old_loss = val[3]
        new_p = method(p_cur, x_data,y_data)
        new_loss = loss_fun(new_p, x_data,y_data)
        return [new_p, val[1]+1,  old_loss, new_loss]
        

    val = [par_init, 0.,0.,jnp.inf]   # on peut faire avec des Tuples, a accorder avec le return de body
    val = jax.lax.while_loop(cond_fun,body,val)

    return val
# -

param, n_iter, loss,_ = minimzer_bis(loss_fun, x_data=xin, y_data=yin, par_init=jnp.array([0., 0.5]),
                    method=gradient_descent_step)

print(f"{n_iter} epochs: p = {param}, loss = {loss:.2e}")


