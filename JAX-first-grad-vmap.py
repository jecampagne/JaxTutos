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
mpl.rcParams["font.family"] = "Times New Roman"
# -

if jax.__version__ < '0.2.26':
    clear_cache = jax.interpreters.xla._xla_callable.cache_clear
else:
    clear_cache = jax._src.dispatch._xla_callable.cache_clear


# # Example de l'usage de l'auto-diff et vectorisation

def f(x):
    return jnp.exp(-x*0.5)*jnp.sin(x)


def fp(x):
    return jnp.exp(-x*0.5)*(2.*jnp.cos(x)-jnp.sin(x))/2.
def fpp(x):
    return -jnp.exp(-x*0.5)*(4.*jnp.cos(x)+3.*jnp.sin(x))/4.


jfp  = grad(f)
jfpp = grad(grad(f))

# +
#jax.make_jaxpr(jfp)(0.)
# -

jfp = vmap(grad(f))          # besion de vectorisation pour appeler avec un array
jfpp = vmap(grad(grad(f)))

fig,_ = plt.subplots(figsize=(8,8))
x=jnp.arange(0,10,0.1)
plt.plot(x,f(x), label="$f(x)$")
plt.plot(x,jfp(x), lw=3,label="$f^{\ \prime}(x)$ (JAX)")
plt.plot(x,fp(x),ls="--",c='k', label="$f^{\ \prime}(x)$")
plt.plot(x,jfpp(x), lw=3,label="$f^{\ \prime\prime}(x)$ (JAX)")
plt.plot(x,fpp(x),ls="--", c='purple',label="$f^{\ \prime\prime}(x)$")
plt.grid()
plt.xlabel("$x$")
plt.legend();


# +
def func(x, y):
    return 2 * x * y

print(grad(func)(3., 4.))  # 8.                 # la fonction evaluee en (3,4)
print(grad(func, argnums=0)(3., 4.))  # 8.      # la derivee par rapport a x evaluee en (3,4)
print(grad(func, argnums=1)(3., 4.))  # 6       # la derivee par rapport a ye valuee en (3,4)
print(grad(func, argnums=(0, 1))(3., 4.))  # (8., 6.)  # ici on fait en mm temps la derivée par rapport a x et a y...


# -

# # Un petit fit par minimisation de la MSE (Mean Squared Error)

# +
def model(p, x):
    return jnp.exp(-x*p[0])*jnp.sin(x*p[1])

def loss_fun(p, xi, yi):
    yhat = model(p, xi)
    return jnp.mean( (yhat - yi)**2 )


# +
# dataset
ptrue = np.array([0.5,1])

xin = np.arange(0,10,1.)
yerr = 0.05
# jnp.exp, jnp.sin ... sont vectorisees donc "model" n'a pas besoin d'unvmap
# jax.random.PRNGKey(seed) gestion des randoms numbers
y_true =  model(ptrue, xin)
yin = y_true + yerr*jax.random.normal(jax.random.PRNGKey(42),shape=xin.shape)
# -

plt.errorbar(xin,yin,yerr=yerr, fmt='o', linewidth=2, capsize=0, c='k', label="data");
plt.plot(xin,y_true, label='True (noiseless)')
plt.legend();


# ### Descente de Gradient

## @jit
def gradient_descent_step(p, xi, yi, lr=0.1):
    return p - lr * jax.grad(loss_fun)(p, xi, yi)


def minimzer(loss_fun, x_data, y_data, par_init, method, verbose=True):
    p_cur = par_init
    new_loss=jnp.inf
    for t in range(5000):
   
        if (t % 100 == 0) and verbose:
            print(t, p_cur,new_loss)

        old_loss = new_loss
        new_p = method(p_cur, x_data,y_data)
        new_loss = loss_fun(new_p, x_data,y_data)

        if np.abs(new_loss-old_loss) < 1e-9:
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

# ### Newton => hessien

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


print(grad(f)({"a":3.,"b":1.},10))
print(vmap(f, in_axes=({"a": None, "b": 0},None))({"a":1.,"b":jnp.array([1.,2.,3.])},10))

vmap(f, in_axes=({"a": None, "b": 0},None))({"a":3.,"b":jnp.array([1.,2.,3.])},10)

# # idem avec un User PyTree

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

# # EXTRA qd on aura pratiqué l'exemple de la fratale de Julia



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

param, n_iter, loss,_ = minimzer_bis(loss_fun, x_data=xin, y_data=yin, par_init=jnp.array([0., 0.5]),
                    method=gradient_descent_step)

print(f"{n_iter} epochs: p = {param}, loss = {loss:.2e}")


