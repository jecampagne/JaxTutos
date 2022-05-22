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
import scipy as sc


import jax
import jax.numpy as jnp
import jax.scipy as jsc   

from jax import grad, jit, vmap
from jax import jacfwd, jacrev, hessian
#from jax.ops import index, index_update
jax.config.update("jax_enable_x64", True)

from functools import partial
from typing import Union, Dict, Callable, Optional, Tuple, Any
# -


jax.__version__

# +
#utile pour clear the cache notament pour voir les recompilation
if jax.__version__ < '0.2.26':
    clear_cache = jax.interpreters.xla._xla_callable.cache_clear
else:
    clear_cache = jax._src.dispatch._xla_callable.cache_clear


def u_print(idx:int, *args)->int:
    print(f"{idx}):",*args)
    idx +=1
    return idx


# -

# # Le thème: JIT et les méthodes de Class
# il s'agit d'élément d'un thread JAX que j'ai initié et qui devient une partie de la [doc](https://jax.readthedocs.io/en/latest/faq.html#strategy-1-jit-compiled-helper-function) (ici c'est une version longue)

# +
class A():
    def __init__(self, a: float):
        print("Nouveau A")
        self.a = a                   # une variable qui ne changera pas ("statique") une fois l'objet cree
        self.b = None                # une variable qui sera déterminée utltérieuremnt ("dynamique")

    def f(self, var: float) -> None:
        self.b = self.a * var

###### 


clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp,objA.a, objA.b)
objA.f(10.)
idp = u_print(idp,objA.a, objA.b)
objA.f(11.)
idp = u_print(idp,objA.a, objA.b)


objA = A(3.0)
idp = u_print(idp,objA.a, objA.b)
objA.f(20.)
idp = u_print(idp,objA.a, objA.b)

# -

# # Et maintenant usage de jit sur f

# + tags=[]
class A():
    def __init__(self, a: float):
        self.a = a                   # une variable qui ne changera pas ("statique")
        self.b = None                # une variable qui sera déterminée utltérieuremnt ("dynamique")
    @jit
    def f(self, var: float) -> None:
        self.b = self.a * var


clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp,objA.a, objA.b)
objA.f(10.)
# -

# ## Le problème est que le premier argument de la fonction est `self`, qui a le type `A` et $\color{red}{\text{JAX ne sait pas comment gérer ce type}}$. Il y a différentes stratégies de base que nous pouvons utiliser dans ce cas, et nous allons les discuter ci-dessous.

# # 1ere strategie: fonction externe ('helper')

# ## petit aparté...

# ## "in Python, type annotations are purely decorative and don't affect runtime values in normal code"

print(1.0*jnp.array([10.]))
def test(v: jnp.array)->jnp.array:
    return 1.0*v
print(test(jnp.array([10.])))
print(test(10.))


@jit
def f(v):
    return 1.0 * v
print(repr(f(10.0)))  # implicit conversion d'un float en JAX device array (a-la-jnp.array)


# +
class A():
    def __init__(self, a):
        print("Nouveau A")
        self.a = a                   # une variable qui ne changera pas ("statique") sauf par le user
        self.b = None                # une variable qui sera déterminée utltérieuremnt ("dynamique")
   
    def f(self, var):
        self.b = _f(self.a,var)

@jit
def _f(a:float, var):
    print("compile...")
    res = a* var
    return res



clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)
objA.f(10.)
idp = u_print(idp, objA.a, objA.b)
objA.f(11.)
idp = u_print(idp, objA.a, objA.b)


objA = A(3.0)
idp = u_print(idp, objA.a, objA.b)
objA.f(20.)
idp = u_print(idp, objA.a, objA.b)
objA.f(21.)
idp = u_print(idp, objA.a, objA.b)


objA = A(4.0)
idp = u_print(idp, objA.a, objA.b)
objA.f(20.)
idp = u_print(idp, objA.a, objA.b)

objA.f(jnp.array([20.]))
idp = u_print(idp, objA.a, objA.b)

objA.f(jnp.array([21.]))
idp = u_print(idp, objA.a, objA.b)

objA.f(21.)
idp = u_print(idp, objA.a, objA.b)

objA.a = 400.  # changement volontaire
objA.f(21.)
idp = u_print(idp, objA.a, objA.b)



# -

# ## Bilan de la méthode `helper`: 
# - ## C'est une $\color{red}{\text{méthode simple et explicite}}$ de mise en oeuvre, et on n'a pas à instruire JAX comment utiliser la class A.
# - ## Maintenant, il devient une affaire de goût de coder un helper par fonction pour utiliser jit. Mais on peut faire de l'encapsulation par fichier pour qu'au moins le code de "A" soit dans celui de la définition de "A".
#
# - ## Pb: que faire si la fonction `f` a besoin par exemple d'une autre fonction (soit membre de `A`, soit externe à `A`)?
#

# # 2nd stratégie: `self` comme static
# ## c'est une procédure classique qui est souvent proposée

# + tags=[]
class A():
    def __init__(self, a: float):
        print("Nouveau A")
        self.a = a
        self.b = None
    
    @partial(jit, static_argnums=(0,))   # on marque bien que "self" est statique    
    def f(self, var: float) -> None:
        print("compile...")
        self.b = self.a * var

    def g(self):
        print("g...:",self.b)
        return self.b*self.b

clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)
objA.f(10.)
idp = u_print(idp, objA.a, objA.b)
res = objA.g()
print(res)


# -

# ## Jit utilise `Traced<ShapedArray>` pour analyser le code, et découvre un `side effect`
# ## !!! $\color{red}{\text{ne pas utiliser}}$ `self.<var> =` dans une fnt jitted

# +
class A():
    def __init__(self, a):
        print("Nouveau A")
        self.a = a               
        self.b = None            

    def set_a(self,x):
        self.a = x 
        print("new a:",self.a)
        
    def set_b(self,x):                  # (*) on va gérérer depuis l'extérieur
        self.b = x
    
    @partial(jit, static_argnums=(0,))   # on marque bien que "self" est statique    
    def f(self, var):
        print("compile...")
        return self.a * var             # voir (*)

clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(10.))
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(11.))
idp = u_print(idp, objA.a, objA.b)

objA = A(3.0)
idp = u_print(idp, objA.a, objA.b)
objA.set_b(objA.f(20.))
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(21.))
idp = u_print(idp, objA.a, objA.b)


objA.set_b(objA.f(jnp.array([20.])))
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(jnp.array([21.])))
idp = u_print(idp, objA.a, objA.b)


# + tags=[]
objA.set_a(4.0)                         # "a" est modifié à la main ...
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(10.))
idp = u_print(idp,objA.a, objA.b)  # oups 4*10 = 20 !!!    il prend 3(old a) * 10 (arg de f)


# -

# ## Il faut faire attention: 
# - ## le caractère "statique" de "a" est un by-product de l'analyse du code par JIT,  mais celà se fait silencieusement et sans crier gare et donc $\color{red}{\text{peut induire en erreur l'utilisateur}}$.

# ## Que se passe-t-il ici ? Le problème est que `static_argnums` s'appuie sur la méthode de hashage (hash) de l'objet pour déterminer s'il a changé entre deux appels, et la méthode $\color{red}{\text{__hash__}}$ par défaut pour une classe définie par l'utilisateur $\color{red}{\text{ne prend pas en compte les valeurs des attributs de classe}}$. Cela signifie qu'au deuxième appel de fonction, JAX n'a aucun moyen de savoir que les attributs de classe ont changé et utilise la valeur statique mise en cache lors de la compilation précédente.
#
#
# ## Pour cette raison, si vous marquez les arguments personnels comme statiques, il est important que vous définissiez une méthode $\color{red}{\text{__hash__}}$ appropriée pour votre classe. Par exemple: 

# +
class A():
    def __init__(self, a):
        print("Nouveau A")
        self.a = a
        self.b = None
        
    def set_a(self,x):
        self.a = x 
        print("new a:",self.a)

        
    def set_b(self,x):
        self.b = x
    
    @partial(jit, static_argnums=0)   # on marque bien que "self" est statique    
    def f(self, var):
        print("compile...")
        return self.a * var
    
    # methode de hashage specifique
    def __hash__(self):
        return hash((self.a,self.b))

    def __eq__(self, other):   # il faut la coder aussi
        return (isinstance(other, A) and
            (self.a, self.b) == (other.a, other.b))
    

clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)

b = objA.f(10.)         #  quel est le type de 'b' ??? (*)
print("b:", b, type(b))
objA.set_b(b)
idp = u_print(idp, objA.a, objA.b)

objA.f(11.)


# -

# ## le problème vient du fait qu'après la compilation `b` devient un `DeviceArray` qui est `unhastable` donc çà coince car alors `self` ne peut ètre hashtable également.

# +
class A():
    def __init__(self, a):
        print("Nouveau A")
        self.a = a
        self.b = None

    def set_a(self,x):
        self.a = x 
        print("new a:",self.a)

    def set_b(self,x):
        self.b = x
    
    @partial(jit, static_argnums=0)
    def f(self, var):
        print("compile...")
        return self.a * var
    
    # methode de hashage specifique
    def __hash__(self):
        return hash((self.a))                 # on ne met pas "b"

    def __eq__(self, other):
        return (isinstance(other, A) and
            (self.a) == (other.a))             # on ne met pas "b"


clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)

b = objA.f(10.)         #  quel est le type de 'b' ??? (*)
print("b:", b, type(b))
objA.set_b(b)
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(11.))
idp = u_print(idp, objA.a, objA.b)


objA = A(3.0)
idp = u_print(idp, objA.a, objA.b)
objA.set_b(objA.f(20.))
idp = u_print(idp, objA.a, objA.b)

# -

objA.set_a(40.) # ou bien objA.a = 40.

objA.set_b(objA.f(11.))
idp = u_print(idp, objA.a, objA.b)   # 40.*11. = 440


# ## Bilan `self`-statique:
# - ## faut définir $\color{red}{\text{__hash__}}$  et $\color{red}{\text{__eq__}}$ (voir pourquoi [Doc. Python](https://docs.python.org/3/reference/datamodel.html#object.__hash__)) avec soin, notament les $\color{red}{\text{DeviceArray sont unhastable}}$.
# - ## 'a' peut-être redéfinit volontairement (avec/sans setter), le résultat de `b` est bien mis à jour

# # 3eme stratégie: PyTree

# +
class A():
    def __init__(self, a, b=None):   # nouvelle signature
        #print("Nouveau A")
        self.a = a
        self.b = b

        
    def set_a(self,x):
        self.a = x 
        print("new a:",self.a)

    def set_b(self, b):
        self.b = b
        
    @jit                              # <------ self, no more static    !                
    def f(self, var):
        print("compile...")
        return self.a * var

    #### PyTree methods....
    def tree_flatten(self):
        children = (self.b,)         # arrays / dynamic values
        aux_data = {'a': self.a}     # static values
        return (children, aux_data)

    @classmethod                     
    def tree_unflatten(cls, aux_data, children):
        b = children
        a = aux_data['a']
        return cls(a=a, b=b ) # doit respected la signature de __init__

# register explicit OU usage d'un decorator
from jax import tree_util
tree_util.register_pytree_node(A,
                               A.tree_flatten,
                               A.tree_unflatten)  

clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)

b = objA.f(10.)         #  quel est le type de 'b' ??? (*)
print("b:", b, type(b))
objA.set_b(b)
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(11.))
idp = u_print(idp, objA.a, objA.b)


objA = A(3.0) 
idp = u_print(idp, objA.a, objA.b)
objA.set_b(objA.f(20.))
idp = u_print(idp, objA.a, objA.b)
objA.set_b(objA.f(30.))
idp = u_print(idp, objA.a, objA.b)


objA.set_a(30.) 
objA.set_b(objA.f(30.))
idp = u_print(idp, objA.a, objA.b)



objA = A(3.0) 
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(20.))
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(30.))
idp = u_print(idp, objA.a, objA.b)

print(">>>> try an array for var")

objA.set_b(objA.f(jnp.array([30.])))
idp = u_print(idp, objA.a, objA.b)

objA.set_b(objA.f(jnp.array([40.])))
idp = u_print(idp, objA.a, objA.b)


# -

# ## une variation

# +
class A():
    def __init__(self, a, b=None):
        ###print("Nouveau A") ## too verbosy
        self.a = a
        self.b = b

    @jit                                       
    def f(self, var):
        print("compile...")
        new_b = self.a * var
        return A(self.a, new_b)       # <---------on retourne un nouvel objet.


# registration with  lambda functions    
tree_util.register_pytree_node(A,
                               lambda x: ((x.a,x.b), None),
                               lambda _, x: A(a=x[0],b=x[1]) 
                              )


clear_cache
idp=0

objA = A(2.0)
idp = u_print(idp, objA.a, objA.b)    #0

objA = objA.f(10.)
idp = u_print(idp,objA.a, objA.b)     #1

objA = objA.f(11.)
idp = u_print(idp, objA.a, objA.b)    #2

####

objA = A(3.0) 
idp = u_print(idp, objA.a, objA.b)     #3
objA = objA.f(20.)
idp = u_print(idp, objA.a, objA.b)     #4
objA= objA.f(30.)
idp = u_print(idp, objA.a, objA.b)     #5


objA.a = 30.  # permis mais volontaire
objA= objA.f(30.)
idp = u_print(idp, objA.a, objA.b)     #6

####

objA = A(3.0) 
idp = u_print(idp, objA.a, objA.b)     #7

objA= objA.f(20.)
idp = u_print(idp, objA.a, objA.b)     #8

objA= objA.f(30.)
idp = u_print(idp, objA.a, objA.b)     #9

print(">>>> try an array for var")

objA= objA.f(jnp.array([30.]))
idp = u_print(idp, objA.a, objA.b)     #10

objA= objA.f(jnp.array([40.]))
idp = u_print(idp, objA.a, objA.b)     #11

objA= objA.f(jnp.array([50.]))         
idp = u_print(idp, objA.a, objA.b)     #12


objA.a = 40. ###
objA= objA.f(jnp.array([50.]))
idp = u_print(idp, objA.a, objA.b)     #13

# -

# ## Q: pourquoi une compilation entre 10) et 11)?

# # Bilan PyTree
#
# - ## on doit mettre en place d'une façon ou d'une autre `tree_flatten` et `tree_unflatten` avec soin
# - ## le changement de `a` est bien assimilé
# - ## choix d'implémention avec `nouvel objet à chaque fois` ou `setter` de la variable interne à changer
#

# # Autres variations sur un autre example

# +
clear_cache

class World:
    def __init__(self, p, v):
        self.p = p
        self.v = v

    @jit
    def step(self, dt):
        print("compile...")
        a = -9.8
        new_v = self.v + a * dt
        new_p = self.p + new_v * dt
        return World(new_p, new_v)

# By registering 'World' as a pytree, it turns into a transparent container and
# can be used as an argument to any JAX-transformed functions.
tree_util.register_pytree_node(World,
                     lambda x: ((x.p, x.v), None),
                     lambda _, tup: World(tup[0], tup[1]))

print("1st... go")

world = World(jnp.array([0., 0.5]), jnp.array([1., 1.5]))

for i in range(10):
    world = world.step(0.01)

print('res:', f"p:{world.p},{world.v}")

######
print("2nd... go")

class World:
    def __init__(self, p, v):
        self.p = p
        self.v = v

    def step(self, dt):
        a = - 9.8
        self.v += a * dt
        self.p += self.v *dt

@jit
def run(init_p, init_v):
    print("compile...")
    world = World(init_p, init_v)
    for i in range(10):
        world.step(0.01)
    return world.p, world.v

out = run(np.array([0, 0.5]), np.array([1, 1.5]))
print(out)


######
print("3rd... go")
#def fori_loop(lower, upper, body_fun, init_val):
#  val = init_val
#  for i in range(lower, upper):
#    val = body_fun(i, val)
#  return val

def body(i,carry):
    print("compile...")
    a = -9.8
    
    p  = carry[0]
    v  = carry[1]
    dt = carry[2]
    
    new_v = v + a * dt
    new_p = p + new_v * dt
    
    return (new_p, new_v, dt)

init_world = (jnp.array([0., 0.5]), jnp.array([1., 1.5]), 0.01)
res = jax.lax.fori_loop(0,10,body, init_world)
print(res)

    
    

# -

# # Takeaway message (JIT dans une classe):
# - plusieurs méthodes: `helper`, `self`-static, `Pytree`
# -  ne faut pas utiliser `self.<var> =` dans une fonction jitted (sinon `side` effect). Au passage l'analyse du code par JIT se fait via `Traced<ShapedArray>`
# - usage des `__hash__` et `__eq__`, et les `DeviceArray` sont `unhastable`
# - quand se déclenche la compilation (JIT) 
# - et `fori_loop` qui déclenche la compilation même si on ne le demande pas...
#
# - Voir également la doc [to-jit-or-not-to-jit](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html?highlight=Traced%3CShapedArray%3E#to-jit-or-not-to-jit)
#
# - Ai-je besoin de l'encapsulation OO (c'est-à-dire faire Class)


