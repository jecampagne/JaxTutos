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
clear_cache()


# + [markdown] tags=[]
# # Auto-diff sur des algorithmes User
# -

# ## Example de l'algorithme recursif de l'extraction de racine carrée.

# +
# version Numpy simple
def sqrt_rec(x):
    val = x
    for i in range(0,10):
        val = (val+x/val)/2.
    return val

# version JAX tout aussi simple
@jit
def jax_sqrt_rec(x):
    def body(i,val):
        return (val+x/val)/2.
    return jax.lax.fori_loop(0,10,body,x)



# -

plt.figure(figsize=(8,8))
x = jnp.arange(0,5,0.01)
plt.plot(x, sqrt_rec(x), label=r"$f(x)=\sqrt{x}$ (algo Numpy)")
plt.plot(x, jax_sqrt_rec(x), lw=3,ls='--', label=r"$f(x)$ (algo JAX)")
plt.plot(x, jnp.sqrt(x), lw=3,ls='--', label=r"$f(x)$ (True)")
plt.plot(x, vmap(grad(jax_sqrt_rec))(x), label=r"$f^\prime(x)$ (Diff. algo JAX)")
#plt.plot(x, vmap(grad(root))(x), label=r"$f^\prime(x)$ (Diff. algo Numpy)")  # Tout a fait possible mais moins Jax-idiom
plt.plot(x, 1/(2*jnp.sqrt(x)), lw=3,ls='--',label=r"$f^\prime(x)$ (True)")
plt.legend()
plt.grid();


# +
#jax.make_jaxpr(jax_root)(1.)
# -

# # Integration

# Version Numpy
def simps(f, a, b, N, f_args=(), f_kargs={}):
    #N doit etre paire
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x, *f_args, **f_kargs)
    w = np.ones_like(y)
    w[2:-1:2] = 2.
    w[1::2]   = 4.
    S = dx / 3. * np.einsum("i...,i...",w,y)
    return S


def ftest(a,b,N):
    x = jnp.linspace(a, b, N + 1) # xi = a + i * (b-a)/N
    return jnp.sum(x) # Sum_i=0^N xi = N*a + (N+1)*(b-a)/2


ftest(0.,1.,10)

gtest = lambda x: ftest(0.,x,10)

gtest(2.)

grad(gtest)(2.)


#version JAX
@partial(jit, static_argnums=(0,3,4,5))  # f et N ne seront des vecteurs
def jax_simps(f, a,b, N=512, f_args=(), f_kargs={}):
    #N doit etre paire
    dx = (b - a) / N
    x = jnp.linspace(a, b, N + 1)
    y = f(x, *f_args, **f_kargs)
    w = jnp.ones_like(y)
    w = w.at[2:-1:2].set(2.)     # w est immutable
    w = w.at[1::2].set(4.)
    S = dx / 3. * jnp.einsum('i...,i...',w,y)
    return S


def func(x):
    return x**(1/10) * jnp.exp(-x)


a = 0.
b = a+0.5

simps(func, a, b, N=2**10)

print(jax_simps(func, a, b, N=2**10))

sc.special.gamma(1.+1./10)*(1.-sc.special.gammaincc(1.+1./10,0.5))

print(jnp.exp(jsc.special.gammaln(1.+1./10)) *(1.-jsc.special.gammaincc(1.+1./10,0.5)))


# ## Le code de Simpson permet des calculs vectorisables

# Famille de functions
@jit
def jax_funcN(x):
    return jnp.stack([x**(i/10) * jnp.exp(-x) for i in range(50)],axis=1)


# lots d'intervalles d'integration [a,a+1/2] a:0,0.1,0.2...
ja = jnp.arange(0,20,0.1)
jb = ja+0.5

res = jax_simps(jax_funcN,ja,jb,  N=2**10)

res.shape

plt.figure(figsize=(8,8))
for i in range(0,50,5):
    plt.plot(ja,res[i,:],label=rf"$f_{{{i}}}$")
plt.xlabel("a")
plt.title(r"$\int_{a}^{a+1/2} f_i(x) dx$")
plt.grid()
plt.legend();


def fn(x,n):
    return x**(n/10) * jnp.exp(-x)
def test(n,z):
    return jax_simps(lambda z: fn(z,n),z,z+0.5,  N=2**10)


test(1.,0.)

z = jnp.linspace(0.6,0.8,100)

plt.plot(z,test(0.,z))
plt.plot(z,test(1.,z))
plt.plot(z,test(10.,z))
plt.plot(z,test(20.,z))
plt.grid()

# ## Vmap emboités
# On derive `test` par rapport à `n` et on veut calculer ce gradiant pour ($n_i$) et $(z_j)$ collection de $i,j$

z = jnp.linspace(0.1,1.0,100)

res = vmap(vmap(grad(test, argnums=0), in_axes=(None,0)),in_axes=(0,None))(jnp.array([0.,10., 20., 50.]),z)

res.shape

plt.plot(z,res[0,:])
plt.plot(z,res[1,:])
plt.plot(z,res[2,:])
plt.plot(z,res[3,:])
plt.ylim([-0.01,0.01])
plt.title(r"$\frac{\partial}{\partial n} (\int_{z}^{z+1/2} f_n(x) dx)$")
plt.xlabel(r"$z$")
plt.grid();

# ## Gradient de l'integration home made

f5 = lambda x: jnp.sqrt(x) * jnp.exp(-x)

intf5 = lambda x: jax_simps(f5,1e-7,x,N=2**10)

vintf5 = vmap(intf5)

plt.figure(figsize=(8,6))
x = jnp.arange(0,10,0.001)
plt.plot(x,f5(x),label=r"$f(x)=\sqrt{x}e^{-x}$")
plt.plot(x,vintf5(x), label=r"$\int_0^x f(z)dz$")
plt.hlines(np.sqrt(np.pi)/2,x.min(),x.max(),colors='k',ls='--',label=r"$\sqrt{\pi}/2$")
plt.plot(x,vmap(grad(intf5))(x), ls="--", lw=3, label="$\partial_x \int_0^x f(z)dz$")
plt.legend();


# # Une methode plus efficace d'integration

from CCquadInteg import *

# +
quad=ClenshawCurtisQuad(150)  # 300 pts
# function familly
@jit
def jax_funcN(x):
    return jnp.stack([x**(i/10) * jnp.exp(-x) for i in range(50)],axis=1)

# set of integration intervals [a,a+1/2] a:0,0.1,0.2...
ja = jnp.arange(0,10,0.5)
jb = ja+0.5

# -

#warm
# %time tmp=jax_simps(jax_funcN,ja,jb,N=2**15).block_until_ready()

# +

# %timeit  res_sim = jax_simps(jax_funcN,ja,jb,N=2**15).block_until_ready()
# -

#warm
# %time tmp=quadIntegral(jax_funcN,ja,jb,quad).block_until_ready()

# %timeit  res_cc=quadIntegral(jax_funcN,ja,jb,quad).block_until_ready()

res_sim = jax_simps(jax_funcN,ja,jb,N=2**15)
res_cc=quadIntegral(jax_funcN,ja,jb,quad)

np.allclose(res_cc,res_sim,rtol=0.,atol=1e-6)  # atol=1e-7 => False


def incremental_int(fn,y0,t, order=3):
    quad = ClenshawCurtisQuad(order)
    def integ(carry,t):
        y, t_prev = carry
        y = y + quadIntegral(fn,t_prev,t,quad)
        return (y,t),y
    (yf, _), y = jax.lax.scan(integ, (y0, jnp.array(t[0])), t)
    return y



quad=ClenshawCurtisQuad(150) 
tdef = jnp.linspace(0,10,4096)
res_cc_def = quadIntegral(f5,0.0,tdef,quad)

tinc = jnp.linspace(0,10,4096)
res_cc_inc = incremental_int(f5,0.0,tinc)

true_intf5 = lambda t: jnp.exp(jsc.special.gammaln(1.5))\
        *(1.-jsc.special.gammaincc(1.5,t))

t_val=jnp.linspace(0,10,4096)

fig, ax=plt.subplots(1,2,figsize=(15,6))
ax[0].plot(t_val, vmap(true_intf5)(t_val), label="True Integral")
ax[0].plot(t_val, jnp.interp(t_val,tdef,res_cc_def),lw=3,ls="--", label="CC Default")
ax[0].plot(t_val, jnp.interp(t_val,tinc,res_cc_inc),lw=3,ls="--", label="CC Increment")
ax[0].set_xlabel(r"$x$")
ax[0].legend()
ax[0].grid()
ax[0].set_title(r"$\int_0^x \sqrt{z} e^{-z} dz$");
#
ax[1].plot(t_val, jnp.abs(res_cc_def-vmap(true_intf5)(t_val)), label="CC def -True ")
ax[1].plot(t_val, jnp.abs(res_cc_inc-vmap(true_intf5)(t_val)), label="CC inc -True ")
ax[1].set_yscale("log")
ax[1].legend()
ax[1].grid()


# # Un peu de Cosmo: qq etudes autour des distances...

# +
from dataclasses import dataclass
from jax.tree_util import register_pytree_node_class

@dataclass(frozen=True)
class Constante:
    rh: float = 2997.92458  # h^{-1} Mpc
    H0: float = 100.0  # km/s/( h^{-1} Mpc)

const = Constante()


#The set of Python types that are considered internal PyTree nodes is extensible
@register_pytree_node_class
class Cosmology:
    def __init__(self, Omega_c, Omega_b, h, Omega_k, w0, wa):
        """
            Omega_c, float
              Cold dark matter density fraction.
            Omega_b, float
              Baryonic matter density fraction.
            h, float
              Hubble constant divided by 100 km/s/Mpc; unitless.
            Omega_k, float
              Curvature density fraction.
            w0, float
              First order term of dark energy equation
            wa, float
              Second order term of dark energy equation of state
        """
        # Store primary parameters
        self._Omega_c = Omega_c
        self._Omega_b = Omega_b
        self._h = h
        self._Omega_k = Omega_k
        self._w0 = w0
        self._wa = wa
    
        # Create a workspace where functions can store some precomputed
        # results
        self._workspace = {}

    def __str__(self):
        return (
              " h:"+ str(self.h)
            + " Omega_b:"+ str(self.Omega_b)
            + " Omega_c:"+ str(self.Omega_c)
            + " Omega_k:"+ str(self.Omega_k)
            + " w0:"+ str(self.w0)
            + " wa:"+ str(self.wa)
        )

    def __repr__(self):
        return self.__str__()

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        params = (
            self._Omega_c,
            self._Omega_b,
            self._h,
            self._Omega_k,
            self._w0,
            self._wa,
        )
        aux_data = None
        return (params, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # Retrieve base parameters
        Omega_c, Omega_b, h, Omega_k, w0, wa = children

        return cls(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            Omega_k=Omega_k,
            w0=w0,
            wa=wa,
        )
    
    # Cosmological parameters, base and derived
    @property
    def Omega(self):
        return 1.0 - self._Omega_k

    @property
    def Omega_b(self):
        return self._Omega_b

    @property
    def Omega_c(self):
        return self._Omega_c

    @property
    def Omega_m(self):
        return self._Omega_b + self._Omega_c

    @property
    def Omega_de(self):
        return self.Omega - self.Omega_m

    @property
    def Omega_k(self):
        return self._Omega_k

    @property
    def k(self):
        return -jnp.sign(self._Omega_k).astype(jnp.int8)

    @property
    def sqrtk(self):
        return jnp.sqrt(jnp.abs(self._Omega_k))

    @property
    def h(self):
        return self._h

    @property
    def w0(self):
        return self._w0

    @property
    def wa(self):
        return self._wa
# +
def z2a(z):
    """converts from redshift to scale factor"""
    return 1.0 / (1.0 + z)

def a2z(a):
    """converts from scale factor to  redshift"""
    return 1.0 / a - 1.0

def w(cosmo, a):
    """
    Linder 2003
     w(a) = w_0 + w_a (1 - a)
    """
    return cosmo.w0 + (1.0 - a) * cosmo.wa 

def f_de(cosmo, a):
    """
        f(a) = -3 (1 + w_0 + w_a) \ln(a) + 3 w_a (a - 1)
    """
    return -3.0 * (1.0 + cosmo.w0 + cosmo.wa) * jnp.log(a) + 3.0 * cosmo.wa * (a - 1.0)

def Esqr(cosmo, a):
    """
      E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} e^{f(a)} =H(a)^2/H0^2
    """
    return (
        cosmo.Omega_m * jnp.power(a, -3)
        + cosmo.Omega_k * jnp.power(a, -2)
        + cosmo.Omega_de * jnp.exp(f_de(cosmo, a))
    )

def H(cosmo, a):
    """
    H(a) = H0 * E(a)
    """
    return const.H0 * jnp.sqrt(Esqr(cosmo, a))

def dchioverda(cosmo, a):
    """
    \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}
    """
    return const.rh / (a**2 * jnp.sqrt(Esqr(cosmo, a)))

def radial_comoving_distance(cosmo, a, log10_amin=-3, steps=256):
    """
    \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
    """
    # Check if distances have already been computed
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = jnp.logspace(log10_amin, 0.0, steps)

        def dchioverdlna_ode(y,x):
            xa = jnp.exp(x)
            return dchioverda(cosmo, xa) * xa

        def dchioverdlna(x):
            xa = jnp.exp(x)
            return dchioverda(cosmo, xa) * xa

        
        chitab_cc_inc = incremental_int(dchioverdlna, 0.0, jnp.log(atab), order=3)
        chitab_cc_inc = chitab_cc_inc[-1]-chitab_cc_inc
        
        cache = {"a": atab, "chi":chitab_cc_inc}
        cosmo._workspace["background.radial_comoving_distance"] = cache
    else:
        cache = cosmo._workspace["background.radial_comoving_distance"]

    #a = jnp.atleast_1d(a)
    # Return the results as an interpolation of the table
    return jnp.clip(jnp.interp(a, cache["a"], cache["chi"]), 0.0)

def transverse_comoving_distance(cosmo, a):
    r"""Transverse comoving distance in [Mpc/h] for a given scale factor.

    f_k : ndarray, or float if input scalar
        Transverse comoving distance corresponding to the specified
        scale factor.
    Notes
    -----
    The transverse comoving distance depends on the curvature of the
    universe and is related to the radial comoving distance through:
    .. math::
        f_k(a) = \left\lbrace
        \begin{matrix}
        R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for }\Omega_k > 0 \\
        \chi(a)&
            \mbox{for } \Omega_k = 0 \\
        R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for } \Omega_k < 0
        \end{matrix}
        \right.
    """  
    index = cosmo.k + 1

    def open_universe(chi):
        return const.rh / cosmo.sqrtk * jnp.sinh(cosmo.sqrtk * chi / const.rh)

    def flat_universe(chi):
        return chi

    def close_universe(chi):
        return const.rh / cosmo.sqrtk * jnp.sin(cosmo.sqrtk * chi / const.rh)

    branches = (open_universe, flat_universe, close_universe)

    chi = radial_comoving_distance(cosmo, a)

    #def switch(index, branches, operand):
    #index = clamp(0, index, len(branches) - 1)
    #return branches[index](operand)

    return jax.lax.switch(cosmo.k + 1, branches, chi)

def angular_diameter_distance(cosmo, a):
    """Angular diameter distance in [Mpc/h] for a given scale factor.
        d_A(a) = a f_k(a)
    """
    return a * transverse_comoving_distance(cosmo, a)

def luminosity_distance(cosmo, a):
    """Angular diameter distance in [Mpc/h] for a given scale factor.
        d_L(a) = f_k(a) / a
    """
    return transverse_comoving_distance(cosmo, a)/a


# -

# $$
# f_k(a) = \left\lbrace
#         \begin{matrix}
#         R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
#             \mbox{for }\Omega_k > 0 \\
#         \chi(a)&
#             \mbox{for } \Omega_k = 0 \\
#         R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
#             \mbox{for } \Omega_k < 0
#         \end{matrix}
#         \right.
# $$ 

cosmo_jax = Cosmology(
    Omega_c=0.2545,
    Omega_b=0.0485,
    h=0.682,
    Omega_k=0.0,
    w0=-1.0,
    wa=0.0  
)

cosmo_jax.Omega_de   # 1 - Omega_k - cosmo_jax._Omega_m

cosmo_jax.Omega_m,  cosmo_jax._Omega_b + cosmo_jax._Omega_c  # cosmo_jax._Omega_b + cosmo_jax._Omega_c

1 - cosmo_jax.Omega_k - cosmo_jax.Omega_m

luminosity_distance(cosmo_jax,0.1)

cosmo_jax.Omega_de + cosmo_jax.Omega_m

z = jnp.logspace(-3, 3,100)
chi_z = radial_comoving_distance(cosmo_jax, z2a(z))/ cosmo_jax.h
trans_com_z = transverse_comoving_distance(cosmo_jax, z2a(z))/ cosmo_jax.h
ang_diam_z= angular_diameter_distance(cosmo_jax, z2a(z))/ cosmo_jax.h
lum_z = luminosity_distance(cosmo_jax, z2a(z)) / cosmo_jax.h

plt.figure(figsize=(8,8))
factMpc2Gly = 0.0032615 #Mpc -> Gly
plt.plot(z,factMpc2Gly * chi_z, label=r"$\chi(z)$")
plt.plot(z,factMpc2Gly * const.rh/cosmo_jax.h * z, label="Naive Hubble law")
plt.plot(z,factMpc2Gly * trans_com_z,ls="--",lw=3, label=rf"$Trans. Com. dist.$ (k={cosmo_jax.k})")
plt.plot(z,factMpc2Gly * ang_diam_z, label=r"$d_A(z)$")
plt.plot(z,factMpc2Gly * lum_z, label=r"$d_L(z)$")
plt.xlabel("redshift (z)")
plt.ylabel("Distance (Gly)")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.grid()

omc=jnp.linspace(0.001,0.5,5)
#cosmo_jax = Cosmology(
#    Omega_c=0.2545,
#    Omega_b=0.0485,
#    h=0.682,
#    Omega_k=0.0,
#    w0=-1.0,
#    wa=0.0  
#)

def test(cosmo,z):
    return cosmo.Omega_c * z


vmap(grad(test), in_axes=(None,0))(cosmo_jax,jnp.array([0.01,0.1,1.0])).Omega_c

vmap(grad(test), in_axes=(None,0))(cosmo_jax,jnp.array([0.01,0.1,1.0])).Omega_b

vlum = vmap(luminosity_distance, in_axes=(Cosmology(0, None,None,None,None,None), None))
vang_diam = vmap(angular_diameter_distance, in_axes=(Cosmology(0, None,None,None,None,None), None))

cosmo_cs = Cosmology(omc,cosmo_jax.Omega_b,cosmo_jax.h,cosmo_jax.Omega_k,cosmo_jax.w0,cosmo_jax.wa)

cosmo_cs.Omega_de[0]

data_lum=vlum(cosmo_cs,z2a(z))
data_lum /= cosmo_jax.h

data_ang_diam=vang_diam(cosmo_cs,z2a(z))
data_ang_diam /= cosmo_jax.h

plt.figure(figsize=(10,10))
color = plt.cm.rainbow(np.linspace(0, 1, len(omc)))
for i,c in enumerate(omc):
    plt.plot(z, factMpc2Gly*data_lum[i,:], c=color[i],
             label=fr"$\Omega_c={c:.2}\rightarrow \Omega_{{\Lambda}}={cosmo_cs.Omega_de[i]:.2}$")
    plt.plot(z, factMpc2Gly*data_ang_diam[i,:], c=color[i])
    plt.plot(z, factMpc2Gly*np.sqrt(data_lum[i,:]*data_ang_diam[i,:]), c=color[i])
plt.legend()
plt.xlabel("z")
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r"$d_L$, $d_A$, $f_k=\sqrt{d_L d_A}$ Gly)")
plt.grid();

grad_lum = vmap(grad(luminosity_distance), in_axes=(None,0))(cosmo_jax,z2a(z))

func = lambda a: -0.5*const.rh * (-1.+1./a**2) / (a**2 * Esqr(cosmo_jax, a)**1.5)
dLdOmega_k = quadIntegral(func,z2a(z),1.0,ClenshawCurtisQuad(100))*(1+z)

#cosmo_jax = Cosmology(
#    Omega_c=0.2545,
#    Omega_b=0.0485,
#    h=0.682,
#    Omega_k=0.0,
#    w0=-1.0,
#    wa=0.0  
#)
plt.figure(figsize=(10,10))
plt.plot(z,grad_lum.Omega_c/cosmo_jax.h,ls='-',lw=3,label=r"$\frac{\partial d_L}{\partial \Omega_c}$")
plt.plot(z,grad_lum.Omega_b/cosmo_jax.h,ls='--',lw=3, label=r"$\frac{\partial d_L}{\partial \Omega_b}$")
plt.plot(z,grad_lum.h/cosmo_jax.h,ls='--',lw=3, label=r"$\frac{\partial d_L}{\partial h}$")
plt.plot(z,grad_lum.Omega_k/cosmo_jax.h,ls='-',lw=3, label=r"$\frac{\partial d_L}{\partial \Omega_k}$")
plt.plot(z,grad_lum.w0/cosmo_jax.h,ls='--',lw=3, label=r"$\frac{\partial d_L}{\partial w_0}$")
plt.plot(z,grad_lum.wa/cosmo_jax.h,ls='--',lw=3, label=r"$\frac{\partial d_L}{\partial w_a}$")
plt.plot(z,dLdOmega_k/cosmo_jax.h, ls='--',lw=3,c='k',label=r"Direct Integ: $\frac{\partial d_L}{\partial \Omega_k}$")
plt.yscale("symlog")
plt.xscale("log")
plt.legend()
plt.grid();


