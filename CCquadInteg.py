
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
jax.config.update("jax_enable_x64", True)

from functools import partial



class ClenshawCurtisQuad:
    """
        Clenshaw-Curtis quadrature computed by FFT
    """

    def __init__(self,order=5):
        # 2n-1 quad
        self._order = jnp.int64(2*order-1)
        self._absc, self._absw, self._errw = self.ComputeAbsWeights()
        self.rescaleAbsWeights()
    
    def __str__(self):
        return f"xi={self._absc}\n wi={self._absw}\n errwi={self._errw}"
    
    @property
    def absc(self):
        return self._absc
    @property
    def absw(self):
        return self._absw
    @property
    def errw(self):
        return self._errw
        
    def ComputeAbsWeights(self):
        x,wx = self.absweights(self._order)
        nsub = (self._order+1)//2
        xSub, wSub = self.absweights(nsub)
        errw = jnp.array(wx, copy=True)                           # np.copy(wx)
        errw=errw.at[::2].add(-wSub)             # errw[::2] -= wSub
        return x,wx,errw
  
    
    def absweights(self,n):

        points = -jnp.cos((jnp.pi * jnp.arange(n)) / (n - 1))

        if n == 2:
            weights = jnp.array([1.0, 1.0])
            return points, weights
            
        n -= 1
        N = jnp.arange(1, n, 2)
        length = len(N)
        m = n - length
        v0 = jnp.concatenate([2.0 / N / (N - 2), jnp.array([1.0 / N[-1]]), jnp.zeros(m)])
        v2 = -v0[:-1] - v0[:0:-1]
        g0 = -jnp.ones(n)
        g0 = g0.at[length].add(n)     # g0[length] += n
        g0 = g0.at[m].add(n)          # g0[m] += n
        g = g0 / (n ** 2 - 1 + (n % 2))

        w = jnp.fft.ihfft(v2 + g)
        ###assert max(w.imag) < 1.0e-15
        w = w.real

        if n % 2 == 1:
            weights = jnp.concatenate([w, w[::-1]])
        else:
            weights = jnp.concatenate([w, w[len(w) - 2 :: -1]])
            
        #return
        return points, weights
    
    def rescaleAbsWeights(self, xInmin=-1.0, xInmax=1.0, xOutmin=0.0, xOutmax=1.0):
        """
            Translate nodes,weights for [xInmin,xInmax] integral to [xOutmin,xOutmax] 
        """
        deltaXIn = xInmax-xInmin
        deltaXOut= xOutmax-xOutmin
        scale = deltaXOut/deltaXIn
        self._absw *= scale
        tmp = jnp.array([((xi-xInmin)*xOutmax
                         -(xi-xInmax)*xOutmin)/deltaXIn for xi in self._absc])
        self._absc=tmp
        
@partial(jit, static_argnums=(0,3))
def quadIntegral(f,a,b,quad):
    a = jnp.atleast_1d(a)
    b = jnp.atleast_1d(b)
    d = b-a
    xi = a[jnp.newaxis,:]+ jnp.einsum('i...,k...->ik...',quad.absc,d)
    fi = f(xi)
    S = d * jnp.einsum('i...,i...',quad.absw,fi)
    return S.squeeze()