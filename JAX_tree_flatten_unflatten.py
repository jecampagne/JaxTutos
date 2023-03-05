
import jax
from jax import lax, grad, jit, vmap
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class State:
  def __init__(self, n: int):
    self.n = n
    self.stateful()
    jax.debug.print('   n = {} # init', self.n)

  def stateful(self):
    self.n += 100

  def tree_flatten(self):
    jax.debug.print('   n = {} # flatten', self.n + 10)
    return (self.n,), {}

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    n, = children
    jax.debug.print('---')
    jax.debug.print('   n = {} # unflatten', n + 1)
    return cls(n)  # runs `__init__` and in turn `stateful`

def body_fun(carry, i):
  # unflatten
  # - __init__
  jax.debug.print('{}: n = {} # body', i, carry.n)
  # flatten
  return carry, carry.n

s = State(0)
print('---')
_, y = lax.scan(body_fun, s, jnp.arange(3, dtype=jnp.int_))
y

# Output:
#    n = 100 # init
# ---
#    n = 110 # flatten
#    n = 110 # flatten
# ---
#    n = 101 # unflatten
#    n = 200 # init
# 0: n = 200 # body
#    n = 210 # flatten
# ---
#    n = 201 # unflatten
#    n = 300 # init
# 1: n = 300 # body
#    n = 310 # flatten
# ---
#    n = 301 # unflatten
#    n = 400 # init
# 2: n = 400 # body
#    n = 410 # flatten
# ---
#    n = 401 # unflatten
#    n = 500 # init
