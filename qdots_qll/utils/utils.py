import jax.numpy as jnp


def ensure_array(array):
    array = jnp.array(array)
    if array.shape == ():
        return jnp.array([array])
    return array


def ensure_particles_shape(x):
    if len(x.shape) == 1:
        return x[:, None]
    else:
        return x
