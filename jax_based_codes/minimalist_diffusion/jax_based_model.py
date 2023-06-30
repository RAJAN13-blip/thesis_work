import jax 
import jax.numpy as jnp 
import flax 
import flax.linen as nn
import optax 
import functools
import numpy as np 
import random
from typing import Any, Tuple


class alpha_step_projection(nn.Module):
    embed_dim : int
    scale : float = 30.
    
    @nn.compact
    def __call__(self, x):
        W = self.param('W', jax.nn.initializers.normal(stddev=self.scale),
                       (self.embed_dim//2,))
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None]* W[None, :] * 2* jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis = -1)
    
class Dense(nn.Module):
    """A fully connected layer that reshapes outputs for addition"""
    output_dim : int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)
    
class diffusion_model(nn.Module):
    channels : Tuple[int] = (64,64,64,64,64,2)
    embed_dim : int = 256
    train : bool = True

    @nn.compact
    def __call__(self, x, alpha):
        act_swish = nn.swish
        act_relu = nn.relu
        embed = act_swish(nn.Dense(self.embed_dim)(alpha_step_projection(embed_dim=self.embed_dim)(alpha)))

        h1 = nn.Dense(features=self.channels[0])(x)
        h1+= Dense(output_dim=self.channels[0])(embed)
        h1 = act_swish(h1)
        # h1 = nn.BatchNorm(use_running_average= not self.train)(h1)

        h2 = nn.Dense(features=self.channels[1])(h1)
        h2+= Dense(output_dim=self.channels[1])(embed)
        h2 = act_swish(h2)
        # h2 = nn.BatchNorm(use_running_average= not self.train)(h2)

        h3 = nn.Dense(features=self.channels[2])(h2)
        h3+= Dense(output_dim=self.channels[2])(embed)
        h3 = act_swish(h3)
        # h3 = nn.BatchNorm(use_running_average= not self.train)(h3)

        h4 = nn.Dense(features=self.channels[3])(h3)
        h4 += Dense(output_dim=self.channels[3])(embed)
        h4 = act_relu(h4)
        # h4 = nn.BatchNorm(use_running_average= not self.train)(h4)

        h5 = nn.Dense(features=self.channels[4])(h4)
        h5 += Dense(output_dim=self.channels[4])(embed)
        h5 = act_relu(h5)
        # h5 = nn.BatchNorm(use_running_average= not self.train)(h5)

        h6 = nn.Dense(features=self.channels[5])(h5)
        h6 += Dense(output_dim=self.channels[5])(embed)
        h6 = act_relu(h6)
        # h6 = nn.BatchNorm(use_running_average= not self.train)(h6)
        return h6
    
rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((1,2))
fake_alpha = jnp.ones(1)
score_model = diffusion_model()
params = score_model.init({'params': rng}, fake_input, fake_alpha)
op = score_model.apply(params, fake_input, fake_alpha)
