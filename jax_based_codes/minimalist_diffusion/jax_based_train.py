from jax_based_model import diffusion_model
from data import generate_data, PROB
from torch.utils.data import DataLoader
import flax.linen as nn
from jax import jit
import jax.numpy as jnp
from flax.serialization import to_bytes, from_bytes
from torch.utils.data import DataLoader
import flax
from flax.training import train_state
import functools
import numpy as np
import optax 
import jax
import torch

batch_size = 1
p0 , p1 = generate_data()

dataset = PROB(p0[0], p1[0])
dataloader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=4)
    
rng = jax.random.PRNGKey(0)
fake_input = jnp.ones((1,2))
fake_alpha = jnp.ones(1)
score_model = diffusion_model()
params = score_model.init({'params': rng}, fake_input, fake_alpha)


# @jax.jit
# def train_step(state, x0, x1):
#     def loss_fn(rng, params):
#         rng, step_rng = jax.random.split(rng)
#         alpha = jax.random.uniform(step_rng, (x0.shape[0],), minval=1e-5, maxval = 1.)
#         x_alpha = (1 - alpha)*x0 + alpha*x1
#         output = diffusion_model.apply({'params':params}, x_alpha, alpha)
#         loss = jnp.mean(jnp.square(output - (x1-x0)))
#         return loss, output
    
#     (loss, outs), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
#     state = state.apply_gradients(grads=grads)
#     return state, loss
    

# def train_one_epoch(state, dataloader, epoch):
#     for i, (x0, x1) in enumerate(dataloader):
#         x0 = x0.numpy()
#         x1 = x1.numpy()
#         state, loss = train_step(state, x0, x1)

#     return state, loss

# def create_train_state(key, learning_rate):
#     model = diffusion_model()
#     params = model.init(key, jnp.ones(1, 2), jnp.ones(1))['params']
#     adam_opt = optax.adam(learning_rate)
#     return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adam_opt)



    
seed = 0 
learning_rate = 1e-5
num_epochs = 1e5

# train_state = create_train_state(jax.random.PRNGKey(seed), learning_rate)

# for epoch in range(1, num_epochs+1):
#     train_state, loss = train_one_epoch(train_state, dataloader, epoch)
#     print(f"Train epoch : {epoch}, loss: {loss}")


def loss_fn(rng, model, params, x0, x1):
    rng, step_rng = jax.random.split(rng)
    alpha = jax.random.uniform(step_rng, (x0.shape[0],), minval=1e-5, maxval =1.)
    rng, step_rng = jax.random.split(rng)
    x_alpha = (1- alpha)*x0 + alpha*x1
    model_output = model.apply(params, x_alpha, alpha)
    loss = jnp.mean(jnp.square(model_output- (x1-x0)))

    return loss

def get_train_step_fn(model):

    val_and_grad_fn  = jax.value_and_grad(loss_fn, argnums=2)
    
    def step_fn(rng, x, opt_var, opt_state):

        pass

    pass

opt_adam = optax.adam(learning_rate=learning_rate)
opt_state = opt_adam.init(params)

print(opt_state)