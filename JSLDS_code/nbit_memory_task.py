import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt



"""Routines for creating 3-bit memory task."""

def build_one_bit_memory(input_params, key):
  """Build white noise input and integration targets."""
  ntimesteps = input_params[0]

  # Generate random pulse inputs.
  key, skey = jax.random.split(key, 2)
  pre_inputs_t = jax.random.randint(skey, (ntimesteps,), 0, 3) - 1.0

  # For each input stream, set the target output to be the most recent
  # nonzero input.
  inputs_1 = jnp.expand_dims(jnp.where(pre_inputs_t < -0.5, 1.0, 0.0), axis=1)
  inputs_2 = jnp.expand_dims(jnp.where(pre_inputs_t > 0.5, 1.0, 0.0), axis=1)
  inputs_tx2 = jnp.concatenate((inputs_1, inputs_2), axis=1)
  targ = -1.0  # always start in off position
  targets_t = []
  for t in range(ntimesteps):
    targ = jnp.where(pre_inputs_t[t] < -0.5, -1.0, targ)
    targ = jnp.where(pre_inputs_t[t] > 0.5, 1.0, targ)
    targets_t.append(targ)
 
  ntosettle = 5
  target_mask = jnp.arange(ntosettle, ntimesteps)
  return inputs_tx2, jnp.array(targets_t), target_mask

def build_nbit_memory(input_params, key):
  """Top level function for building inputs."""
  ntimesteps, ninputs, _ = input_params
  inputs = []
  targets = []
  masks = []
  skeys = jax.random.split(key, ninputs)
  for skey in skeys:
    inputs_tx2, targets_t, mask = build_one_bit_memory(input_params, skey)
    inputs.append(inputs_tx2.T)
    targets.append(targets_t)
    masks.append(mask)
  inputs = jnp.transpose(jnp.array(inputs), [2, 0, 1])
  return jnp.reshape(inputs, [ntimesteps, -1]), jnp.array(targets).T, jnp.array(masks).T


nbit_memory_build_inputs_and_targets = \
    jax.vmap(build_nbit_memory, in_axes=(None, 0))

def nbit_memory_plot_batch(ntimesteps, input_bxtxu, target_bxtxo=None, 
                           output_bxtxo=None, errors_bxtxo=None):
  """Plot some white noise / integrated white noise examples."""
  bidx = 0
  ninputs = input_bxtxu.shape[2]
  ntargets = target_bxtxo.shape[2]
  plt.figure(figsize=(10, 7))
  plt.subplot(311)
  expander = 2*jnp.expand_dims(jnp.arange(ninputs), axis=0)
  plt.plot(input_bxtxu[bidx, :, :] + expander)
  plt.xlim([0, ntimesteps-1])
  plt.ylabel("Input Blips")
  plt.subplot(312)

  if output_bxtxo is not None:
    expander = 2 * jnp.expand_dims(jnp.arange(ntargets), axis=0)
    plt.plot(output_bxtxo[bidx, :, :] + expander, label='JSLDS Output')
    plt.xlim([0, ntimesteps-1])

  if target_bxtxo is not None:
    expander = 2 * jnp.expand_dims(jnp.arange(ntargets), axis=0)
    plt.plot(target_bxtxo[bidx, :, :] + expander, "--", label='Target')
    plt.xlim([0, ntimesteps-1])
    plt.ylabel("Memory")
  
  plt.legend(loc = 'upper right')

  if errors_bxtxo is not None:
    expander = 2 * jnp.expand_dims(jnp.arange(ntargets), axis=0)
    plt.subplot(313)
    plt.plot(errors_bxtxo[bidx, :, :] + expander, "--")
    plt.xlim([0, ntimesteps-1])
    plt.ylabel("|Errors|")
  plt.xlabel("Timesteps")
