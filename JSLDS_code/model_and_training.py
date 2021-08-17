import functools
import jax
from jax.ops import index_update, index
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np
import math

def keygen(key, nkeys):
  """Generate randomness that JAX can use by splitting the JAX keys.
  Args:
    key : the random.PRNGKey for JAX
    nkeys : how many keys in key generator
  Returns:
    2-tuple (new key for further generators, key generator)
  """
  keys = jax.random.split(key, nkeys+1)
  return keys[0], (k for k in keys[1:])


def get_warmup_fun(warmup_start, warmup_end, val_min, val_max):
  """Warmup function to avoid pathological conditions early in training.

  Args:
    warmup_start: index to start warmup
    warmup_end: index to stop warmup
    val_min: minimal value for warmup at beginning
    val_max: maximal value for wamrup at end

  Returns:
    a function which yields the warmup value
  """

  def warmup(batch_idx):
    progress_frac = ((batch_idx - warmup_start) / (warmup_end - warmup_start))
    warmup = jnp.where(batch_idx < warmup_start,
                       val_min,
                       (val_max - val_min) * progress_frac + val_min)
    return jnp.where(batch_idx > warmup_end, val_max, warmup)
  return warmup



def affine_params(key, **affine_hps):
  """Params for y = W x + b.

  Args:
    key: random.PRNGKey for random bits
    **affine_hps: dict of hps such as 'u' for input size, 'o' for output size,
      'ifactor' for scaling factor

  Returns:
    a dictionary of parameters
  """

  key, skeys = keygen(key, 1)
  u = affine_hps['u']
  o = affine_hps['o']
  i_factor = affine_hps['i_factor']
  i_factor = i_factor / jnp.sqrt(u)
  return {'W': jax.random.normal(next(skeys), (o, u)) * i_factor,
          'b': jnp.zeros((o,))}

def mlp_params(key, nlayers, n):  # pylint: disable=unused-argument
  """Build a very specific multilayer perceptron for picking fixed points.

  Args:
    key: random.PRNGKey for randomness
    nlayers: number of layers in the MLP.
    n: MLP dimension

  Returns:
    List of dictionaries, where list index is the layer (indexed by integer)
      and each dict is the params of the layer, i.e. weights and bias.
  """
  params = [None] * nlayers
  for l in range(nlayers):
    # Below we build against identity, so zeros appropriate here.
    params[l] = {'W': jnp.zeros([n, n]), 'b': jnp.zeros(n)}
  return params

def gru_params(key, **rnn_hps):
  """Generate GRU parameters.

  Args:
    key: random.PRNGKey for random bits
    **rnn_hps: key-value pairs in dict as
       n, hidden state size
       u, input size
       i_factor, scaling factor for input weights
       h_factor, scaling factor for hidden -> hidden weights
       h_scale, scale on h0 initial condition

  Returns:
    a dictionary of parameters
  """
  key, skeys = keygen(key, 5)
  u = rnn_hps['u']
  n = rnn_hps['n']

  ifactor = rnn_hps['i_factor'] / jnp.sqrt(u)
  hfactor = rnn_hps['h_factor'] / jnp.sqrt(n)
  hscale = rnn_hps['h_scale']

  wRUH = jax.random.normal(next(skeys), (n + n, n)) * hfactor
  wRUX = jax.random.normal(next(skeys), (n + n, u)) * ifactor
  wRUHX = jnp.concatenate([wRUH, wRUX], axis=1)

  wCH = jax.random.normal(next(skeys), (n, n)) * hfactor
  wCX = jax.random.normal(next(skeys), (n, u)) * ifactor
  wCHX = jnp.concatenate([wCH, wCX], axis=1)

  return {'h0': jax.random.normal(next(skeys), (n,)) * hscale,
          'wRUHX': wRUHX,
          'wCHX': wCHX,
          'bRU': jnp.zeros((n+n,)),
          'bC': jnp.zeros((n,))}


def vrnn_params(key, **rnn_hps):
  """Generate random Vanilla RNN parameters.

  Args:
    key: random.PRNGKey for randomness
    **rnn_hps: dictionary of hps for VRNN, re size and scaling

  Returns:
    a dict of params appropriate for a 1-layer VRNN
  """

  u = rnn_hps['u']
  n = rnn_hps['n']
  key, skeys = keygen(key, 4)
  hscale = 0.1
  ifactor = 1.0 / jnp.sqrt(u)
  # pfactor = 1.0 / jnp.sqrt(n)

  if rnn_hps['recurrent_is_identity']:
    wR = rnn_hps['h_factor'] * jnp.eye(n)
  else:
    g = rnn_hps['h_factor'] / jnp.sqrt(n)
    wR = jax.random.normal(next(skeys), (n, n)) *  g

  return {'h0': jax.random.normal(next(skeys), (n,)) * hscale,
          'wI': jax.random.normal(next(skeys), (n, u)) * ifactor,
          'wR': wR,
          'bR': jnp.zeros([n])}

def jslds_rnn_params(key, **rnn_hps):
  """Generate parameters for Jacobian Switching Linear Dynamical Systems model.

  Args:
    key: random.PRNGKey for randomness
    **rnn_hps: (nested) dictionary of hyperparameters relevant for JSLDS model.

  Returns:
    A (nested) dictionary of all params relevant for JSLDS RNN.
  """
  key, skeys = keygen(key, 3)
  rnn_params_fun = rnn_hps['rnn']['params_fun']
  return {'mlp': mlp_params(next(skeys), **rnn_hps['mlp']),
          'rnn': rnn_params_fun(next(skeys), **rnn_hps['rnn']),
          'out': affine_params(next(skeys), **rnn_hps['out'])}


def sigmoid(x):
  return 0.5 * (jnp.tanh(x / 2.) + 1)


def affine(params, x):
  """Implement y = W x + b."""
  return jnp.dot(params['W'], x) + params['b']


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims)
# So map over first dimension to handle t_x_m.
# I.e. if affine yields n_y_1 = dot(n_W_m, m_x_1), then
# batch_affine yields t_y_n.
# And so the vectorization pattern goes for all batch_* functions.
batch_affine = jax.vmap(affine, in_axes=(None, 0))


def mlp_tanh(params, x):
  """Multilayer perceptron with tanh nonlinearity.

  Args:
    params: dict of params for MLP
    x: input

  Returns:
    hidden state after applying the MLP
  """
  h = x
  for layer in params:
    h = jnp.tanh(h + jnp.dot(layer['W'], h) + layer['b'])
  return h

def mlp_relu(params, x, b=0.01):
  """Multilayer perceptron with relu nonlinearity.

  Args:
    params: dict of params for MLP
    x: input
    b: static bias for each layer

  Returns:
    hidden state after applying the MLP
  """
  h = x
  for layer in params:
    a = h + jnp.dot(layer['W'], h) + layer['b'] + b
    h = jnp.where(a > 0.0, a, 0.0)
  return h


mlp = mlp_tanh
batch_mlp = jax.vmap(mlp, in_axes=(None, 0))


def vrnn_tanh(params, h, x):
  """Run the Vanilla RNN (tanh) one step.

  Args:
    params: dict of params for VRNN
    h: hidden state of the RNN
    x: input for the RNN

  Returns:
    the hidden state of the RNN after running one step forward in time
  """
  a = jnp.dot(params['wI'], x) + params['bR'] + jnp.dot(params['wR'], h)
  return jnp.tanh(a)


def vrnn_relu(params, h, x):
  """Run the Vanilla RNN (relu) one step.

  Args:
    params: dict of params for VRNN
    h: hidden state of the RNN
    x: input for the RNN

  Returns:
    the hidden state of the RNN after running one step forward in time
  """
  a = jnp.dot(params['wI'], x) + params['bR'] + jnp.dot(params['wR'], h)
  return jnp.where(a > 0.0, a, 0.0)


def gru(params, h, x, bfg=0.0):
  """Implement the GRU equations.

  Arguments:
    params: dictionary of GRU parameters
    h: np array of  hidden state
    x: np array of input
    bfg: bias on forget gate (useful for learning if > 0.0)

  Returns:
    np array of hidden state after GRU update
  """

  hx = jnp.concatenate([h, x], axis=0)
  ru = jnp.dot(params['wRUHX'], hx) + params['bRU']
  r, u = jnp.split(ru, 2, axis=0)
  u = u + bfg
  u = sigmoid(u)
  r = sigmoid(r)
  rhx = jnp.concatenate([r * h, x])
  c = jnp.tanh(jnp.dot(params['wCHX'], rhx) + params['bC'])
  return u * h + (1.0 - u) * c


def taylor(f, order):
  """Compute nth order Taylor series approximation of f.

  Args:
    f: the function to compute the Taylor series expansion on, with signature
        f:: h, x -> h
    order: the order of the expansion (int)

  Returns:
    order-order Taylor series approximation as a function with signature
      T[f]: h, x -> h
  """

  def jvp_first(f, primals, tangent):
    """Jacobian-vector product of first argument element."""
    x, xs = primals[0], primals[1:]
    return jax.jvp(lambda x: f(x, *xs), (x,), (tangent,))

  def improve_approx(g, k):
    """Improve taylor series approximation step-by-step."""
    return lambda x, v: jvp_first(g, (x, v), v)[1] + f(x) / math.factorial(k)

  approx = lambda x, v: f(x) / math.factorial(order)
  for n in range(order):
    approx = improve_approx(approx, order - n - 1)
  return approx

def taylor_approx_rnn(rnn, params, h_star, x_star, h_approx_tm1, x_t, order):
  xdim = x_t.shape[0]
  hx_star = jnp.concatenate([h_star, x_star], axis=0)
  hx = jnp.concatenate([h_approx_tm1, x_t], axis=0)

  Fhx = lambda hx: rnn(params, hx[:-xdim], hx[-xdim:])
  return taylor(Fhx, order)(hx_star, hx - hx_star)


def staylor_rnn(rnn, params, order, h_tm1, h_approx_tm1, x_t, x_star):
  """Run the switching taylor rnn."""
  h_star = mlp(params['mlp'], h_approx_tm1)
  F_star = rnn(params['rnn'], h_star, x_star)

  # Taylor series expansion includes 0 order, so we subtract it off,
  # using the learned MLP point instead. This makes sense because we
  # expanded around (h*,x*), and if the MLP produces a fixed point (thanks
  # to the fixed point regularization pressure), it is equal to F(h*,x*).
  h_staylor_t = taylor_approx_rnn(rnn, params['rnn'], h_star, x_star,
                                  h_approx_tm1, x_t, order)
  h_approx_t = h_staylor_t - F_star + h_star
  o_approx_t = affine(params['out'], h_approx_t)

  h_t = rnn(params['rnn'], h_tm1, x_t)
  o_t = affine(params['out'], h_t)
  return h_star, F_star, h_t, h_approx_t, o_t, o_approx_t

def jslds_rnn_x_star_is_zeros(rnn, params, h_tm1, h_approx_tm1, x_t):
  """define JSLDS with x_star set to all zeros
    (this is the typical usage) """
  x_star = jnp.zeros_like(x_t)
  return staylor_rnn(rnn, params, 1, h_tm1, h_approx_tm1, x_t, x_star)


def jslds_rnn_x_star_context(rnn, params, h_tm1, h_approx_tm1, x_t):
  """define x_star to take static context into account 
  (for context integration) """
  x_star = jnp.zeros_like(x_t)
  x_star = index_update(x_star, index[2:4], x_t[2:4])
  return staylor_rnn(rnn, params, 1, h_tm1, h_approx_tm1, x_t, x_star)

def jslds_rnn_scan(rnn, jslds_rnn, params, state, x_t):
  """Run the JSLDS network 1 step adapting the inputs and outputs for scan."""
  h_tm1, h_approx_tm1 = state
  state_and_returns = jslds_rnn(rnn, params, h_tm1, h_approx_tm1, x_t)
  h_star, F_star, h_t, h_approx_t, o_t, o_approx_t = state_and_returns
  state = h_t, h_approx_t
  return state, state_and_returns


def jslds_rnn_run(params, x_t, rnn, jslds_rnn):
  """Run the RNN T steps, where T is shape[0] of input."""
  state0 = (params['rnn']['h0'], params['rnn']['h0'])
  # Convert signature to f: (state, x) -> (state, returns)
  this_jslds_rnn = functools.partial(jslds_rnn_scan, *(rnn, jslds_rnn, params))
  _, state_and_returns = jax.lax.scan(this_jslds_rnn, state0, x_t)
  return state_and_returns


def get_batch_rnn_run_fun(rnn, jslds_rnn):
  return jax.vmap(functools.partial(jslds_rnn_run, rnn=rnn, jslds_rnn=jslds_rnn),
                  in_axes=(None, 0))


def loss(params, inputs_bxtxu, targets_bxtxo, targets_mask_t, 
         out_nl_reg, out_jslds_reg, taylor_reg, fp_reg, l2_reg, rnn, jslds_rnn):
  """Compute the least squares loss of the output, plus L2 regularization."""
  batch_rnn_run = get_batch_rnn_run_fun(rnn, jslds_rnn)
  hstar_bxtxn, F0_bxtxn, h_bxtxn, h_approx_bxtxn, o_bxtxo, o_approx_bxtxo = \
      batch_rnn_run(params, inputs_bxtxu)

  l2_loss = l2_reg * optimizers.l2_norm(params)**2
  fp_loss = fp_reg * jnp.mean((F0_bxtxn - hstar_bxtxn)**2)

  fo_loss = taylor_reg * jnp.mean((h_bxtxn - h_approx_bxtxn)**2)

  o_bxsxo = o_bxtxo[:, targets_mask_t, :]
  o_approx_bxsxo = o_approx_bxtxo[:, targets_mask_t, :]
  targets_bxsxo = targets_bxtxo[:, targets_mask_t, :]

  lms_nl_loss = out_nl_reg * jnp.mean((o_bxsxo - targets_bxsxo)**2)
  lms_jslds_loss = out_jslds_reg * jnp.mean((o_approx_bxsxo - targets_bxsxo)**2)

  total_loss = lms_jslds_loss + lms_nl_loss + l2_loss + fp_loss + fo_loss
  return {'total': total_loss,
          'lms_jslds': lms_jslds_loss, 'lms_nl': lms_nl_loss,
          'l2': l2_loss, 'fixed_point': fp_loss, 'taylor': fo_loss}


loss_jit = jax.jit(loss, static_argnums=(9,10,))



def update_w_gc(i, opt_state, opt_update, get_params, x_bxt, f_bxt, f_mask_bxt,
                max_grad_norm, out_nl_reg, out_jslds_reg, taylor_reg,
                fp_reg, l2_reg, rnn, jslds_rnn):
  """Update the parameters w/ gradient clipped, gradient descent updates."""
  params = get_params(opt_state)

  def training_loss(params):
    return loss(params, x_bxt, f_bxt, f_mask_bxt, out_nl_reg, out_jslds_reg, 
                taylor_reg, fp_reg, l2_reg, rnn, jslds_rnn)['total']

  grads = jax.grad(training_loss)(params)

  clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
  return opt_update(i, clipped_grads, opt_state)

update_w_gc_jit = jax.jit(update_w_gc, static_argnums=(2, 3, 13, 14))


def run_trials(batch_run_fun, inputs_targets_h0s_fun, nbatches, batch_size, key):
  """Run a bunch of trials and save everything in a dictionary."""
  inputs = []
  h_stars = []
  F0_stars = []
  nl_hiddens = []
  nl_outputs = []
  hiddens = []
  outputs = []
  targets = []
  masks = []
  h0s = []
  for _ in range(nbatches):
    key, skey = jax.random.split(key)
    keys = jax.random.split(skey, batch_size)
    input_b, target_b, masks_b, h0s_b = inputs_targets_h0s_fun(keys)
    if h0s_b is None:
      h_star_b, F0_star_b, h_b, h_approx_b, o_b, o_approx_b = \
          batch_run_fun(input_b)
    else:
      h_star_b, F0_star_b, h_b, h_approx_b, o_b, o_approx_b = \
          batch_run_fun(input_b, h0s_b)
      h0s.append(h0s_b)

    inputs.append(input_b)
    h_stars.append(h_star_b)
    F0_stars.append(F0_star_b)
    nl_hiddens.append(h_b)
    nl_outputs.append(o_b)
    hiddens.append(h_approx_b)
    outputs.append(o_approx_b)
    targets.append(target_b)
    masks.append(masks_b)

  trial_dict = {'inputs': np.vstack(inputs),
                'h_stars': np.vstack(h_stars),
                'F0_stars': np.vstack(F0_stars),
                'nl_hiddens': np.vstack(nl_hiddens),
                'nl_outputs': np.vstack(nl_outputs),
                'hiddens': np.vstack(hiddens),
                'outputs': np.vstack(outputs),
                'targets': np.vstack(targets), 
                'masks': np.vstack(masks)}

  if h0s_b is not None:
    trial_dict['h0s'] = np.vstack(h0s)
  else:
    trial_dict['h0s'] = None
  return trial_dict


def run_trials_given_inputs(batch_run_fun, inputs, targets, nbatches, batch_size):
  """Run a bunch of trials and save everything in a dictionary."""
  inputs = []
  h_stars = []
  F0_stars = []
  nl_hiddens = []
  nl_outputs = []
  hiddens = []
  outputs = []
  masks = []
  h0s = []
  for _ in range(nbatches):
    input_b = inputs

    h_star_b, F0_star_b, h_b, h_approx_b, o_b, o_approx_b = \
          batch_run_fun(input_b)

    inputs.append(input_b)
    h_stars.append(h_star_b)
    F0_stars.append(F0_star_b)
    nl_hiddens.append(h_b)
    nl_outputs.append(o_b)
    hiddens.append(h_approx_b)
    outputs.append(o_approx_b)

  trial_dict = {'inputs': np.vstack(inputs),
                'h_stars': np.vstack(h_stars),
                'F0_stars': np.vstack(F0_stars),
                'nl_hiddens': np.vstack(nl_hiddens),
                'nl_outputs': np.vstack(nl_outputs),
                'hiddens': np.vstack(hiddens),
                'outputs': np.vstack(outputs)}

  return trial_dict
