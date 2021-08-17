import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np            


def plot_params(params, hps):
  """Plot the parameters of the vanilla RNN."""
  plt.figure(figsize=(16, 12))
  plt.subplot(331)
  plt.stem(params['out']['W'][0, :])
  plt.title('wO - output weights')

  plt.subplot(332)
  plt.stem(params['rnn']['h0'])
  plt.title('h0 - initial hidden state')

  rnn_fun = hps['rnn']['fun']
  rnn_fun_h = lambda h: rnn_fun(params['rnn'], h, np.zeros(hps['rnn']['u']))
  wR = jax.jacrev(rnn_fun_h)(params['rnn']['h0'])
  plt.subplot(333)
  plt.imshow(wR, interpolation=None)
  plt.title('dF/dh(h0)')
  plt.colorbar()

  plt.subplot(334)
  plt.title('wI - input weights')

  plt.subplot(335)
  
  plt.title('bR - recurrent biases')

  plt.subplot(336)
  evals, _ = np.linalg.eig(wR)
  x = np.linspace(-1, 1, 1000)
  plt.plot(x, np.sqrt(1-x**2), 'k')
  plt.plot(x, -np.sqrt(1-x**2), 'k')
  plt.plot(np.real(evals), np.imag(evals), '.')
  plt.axis('equal')
  plt.title('Eigenvalues of dF/dh(h0)')

  plt.subplot(337)
  plt.imshow(params['mlp'][0]['W'])
  plt.title('MLP W layer 0')
  plt.subplot(338)
  plt.imshow(params['mlp'][1]['W'])
  plt.title('MLP W layer 1')
  plt.subplot(339)
  plt.stem(params['mlp'][0]['b'])
  plt.stem(params['mlp'][1]['b'])
  plt.title('MLP W layer biases')


def plot_examples(ntimesteps, rnn_internals, key, nexamples=1, do_plot_nl=True,
                  do_plot_slds=True):
  """Plot some input/hidden/output triplets."""
  nplots = 4 if do_plot_slds else 3
  ridx = 0
  batch_size = rnn_internals['inputs'].shape[0]
  example_idxs = jax.random.randint(key, shape=(nexamples,),minval=0, maxval=batch_size)
  fig = plt.figure(figsize=(nexamples*5, 20))
  input_dim = rnn_internals['inputs'].shape[2]
  expand_val = 3 * np.std(rnn_internals['inputs'][:, :, 0])
  expander = expand_val * np.expand_dims(np.arange(input_dim), axis=0)
  for eidx, bidx in enumerate(example_idxs):
    plt.subplot(nplots, nexamples, ridx*nexamples + eidx + 1)
    plt.plot(rnn_internals['inputs'][bidx, :] + expander, 'k')
    plt.xlim([0, ntimesteps])
    plt.title('Example %d' % (bidx))
    if eidx == 0:
      plt.ylabel('Input')
  ridx += 1

  ntoplot = 10
  closeness = 0.25
  for eidx, bidx in enumerate(example_idxs):
    plt.subplot(nplots, nexamples, ridx * nexamples + eidx + 1)
    if do_plot_slds:
      plt.plot(rnn_internals['hiddens'][bidx, :, 0:ntoplot] +
               closeness * np.arange(ntoplot), 'b')
    if do_plot_nl:
      plt.plot(rnn_internals['nl_hiddens'][bidx, :, 0:ntoplot] +
               closeness * np.arange(ntoplot), 'r-.')

    plt.xlim([0, ntimesteps])
    if eidx == 0:
      plt.ylabel('JSLDS Hidden / NL Hidden')
  ridx += 1

  ntoplot = 10
  closeness = 0.25
  if do_plot_slds:
    for eidx, bidx in enumerate(example_idxs):
      plt.subplot(nplots, nexamples, ridx * nexamples + eidx + 1)
      plt.plot(rnn_internals['h_stars'][bidx, :, 0:ntoplot] +
               closeness * np.arange(ntoplot), 'b')
      plt.plot(rnn_internals['F0_stars'][bidx, :, 0:ntoplot] +
               closeness * np.arange(ntoplot), 'r-.')
      plt.xlim([0, ntimesteps])
      if eidx == 0:
        plt.ylabel('e*, F(e*, x*)')
    ridx += 1

  target_dim = rnn_internals['targets'].shape[2]
  expand_val = 3 * np.std(rnn_internals['targets'][:, :, 0])
  expander = expand_val * np.expand_dims(np.arange(target_dim), axis=0)
  for eidx, bidx in enumerate(example_idxs):
    plt.subplot(nplots, nexamples, ridx * nexamples + eidx + 1)
    if do_plot_slds:
      plt.plot(rnn_internals['outputs'][bidx, :, :] + expander, 'r')
    if do_plot_nl:
      plt.plot(rnn_internals['nl_outputs'][bidx, :, :] + expander, 'm-.')
    plt.plot(rnn_internals['targets'][bidx, :, :] + expander, 'k')
    plt.xlim([0, ntimesteps])
    plt.xlabel('Timesteps')
    if eidx == 0:
      plt.ylabel('Output / NL Output')
  ridx += 1

  plt.plot(0,0, 'r', label='JSLDS output')
  plt.plot(0,0, 'm-.', label='RNN output')
  plt.plot(0,0, 'k', label='Target')
  fig.legend(loc=[.91,.1])
