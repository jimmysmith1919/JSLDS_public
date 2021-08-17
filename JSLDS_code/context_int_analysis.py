import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np             # original CPU-backed NumPy
from sklearn.decomposition import PCA
import model_and_training as JSLDS
import integrator
import fixed_point_finding as num_fps
import seaborn as sns
import matplotlib.gridspec as gridspec

def run_trials_fixed_bias(key,eval_batch_size, motions,
                          colors, batch_rnn_run, params, ntimesteps, n, u):
  """Run JSLDS-RNN system on context-dependent integration trials, fixing the 
     bias levels """
  m_hiddens = np.zeros((36,eval_batch_size,ntimesteps, n))
  m_hstars = np.zeros((36,eval_batch_size,ntimesteps, n))
  m_inputs = np.zeros((36, eval_batch_size,ntimesteps, u))

  c_hiddens = np.zeros((36, eval_batch_size,ntimesteps, n))
  c_hstars = np.zeros((36,eval_batch_size,ntimesteps, n))
  c_inputs = np.zeros((36, eval_batch_size,ntimesteps, u))

  i = 0
  for m in motions:
    for c in colors:
      T = 1.0          # Arbitrary amount time, roughly physiological.
      ntimesteps = 25  # Divide T into this many bins
      motion= m
      color = c
      bval = np.array([motion, color])[None,...]
      sval = 0.025     # standard deviation (before dividing by sqrt(dt))

      #motion context
      context = True
      input_params = (bval, context, sval, T, ntimesteps)

      build_partial = lambda skeys: integrator.build_inputs_and_targets_fix_bias(input_params, skeys)
      build_inputs_and_targets = jax.jit(build_partial)

      def inputs_targets_no_h0s(keys):
          inputs_b, targets_b, masks_b = \
              build_inputs_and_targets(keys)
          h0s_b = None # Use trained h0
          return inputs_b, targets_b, masks_b, h0s_b

      rnn_run = lambda inputs: batch_rnn_run(params, inputs)
      key, skey = jax.random.split(key)
      rnn_internals = JSLDS.run_trials(rnn_run, inputs_targets_no_h0s, 1, 
                                       eval_batch_size, skey)

      m_hiddens[i] = rnn_internals['hiddens']
      m_hstars[i] = rnn_internals['h_stars']
      m_inputs[i] = rnn_internals['inputs']

      #color context
      context = False
      input_params = (bval, context, sval, T, ntimesteps)
      build_partial = lambda skeys: integrator.build_inputs_and_targets_fix_bias(input_params, skeys)
      build_inputs_and_targets = jax.jit(build_partial)

      def inputs_targets_no_h0s(keys):
          inputs_b, targets_b, masks_b = \
              build_inputs_and_targets(keys)
          h0s_b = None # Use trained h0
          return inputs_b, targets_b, masks_b, h0s_b

      rnn_run = lambda inputs: batch_rnn_run(params, inputs)
      key, skey = jax.random.split(key)
      rnn_internals = JSLDS.run_trials(rnn_run, inputs_targets_no_h0s, 1,
                                       eval_batch_size,skey)

      c_hiddens[i] = rnn_internals['hiddens']
      c_hstars[i] = rnn_internals['h_stars']
      c_inputs[i] = rnn_internals['inputs']
      
      i +=1
  return m_hiddens,m_hstars,m_inputs,c_hiddens,c_hstars,c_inputs


def get_reduce_inds_(key, batch_size, num_samples):
  return jax.random.choice(key, np.arange(batch_size), replace=False,shape=(num_samples,))

def get_reduce_inds(key, num_trials, batch_size, num_samples):
  """Helper function to sample trials for each bias level """
  keys = jax.random.split(key,num_trials)
  return jax.vmap(get_reduce_inds_, (0,None,None))(keys, batch_size, num_samples)

def get_sample_expansion_points(key, num_trials, num_samples, eval_batch_size,
                                hiddens, hstars, ntimesteps,n):
  """Get expansion points and states from sample trials. We use this
  to speed up the eigendecompositions required to construct the subspace below """
  key, skey = jax.random.split(key)
  reduce_inds = get_reduce_inds(skey, num_trials, eval_batch_size, num_samples)

  hiddens_approx_c_ = np.zeros((num_trials, num_samples, ntimesteps, n))
  hstars_c_ = np.zeros((num_trials, num_samples, ntimesteps, n))
  for i in range(num_trials):
    hiddens_approx_c_[i] = hiddens[i,reduce_inds[i]]
    hstars_c_[i] = hstars[i,reduce_inds[i]]

  hiddens_approx_c = hiddens_approx_c_.reshape(-1,n)
  hstars_c = hstars_c_.reshape(-1,n)
  return hiddens_approx_c, hstars_c

def get_subspace_fixed_bias(params,rnn_fun, hiddens_approx_c, hstars_c,
                            context, ntimesteps,offset=2):

  """Computes the orthgonalized subspace """
  if context == 'motion':
    x_star = np.array([0,0,1,0])
  elif context == 'color':
    x_star = np.array([0,0,0,1])
  
  rnn_fun_h = lambda h : rnn_fun(params['rnn'], h, x_star)
  hstar_jac = num_fps.compute_jacobians(rnn_fun_h, hstars_c)
  eig_decomps_c = num_fps.compute_eigenvalue_decomposition(hstar_jac, sort_by='real',
                                    do_compute_lefts=True)
  
  new_es_c = np.arange(offset,ntimesteps)
  for i in range(ntimesteps,hiddens_approx_c.shape[0],ntimesteps):
    new_es_c = np.concatenate((new_es_c, np.arange(i+offset, i+ntimesteps)))

  r1 = 0
  for i in new_es_c:
    r1 += eig_decomps_c[i]['R'][:,0]
  r1 = r1/len(new_es_c)

  b1 = params['rnn']['wI'][:,0]
  b2 = params['rnn']['wI'][:,1]
  c_axes = np.concatenate((r1[...,None], b1[...,None], b2[...,None]), axis=1)

  proj_mat_c,_= np.linalg.qr(c_axes)
  proj_hiddens_c = hiddens_approx_c[new_es_c] @ proj_mat_c
  proj_hstars_c = hstars_c[new_es_c] @ proj_mat_c

  return eig_decomps_c, new_es_c, proj_mat_c, proj_hiddens_c, proj_hstars_c



def get_subspace(params,rnn_fun,inputs, hiddens_approx, hstars, context, 
                 ntimesteps,offset=2):
  """Computes the orthogonalized subspace when the biases are not fixed """
  if context == 'motion':
    inds = np.argwhere(inputs[:,2]==1)[:,0]
    x_star = np.array([0,0,1,0])
  elif context == 'color':
    inds = np.argwhere(inputs[:,2]==0)[:,0]
    x_star = np.array([0,0,0,1])
  
  
  hiddens_approx_c = hiddens_approx[inds]
  hstars_c = hstars[inds]
  rnn_fun_h = lambda h : rnn_fun(params['rnn'], h, x_star)
  hstar_jac = num_fps.compute_jacobians(rnn_fun_h, hstars_c)
  eig_decomps_c = num_fps.compute_eigenvalue_decomposition(hstar_jac,
                                                           sort_by='real',
                                                        do_compute_lefts=True)
  
  new_es_c = np.arange(offset,ntimesteps)
  for i in range(ntimesteps,hiddens_approx_c.shape[0],ntimesteps):
    new_es_c = np.concatenate((new_es_c, np.arange(i+offset, i+ntimesteps)))

  r1 = 0
  for i in new_es_c:
    r1 += eig_decomps_c[i]['R'][:,0]
  r1 = r1/len(new_es_c)

  b1 = params['rnn']['wI'][:,0]
  b2 = params['rnn']['wI'][:,1]
  c_axes = np.concatenate((r1[...,None], b1[...,None], b2[...,None]), axis=1)

  proj_mat_c,_= np.linalg.qr(c_axes)
  proj_hiddens_c = hiddens_approx_c[new_es_c] @ proj_mat_c
  proj_hstars_c = hstars_c[new_es_c] @ proj_mat_c

  return eig_decomps_c, new_es_c, proj_mat_c, proj_hiddens_c, proj_hstars_c


def selection_vector_plot(offset1, offset2, new_es_c1, new_es_c2,
                          proj_mat_c1, proj_mat_c2,
                          proj_hstars_c1, proj_hstars_c2,
                          eig_decomps_c1,eig_decomps_c2, 
                          scale1, scale2,
                          alpha1, alpha2,ntimesteps,
                          num_trials=14):
  """Produce the selection vector plot"""
  fs=12
  fig, ax = plt.subplots(1, 1, figsize=(8,8))
  for i in range(num_trials):
    add1 = ntimesteps-offset1
    erange = np.arange(i*add1,i*add1+add1)
    ax.scatter(proj_hstars_c1[erange,1], proj_hstars_c1[erange,2], 
                color='red', s=10)
    for j in range(len(erange)): 
        ex = new_es_c1[j]
        l0 = eig_decomps_c1[ex]['L'][:,0]
        a = l0 @ proj_mat_c1
        x = proj_hstars_c1[erange,1][j].real
        y = proj_hstars_c1[erange,2][j].real
        xplusdx = proj_hstars_c1[erange,1][j].real+scale1*a[1].real
        yplusdy = proj_hstars_c1[erange,2][j].real+scale1*a[2].real
        xmindx = proj_hstars_c1[erange,1][j].real-scale1*a[1].real
        ymindy = proj_hstars_c1[erange,2][j].real-scale1*a[2].real
        
        ax.arrow(x,y, xplusdx-x, yplusdy-y, width = 1e-15,
                  **dict(linestyle='--', color='green', alpha=alpha1,
                         head_width=0.0, head_length=0.0))
        ax.arrow(x,y, xmindx-x, ymindy-y, width = 1e-15,
              **dict(linestyle='--', color='green', alpha=alpha1, 
                     head_width=0.0, head_length=0.0))


    add2 = ntimesteps-offset2
    erange = np.arange(i*add2,i*add2+add2)
    ax.scatter(proj_hstars_c2[erange,1], proj_hstars_c2[erange,2], 
                color='orange', s=10)
    for j in range(len(erange)): 
        ex = new_es_c2[j]
        l0 = eig_decomps_c2[ex]['L'][:,0]
        a = l0 @ proj_mat_c2
        x = proj_hstars_c2[erange,1][j].real
        y = proj_hstars_c2[erange,2][j].real
        xplusdx = proj_hstars_c2[erange,1][j].real+scale2*a[1].real
        yplusdy = proj_hstars_c2[erange,2][j].real+scale2*a[2].real
        xmindx = proj_hstars_c2[erange,1][j].real-scale2*a[1].real
        ymindy = proj_hstars_c2[erange,2][j].real-scale2*a[2].real
        
        ax.arrow(x,y, xplusdx-x, yplusdy-y, width = 1e-15,
                  **dict(linestyle='--', color='green', alpha = alpha2,
                         head_width=0.0, head_length=0.0))
        ax.arrow(x,y, xmindx-x, ymindy-y, width = 1e-15,
              **dict(linestyle='--', color='green', alpha=alpha2,
                     head_width=0.0, head_length=0.0))
        

  ax.scatter(proj_hstars_c1[erange,1], proj_hstars_c1[erange,2], 
                  color='red', s=10, label = 'Motion Context Expansion points')
  ax.scatter(proj_hstars_c2[0,1], proj_hstars_c2[0,2], 
                  color='orange', s=10, label='Color Context Expansion points')
  ax.plot(proj_hstars_c2[0,1],proj_hstars_c2[0,2], '-', color='green', 
          label='selection vector')



  ax.set_xlabel('motion')
  ax.set_ylabel('color')
  ax.axes.xaxis.set_ticks([])
  ax.axes.yaxis.set_ticks([])
  ax.legend(frameon=False, fontsize=fs)


def plot_subspace_projection(proj_hstars_c1, proj_hstars_c2, 
                             m_project, c_project, offset1, offset2):
  """Produce the subspace projection plot """
  sns.set_style("white")
  sns.set_context("paper")


  fig = plt.figure(figsize=(14, 14))
  gs = fig.add_gridspec(3,3)
  fs=11

  xstart = -0.01
  ystart = 1.01

  ##########################################################################
  #Plot D
  ax0 = fig.add_subplot(gs[0,0])


  m_hstars_fig = proj_hstars_c1.real

  m_project_mean = np.zeros((6, 25, 3))
  for i in range(6):
    m_project_mean[i] = np.mean(m_project[i*6:i*6+6], axis=0)

  ax0.scatter(m_hstars_fig[:,0], m_hstars_fig[:,1], 
              color='red', s=10, label='JSLDS Expansion points')

  sn1 = 0
  ax0.plot(m_project_mean[sn1,offset1:,0], m_project_mean[sn1,offset1:,1], 
              '-o',color='darkgreen',markeredgecolor='k',
            label='Strong negative Motion')
  mn1= 1
  ax0.plot(m_project_mean[mn1,offset1:,0], m_project_mean[mn1,offset1:,1], 
              '-o',color='mediumseagreen', markeredgecolor='k',
          label='Medium Negative Motion')
  wn1 = 2
  ax0.plot(m_project_mean[wn1,offset1:,0], m_project_mean[wn1,offset1:,1], 
              '-o',color='mediumspringgreen',markeredgecolor='k', 
          label='Weak Negative Motion')
  sp1 = 5
  ax0.plot(m_project_mean[sp1,offset1:,0], m_project_mean[sp1,offset1:,1], 
              'o-',color='black', label='Strong positive Motion',
            markeredgecolor='k')
  mp1 = 4
  ax0.plot(m_project_mean[mp1,offset1:,0], m_project_mean[mp1,offset1:,1], 
              'o-',color='grey', markeredgecolor='k',
            label='Medium positive Motion')
  wp1 = 3
  ax0.plot(m_project_mean[wp1,offset1:,0], m_project_mean[wp1,offset1:,1], 
              '-o',color='black',mfc = 'white',markeredgecolor='k', 
          label='Weak positive Motion')

  # ax0.legend(frameon=False, fontsize=fs)
  # ax0.axes.xaxis.set_ticks([])
  # ax0.axes.yaxis.set_ticks([])
  ax0.set_xlabel('Choice',labelpad=0.1,fontsize=fs)
  ax0.set_ylabel('Motion',labelpad=0.1, fontsize=fs)
  ax0.spines['right'].set_visible(False)
  ax0.spines['top'].set_visible(False)
  bbox = ax0.get_tightbbox(fig.canvas.get_renderer())
  ax0.text(xstart, ystart,  'D', transform=ax0.transAxes, 
              size=20,weight='bold')

  #########################################################################
  #plot E
  ax1 = fig.add_subplot(gs[0,1])

  ax1.scatter(m_hstars_fig[:,0], m_hstars_fig[:,2], 
              color='red', s=10, label='JSLDS Expansion points')

  ax1.plot(m_project_mean[sn1,offset1:,0], m_project_mean[sn1,offset1:,2], 
              '-o',color='darkgreen',markeredgecolor='k', 
           label='Strong negative Motion')

  ax1.plot(m_project_mean[mn1,offset1:,0], m_project_mean[mn1,offset1:,2], 
              '-o',color='mediumseagreen', markeredgecolor='k',
           label='Medium Negative Motion')

  ax1.plot(m_project_mean[wn1,offset1:,0], m_project_mean[wn1,offset1:,2], 
              '-o',color='mediumspringgreen',markeredgecolor='k', 
           label='Weak Negative Motion')
  ax1.plot(m_project_mean[sp1,offset1:,0], m_project_mean[sp1,offset1:,2], 
              'o-',color='black', label='Strong positive Motion',
            markeredgecolor='k')
  ax1.plot(m_project_mean[mp1,offset1:,0], m_project_mean[mp1,offset1:,2], 
              'o-',color='grey', markeredgecolor='k', 
           label='Medium positive Motion')
  ax1.plot(m_project_mean[wp1,offset1:,0], m_project_mean[wp1,offset1:,2], 
              '-o',color='black',mfc = 'white',markeredgecolor='k', 
           label='Weak positive Motion')
  # ax1.axes.xaxis.set_ticks([])
  # ax1.axes.yaxis.set_ticks([])
  ax1.set_xlabel('Choice',labelpad=0.1,fontsize=fs)
  ax1.set_ylabel('Color',labelpad=0.1, fontsize=fs)
  ax1.spines['right'].set_visible(False)
  ax1.spines['top'].set_visible(False)
  bbox = ax1.get_tightbbox(fig.canvas.get_renderer())
  ax1.text(xstart, ystart,  'E', transform=ax1.transAxes, 
              size=20,weight='bold')

  ##############################################################################
  #Plot F

  ax2 = fig.add_subplot(gs[0,2])

  m_project_c_mean_mn = np.zeros((6,25,3))
  for i in range(6):
    inds = [i, i+6, i+12]
    m_project_c_mean_mn[i] =  np.mean(m_project[inds], axis=0)

  m_project_c_mean_mp = np.zeros((6,25,3))
  for i in range(6):
    inds = [18+i, 18+i+6, 18+i+12]
    m_project_c_mean_mp[i] =  np.mean(m_project[inds], axis=0)

  ax2.scatter(m_hstars_fig[:,0], m_hstars_fig[:,2], 
              color='red', s=10, label='JSLDS Expansion points')
  ax2.plot(m_project_c_mean_mn[0,offset1:,0], m_project_c_mean_mn[0,offset1:,2], 
              '-o',color='blue', label='Strong negative Color')
  ax2.plot(m_project_c_mean_mp[0,offset1:,0], m_project_c_mean_mp[0,offset1:,2], 
              'o-',color='blue')
  ax2.plot(m_project_c_mean_mn[0,offset1:,0], m_project_c_mean_mn[1,offset1:,2], 
              '-o',color='royalblue', label='Medium negative Color')
  ax2.plot(m_project_c_mean_mp[0,offset1:,0], m_project_c_mean_mp[1,offset1:,2], 
              '-o',color='royalblue')
  ax2.plot(m_project_c_mean_mn[2,offset1:,0], m_project_c_mean_mn[2,offset1:,2], 
              '-o',color='lightskyblue', label='Weak Negative Color')
  ax2.plot(m_project_c_mean_mp[2,offset1:,0], m_project_c_mean_mp[2,offset1:,2], 
              '-o',color='lightskyblue')
  ax2.plot(m_project_c_mean_mn[3,offset1:,0], m_project_c_mean_mn[3,offset1:,2], 
              '-o',color='thistle', label='Weak Positive Color')
  ax2.plot(m_project_c_mean_mp[3,offset1:,0], m_project_c_mean_mp[3,offset1:,2], 
              '-o',color='thistle')
  ax2.plot(m_project_c_mean_mn[4,offset1:,0], m_project_c_mean_mn[4,offset1:,2], 
              '-o',color='darkviolet', label='Medium Positve Color')
  ax2.plot(m_project_c_mean_mp[4,offset1:,0], m_project_c_mean_mp[4,offset1:,2], 
              '-o',color='darkviolet')
  ax2.plot(m_project_c_mean_mn[5,offset1:,0], m_project_c_mean_mn[5,offset1:,2], 
              '-o',color='indigo', label='Strong Positive Color')
  ax2.plot(m_project_c_mean_mp[5,offset1:,0], m_project_c_mean_mp[5,offset1:,2], 
              '-o',color='indigo')
  # ax2.axes.xaxis.set_ticks([])
  # ax2.axes.yaxis.set_ticks([])
  ax2.set_xlabel('Choice',labelpad=0.1,fontsize=fs)
  ax2.set_ylabel('Color',labelpad=0.1, fontsize=fs)
  ax2.spines['right'].set_visible(False)
  ax2.spines['top'].set_visible(False)

  bbox = ax2.get_tightbbox(fig.canvas.get_renderer())
  ax2.text(xstart, ystart,  'F', transform=ax2.transAxes, 
              size=20,weight='bold')

  ##############################################################################
  #Plot I
  ax3 = fig.add_subplot(gs[1,2])

  c_project_mean = np.zeros((6, 25, 3))
  c_hstars_fig = proj_hstars_c2.real
  for i in range(6):
    inds = [i, i+6, i+12, i+18, i+24, i+30]
    c_project_mean[i] = np.mean(c_project[inds], axis=0)

  ax3.scatter(c_hstars_fig[:,0], c_hstars_fig[:,2], 
              color='orange', s=10, label='JSLDS Expansion points')
  ax3.plot(c_project_mean[0,offset1:,0], c_project_mean[0,offset1:,2], 
              '-o',color='blue', label='Strong negative Color')
  ax3.plot(c_project_mean[1,offset1:,0], c_project_mean[1,offset1:,2], 
              '-o',color='royalblue', label='Medium Negative Color')
  ax3.plot(c_project_mean[2,offset1:,0], c_project_mean[2,offset1:,2], 
              '-o',color='lightskyblue', label='Weak Negative Color')
  ax3.plot(c_project_mean[5,offset1:,0], c_project_mean[5,offset1:,2], 
              '-o',color='indigo', label='Strong Positive Color')
  ax3.plot(c_project_mean[4,offset1:,0], c_project_mean[4,offset1:,2], 
              '-o',color='darkviolet', label='Medium Positive Color')
  ax3.plot(c_project_mean[3,offset1:,0], c_project_mean[3,offset1:,2], 
              '-o',color='thistle', label='Weak Positive Color')
  ax3.set_xlabel('Choice',labelpad=0.1,fontsize=fs)
  ax3.set_ylabel('Color',labelpad=0.1,fontsize=fs)
  # ax3.legend(frameon=False, fontsize=fs)
  ax3.spines['right'].set_visible(False)
  ax3.spines['top'].set_visible(False)

  bbox = ax3.get_tightbbox(fig.canvas.get_renderer())
  ax3.text(xstart, ystart,  'I', transform=ax3.transAxes, 
              size=20,weight='bold')

  ##############################################################################
  #Plot H
  ax4 = fig.add_subplot(gs[1,1])

  ax4.scatter(c_hstars_fig[:,0], c_hstars_fig[:,1], 
              color='orange', s=10, label='JSLDS Expansion points')
  ax4.plot(c_project_mean[0,offset1:,0], c_project_mean[0,offset1:,1], 
              '-o',color='blue', label='Strong negative Color')
  ax4.plot(c_project_mean[1,offset1:,0], c_project_mean[1,offset1:,1], 
              '-o',color='royalblue', label='Medium Negative Color')
  ax4.plot(c_project_mean[2,offset1:,0], c_project_mean[2,offset1:,1], 
              '-o',color='lightskyblue', label='Weak Negative Color')
  ax4.plot(c_project_mean[5,offset1:,0], c_project_mean[5,offset1:,1], 
              '-o',color='indigo', label='Strong Positive Color')
  ax4.plot(c_project_mean[4,offset1:,0], c_project_mean[4,offset1:,1], 
              '-o',color='darkviolet', label='Medium Positive Color')
  ax4.plot(c_project_mean[3,offset1:,0], c_project_mean[3,offset1:,1], 
              '-o',color='thistle', label='Weak Positive Color')
  ax4.set_xlabel('Choice',labelpad=0.1,fontsize=fs)
  ax4.set_ylabel('Motion',labelpad=0.1,fontsize=fs)

  # ax4.axes.xaxis.set_ticks([])
  # ax4.axes.yaxis.set_ticks([])
  ax4.spines['right'].set_visible(False)
  ax4.spines['top'].set_visible(False)

  bbox = ax4.get_tightbbox(fig.canvas.get_renderer())
  ax4.text(xstart, ystart,  'H', transform=ax4.transAxes, 
              size=20,weight='bold')
  ###########################################################################
  #Plot G

  ax5 = fig.add_subplot(gs[1,0])

  c_project_m_mean_cn = np.zeros((6,25,3))
  for i in range(6):
    c_project_m_mean_cn[i] = np.mean(c_project[i*6:i*6+3],axis=0)

  c_project_m_mean_cp = np.zeros((6,25,3))
  for i in range(6):
    c_project_m_mean_cp[i] = np.mean(c_project[i*6+3:i*6+6],axis=0)



  ax5.scatter(c_hstars_fig[:,0], c_hstars_fig[:,1], 
              color='orange', s=10, label='JSLDS Expansion points')

  ax5.plot(c_project_m_mean_cn[0,offset1:,0], c_project_m_mean_cn[0,offset1:,1], 
              '-o',color='darkgreen',markeredgecolor='k', 
           label='Strong negative Motion')
  ax5.plot(c_project_m_mean_cp[0,offset1:,0], c_project_m_mean_cp[0,offset1:,1], 
              '-o',color='darkgreen',markeredgecolor='k' )
  ax5.plot(c_project_m_mean_cn[1,offset1:,0], c_project_m_mean_cn[1,offset1:,1], 
              '-o',color='mediumseagreen', markeredgecolor='k',
           label='Medium Negative Motion')
  ax5.plot(c_project_m_mean_cp[1,offset1:,0], c_project_m_mean_cp[1,offset1:,1], 
              '-o',color='mediumseagreen', markeredgecolor='k')
  ax5.plot(c_project_m_mean_cn[2,offset1:,0], c_project_m_mean_cn[2,offset1:,1], 
              '-o',color='mediumspringgreen',markeredgecolor='k', 
           label='Weak Negative Motion')
  ax5.plot(c_project_m_mean_cp[2,offset1:,0], c_project_m_mean_cp[2,offset1:,1], 
              '-o',color='mediumspringgreen',markeredgecolor='k')
  ax5.plot(c_project_m_mean_cn[3,offset1:,0], c_project_m_mean_cn[3,offset1:,1], 
              '-o',color='k',mfc = 'white',markeredgecolor='k', 
           label='Weak positive Motion')
  ax5.plot(c_project_m_mean_cp[3,offset1:,0],c_project_m_mean_cp[3,offset1:,1], 
              '-o',color='k',mfc = 'white',markeredgecolor='k')
  ax5.plot(c_project_m_mean_cn[4,offset1:,0], c_project_m_mean_cn[4,offset1:,1], 
              'o-',color='grey', markeredgecolor='k',
            label='Medium positive Motion')
  ax5.plot(c_project_m_mean_cp[4,offset1:,0], c_project_m_mean_cp[4,offset1:,1], 
              'o-',color='grey', markeredgecolor='k')
  ax5.plot(c_project_m_mean_cn[5,offset1:,0], c_project_m_mean_cn[5,offset1:,1], 
              'o-',color='black',  label='Strong positive Motion',
            markeredgecolor='k')
  ax5.plot(c_project_m_mean_cp[5,offset1:,0], c_project_m_mean_cp[5,offset1:,1], 
              'o-',color='black',  markeredgecolor='k')
  ax5.set_xlabel('Choice',labelpad=0.1,fontsize=fs)
  ax5.set_ylabel('Motion',labelpad=0.1,fontsize=fs)
  # ax5.axes.xaxis.set_ticks([])
  # ax5.axes.yaxis.set_ticks([])
  ax5.spines['right'].set_visible(False)
  ax5.spines['top'].set_visible(False)
  bbox = ax5.get_tightbbox(fig.canvas.get_renderer())
  ax5.text(xstart, ystart,  'G', transform=ax5.transAxes, 
              size=20,weight='bold')
  # #########################################################################
  #motion input labels
  ms=10
  ax6 = fig.add_subplot(gs[2,0])
  ax6.plot([],[],' ', label='Motion Inputs:')
  ax6.plot([],[],
              '-o',color='black', label='Strong Choice 1', markersize=ms)
  ax6.plot([],[],
              '-o',color='grey', markeredgecolor='k', label='Medium Choice 1',
          markersize=ms)
  ax6.plot([],[],
              '-o',color='black',mfc = 'white',markeredgecolor='k',
          label='Weak Choice 1',markersize=ms)
  ax6.plot([],[],
              '-o',color='mediumspringgreen',markeredgecolor='k', 
          label='Weak Choice 2',markersize=ms)
  ax6.plot([],[],
              '-o',color='mediumseagreen', markeredgecolor='k', 
          label='Medium Choice 2',markersize=ms)
  ax6.plot([],[],
              '-o',color='darkgreen',markeredgecolor='k',
           label='Strong Choice 2',
          markersize=ms)

  ax6.legend(loc='upper center',frameon=False, fontsize=fs)
  ax6.axis('off')

  ############################################################
  #color inputs
  ax7 = fig.add_subplot(gs[2,2])


  ax7.plot([],[],' ', label='Color Inputs:')
  ax7.plot([],[],
              '-o',color='blue', label='Strong Choice 1',markersize=ms,
           markeredgecolor='k')
  ax7.plot([],[],
              '-o',color='royalblue', label='Medium Choice 1',
           markeredgecolor='k',markersize=ms)
  ax7.plot([],[],
              '-o',color='lightskyblue', label='Weak Choice 1',
          markersize=ms,markeredgecolor='k')
  ax7.plot([],[],
              '-o',color='thistle', label='Weak Choice 2',
           markeredgecolor='k',markersize=ms)
  ax7.plot([],[],
              '-o',color='darkviolet', label='Medium Choice 2',markersize=ms,
          markeredgecolor='k')
  ax7.plot([],[],
              '-o',color='indigo', label='Strong Choice 2',
           markeredgecolor='k',markersize=ms)
  ax7.legend(loc='upper center',frameon=False, fontsize=fs)
  ax7.axis('off')

  plt.show()
