from scipy.io import loadmat
import jax.numpy as jnp
from sklearn.utils import resample
from jax import jit, vmap
from gcmi import copnorm
import itertools
import pandas as pd
import numpy as np
import math 
import scipy


def construct_batches(n,k,batch_size):
    combinations_slices = []
    # Calculate number of batches
    n_batches = math.ceil(scipy.special.comb(n,k,exact=True)/batch_size)

    # Construct iterator for combinations
    combinations = itertools.combinations(range(n),k)

    while len(combinations_slices) < n_batches:
        combinations_slices.append(itertools.islice(combinations,batch_size))

    return combinations_slices

@jit
def vmap_batched_o_info(x, comb):
    return vmap(_o_info)(x = x, comb = comb)

#@jit 
def jax_ent_g(x):
    """Jax implementation of the entropy of a Gaussian variable in bits.
    """
    nvarx, ntrl = x.shape

    # covariance
    c = jnp.dot(x, x.T) / float(ntrl - 1)
    chc = jnp.linalg.cholesky(c)

    # entropy in nats
    hx = jnp.sum(jnp.log(jnp.diag(chc))) + 0.5 * nvarx * (
        jnp.log(2 * jnp.pi) + 1.0)
    return hx

def _o_info(x, comb, return_comb=True):
    nvars, _ = x.shape

    # (n - 2) * H(X^n)
    o = (nvars - 2) * jax_ent_g(x)

    for j in range(nvars):
        # sum_{j=1...n}( H(X_{j}) - H(X_{-j}^n) )
        o += jax_ent_g(x[[j], :]) - jax_ent_g(jnp.delete(x, j, axis=0))

    if return_comb:
        return o, comb
    else:
        return o


def combinations(n, k, groups=None):
    assert isinstance(n, int)
    if isinstance(k, int): k = [k]
    assert isinstance(k, (list, tuple, jnp.ndarray))

    iterable = jnp.arange(n)

   
    combs = []
    for i in k:
        for test in itertools.combinations(iterable, i):
            print(test)
        exit(0)
        combs += [itertools.combinations(iterable, i) for i in k]
    comb = itertools.chain(*tuple(combs))

    for i in comb:
        if isinstance(groups, (list, tuple)):
            if all([k in i for k in groups]):
                yield i
        else:
            yield i

def exhaustive_loop_zerolag(ts, maxsize=5, n_best=10, groups=None, n_jobs=-1,
                            n_boots=None, alpha=0.05, batch_size = 10000):
    """Simple implementation of the Oinfo.

    Parameters
    ----------
    ts : array_like
        Time-series of shape (n_variables, n_samples) (e.g (n_roi, n_trials))
    """
    # copnorm and demean the data
    
    x = copnorm(ts)
    x = (x - x.mean(axis=1)[:, jnp.newaxis]).astype(jnp.float32)
    nvars, nsamp = x.shape

    # get the maximum size of the multiplets investigated
    if not isinstance(maxsize, int):
        maxsize = nvars
    maxsize = max(1, maxsize)


    oinfo, combs = [], []
    combinations_slices = construct_batches(nvars,maxsize,batch_size)
    for s in combinations_slices:
        #print(list(s))
        all_comb = jnp.array(list(s))
        total_x = jnp.array([x[comb, :] for comb in all_comb])
        outs = vmap_batched_o_info(total_x, all_comb)
        slice_oinfo, slice_combs = outs
        oinfo.extend(slice_oinfo)
        combs.extend(slice_combs)
      
    # dataframe conversion
    df = pd.DataFrame({
        'Combination': [tuple(x) for x in combs],
        'Oinfo': oinfo,
        'Size': [len(c) for c in combs]
    })
    df.sort_values('Oinfo', inplace=True, ascending=False)
    
    # n_best selection
    if isinstance(n_best, int):
        # redundancy selection
        red_ind = np.zeros((len(df),), dtype=bool)
        red_ind[0:n_best] = True
        red_ind = np.logical_and(red_ind, df['Oinfo'] > 0)
        # synergy selection
        syn_ind = np.zeros((len(df),), dtype=bool)
        syn_ind[-n_best::] = True
        syn_ind = np.logical_and(syn_ind, df['Oinfo'] < 0)
        # merge both
        redsyn_ind = np.logical_or(red_ind, syn_ind)
        df = df.loc[redsyn_ind]

    return df


#TODO: bootstrapping in jax
def bootci(o, x, comb, alpha, n_boots, rnd=0, n_jobs=-1):
    # bootstrap computations
    _, nsamps = x.shape
    #oboot = Parallel(n_jobs=n_jobs)(delayed(_o_info)(
    #        resample(x.T, n_samples=nsamps, random_state=rnd + i).T, comb,
    #        return_comb=False) for i in range(n_boots))
    #oboot = np.asarray(oboot)


    total_x = jnp.array([resample(x.T, n_samples=nsamps, random_state=rnd + i).T for i in range(n_boots)])
    total_comb = jnp.array([comb for i in range(n_boots)])

    oboot = vmap_batched_o_info(total_x, total_comb, return_comb = False)
    # confidence interval
    lower = np.percentile(oboot, (alpha / 2.) * 100.)
    upper = np.percentile(oboot, (1 - (alpha / 2.)) * 100.)

    # p-value inference
    indices = oboot < 0 if o > 0 else oboot > 0
    pv = (1 + np.sum(indices)) / (n_boots + 1)

    return pv, lower, upper
 
