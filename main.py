from scipy.io import loadmat
import random
import jax.numpy as jnp
from sklearn.utils import resample
from jax import grad, jit, vmap
from jax import random
from gcmi import copnorm, ent_g
import itertools
import pandas as pd
import timeit
import numpy as np
from stats import fdr_correction
from oinfo import exhaustive_loop_zerolag

###########################################################################
# file = 'Ex1_syn'
file = 'Ex1_red'
###########################################################################


path = './data/%s'
mat = loadmat(path % ('%s.mat' % file))
ts = mat['data'][:, :].T



starting_time = timeit.default_timer()

df = exhaustive_loop_zerolag(ts, maxsize=4, n_jobs=-1, n_boots=0,
                                 n_best=20, groups=4)

print("Time difference :", timeit.default_timer() - starting_time)

