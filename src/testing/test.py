import numpy as np

from src.synthdid import *
from src.solver import *
from src.utils import *

N = 30
N0 = 1
T = 25
T0 = 12

N1 = N - N0
T1 = T - T0

beta = np.atleast_2d(np.array([0.3, 0, 0.7])).T
omega = np.random.rand(N1, 1)

omega = sparsify_function(omega)

