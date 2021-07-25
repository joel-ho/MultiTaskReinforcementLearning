import numpy as np

def poisson(lam, minimum=1, maximum=8):
  # Poisson distr
  p = lam**np.arange(minimum, maximum+1)*np.exp(-lam)
  fac = np.math.factorial(minimum-1)
  for i, k in enumerate(range(minimum, maximum+1)):
    fac *= k
    p[i] /= fac
  delta = 1-np.sum(p)
  p += delta/(maximum+1-minimum)
  return minimum, maximum, p
  
def get_n_replay(lam):
  minimum, maximum, p = poisson(lam)
  return np.random.choice(np.arange(minimum, maximum+1), 1, p=p)[0]