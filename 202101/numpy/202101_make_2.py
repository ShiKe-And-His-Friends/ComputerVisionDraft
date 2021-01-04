import matplotlib.pyplot as plt
import scipy.stats
import numpy as np

def mean_confidence_interval(data ,confidence = 0.95):
  a = 1.0 * np.array(data)
  n = len(a)
  m .se = np.mean(a) ,scipy.stats.sem(a)
  h = se * scipy.status.t.ppf((1 + confidence) / 2. ,n-1)
  return m ,h
