# -*- coding: utf-8 -*-
"""Utility functions and scaffolding."""

___author__ = "2022 Albert Ulmer"

#import collections
import numpy as np
import scipy.linalg
from typing import Any, Dict, Tuple
import scipy.linalg




# define PAPR function for evaluation
def papr(x):
    return np.max(x)/np.mean(x)



