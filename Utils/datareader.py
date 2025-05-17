import os
import csv
import codecs
import torch
import numpy as np
import scipy.io as scio

path = '../sess01/sess01_subj01_EEG_SSVEP.mat'
data = scio.loadmat(path)
print('ok')



