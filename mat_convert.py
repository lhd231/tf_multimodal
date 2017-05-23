import scipy.io as io
import numpy as np

M = io.loadmat('/home/lhd/upenn/UPENN_phen.mat')
print M.keys()
R = M['up_temp']

np.savetxt('/home/lhd/tensorflow/For_Tensorflow_Multimodal/snps/snp_text/labels_UPENN.txt',R,delimiter=',')