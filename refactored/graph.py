import numpy as np
import pandas as pd
import seaborn
import pylab as plt

mri_raw = np.loadtxt("SNP_BdRNN_only_NEW_2_L.txt",delimiter=',')

#comb_raw = np.load("MULTIMODAL_05_06:14:32_dropout:_0.5.txt")

comb_raw = np.loadtxt("/export/mialab/users/nlewis/tf_multimodal/refactored/multimodal_output/MULTIMODAL_05_06:22:24_dropout:_0.5_SECONDHALF.txt",delimiter=',')
comb_raw = np.concatenate((comb_raw,np.loadtxt("/export/mialab/users/nlewis/tf_multimodal/refactored/multimodal_output/MULTIMODAL_06_06:13:51_dropout:_0.5_FIRSTHALF.txt",delimiter=',')),axis=0)
print comb_raw.shape
pal = seaborn.color_palette("hls",7)

SNP_bdRNN_raw = np.loadtxt("SNP_BdRNN_only_NEW_2_R.txt",delimiter=',')
print "raw shape: "+str(mri_raw.shape)
comb_median = np.median(comb_raw,axis=0)
comb_max_col = np.argmax(comb_median)

mri_median = np.median(mri_raw,axis=0)
mri_max_col = np.argmax(mri_median)

SNP_median = np.median(SNP_bdRNN_raw,axis=0)
snp_max_col = np.argmax(SNP_median)

mri_target = mri_raw[:,mri_max_col]
comb_target = comb_raw[:,comb_max_col]
SNP_target = SNP_bdRNN_raw[:,snp_max_col]
print comb_target.shape
FW = np.column_stack((comb_target,mri_target,SNP_target))
print FW.shape
s = pd.DataFrame(FW, columns=['combined','sMRI','SNP'])

#print s.shape
#np.savetxt("outs_0H_LEAVEONEOUT.txt",R)
#s["class"] = pd.Series(train_keys + valid_keys, index = s.index,dtype="category")
g = seaborn.boxplot(data=s)
g.set(xlabel='model',ylabel='accuracy')
#seaborn.plt.savefig("balanced_normalized_0H_LEAVEONEOUT.svg",bbox_inches="tight",pad_inches=0)
seaborn.plt.show()