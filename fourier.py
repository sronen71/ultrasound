import cv2
import numpy as np
from matplotlib import pyplot as plt
from  elliptic_fourier_descriptors import elliptic_fourier_descriptors as efd
from  elliptic_fourier_descriptors import reconstruct


bin_im = cv2.imread('./raw/train/10_100_mask.tif',cv2.IMREAD_GRAYSCALE);
#plt.imshow(bin_im)
#plt.show()
degree = 3
efds,K,T = efd(bin_im,degree)
print K,T
efds1 = efds[0]
print efds1

numpoints = 200
recon = reconstruct(efds1,numpoints)
recon = recon.astype(np.int32)
contours, hierarchy = cv2.findContours(bin_im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

im_color = cv2.applyColorMap(bin_im, cv2.COLORMAP_JET)
cv2.fillPoly(im_color,[recon],(255,0,0))
cv2.drawContours(im_color,contours,-1,(0,255,0),3)
plt.imshow(im_color)
plt.show()
