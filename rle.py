import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from itertools import chain;

def run_length(label):
    x = label.transpose().flatten();
    y = np.where(x>0.5)[0];
    if len(y)<10:# consider as empty
        return [];
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    leng = end - start;
    res = [[s+1,l+1] for s,l in zip(list(start),list(leng))]
    res = list(chain.from_iterable(res));
    return res;
    
mask = cv2.imread('./raw/train/1_1_mask.tif',cv2.IMREAD_GRAYSCALE)
mask_rle = run_length(mask);
print(mask_rle[:200])
