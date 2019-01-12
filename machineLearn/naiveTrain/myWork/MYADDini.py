import numpy as np

OUT_SHAPE=(4,4)
CAND=16
map_table={2**i:i for i in range(1,CAND)}
map_table[0]=0

def grid_ohe(arr):
    ret=np.zeros(shape=OUT_SHAPE+(CAND,),dtype=bool)
    for r in range(OUT_SHAPE[0]):
        for c in range(OUT_SHAPE[1]):
            ret[r,c,map_table[arr[r,c]]]=1
    return ret
