# from numba import jit 
import numpy as np
import pandas as pd
from scipy.special import erfinv




class GaussRank():
    """
    CPU & GPU codes are mostly the same except for the imported libraries.
    GPU codes are executed automatically if the input tensor is on GPU.
    """
    def __init__(self,epsilon=0.001):
        self.epsilon = epsilon
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower

    def fit(self,x):
        self.fit_transform(x)
        return self    
        
    def fit_transform(self,x):
        # x is a 1D numpy/cupy array
        msg = 'input must be a 1D numpy/cucpy array'
        
        assert isinstance(x,np.ndarray) or isinstance(x,cp.ndarray),msg
        erfinv_ = erfinv if isinstance(x,np.ndarray) else cupy_erfinv
        DataFrame = pd.DataFrame if isinstance(x,np.ndarray) else gd.DataFrame
        
        j = x.argsort().argsort()

        assert (j.min() == 0).all()
        assert (j.max() == len(j) - 1).all()

        j_range = len(j) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv_(transformed)
        
        transformed_map = DataFrame()
        transformed_map['src'] = x
        transformed_map['tgt'] = transformed
        transformed_map = transformed_map.sort_values(by='src',ascending=True)
        self.transformed_map = transformed_map
        return transformed
    
    def transform(self,x):
        return self._transform(x,src_col='src',tgt_col='tgt')
    
    def inverse_transform(self,transformed):
        return self._transform(transformed,src_col='tgt',tgt_col='src')
    
    def _transform(self,x,src_col,tgt_col):
        msg = 'input must be a 1D numpy/cucpy array'
        assert isinstance(x,np.ndarray) or isinstance(x,cp.ndarray),msg
        transformed_map = self.transformed_map
        N = len(transformed_map)
        pos = transformed_map[src_col].searchsorted(x, side='left')
        
        pos[pos>=N] = N-1
        pos[pos-1<=0] = 0

        x1 = transformed_map[src_col].values[pos]
        x2 = transformed_map[src_col].values[pos-1]
        y1 = transformed_map[tgt_col].values[pos]
        y2 = transformed_map[tgt_col].values[pos-1]

        relative = (x-x2)  / (x1-x2)
        return (1-relative)*y2 + relative*y1
    
    
def rank_gauss(x):
    from scipy.special import erfinv
    N = x.shape[0]
    temp = x.argsort()
    rank_x = temp.argsort() / N
    rank_x -= rank_x.mean()
    rank_x *= 2
    efi_x = erfinv(rank_x)
    efi_x -= efi_x.mean()
    return efi_x
