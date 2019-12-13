import scipy.ndimage

sigma_fac = 0.001
sigma_base = 4

eps = 0.00000001

def get_count(X_all):
    features_count = np.zeros((X_all.shape[0], len(features)))
    features_density = np.zeros((X_all.shape[0], len(features)))
    features_deviation = np.zeros((X_all.shape[0], len(features)))

    sigmas = []

    for i,var in enumerate(tqdm(features)):
        X_all_var_int = (X_all[var].values * 10000).round().astype(int)
        X_fake_var_int = (X_fake[var].values * 10000).round().astype(int)
        lo = X_all_var_int.min()
        X_all_var_int -= lo
        X_fake_var_int -= lo
        hi = X_all_var_int.max()+1
        counts_all = np.bincount(X_all_var_int, minlength=hi).astype(float)
        zeros = (counts_all == 0).astype(int)
        before_zeros = np.concatenate([zeros[1:],[0]])
        indices_all = np.arange(counts_all.shape[0])
        # Geometric mean of twice sigma_base and a sigma_scaled which is scaled to the length of array 
        sigma_scaled = counts_all.shape[0]*sigma_fac
        sigma = np.power(sigma_base * sigma_base * sigma_scaled, 1/3)
        sigmas.append(sigma)
        counts_all_smooth = scipy.ndimage.filters.gaussian_filter1d(counts_all, sigma)
        deviation = counts_all / (counts_all_smooth+eps)
        indices = X_all_var_int
        features_count[:,i] = counts_all[indices]
        features_density[:,i] = counts_all_smooth[indices]
        features_deviation[:,i] = deviation[indices]

        
    features_count_names = [var+'_count' for var in features]
    features_density_names = [var+'_density' for var in features]
    features_deviation_names = [var+'_deviation' for var in features]

    X_all_count = pd.DataFrame(columns=features_count_names, data = features_count)
    X_all_count.index = X_all.index
    X_all_density = pd.DataFrame(columns=features_density_names, data = features_density)
    X_all_density.index = X_all.index
    X_all_deviation = pd.DataFrame(columns=features_deviation_names, data = features_deviation)
    X_all_deviation.index = X_all.index
    X_all = pd.concat([X_all,X_all_count, X_all_density, X_all_deviation], axis=1)
       

    features_count = features_count_names
    features_density = features_density_names
    features_deviation = features_deviation_names
    
    return X_all, features_count, features_density, features_deviation

X_all, features_count, features_density, features_deviation, X_fake = get_count(X_all, X_fake)
print(X_all.shape)


from sklearn.base import TransformerMixin
from itertools import repeat
import scipy


class ThermometerEncoder(TransformerMixin):
    """
    Assumes all values are known at fit
    """
    def __init__(self, sort_key=None):
        self.sort_key = sort_key
        self.value_map_ = None
    
    def fit(self, X, y=None):
        self.value_map_ = {val: i for i, val in enumerate(sorted(X.unique(), key=self.sort_key))}
        return self
    
    def transform(self, X, y=None):
        values = X.map(self.value_map_)
        
        possible_values = sorted(self.value_map_.values())
        
        idx1 = []
        idx2 = []
        
        all_indices = np.arange(len(X))
        
        for idx, val in enumerate(possible_values[:-1]):
            new_idxs = all_indices[values > val]
            idx1.extend(new_idxs)
            idx2.extend(repeat(idx, len(new_idxs)))
            
        result = scipy.sparse.coo_matrix(([1] * len(idx1), (idx1, idx2)), shape=(len(X), len(possible_values)), dtype="int8")
            
        return result
        
 # thermos=[]
# for col in ["cloud_coverage"]:
#     if col=="cloud_coverage":
#         sort_key=[-1, 0, 1,2,3,4,5,6,7,8,9].index
#     else:
#         raise ValueError(col)
    
#     enc=ThermometerEncoder(sort_key=sort_key)
#     thermos.append(enc.fit_transform(train[col]))

# thermos[0].todense().shape

# cloudmat = pd.DataFrame(data=thermos[0].todense(), columns=['cloud'+str(i) for i in range(11)])
