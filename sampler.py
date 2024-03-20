from typing import Dict, List
import numpy as np

class Sampler:

    def __init__(
        self, 
        strategy = 'random', 
        seed = None, 
        test_crossval = False,
        num_tries_inner = 1,
        **kwargs
    ):
        self.strategy = strategy
        self.seed = seed
        self.num_tries_inner = num_tries_inner
        self.test_crossval = test_crossval
        self.n = kwargs.get('n', None)
        assert(self.n is not None)
        if seed:
            np.random.seed(self.seed)
        
        if r_test := kwargs.get('r_test', None):
            self.n_test = np.floor(r_test * self.n).astype(np.int32)
        else:
            self.n_test = kwargs.get('n_test', None)
            assert(self.n_test is not None)

        self.r_train = kwargs.get('r_train', None)
        self.n_train = kwargs.get('n_train', None)
        assert(bool(self.n_train) ^ bool(self.r_train))

        self.sampling_results = [None]

    def split_function(self):
        raise NotImplementedError

    def test_split(self):
        total = np.arange(self.n)
        np.random.shuffle(total)
        if self.test_crossval:
            subarrays = np.array_split(total, 5)
            complements = [np.concatenate(subarrays[:i] + subarrays[i+1:]) for i in range(len(subarrays))]
            self.sampling_results[0] = {
                "crossval": [{"test_idx": subarrays[i], "rest_idx": complements[i]} for i in range(len(subarrays))],
            }
        else:
            self.sampling_results[0] = {
                'test_idx': total[:self.n_test], 'rest_idx': total[self.n_test:]
            }

    # i have made some mistake here, so i add these notes for easy understanding.
    # former time, i use a generator as a return value, know all the sampling rsults are stored in a list.
    def sampling_split(self, direct = False, refer_dict = None, k = 0): # we assumpt that the test_split is always called before this function
        if direct or not self.test_crossval:
            if not self.test_crossval:
                refer_dict = self.sampling_results[0]
            test_set, rest_set = refer_dict['test_idx'], refer_dict['rest_idx']

            assert(self.num_tries_inner)
            trn_set = []
            val_set = []
            for tr in range(self.num_tries_inner):
                np.random.shuffle(rest_set)
                if self.r_train:
                    n_train = np.floor(self.r_train * rest_set.shape[0]).astype(np.int32)
                else:
                    n_train = self.n_train
                if self.strategy == "fps":
                    t, v = self.split_function(rest_set.shape[0], n_train, feature = self.feature[rest_set])
                else:
                    t, v = self.split_function(rest_set.shape[0], n_train)
                trn_set.append(rest_set[t])
                val_set.append(rest_set[v])
            refer_dict['train_idx'] = trn_set
            refer_dict['val_idx'] = val_set
            if not self.test_crossval:
                self.sampling_results[0] = refer_dict
            else:
                self.sampling_results[0]["crossval"][k] = refer_dict
        else:
            for k, refer_dict in enumerate(self.sampling_results[0]["crossval"]):
                self.sampling_split(True, refer_dict, k)
    
class RandomSampler(Sampler):

    def __init__(self, **kwargs):
        super(RandomSampler, self).__init__(**kwargs)

    def split_function(self, n, b, **kwargs):
        return random_split(n, b)

class FPSSampler(Sampler):

    def __init__(self, **kwargs):
        super(FPSSampler, self).__init__(**kwargs)
        self.feature = kwargs.get('feature', None)

    def split_function(self, n, b, **kwargs):
        assert(kwargs.get('feature') is not None)
        return fps_split(n, b, kwargs.get('feature'))


def random_split(n: int, b: int):
    idx = np.arange(n)
    np.random.shuffle(idx)

    return idx[:b], idx[b:]

def fps_split(n: int, b: int, feature: np.ndarray): # feature: b * F matrix
    idx = np.arange(n)
    fn, fm = feature.shape
    if not fn==n:
        raise Exception('Feature is not matched with N')
    
    feature_temp = np.zeros([b, fm], dtype=np.float64)
    idx_unsplit = np.ones(n, dtype=bool)
    idx_temp = np.zeros(b, dtype=np.int32)
    modulus = np.zeros(n)

    idx_init = np.random.randint(n)
    
    idx_unsplit[idx_init] = 0
    idx_temp[0] = idx_init
    feature_temp[0, :] = feature[idx_init, :]

    for i in range(b - 1):
        mod_array = np.linalg.norm(feature[:, :] - feature_temp[i, :], 2, axis=1)
        mod_array[mod_array == 0] = 1e-8 # add torlerance
        if not i==0:
            mod_array = np.minimum(modulus, mod_array)
        modulus[idx_unsplit] = mod_array[idx_unsplit]
        # for j in  range(n):
        #     if not idx_unsplit[j]:
        #         continue

        #     if i==0:
        #         modulus[j] = mod_array[j]
        #     else:
        #         modulus[j] = modulus[j] if modulus[j] < mod_array[j] else mod_array[j]

        idx_next = np.argmax(modulus)

        idx_temp[i+1] = idx_next
        idx_unsplit[idx_next] = False
        feature_temp[i+1, :] = feature[idx_next, :]
        modulus[idx_next] = 0
    
    return idx_temp, np.setdiff1d(idx, idx_temp, assume_unique=True)