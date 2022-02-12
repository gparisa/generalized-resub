import numpy as np
from scipy import stats
from sklearn import metrics

class bolstering():
    def __init__(self, x, y, bolstering_type='original', factor=1):
        self.x = x
        self.y = y
        self.dim = x.shape[1]
        self.bolstering_type = bolstering_type
        self.factor = factor
        
        self.xx = self.get_x_perClass(self.x, self.y)

        self.alpha_orig = stats.chi.median(self.dim)
        self.alpha_nB = stats.chi.median(1)
        
    def get_x_perClass(self, x, y):
        xtr = np.array(x)
        ytr = np.array(y)
        classes = sorted(set(ytr.reshape(len(ytr),)))
        xx = {y_clas: xtr[ytr == y_clas, :] for y_clas in classes}
        return xx
    
    def get_sigma(self):
        def get_sigma_perclass(x_cls):
            if self.bolstering_type == 'original':
                dist = metrics.pairwise_distances(x_cls)
                sig = np.mean(np.amin(dist+np.diag(np.sum(dist,axis=1)),axis=1))/self.alpha_orig
                return [(sig*self.factor)**2 for _ in range(self.dim)] 
                # cov = sig*sig*np.identity(d) 
            elif self.bolstering_type == 'naive_Bayes':
                def get_diff(xi):
                    sorted_tmp = sorted(set(xi))
                    if len(sorted_tmp) > 1:
                        target = {sorted_tmp[i]: min(abs(sorted_tmp[i] - sorted_tmp[i-1]),
                                                      abs(sorted_tmp[i+1] - sorted_tmp[i]))
                              for i in range(1, len(sorted_tmp) - 1)}
                        target[sorted_tmp[0]] = sorted_tmp[1] - sorted_tmp[0]
                        target[sorted_tmp[-1]] = sorted_tmp[-1] - sorted_tmp[-2]
                        return np.mean([target[z] for z in xi])
                    else:
                        return np.max([1e-30, sorted_tmp[-1]])
    
                tmp = [(get_diff(self.x[:, j].tolist())/self.alpha_nB).tolist()
                       for j in range(self.x.shape[1])]
            else:
                raise TypeError('unknown bolstering_type: {\'original\' or \'naive_Bayes\'}')
            return [(i*self.factor)** 2 for i in tmp]
    
        return {y_cls: get_sigma_perclass(x_cls) for y_cls, x_cls in self.xx.items()}

    def generate_mc_sample(self, n_mc_sample, 
                           save_to_file=True,
                           input_shape=None,
                           filename= None):
        if input_shape is None:
            input_shape = self.x.shape[1:]
            
        if  save_to_file:
            if filename is None:
                raise TypeError("'save_to_file is 'True', but filename is None.")
            else:
                filename = str(filename)+'.hdf5'
                
            import h5py
            saved_sampl = h5py.File(filename, 'a')
        
        sigma = self.get_sigma()
        mc_smpls = {}
        for y_clas, x_cls in self.xx.items():
            sig = np.diag(sigma[y_clas])
            mc_smpl_clas = np.random.multivariate_normal(x_cls[0,:], sig, n_mc_sample)
            for i in range(1, x_cls.shape[0]):  
                mc_tmp = np.random.multivariate_normal(x_cls[i,:], sig, n_mc_sample)
                mc_smpl_clas= np.append(mc_smpl_clas, mc_tmp, axis=0)
                del mc_tmp
            mc_smpls[y_clas]= mc_smpl_clas
            if save_to_file:
                saved_sampl.create_dataset(str(y_clas), mc_smpl_clas.shape,
                                          data= mc_smpl_clas)
        return mc_smpls
            
