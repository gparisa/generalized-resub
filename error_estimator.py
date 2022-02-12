
#############################################################################
#############################################################################
#############################################################################
import numpy as np

class generalized_resub():
    def __init__(self, x, y,
                 classifier=None,
                 mc_samples=None,
                 x_test=None):

        self.x = x
        self.y = y
        self.mc_bol = mc_samples
        self.mdl = classifier
        self.x_test = x_test

        if self.mdl is None:
            raise TypeError("missing object property: 'classifier' ")

        self.xx_tr = self.get_x_perClass(self.x, self.y)

        self.xtr_flat = np.array([self.x[i].flatten()
                                 for i in range(self.x.shape[0])])
        self.xx_tr_flat = self.get_x_perClass(self.xtr_flat, self.y)

        if self.x_test is None:
            self.x_test = self.x
            self.y_test = self.y
            self.xx_ts = self.xx_tr
            self.xts_flat = self.xtr_flat
            self.xx_ts_flat = self.xx_tr_flat
        else:
            self.xx_ts = self.xx_tr
            self.xts_flat = np.array([self.x_test[i].flatten()
                                     for i in range(self.x_test.shape[0])])
            self.xx_ts_flat = self.get_x_perClass(self.xts_flat, self.y_test)

        self.post_prob, self.psi_r = self.get_psi(x=self.x, y=self.y)

        self.clas_post_prob_psi_r = {clas: self.get_psi(x=x_clas, y=clas)
                                     for clas, x_clas in self.xx_tr.items()}

    #------------------------------------------------------------------------------------------
    def get_x_perClass(self, x, y):
        xtr = np.array(x)
        ytr = np.array(y)
        classes = sorted(set(ytr.reshape(len(ytr),)))
        xx = {y_clas: xtr[ytr == y_clas, :] for y_clas in classes}
        return xx

    #------------------------------------------------------------------------------------------
    def get_psi(self, x=None, y=None):
        if x is None:
            x = self.x
        post_prob = self.mdl.predict(x, verbose=0)
        psi = np.argmax(post_prob, axis=1)
        return post_prob, psi

    #------------------------------------------------------------------------------------------
    def standard(self, x_test=None, y_test=None):
        if x_test is None:
            psi_rr = self.psi_r
            y_test = self.y
        else:
            _, psi_rr = self.get_psi(x_test, y_test)
        return np.mean(psi_rr != y_test)

    #------------------------------------------------------------------------------------------
    def compute_mc_pp_psi(self, mc_samples):
        pp_mc, psi_mc = {}, {}
        for y_clas, mc_smpl_clas in mc_samples.items():
            pp_mc[y_clas], psi_mc[y_clas] = self.get_psi(mc_smpl_clas, y_clas)
        return pp_mc, psi_mc
    
    #------------------------------------------------------------------------------------------
    def bolster(self, mc_samples, psi_mc):
        err_clas = 0.0
        for y_clas, mc_smpl_clas in mc_samples.items():
            err_clas += np.mean(psi_mc[y_clas] != y_clas)
        return err_clas/len(mc_samples)
    
    #------------------------------------------------------------------------------------------
    def pp(self):
        n_sample = self.x.shape[0]
        err_correctlyClassified, err_misClassified = 0.0, 0.0
        for clas, val in self.clas_post_prob_psi_r.items():
            post_prob = val[0]
            psi = val[1]
            err_correctlyClassified += sum(1 -
                                           post_prob[:, clas][(psi == clas)])
            err_misClassified += sum(post_prob[:, clas][(psi != clas)])
        return (err_correctlyClassified + err_misClassified) / n_sample

 #------------------------------------------------------------------------------------------

    def bolster_pp(self, mc_samples, pp_mc, psi_mc):
        err_clas = 0.0
        for clas, mc_smpl_clas in mc_samples.items():
            post_prob_clas = pp_mc[clas]
            psi_mc_clas = psi_mc[clas]
            n_mc = len(psi_mc_clas)
            err_correctlyClassified = sum(
                1 - post_prob_clas[:, clas][(psi_mc_clas == clas)])/n_mc
            err_misClassified = sum(
                post_prob_clas[:, clas][(psi_mc_clas != clas)])/n_mc

            err_clas += err_correctlyClassified + err_misClassified
        return err_clas/len(mc_samples)

