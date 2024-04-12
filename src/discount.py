import numpy as np


class kDISCount():
    def __init__(self, g, eps=4e-2):
        '''
        Counting in large image collections with detector-based importance sampling.

        input:
        - g (list): detector counts of size |\Omega|
        '''

        if min(g) == 0:
            g = np.array(g) + max(max(g)*eps, 1) # to guarantee g(s)>0 for all f(s)>0

        self.g = g # detector counts
        self.q = np.array(g)/np.sum(g) # proposal distribution

    def sample(self, n):
        '''
        return n samples from q
        '''
        self.samples = list(np.random.choice(np.arange(len(self.g)), n, p=self.q, replace=True))
        return self.samples

    def load(self, screened_samples):
        '''
        loads screened samples to estimator
        '''
        self.f = [np.nan]*len(self.g)
        for i, s_i in enumerate(self.samples):
            self.f[s_i] = screened_samples[i]

    def estimate(self, regions, ci_all_samples=False):
        '''
        input:
        - regions (list): list with regions. Each region is another list with indices 
                          of elements s in the region. Can overlap or be disjoint with
                          each other.
        - ci_all_samples (bool): True to calculate confidence intervals with all samples
                                 in \Omega

        output:
        - F_hat: (list): list of tuples with estimated counts and confidence intervals 
                         for each region. len(F_hat) = len(regions)
        '''
    
        F_hat, CI = [], []
        for region in regions:
            S = [s_i for s_i in self.samples if s_i in region] # get samples inside region S
            G_S = sum([self.g[i] for i in region]) # get G(S)
            if len(S) == 0: 
                F_hat.append(0.) # F_hat=0 if n(S)=0
                CI.append(0.)
                continue

            # Calculate count estimate
            w_bar = 0
            for s_i in S:
                w_bar += self.f[s_i]/self.g[s_i] # importance weight for region S_i
            F_hat_i = G_S*w_bar/len(S) # region F_hat--> G(S_i)*w_bar(S_i)
            
            # Calculate confidence intervals
            w_ci = 0
            if ci_all_samples: # use all samples to calculate variance
                S_ci = self.samples.copy()
            else:
                S_ci = S
            for s_i in S_ci:
                w_ci += (G_S*self.f[s_i]/self.g[s_i] - F_hat_i)**2
            var_hat = w_ci/len(S_ci) # estimated variance
            CI_i = 1.96*np.sqrt(var_hat/len(S_ci)) # 95% confidence intervals

            F_hat.append(F_hat_i)
            CI.append(CI_i)
        return F_hat, CI
