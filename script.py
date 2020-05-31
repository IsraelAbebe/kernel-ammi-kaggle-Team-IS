import sklearn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score


import os
import optuna
import random
import cvxopt
import cvxopt.solvers
import sklearn

cvxopt.solvers.options['show_progress'] = False


np.random.seed(42)
random.seed(42)

X_test_ = pd.read_csv('data/Xte.csv',sep=',',index_col=0)
X_train_ = pd.read_csv('data/Xtr.csv',sep=',',index_col=0)

X_test_mat100 = pd.read_csv('data/Xte_mat100.csv',sep=' ',header=None).values
X_train_mat100 = pd.read_csv('data/Xtr_mat100.csv',sep=' ',header=None).values

y = pd.read_csv('data/Ytr.csv',sep=',',index_col=0)

def get_label(type=0):
    y = pd.read_csv('data/Ytr.csv',sep=',',index_col=0)
    if type == 0:
        y = y.Bound.values
        return y
    else:
        y['Bound'] = y.Bound.apply(lambda x: -1 if x == 0 else 1)
        y = y.Bound.values
        return y

def get_train_test(X,y,p):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=p, random_state=42)
    print(X_train.shape,X_test.shape,y_train.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def getKmers(sequence, size=3):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

def base2int(c):
    return {'a':0,'c':1,'g':2,'t':3}.get(c,0)

def index(kmer):
    base_idx = np.array([base2int(base) for base in kmer])
    multiplier = 4** np.arange(len(kmer))
    kmer_idx = multiplier.dot(base_idx)
    return kmer_idx
    
    
def spectral_embedding(sequence,kmer_size=3):
    kmers = getKmers(sequence,kmer_size)
    kmer_idxs = [index(kmer) for kmer in kmers]
    one_hot_vector = np.zeros(4**kmer_size)
    
    for kmer_idx in kmer_idxs:
        one_hot_vector[kmer_idx] += 1
    return one_hot_vector


def get_data(kmer_size):
    data = pd.DataFrame(pd.concat([X_train_.seq,X_test_.seq],axis=0))
    train_text = data.seq.values
    # X_train_['kmers'] = X_train_.seq.apply(lambda x:list(spectral_embedding(x,kmer_size=3)))
    kmer_data = []
    for i in train_text:
        kmer_data.append(spectral_embedding(i,kmer_size=kmer_size))

    return np.array(kmer_data)

def sigma_from_median(X):
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def polynomial_kernel(X1, X2, power=2):
    return np.power((1 + linear_kernel(X1, X2)),power)

def rbf_kernel(X1, X2, sigma=10):
    X2_norm = np.sum(X2 ** 2, axis = -1)
    X1_norm = np.sum(X1 ** 2, axis = -1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K

class KernelMethodBase(object):
    '''
    Base class for kernel methods models
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel,
        'gaussian':gaussian_kernel
    }
    def __init__(self, kernel='linear', **kwargs):
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf' or self.kernel_name == 'gaussian':
            params['sigma'] = kwargs.get('sigma', None)
        if self.kernel_name == 'polynomial':
            params['power'] = kwargs.get('power', None)
            
        
        return params

    def fit(self, X, y, **kwargs):
        return self
        
    def decision_function(self, X):
        pass

    def predict(self, X):
        pass


class KernelRidgeRegression(KernelMethodBase):
    '''
    Kernel Ridge Regression
    '''
    def __init__(self, lambd=0.1, **kwargs):
        self.lambd = lambd
        # Python 3: replace the following line by
        # super().__init__(**kwargs)
        super(KernelRidgeRegression, self).__init__(**kwargs)

    def fit(self, X, y, sample_weights=None):
        n, p = X.shape
        assert (n == len(y))
    
        self.X_train = X
        self.y_train = y
        
        if sample_weights is not None:
            w_sqrt = np.sqrt(sample_weights)
            self.X_train = self.X_train * w_sqrt[:, None]
            self.y_train = self.y_train * w_sqrt
        
        A = self.kernel_function_(X,X,**self.kernel_parameters)
        A[np.diag_indices_from(A)] = np.add(A[np.diag_indices_from(A)],n*self.lambd)
        # self.alpha = (K + n lambda I)^-1 y
        self.alpha = np.linalg.solve(A , self.y_train)

        return self
    
    def decision_function(self, X):
        K_x = self.kernel_function_(X,self.X_train, **self.kernel_parameters)
        return K_x.dot(self.alpha)
    
    def predict(self, X):
        return self.decision_function(X)

def cross_validate(x_data,y_data,kernel=None,lambd=0.2,C=3,sigma=0.5,k=5,power=2):
    if len(x_data)%k != 0:
        print('cant vsplit',len(x_data),' by ',k)
        return
    
    x_data_splitted = np.vsplit(x_data,k)
    y_data_splitted = np.vsplit(y_data.reshape(-1,1),k)
    
    aggrigate_result = []
    for i in range(len(x_data_splitted)):
        train = []
        test = []
        items = [j for j in range(len(x_data_splitted)) if j !=i ]
        x_test = x_data_splitted[i]
        y_test = y_data_splitted[i]
        for item in items:
            if len(train) == 0:
                x_train = x_data_splitted[item]
                y_train = y_data_splitted[item]
            else:
                x_train = np.concatenate((x_train,x_data_splitted[item]), axis=0)
                y_train = np.concatenate((y_train,y_data_splitted[item]), axis=0)
        
        model = KernelRidgeRegression(
                    kernel=kernel,
                    lambd=lambd,
                    sigma=sigma,
                    power=power
                ).fit(x_train, y_train)
        result =sum(np.sign(model.predict(x_test))==y_test)/len(y_test)#roc_auc_score(np.sign(model.predict(x_test)),y_test) #
            
        aggrigate_result.append(result)
        
        value = sum(aggrigate_result)/len(aggrigate_result)
    return value



X_train, X_test, y_train, y_test = get_train_test(get_data(7)[:2000,:],get_label(type=-1),0.3)

# 0.6585	00:00:03.992723	27.187947	4	7	1.418356	2

# model = KernelRidgeRegression(
#                 kernel='linear',
#                 lambd=0.688381,
#                 sigma=93.801110
#             ).fit(X_train, y_train)

model = KernelRidgeRegression(
                kernel='polynomial',
                lambd=1.418356,
                sigma=2,
                power=2,
                C= 27.187947
            ).fit(X_train, y_train)
result = sum(np.sign(model.predict(X_test).flatten())==y_test.flatten())/len(y_test.flatten())
cv_result = cross_validate(get_data(7)[:2000,:],get_label(type=-1),
                kernel='polynomial',
                lambd=1.418356,
                sigma=2,
                C= 27.187947,
                power=2,
                k = 4)
print(result,cv_result)

X_test_final  = np.sign(model.predict(get_data(7)[2000:,:]))
sumbission = []
for i in range(len(X_test_final)):
    r1 = X_test_final[i]
    if r1 == 1:
        sumbission.append([i,int(r1)])
    elif r1 == -1:
        sumbission.append([i,0])
    else:
        print('problem')
        
    
# sumbission
df = pd.DataFrame(sumbission)
df.columns = ['Id','Bound']
df.to_csv('cv_new.csv',index=False)

df.head(15)