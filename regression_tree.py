'''
Created on 15.08.2016

@author: Tobias
'''
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from tree import decisionnode, printtree
from helpers import divide_set, average
from numpy import unique
from numpy.random import choice
from sklearn.linear_model.base import LinearRegression
from sklearn.preprocessing.data import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
from sklearn.metrics.regression import mean_squared_error, mean_absolute_error
import time
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.gaussian_process.gaussian_process import GaussianProcess
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV

class regression_forest(object):
    '''
    regression random forest
    '''
    def __init__(self, n_estimators=10, max_depth=4, max_features=None, bootstrap=True, min_samples=100, sample_ratio=1.0, seed=2016, treetype="linear"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.treetype = treetype
        self.min_samples = min_samples
        
    def fit(self,X,y):
        self._trees = []
        self.n_samples = X.shape[0]
        self.classes = np.unique(y).tolist()
        self.n_classes = len(self.classes)
        self.min_samples = (self.n_samples/self.n_classes)/10
        np.random.seed(self.seed)
        
        for _ in range(self.n_estimators):
            ind = np.random.choice(X.shape[0], int(X.shape[0]*self.sample_ratio), replace=self.bootstrap)
            X_temp = X[ind,:]
            y_temp = y[ind]
            if self.treetype == "linear":
                tree = linear_regression_tree(max_depth=self.max_depth, max_features=self.max_features, random_state=self.seed, min_samples=self.min_samples).fit(X_temp,y_temp)
            else:
                print("Select an existing type of tree!")
            self._trees.append(tree)
        return self    
        
    def predict(self,X):
        predictions = []
        for tree in self._trees:
            y_pred = tree.predict(X)
            predictions.append(y_pred)
        y_out = average(predictions)
        return y_out

class linear_regression_tree(object):
    '''
    linear regression tree
    '''
    def __init__(self, max_depth=4, max_features=None, min_samples=100, kerneltype="poly", gridsearch=True, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples = min_samples
        self.kerneltype = kerneltype
        self.gridsearch = gridsearch
        self.random_state = random_state
        np.random.seed(random_state)
        
    def fit(self,X,y):
        if self.max_features == None:
            self.max_features = X.shape[1]
        elif self.max_features == "sqrt":
            self.max_features = int(np.sqrt(X.shape[1]))
        elif self.max_features == "log2":
            self.max_features = int(np.log2(X.shape[1]))
                    
        self.classes = np.unique(y).tolist()
        self._tree = self._build_tree(X,y)
        return self
    
    def predict(self,X):
        y_pred = []
        for i in range(X.shape[0]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = self._predict_one(X[i], self._tree)
            y_pred.append(pred)
        return np.array(y_pred)
    
    def _predict_one(self, sample, tree):
        m = 0
        if isinstance(tree.results,np.float32) or isinstance(tree.results,np.float64):
            m = 1
            return tree.results
        elif tree.results != None and m==0:
            return tree.results.predict(sample)[0]
        else:
            v = sample[tree.feature]
            branch = None
            if isinstance(v,int) or isinstance(v,float):
                if v >= tree.value: 
                    branch = tree.tb
                else: 
                    branch = tree.fb
            return self._predict_one(sample, branch)
    
    def _build_tree(self,X,y,current_depth=0):
        current_error = X.shape[0]*np.var(y)
        best_error = current_error
        best_criteria = None
        best_sets = None
        best_labels = None
        selected_features = choice(X.shape[1], self.max_features)
        for feature in selected_features:
            for value in unique(X[:,feature])[1:len(unique(X[:,feature]))-1]:
                lower_set,lower_set_target, higher_set, higher_set_target = divide_set(X, y, feature, value)
                error = lower_set.shape[0]*np.var(lower_set_target) + higher_set.shape[0]*np.var(higher_set_target)
                if error < best_error and lower_set.shape[0] > 2 and higher_set.shape[0] > 2:
                    best_error = error
                    best_criteria = (feature,value)
                    best_sets = (lower_set,higher_set)
                    best_labels = (lower_set_target,higher_set_target)
        if best_error < current_error and current_depth < self.max_depth and X.shape[0] > self.min_samples:
            trueBranch = self._build_tree(best_sets[0], best_labels[0], current_depth=current_depth+1)
            falseBranch = self._build_tree(best_sets[1], best_labels[1], current_depth=current_depth+1)
            return decisionnode(feature=best_criteria[0], value=best_criteria[1], tb=trueBranch, fb=falseBranch, depth=current_depth)
        else:
            if X.shape[0] > self.min_samples:
                kernel_ridge = KernelRidge(kernel=self.kerneltype, degree=2, coef0=1)
                pipe = Pipeline([
                                ("scaler", StandardScaler()),
                                ("regressor", kernel_ridge)
                                ])
                if self.gridsearch == True:
                    params = {
                              "regressor__alpha": [0.01, 0.1, 1.0, 10.0]
                            }
                    regressor = GridSearchCV(pipe, params, refit=True, cv=3)
                else:
                    regressor = pipe
                regressor.fit(X,y)
                #print(regressor.best_params_)
                return decisionnode(results=regressor)
            else:
                #regressor = LinearRegression(normalize=True).fit(X,y)
                return decisionnode(results=np.mean(y))
                

if __name__ == "__main__":
    boston = datasets.load_boston()
    y = boston.target
    X = boston.data
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5, random_state=2016)
    
    print("Linear RT")
    t0 = time.time()
    lrt = linear_regression_tree(max_depth=3, max_features="sqrt", min_samples=20, kerneltype="poly", gridsearch=True, random_state=2016).fit(X_train,y_train)
    y_pred = lrt.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = mean_absolute_error(y_test, y_pred)
    print("Error: %0.3f" %score)
    print("")
    
    #printtree(lrt._tree, indent="  ")
    
    print("Linear RF")
    t0 = time.time()
    lrt = regression_forest(n_estimators=10, max_depth=4, max_features="sqrt", seed=2016, min_samples=100).fit(X_train,y_train)
    y_pred = lrt.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = mean_absolute_error(y_test, y_pred)
    print("Error: %0.3f" %score)
    print("")
    
    print("Sklearn RT")
    t0 = time.time()
    rt_sklearn = DecisionTreeRegressor(max_depth=7, max_features="sqrt", random_state=2016).fit(X_train,y_train)
    y_pred = rt_sklearn.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = mean_absolute_error(y_test, y_pred)
    print("Error: %0.3f" %score)
    print("")
    
    print("Skearn GP")
    gp = GaussianProcess(regr="constant", corr='absolute_exponential', beta0=None, storage_mode='full',
                     verbose=False, theta0=0.1, thetaL=None, thetaU=None, optimizer='fmin_cobyla', 
                     random_start=1, normalize=True, nugget=0.05, random_state=2016).fit(X_train,y_train)
    y_pred = gp.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = mean_absolute_error(y_test, y_pred)
    print("Error: %0.3f" %score)
    print("")