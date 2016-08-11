'''
Created on 07.08.2016

@author: Tobias Sterbak: sterbak-it@outlook.com
'''
import numpy as np
from tree import decision_tree_c45, foggy_decision_tree, simple_tree, logistic_tree,\
    rbf_tree
from helpers import voting

from sklearn.datasets.base import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.metrics.classification import accuracy_score
import time

class random_forest(object):
    '''
    standard random forest
    '''
    def __init__(self, n_estimators=10, max_depth=4, max_features=None, bootstrap=True, sample_ratio=1.0, seed=2016, treetype="standard"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.treetype = treetype
    
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
            if self.treetype == "simple":
                tree = simple_tree(max_depth=self.max_depth, max_features=self.max_features).fit(X_temp,y_temp)
            elif self.treetype == "rbf":
                tree = rbf_tree(max_depth=self.max_depth, max_features=self.max_features, min_samples=100).fit(X_temp,y_temp)
            elif self.treetype == "logistic":
                tree = logistic_tree(max_depth=self.max_depth, max_features=self.max_features, min_samples=100).fit(X_temp,y_temp)
            elif self.treetype == "standard":
                tree = decision_tree_c45(max_depth=self.max_depth, max_features=self.max_features).fit(X_temp,y_temp)
            else:
                print("Select an existing type of tree!")
            self._trees.append(tree)
        return self    
        
    def predict(self,X):
        predictions = []
        for tree in self._trees:
            y_pred = tree.predict(X)
            predictions.append(y_pred)
        y_out = voting(predictions)
        return y_out

class foggy_forest(object):
    '''
    forest of randomized foggy trees
    '''
    def __init__(self,n_estimators=10, max_depth=4, max_features=None, bootstrap=True, sample_ratio=1.0, var=0.5, seed=2016):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.sample_ratio = sample_ratio
        self.seed = seed
        self.var = var
        self.max_features = max_features
    
    def fit(self,X,y):
        self._trees = []
        np.random.seed(self.seed)
        for _ in range(self.n_estimators):
            ind = np.random.choice(X.shape[0], int(X.shape[0]*self.sample_ratio), replace=self.bootstrap)
            X_temp = X[ind,:]
            y_temp = y[ind]
            tree = foggy_decision_tree(max_depth=self.max_depth, max_features=self.max_features, var=self.var).fit(X_temp,y_temp)
            self._trees.append(tree)
        return self    
        
    def predict(self,X):
        predictions = []
        for tree in self._trees:
            y_pred = tree.predict(X)
            predictions.append(y_pred)
        y_out = voting(predictions)
        return y_out

if __name__ == "__main__":
    digits = load_digits(n_class=10)
    X = digits.data
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.5, random_state=2016)
    
    print("Simple random forest")
    t0 = time.time()
    forest = random_forest(max_depth=10, n_estimators=20, max_features=20, treetype="simple").fit(X_train,y_train)
    #forest = foggy_forest(max_depth=10, n_estimators=20, var=3, max_features=20).fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = accuracy_score(y_test, y_pred)
    print("Score: %0.3f" %score)
    print("")
    
    print("Logistic random forest")
    t0 = time.time()
    forest = random_forest(max_depth=4, n_estimators=20, max_features=None, treetype="logistic").fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = accuracy_score(y_test, y_pred)
    print("Score: %0.3f" %score)
    print("")
    
    print("RBF forest")
    t0 = time.time()
    forest = random_forest(max_depth=4, n_estimators=20, max_features=None, treetype="rbf").fit(X_train,y_train)
    #forest = foggy_forest(max_depth=10, n_estimators=20, var=3, max_features=2).fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = accuracy_score(y_test, y_pred)
    print("Score: %0.3f" %score)
    print("")
    
    print("Random forest")
    t0 = time.time()
    forest = random_forest(max_depth=10, n_estimators=20, max_features=20, treetype="standard").fit(X_train,y_train)
    #forest = foggy_forest(max_depth=10, n_estimators=20, var=3, max_features=2).fit(X_train,y_train)
    y_pred = forest.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = accuracy_score(y_test, y_pred)
    print("Score: %0.3f" %score)
    print("")
    
    # printtree(tree._tree,indent='')
    
    print("Sklearn Baseline")
    t0 = time.time()
    sklearn_forest = RandomForestClassifier(criterion="entropy", max_depth=10, n_estimators=20, random_state=2016, max_features=20, min_samples_split=1).fit(X_train, y_train)
    y_pred = sklearn_forest.predict(X_test)
    print("Time taken: %0.3f" %(time.time() - t0))
    score = accuracy_score(y_test, y_pred)
    print("Score: %0.3f" %score)
    print("")