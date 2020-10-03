import numpy as np
import pandas as pd 
from decision_tree import DecisionTreeClassif

class RandomForestClassif:
    
    def __init__(self, max_depth=float('inf'), max_features="auto", n_estimators=5):
        
        self.max_depth = max_depth + 1
        self.max_features = max_features
        self.n_estimators = n_estimators
        
        
    def fit(self, X, y, target_name=None):
        
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        
        all_tree = []

        for i in range(self.n_estimators):
            X_boot, y_boot = self.__bootstrap__(X, y)

            X_boot.reset_index(drop=True, inplace=True)
            y_boot.reset_index(drop=True, inplace=True)

            clf = DecisionTreeClassif(max_features=self.max_features)
            clf.fit(X_boot, y_boot, target_name=target_name)

            all_tree.append(clf)

        self.__all_tree__ = all_tree
    
    def __bootstrap__(self, X, y):
        indices = np.random.randint(len(y), size=len(y))
        return X.loc[indices], y.loc[indices]
    
    
    def predict(self, X):
        X.reset_index(drop=True, inplace=True)
        return [self._predict_(inputs, X) for inputs in X.index]


    def _predict_(self, inputs, X):
        
        all_predicts = []
        
        for tree in self.__all_tree__ :
            node = tree.tree
            
            while node.node.left:
                if X.loc[inputs, node.node.var] < node.node.threshold:
                    node = node.node.left

                else:
                    node = node.node.right

                if node.node is None:
                    break

            all_predicts.append(node.classe_predict)
            
        return pd.Series(all_predicts).mode()[0]