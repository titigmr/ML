import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""

Implémentation de l'algorithme CART : Arbre de Décision pour de la classification.
La métrique de split utilisée est l'indice de Gini. 

"""


class DecisionTreeClassif:

    def __init__(self, max_depth=float('inf'), max_features='all', criterion='gini'):
        self.max_depth = max_depth + 1
        self.max_features = max_features
        self.criterion = criterion
    

    def fit(self, X, y, target_name=None):

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

        if target_name is None:
            target_name = np.unique(y)

        self.all_classes = np.unique(y)
        self.tree = self._tree_growth_(X, y, target_name=target_name)

    

    def predict(self, X):
        X.reset_index(drop=True, inplace=True)
        return [self._predict_(inputs, X) for inputs in X.index]


    
    def _predict_(self, inputs, X):
        node = self.tree
        
        while node.node.left:

            if X.loc[inputs, node.node.var] < node.node.threshold:
                node = node.node.left
            
            else:
                node = node.node.right

            if node.node is None:
                break
            
        return node.classe_predict
        
    

    def _gini_(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2
                         for c in self.all_classes)

    

    def __make_threshold__(self, liste):

        liste_sorted = sorted(liste)
        return np.array([np.mean((liste_sorted[i], liste_sorted[i+1]))
                         for i in range(len(liste_sorted))[:-1]])

    

    def _tree_growth_(self, X, y, target_name, depth=0):

        if depth < self.max_depth:

            node = self._find_best_node_(X, y, target_name)

            if node.node is not None:

                index_left = node.node._sample_left
                index_right = node.node._sample_right

                X_left, y_left = X[index_left], y[index_left]
                X_right, y_right= X[index_right], y[index_right]

                node.node.left = self._tree_growth_(
                    X_left, y_left, target_name, depth + 1)
                node.node.right = self._tree_growth_(
                    X_right, y_right, target_name, depth + 1)

            return node

    
    def _select_features_(self, columns, metric='all'):
    
        if metric == 'auto':
            n = int(np.sqrt(len(columns)))
            selected_features = np.random.choice(columns, size=n, replace=False)
        
        if metric == 'all':
            selected_features = columns
            
        if type(metric) == int:
            n = metric
            selected_features = np.random.choice(columns, size=n, replace=False)
            
        return selected_features


    def _find_best_node_(self, X, y, target_name):
        """
        Recherche le meilleur noeud possible (parmi toutes les variables et les seuils possibles) 
        selon la métrique choisie.
        
        Paramètres
        -----
        X : features (pd.DataFrame)
        y : target (pd.Series)


        """

        if self.criterion == "gini":
            best_gini = self._gini_(y)

        if self.criterion == "entropy":
            best_gain = 0

        leaf = Leaf(X, y, target_name, self.all_classes)
        leaf.node = None

        selected_features = self._select_features_(X.columns, self.max_features)

        for var in np.array(selected_features):

            all_value = np.unique(X[var])
            all_treshold = self.__make_threshold__(all_value)

            if len(all_treshold) > 1:

                for threshold in all_treshold:
                    node = Node(X, y, var, threshold, target_name, 
                        self.all_classes, self.criterion)

                    if self.criterion == 'gini':
                        if node.gini_pondere < best_gini:
                            best_gini = node.gini_pondere
                            leaf.node = node
                    
                    if self.criterion == 'entropy':
                        if node.gain_information > best_gain:
                            best_gain = node.gain_information
                            leaf.node = node

        return leaf




class Leaf:

    def __init__(self, X, y, target_name, all_classes):

        self.repartition_parent = {target_name[i]: sum(y == i) for i in all_classes}
        self.classe_predict = self._prediction_(self.repartition_parent)

    
    def _prediction_(self, repartition):
        n = -1
        for key, value in repartition.items():

            if value > n:
                n = value
                classe_pred = key
        return classe_pred




class Node:
    def __init__(self, X, y, var, threshold, target_name, all_classes, criterion):

        classe = X[var] < threshold
        m = len(y)
        number_left = sum(classe)
        number_right = sum(~classe)

        self._all_classes = all_classes
        self.threshold = threshold
        self.var = var

        left, right = self._sample_node_child_(
            X, y, var, threshold, target_name, classe)

        self._sample_left = left[1]
        self.repartition_left = left[0]

        self._sample_right = right[1]
        self.repartition_right = right[0]


        if criterion == "gini":
            self.gini_pondere = self._gini_node_child_(
                X, y, var, threshold, classe, m, number_left, number_right)
        
        if criterion == 'entropy':
            self.gain_information = self._information_gain_(
                X, y, var, threshold, classe)

    
    def _sample_node_child_(self, X, y, var, threshold, target_name, classe):
        """
        Recherche de la répartition et des indices des noeuds fils

        Paramètre 
        --------
        X : DataFrame des features (X)
        y : Série de la valeur à prédire (y)
        threshold : seuil de découpage de l'échantillons
        classe : boolean indexing des données avec la variable threshold

        """

        sample_left_count = {target_name[i]: sum(
            classe[y == i]) for i in self._all_classes}

        sample_right_count = {target_name[i]: sum(
            ~classe[y == i]) for i in self._all_classes}

        sample_left = pd.concat(classe[y == i] for i in self._all_classes)
        sample_right = pd.concat(~classe[y == i] for i in self._all_classes)

        return (sample_left_count, sample_left), (sample_right_count, sample_right)

    

    def _gini_node_child_(self, X, y, var, threshold, classe, m, number_left, number_right):
        """
        Calcul du gini pondéré entre les deux noeuds fils

        Paramètre
        --------
        X : DataFrame des features (X)
        y : Série de la valeur à prédire (y)
        var : nom de la colonne
        threshold : seuil de découpage en deux échantillons
        m : nombre d'observations
        number_left : nombre d'observations pour le noeud fils gauche
        number_right : nombre d'observations pour le noeud fils droit

        """

        gini_left = 1.0 - sum((sum(classe[y == i]) / number_left) ** 2
                              for i in self._all_classes)
        gini_right = 1.0 - sum((sum(~classe[y == i]) / number_right) ** 2
                               for i in self._all_classes)

        gini_pondere = (number_left * gini_left +
                        number_right * gini_right) / m

        return gini_pondere

    
    def _information_gain_(self, X, y, var, threshold, classe):
        
        entropie = self._entropie_(y)

        n_left = sum(classe) / len(y)
        n_right = sum(~classe) / len(y)

        e_left = self._entropie_(y[classe])
        e_right = self._entropie_(y[~classe])

        return entropie - ((n_left * e_left) + (n_right * e_right))

    
    def _entropie_(self, y, normalize=False):
        n = y.size

        entropie = sum(-(sum(c == y) / n) * np.log2(sum(c == y) / n)
                       for c in np.unique(y))

        if normalize:
            n_classe = len(np.unique(y))
            return entropie / np.log2(n_classe)

        return entropie
