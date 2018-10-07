from model.impurity.logistic_impurity import LogisticImpurity
import numpy as np

class LogisticImpurityTree:

    def __init__(self):
        self.__head = LogisticImpurityNode()

    def train(self, X, y, depth, min_data_to_split, sgd_iters, sgd_step_size):
        self.__head._train(X, y, depth, min_data_to_split, sgd_iters, sgd_step_size)
        #TODO: (is a dumb case but could happen) catch the tree already terminating here

    def predict(self, X):
        out = np.zeros(X.shape[0])
        self.__head._predict(X, np.arange(0, X.shape[0]), out)
        return out


class LogisticImpurityNode:
    def __init__(self):
        self.__model = LogisticImpurity()
        self.__children = []
        self.__label = None

    #TODO: add min impurity to split
    def _train(self, X, y, depth, min_data_to_split, sgd_iters, sgd_step_size):
        if depth <= 0 or len(y) < min_data_to_split:
            self.__set_leaf(y)
            return None
        self.__model.train(X,y,sgd_iters,sgd_step_size)
        splits = self.__split(X)
        for split in splits:
            split = split[0]
            if len(split) < min_data_to_split:
                #normal decision trees are never able to have no data fall into a child, but it is possible to occur hereself.
                #is handled here:
                self.__set_leaf(y)
                return None
            else:
                child = LogisticImpurityNode()
                child._train(X[split], y[split], depth-1, min_data_to_split, sgd_iters, sgd_step_size)
                self.__children.append(child)

    def _predict(self, X, inds, predictions):
        if self.__children is None:
            assert(self.__label is not None)
            predictions[inds] = self.__label
            return None
        splits = self.__split(X)
        for i in range(len(splits)):
            if len(splits[i]) != 0:
                self.__children[i]._predict(X[splits[i]], inds[splits[i]], predictions)

    def __split(self, X):
        probs = self.__model.predict(X)
        return [np.where(probs <= 0.5), np.where(probs > 0.5)]

    def __set_leaf(self, y):
        self.__children = None
        labels, counts = np.unique(y, return_counts = True)
        self.__label = labels[np.argmax(counts)]
'''
class LeafNode:

    def _train(self, X, y, depth, min_data_to_split, sgd_iters, sgd_step_size):
        labels, counts = np.unique(y, return_counts = True)
        self.__label = labels[np.argmax(counts)]

    def _predict(self, X, inds, predictions):
        predictions[inds] = self.__label
'''
