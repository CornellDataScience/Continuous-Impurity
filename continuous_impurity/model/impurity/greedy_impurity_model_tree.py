from abc import ABC, abstractmethod
import numpy as np
import function.impurity as impurity
from time import sleep
from model.impurity.greedy_impurity_tree_leaf import GreedyImpurityTreeLeaf
import timeit


class GreedyImpurityModelTree:

    def __init__(self, model):
        self._model = model
        self.__children = None

    #TODO: add param for when to set leaf if input X is already a certain % of one class
    #(prevents training 100% accurate sub-X)
    def train(self, parent, X, y, min_data_to_split, sgd_iters, sgd_step_sizes):
        self._model.train(X,y,sgd_iters,sgd_step_sizes)
        splits = self.__split(X)
        where_splits_eq_child = []
        for child_ind in range(len(self.__children)):
            where_split_eq_child_ind = np.where(splits == child_ind)
            if len(where_split_eq_child_ind[0]) < min_data_to_split:
                parent.set_child_to_leaf(self, y)
                return None
            where_splits_eq_child.append(where_split_eq_child_ind)

        for child_ind in range(len(self.__children)):
            X_child = X[where_splits_eq_child[child_ind]]
            y_child = y[where_splits_eq_child[child_ind]]
            self.__children[child_ind].train(self,X_child,y_child,min_data_to_split,sgd_iters,sgd_step_sizes)

    def set_child_to_leaf(self, child, y):
        ind = self.__children.index(child)
        self.__children[ind] = GreedyImpurityTreeLeaf()
        self.__children[ind].train(None, None, y, None, None, None)



    def predict(self, X):
        predicts = np.zeros(X.shape[0])
        self._predict(X,np.arange(0,X.shape[0]),predicts)
        return predicts

    def _predict(self, X, inds, predicts):
        splits = self.__split(X[inds])
        for child_num in range(len(self.__children)):
            where_splits_eq_child_num = np.where(splits == child_num)
            inds_child_num = inds[where_splits_eq_child_num]
            self.__children[child_num]._predict(X,inds_child_num,predicts)



    '''
    returns a vector, v, where v[i] is the index of the child that X[i] should
    be funneled to.
    '''
    def __split(self, X):
        predictions = self._model.predict(X)
        assert predictions.shape[1] == len(self.__children), \
            "has " + str(len(self.__children)) + " children, but only outputs " +\
            str(predictions.shape[1]) + " subset probabilities"
        split_assigns = np.argmax(predictions, axis = 1)
        return split_assigns


    def add_children(self, children):
        if not isinstance(children, list):
            children = [children]
        if len(children) > 0:
            if self.__children is None:
                self.__children = []
            for child in children:
                self.__children.append(child)
