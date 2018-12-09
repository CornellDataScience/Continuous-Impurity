import numpy as np
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import gc
from performance.stopwatch_profiler import StopwatchProfiler
from model.impurity.global_impurity3.node3 import Node3


class GlobalImpurityNode3(Node3):
    #originally had children = [] as default argument, but turns out that whenever
    #that when self._children = children, and then self._children is modified,
    #the default argument children will also be modified since self._children points
    #to it.
    def __init__(self, parent, model):
        Node3.__init__(self, [])
        self._parent = parent
        self._model = model
        self._ID = None
        self._leaf_predict = None


    '''
    Returns (nodes, nonleaves, leaves)
    Postcondition: Each node has it's ID set to its corresponding index in the list
        representation.
    '''
    def to_list(self):
        nodes = []
        leaves = []
        nonleaves = []
        def f(node, id):
            node._ID = id
            nodes.append(node)
            if node._is_leaf():
                leaves.append(node)
            else:
                nonleaves.append(node)
            return id + 1
        self.fold(f, 0)
        return nodes, nonleaves, leaves


    def predict(self, X):
        inds = np.arange(0, X.shape[0], 1)
        predictions = np.zeros(X.shape[0], dtype = np.int)
        self.__predict(X, predictions, inds)
        return predictions

    def __predict(self, X, predictions, inds):
        if self._is_leaf():
            assert(self._leaf_predict is not None)
            predictions[inds] = self._leaf_predict
            return None
        splits = self.__split(X, inds)
        for child_ind in range(len(splits)):
            self._children[child_ind].__predict(X, predictions, splits[child_ind])

    def _set_node_inds(self, X, inds, node_inds):
        if self._ID >= len(node_inds):
            node_inds.extend([None for i in range(len(node_inds), self._ID+1)])
        assert(node_inds[self._ID] is None)
        node_inds[self._ID] = inds
        if not self._is_leaf():
            splits = self.__split(X, inds)
            for child_ind in range(len(splits)):
                self._children[child_ind]._set_node_inds(X, splits[child_ind], node_inds)

    def __split(self, X, inds):
        f_inds_out = self.f(X[inds])
        split_inds_assign = np.argmax(f_inds_out, axis = 0)
        out = []
        for child_num in range(len(self._children)):
            out.append(inds[np.where(split_inds_assign == child_num)])
        return tuple(out)

    def f(self, X):
        return self._model._f(X)

    def grad_f(self, X, f_outs):
        return self._model._grad_f(X, f_outs)


    def _is_root(self):
        return self._parent is None


    def __depth(self):
        if self._is_root():
            return 0
        return 1 + self._parent.__depth()

    def __repr__(self):
        return "ID: " + str(self._ID)
