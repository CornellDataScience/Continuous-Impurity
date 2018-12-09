import numpy as np
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import gc
from performance.stopwatch_profiler import StopwatchProfiler



class GlobalImpurityNode3:
    #originally had children = [] as default argument, but turns out that whenever
    #that when self._children = children, and then self._children is modified,
    #the default argument children will also be modified since self._children points
    #to it.
    def __init__(self, parent, model):
        self._parent = parent
        self._children = []
        self._model = model
        self._ID = None
        self._leaf_predict = None



    def fold(self, f, acc):
        def traverse(node, acc):
            acc = f(node, acc)
            if node._is_leaf():
                return acc
            for child in node._children:
                acc = traverse(child, acc)
            return acc

        return traverse(self, acc)

    def fold_in_place(self, f):
        def traverse(node):
            f(node)
            for child in node._children:
                traverse(child)
        traverse(self)

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

    def _child_ind(self, child):
        return self._children.index(child)

    def _add_children(self, new_children):
        if not isinstance(new_children, list):
            new_children = [new_children]
        assert(len(new_children) + len(self._children) <= 2), ("children would've been: " + str(len(new_children) + len(self._children)))

        self._children.extend(new_children)

    def __is_root(self):
        return self._parent is None


    def _is_leaf(self):
        is_leaf = len(self._children) == 0
        if is_leaf:
            assert(self._model is None)
        return is_leaf

    def __depth(self):
        if self.__is_root():
            return 0
        return 1 + self._parent.__depth()


    def __repr__(self):
        return "ID: " + str(self._ID)
