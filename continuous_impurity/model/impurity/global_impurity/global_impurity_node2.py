import numpy as np
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity

class GlobalImpurityNode2:

    def __init__(self, parent, model):
        self._parent = parent
        self._children = []
        self._model = model
        self._leaf_predict = None


    '''
    returns dictionary D s.t. D[node_obj] = A, a 1.d. numpy array of all
        indices i s.t. X[i] falls through this node during the classification
        process.

    May want to do some rewriting in predict, given this does something fairly similar,
    but is clunkier and would require predicting being moved into the
    container class for the head, since only it "knows" about all leaves of
    the model
    '''
    def _fill_node_ind_dict(self, dict, X, in_node_inds):
        dict[self] = in_node_inds
        if self._is_leaf():
            return None
        inds_splits = self.__split(X, in_node_inds)
        for child_num in range(len(self._children)):
            self._children[child_num]._fill_node_ind_dict(dict, X, inds_splits[child_num])


    def _predict(self, X, inds, predictions):
        if self._is_leaf():
            predictions[inds] = self._leaf_predict
            return None
        splits = self.__split(X, inds)
        for child_num in range(len(self._children)):
            self._children[child_num]._predict(X, splits[child_num], predictions)


    def __split(self, X, inds):
        X_inds = X[inds]
        f_X_inds_outs = np.column_stack([self._model._f(k, X_inds) for k in range(len(self._children))])
        X_inds_assigned_child = np.argmax(f_X_inds_outs, axis = 1)
        splits = []
        for child_num in range(len(self._children)):
            splits.append(inds[np.where(X_inds_assigned_child == child_num)])
        return splits


    def add_children(self, *to_add):
        self._children.extend(to_add)

    def _to_list(self):
        def build(node, acc):
            acc.append(node)
            for child in node._children:
                build(child, acc)
        out = []
        build(self, out)
        return out

    def _get_leaves(self):
        out = []
        for node in self._to_list():
            if node._is_leaf():
                out.append(node)
        return out

    def _get_nonleaves(self):
        out = []
        for node in self._to_list():
            if not node._is_leaf():
                out.append(node)
        return out

    def _f(self, child, X):
        return self._model._f(self.__child_index(child), X)

    def _grad_f(self, child, X):
        return self._model._grad_f(self.__child_index(child), X)

    def __child_index(self, child):
        return self._children.index(child)

    def _step_params(self, params_grad, learn_rate):
        for p in params_grad:
            self._model._params_dict[p] -= learn_rate*params_grad[p]

    def _is_root(self):
        return self._parent is None

    #is O(max depth) complexity.
    def _depth(self):
        if self._is_root():
            return 0
        return 1 + self._parent._depth()

    def _is_leaf(self):
        out = len(self._children) == 0
        if out:
            assert(self._model is None)
        else:
            assert(self._model is not None)
        return out
