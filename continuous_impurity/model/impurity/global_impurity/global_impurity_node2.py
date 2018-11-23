import numpy as np
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity

class GlobalImpurityNode2:

    def __init__(self, parent, model):
        self._parent = parent
        self._children = []
        self._model = model

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
            self._model._params_dict[p] += learn_rate*params_grad[p]

    def _is_root(self):
        return self._parent is None

    def _is_leaf(self):
        return len(self._children) == 0
