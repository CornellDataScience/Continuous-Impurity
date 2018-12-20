import numpy as np
import function.stable_func as stable_func

class RelaxedGlobalSigmoidImpurityTree:

    #assumes x is affined
    def __init__(self, x_length, depth):
        self.__depth = depth
        self.__head = ModelTree.construct(x_length, self.__depth)
        self.__leaves = self.__head._get_leaves()
        self.__set_leaf_ids()
        self.__set_rooted_leaves()

    def __set_leaf_ids(self):
        for i in range(len(self.__leaves)):
            self.__leaves[i].__leaf_id = i

    def __set_rooted_leaves(self):
        def f(node):
            if node._is_leaf():
                node_leaves = node._get_leaves()
                node.__rooted_leaves = np.zeros(len(node_leaves), dtype = np.int)
                for i in range(len(node_leaves)):
                    node.__rooted_leaves[i] = node_leaves[i].__leaf_id
        self.__head.fold_in_place(f)

    '''all that is needed are leaf probabilities and all split probabilities'''
    def calc_leaf_probs(self, x, p0):
        out = np.zeros(len(self.__leaves), dtype = np.float64)

        def autofill_leaves(node, p_parent, d):
            out[node.__rooted_leaves] = p_parent*(0.5**(self.__depth - d + 1))

        x_wrapped = np.asarray([x])
        def f(node, p_parent, d):
            parent_f_out = node._parent._f(x_wrapped)

            if node._is_leaf():
                child_ind = node._parent._child_ind(node)
                out[node.__leaf_id] = p_parent * parent_f_out[child_ind]
                return None

            for child_ind in range(len(parent_f_out)):
                if parent_f_out[child_ind][0] >= 0.5 or p_parent > p0:
                    f(node._children[child_ind], p_parent * parent_f_out[child_ind][0], d+1)
                else:
                    autofill_leaves(node._children[child_ind], p_parent * parent_f_out[child_ind][0], d+1)
        for head_child in self.__head._children:
            f(head_child, 1.0, p0)

        assert(np.sum(out) >= .99 and np.sum(out) <= 1.01)
        return out







from model.impurity.global_impurity3.node3 import Node3

class BinaryTree(Node3):

    def __init__(self, parent):
        Node3.__init__(self, [])
        self._parent = parent

    def _get_leaves(self):
        acc = []
        def f(node):
            if node._is_leaf():
                acc.append(node)
        self.fold_in_place(f)
        return acc


    def _is_root(self):
        return self._parent is None




class ModelTree(BinaryTree):

    def __init__(self, parent, theta):
        BinaryTree.__init__(self, parent)
        self.__theta = theta

        self.__leaf_id = None#TODO

    def __rand_theta(x_length):
        return 0.0001*(np.random.rand(x_length) - 0.5)

    def construct(x_length, max_depth):
        def f(node, depth):
            if depth >= max_depth:
                return None
            if depth == max_depth - 1:
                node._add_children([ModelTree(node, None), \
                    ModelTree(node, None)])
            else:
                node._add_children([ModelTree(node, ModelTree.__rand_theta(x_length)), \
                    ModelTree(node, ModelTree.__rand_theta(x_length))])
            for child in node._children:
                f(child, depth+1)
        head = ModelTree(None, ModelTree.__rand_theta(x_length))
        f(head, 0)
        return head

    def _f(self, X):
        left = stable_func.sigmoid(np.dot(X, self.__theta))
        return np.asarray([left, 1-left])
