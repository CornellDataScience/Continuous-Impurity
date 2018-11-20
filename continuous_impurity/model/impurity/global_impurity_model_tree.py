import numpy as np
from abc import ABC, abstractmethod
import function.impurity as impurity
#TODO: Implement this with lots of for loops for simplicity :(
#TODO: Prune tree after training in order to prevent arbitray split planes when
#a node desn'trequire more children for further separation, but is forced to train
#them anyway

#TODO: Prune tree after training to eliminate nodes that have 0 training data
#fall into them.

#TODO: Find a way to initialize weights such that leaves all have X falling into
#them -- probably will speed up train time

#For now: Implementing this in such a way that it is easy to convert it into
#abstract class, but has implemented the functions in this class
class GlobalImpurityModelTree:

    def __init__(self, head):
        self._head = head

    def train(self, X, y, iters, learn_rate):
        unique_labels = np.unique(y)
        for iter in range(iters):
            head_nonleaves = self._head._get_non_leaves()
            leaves = self._head._get_leaves()
            head_clone = self._head._clone(None)
            head_clone_nonleaves = head_clone._get_non_leaves()
            for q_ind in range(len(head_nonleaves)):
                q = head_nonleaves[q_ind]
                q_grad = self._calc_gradient(q, X, y, unique_labels, leaves)
                head_clone_nonleaves[q_ind]._step_params(-learn_rate*q_grad)
            self._head = head_clone
            if iter%100 == 0:
                print("iter: ", iter)
                print("falling leaves: ", self._leaf_maxes(X))
                print("expected GINI: ", impurity.expected_gini(self._leaf_probs(X), y))
                print("----------------------------------")
        self.__set_leaf_labels(X, y)

    def __set_leaf_labels(self, X, y):
        len_leaves = len(self._head._get_leaves())
        leaf_prob_maxes = self._leaf_maxes(X)
        self.__leaf_labels = np.zeros(len_leaves, dtype = np.int)
        for leaf_ind in range(self.__leaf_labels.shape[0]):
            where_leaf_prob_maxes_eq_leaf_ind = np.where(leaf_prob_maxes == leaf_ind)
            y_in_leaf = y[where_leaf_prob_maxes_eq_leaf_ind]
            unq, counts = np.unique(y_in_leaf, return_counts = True)
            if len(unq) == 0:
                #NO LABELS FALL INTO THIS LEAF, MEANING THAT EITHER EITHER THE OTHER
                #SIBLING SHOULD REPLACE THE PARENT OF THIS LEAF, OR
                #THE PARENT SHOULD BE SET TO A LEAF (PROBABLY DEPENDS ON WHETHER
                #LOTS OF X FALL THROUGH PARENT. IF NOT MUCH, JUST MAKE PARENT ROOT,
                #IF LOTS FALL THROUGH PARENT, THEN THIS SUGGESTS THE MODEL DID NOT
                #FINISH CONVERGING, BUT COULD BE CHEESED BY SETTING PARENT TO OTHER
                #SIBLING)
                raise ValueError("leaf with no X falling through it. TODO: Fix this case")

            self.__leaf_labels[leaf_ind] = unq[np.argmax(counts)]

    def predict(self, X):
        leaf_prob_maxes = self._leaf_maxes(X)
        return self.__leaf_labels[leaf_prob_maxes]



    def _calc_gradient(self, q, X, y, unique_labels, leaves):
        out = np.zeros(q._model._params.shape, dtype = q._model._params.dtype)
        for k in leaves:
            #otherwise this gradient term would be zero. If causing problems,
            #should be able to remove this check since the gradient of root
            #is 0
            if k._is_super_parent(q):
                p_k = self._p_X(k, X)
                grad_p_k = self._grad_p_X(q, k, X)
                u_k = self._u(p_k)
                v_k = self._v(p_k, y, unique_labels)
                grad_u_k = self._grad_u(p_k, grad_p_k)
                grad_v_k = self._grad_v(p_k, grad_p_k, y, unique_labels)
                #print("p_k: ", p_k)
                #print("grad_p_k: ", grad_p_k)
                #print("u_k: ", u_k)
                #print("v_k: ", v_k)
                #print("grad_u_k: ", grad_u_k)
                #print("grad_v_k: ", grad_v_k)
                out += v_k*grad_u_k + u_k*grad_v_k
        return (-1.0/float(X.shape[0]))*out

    def _leaf_probs(self, X):
        leaves = self._head._get_leaves()
        out = np.zeros((X.shape[0], len(leaves)), dtype = X.dtype)
        for k_ind in range(len(leaves)):
            out[:,k_ind] = self._p_X(leaves[k_ind], X)
        return out

    def _leaf_maxes(self, X):
        return np.argmax(self._leaf_probs(X), axis = 1)

    def _p_X(self, node, X):
        return np.array([self._p(node, X[i]) for i in range(X.shape[0])])

    def _grad_p_X(self, q, k, X):
        assert(k._is_super_parent(q)), "_grad_p assumes q is a super parent of k"
        return np.array([self._grad_p(q, k, X[i]) for i in range(X.shape[0])])



    def _u(self, p_k):
        return 1.0/np.sum(p_k)

    def _v(self, p_k, y, unique_labels):
        out = 0
        for l in unique_labels:
            where_y_eq_l = np.where(y == l)
            out += np.square(np.sum(p_k[where_y_eq_l], axis = 0))
        return out

    def _grad_u(self, p_k, grad_p_k):
        return -np.sum(grad_p_k, axis = 0)/np.square(np.sum(p_k))

    def _grad_v(self, p_k, grad_p_k, y, unique_labels):
        out = np.zeros(grad_p_k.shape[1], dtype = grad_p_k.dtype)
        for l in unique_labels:
            where_y_eq_l = np.where(y == l)
            out += np.sum(p_k[where_y_eq_l])*np.sum(grad_p_k[where_y_eq_l], axis = 0)
        out *= 2
        return out



    #where k is a node
    def _p(self, k, x):
        if k._is_root():
            return 1.0
        k_parent = k._parent
        return k_parent._f(k, x)*self._p(k_parent, x)

    #is grad(node q params, an ndarray) _p(k, x)
    #assumes q is reachable by travelling across parents starting from k
    def _grad_p(self, q, k, x):
        if k._is_root():
            return np.zeros(q._model._params.shape, dtype = q._model._params.dtype)
        k_parent = k._parent
        if k_parent == q:
            return self._p(k_parent, x)*k_parent._grad_f(k,x)
        return k_parent._f(k,x)*self._grad_p(q,k_parent,x)


class Node:
    #where model must be a NodeModel, and MUST be None if this node is a leaf
    def __init__(self, parent, model):
        self._parent = parent
        self._model = model
        self._children = []

    def add_child(self, child):
        assert(self._model is not None)
        assert(child._parent == self)
        self._children.append(child)

    def _init_with_children(self, parent, model, children):
        out = Node(parent, model)
        for child in children:
            out._children.append(child)
        return out

    def _step_params(self, step):
        assert(step.shape == self._model._params.shape)
        self._model._params += step

    #where k is a node that is in self._children
    def _f(self, k, x):
        return self._model._model_func(self._child_num(k), x)

    def _grad_f(self, k, x):
        return self._model._grad_model_func(self._child_num(k), x)

    def _is_root(self):
        return self._parent is None

    def _is_leaf(self):
        out = len(self._children) == 0
        if out:
            assert self._model is None
        return out

    def _child_num(self, child):
        return self._children.index(child)

    def _get_super_parents(self):
        def f(n, acc):
            if n._is_root():
                acc.append(n)
                return acc
            acc.append(n)
            return f(n._parent, acc)
        return f(self, [])

    def _is_super_parent(self, parent_node):
        super_parents = self._get_super_parents()
        for super_parent in super_parents:
            if parent_node == super_parent:
                return True
        return False

    def _get_leaves(self):
        def iter_get_leaves(node, acc):
            if node._is_leaf():
                acc.append(node)
                return None
            for child in node._children:
                iter_get_leaves(child, acc)
        out = []
        iter_get_leaves(self, out)
        return out

    def _get_non_leaves(self):
        def iter_get_non_leaves(node, acc):
            if not node._is_leaf():
                acc.append(node)
                for child in node._children:
                    iter_get_non_leaves(child, acc)
            return None
        out = []
        iter_get_non_leaves(self, out)
        return out

    def _clone(self, parent = None):
        model_clone = None if self._is_leaf() else self._model._clone()
        head_clone = Node(parent, model_clone)
        for child in self._children:
            head_clone._children.append(child._clone(parent = head_clone))
        return head_clone

    def _to_list(self):
        def f(n, acc):
            if n._is_leaf():
                acc.append(n)
                return None
            acc.append(n)
            for child in n._children:
                f(child, acc)
        out = []
        f(self, out)
        return out


    def __str__(self):
        return "(Leaf: " + str(self._is_leaf()) + \
        ", Root: " + str(self._is_root()) + \
        ", ID: " + str(hex(id(self))) + ")\n"

    def __repr__(self):
        return self.__str__()

class NodeModel:
    #model_func must be of the form f(child_node_num, x)
    #grad_model_func must be grad (params) f(child_node_num, x)
    def __init__(self, params, model_func, grad_model_func):
        self._params = params
        self.__func = model_func
        self.__grad_func = grad_model_func

    def _model_func(self, subset_ind, x):
        return self.__func(self._params, subset_ind, x)

    def _grad_model_func(self, subset_ind, x):
        return self.__grad_func(self._params, subset_ind, x)

    def _clone(self):
        return NodeModel(self._params.copy(), self.__func, self.__grad_func)
