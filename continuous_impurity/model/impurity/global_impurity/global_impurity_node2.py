import numpy as np

class GlobalImpurityNode2:

    def __init__(self, parent, model):
        self._parent = parent
        self._children = []
        self.__model = model

    '''
    - p_X_dict[n] (n is a node) is (v, r) where vector v is s.t. v[i] is
      p(n|X[i]) starting from head as root, and r is the transformed version of
      X as it was inputted to n

    Requires: p_X_dict and X_dict has a value for head by default
    Postconditions: p_dict will completely filled for all nodes (including leaves)
    '''
    #if run into memory/maybe speed problems, may want to set the X's of p_X_dict to
    #Nones if guaranteed never to need them again? Or split p_X_dict into two
    #and remove X's from X_dict? (so long as later processes don't rely on
    #having the X's of p_X_dict)
    def __fill_grad_p_X_dict(node, p_X_dict):
        if node._is_root():
            assert(node in p_X_dict)
        elif node._parent in p_X_dict:
            parent_container = p_X_dict[node._parent]
            parent_p = parent_container.p
            parent_L = parent_container.L
            node_p, node_L = node._parent._prob_child_split(node, parent_L)
            node_p *= parent_p
            p_X_dict[node] = NodePXContainer(node_p, node_L)
        else:
            raise ValueError("Made some bug, this shouldn't be called")
        for node_child in node._children:
            GlobalImpurityNode2.__fill_grad_p_X_dict(node_child, p_X_dict)

    

    '''
    returns a dictionary, D, s.t. D[node k] is (p(k|X), the input X of the node)
    '''
    def p_nodes(head, X):
        out = {head: NodePXContainer(np.ones(X.shape[0], dtype = X.dtype), X)}
        GlobalImpurityNode2.__fill_grad_p_X_dict(head, out)
        return out



    def _prob_child_split(self, child, X):
        child_num = self._child_num(child)
        out, X_transformed = self.__model._func(child_num, X)
        return (out, X_transformed)

    def _grad_prob_child_split(self, child, X):
        child_num = self._child_num(child)
        grad_out, grad_X_transformed = self.__model._grad_func(child_num, X)
        return (grad_out, grad_X_transformed)

    def add_children(self, *children):
        for child in children:
            assert(child._parent == self)
            self._children.append(child)

    def _child_num(self, child):
        return self._children.index(child)

    def _is_root(self):
        return self._parent is None

    def _is_leaf(self):
        return len(self._children) == 0

class NodePXContainer:

    def __init__(self, p, L):
        self.p = p
        self.L = L

    def __str__(self):
        return "Prob: " + str(self.p) + ", L: " + str(self.L)

    def __repr__(self):
        return str(self)
