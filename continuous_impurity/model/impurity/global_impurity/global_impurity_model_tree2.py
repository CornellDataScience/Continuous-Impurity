import numpy as np
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity

class GlobalImpurityModelTree2:

    

    #is a dictionary where dict[node] is a vector v where v[i] is p(node|X[i])
    def calc_p_dict(head, X):
        def fill_p_dict(node, p_dict):
            if not node._is_root():
                parent_probs = p_dict[node._parent]
                split_to_node_probs = node._parent._f(node, X)
                p_dict[node] = parent_probs*split_to_node_probs
            for node_child in node._children:
                fill_p_dict(node_child, p_dict)
        p_dict = {head: np.ones(X.shape[0], dtype = X.dtype)}
        fill_p_dict(head, p_dict)
        return p_dict

    #is a dictionary with keys for all leaf nodes that are in the subtree rooted at q,
    #where dict[k] is grad(q's params) p(k|X), which is a dictionary dict2 where
    #dict2[p] is the gradient with respect to q's params_dict[p]
    def calc_grad_p_leaves_dict(q, p_dict, X):
        #could speed this up by keeping track of splits too in p_dict since
        #both calculate them twice (minor speedup for fairly large amount of
        #memory, likely)
        out = {}
        for q_child in q._children:
            q_child_leaves = q_child._get_leaves()
            q_child_split = q._f(q_child, X)
            grad_q_child_split = q._grad_f(q_child, X)

            for q_child_leaf in q_child_leaves:
                to_add = {}
                for param in grad_q_child_split:
                    to_add[param] = numpy_helper.fast_multiply_along_first_axis(\
                        p_dict[q_child_leaf]/q_child_split, \
                        grad_q_child_split[param])
                out[q_child_leaf] = to_add
        return out


    def train(head, X, y, iters, learn_rate):
        unique_labels = np.unique(y)
        nonleaves = head._get_nonleaves()
        for iter in range(iters):
            p_dict = GlobalImpurityModelTree2.calc_p_dict(head, X)
            for node in nonleaves:
                node._step_params(GlobalImpurityModelTree2.__calc_gradient(node, p_dict, X, y, unique_labels), -learn_rate)
            if iter % 100 == 0:
                GlobalImpurityModelTree2.__print_performance(head, p_dict, X, y)

    def __print_performance(head, p_dict, X, y):
        leaves = head._get_leaves()
        subset_assign_probs = np.zeros((X.shape[0], len(leaves)))
        for i in range(len(leaves)):
            subset_assign_probs[:,i] = p_dict[leaves[i]]
        print("EXPECTED GINI: ", impurity.expected_gini(subset_assign_probs, y))

    def __calc_gradient(q, p_dict, X, y, unique_labels):
        grad_p_dict = GlobalImpurityModelTree2.calc_grad_p_leaves_dict(q, p_dict, X)
        out = {p: np.zeros(q._model._params_dict[p].shape, dtype = q._model._params_dict[p].dtype) for p in q._model._params_dict}

        for k in q._get_leaves():
            u_k = GlobalImpurityModelTree2.__u(k, p_dict)
            v_k = GlobalImpurityModelTree2.__v(k, p_dict, y, unique_labels)
            grad_u_k = GlobalImpurityModelTree2.__grad_u(k, p_dict, grad_p_dict)
            grad_v_k = GlobalImpurityModelTree2.__grad_v(k, p_dict, grad_p_dict, y, unique_labels)
            for param in out:
                out[param] += v_k*grad_u_k[param] + u_k*grad_v_k[param]

        for param in out:
            out[param] *= -1.0/float(X.shape[0])
        return out


    def __u(k, p_dict):
        return 1.0/np.sum(p_dict[k])

    def __v(k, p_dict, y, unique_labels):
        out = 0
        for l in unique_labels:
            where_y_eq_l = np.where(y == l)
            out += np.square(np.sum(p_dict[k][where_y_eq_l]))
        return out

    def __grad_u(k, p_dict, grad_p_dict):
        denominator = np.square(np.sum(p_dict[k]))
        out = {}
        for param in grad_p_dict[k]:
            out[param] = -np.sum(grad_p_dict[k][param], axis = 0)/denominator
        return out

    def __grad_v(k, p_dict, grad_p_dict, y, unique_labels):
        #[1:] because the 0th axis is for indexing along X
        out = {p:np.zeros(grad_p_dict[k][p].shape[1:]) for p in grad_p_dict[k]}
        for l in unique_labels:
            where_y_eq_l = np.where(y==l)
            p_where_y_eq_l_sum = np.sum(p_dict[k][where_y_eq_l])
            for param in grad_p_dict[k]:
                out[param] += 2.0*p_where_y_eq_l_sum * np.sum(grad_p_dict[k][param][where_y_eq_l], axis = 0)

        return out
