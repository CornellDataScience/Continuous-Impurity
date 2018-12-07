import numpy as np
import toolbox.numpy_helper as numpy_helper
from performance.stopwatch_profiler import StopwatchProfiler







def take_gradient_descent_step(head, X, y, learn_rate, unique_labels, where_y_eq_ls, leaves, nonleaves):
    stopwatch = StopwatchProfiler()
    stopwatch.start()

    p_dict = calc_p_dict(head, X)

    stopwatch.lap("p_dict calcd")

    f_dict = calc_f_dict(head, X)

    stopwatch.lap("f_dict calcd")

    p_sums_dict = {n: np.sum(p_dict[n], axis = 0) for n in leaves}
    p_sums_where_y_eq_ls_dict = {}
    for n in leaves:
        iter_add = []
        for l_ind in range(len(where_y_eq_ls)):
            iter_add.append(np.sum(p_dict[n][where_y_eq_ls[l_ind]]))
        p_sums_where_y_eq_ls_dict[n] = iter_add

    stopwatch.lap("P_sums calcd")

    for node in nonleaves:
        node_grad = calc_gradient(node, p_dict, f_dict, p_sums_dict, p_sums_where_y_eq_ls_dict, X, where_y_eq_ls)#self.__calc_gradient(node, p_dict, p_sums_dict, p_sums_where_y_eq_ls_dict, X, where_y_eq_ls)
        node._step_params(node_grad, learn_rate)
    stopwatch.lap("node calc_gradients calcd")
    stopwatch.stop()
    print("relative take_gradient_descent_step lap deltas: ", stopwatch.relative_lap_deltas())
    stopwatch.reset()


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

def calc_f_dict(head, X):
    def fill_f_dict(node, f_dict):

        node_elem = {}
        for node_child in node._children:
            node_elem[node_child] = (node._f(node_child, X), node._grad_f(node_child, X))
            fill_f_dict(node_child, f_dict)
        f_dict[node] = node_elem
    f_dict = {}
    fill_f_dict(head, f_dict)
    return f_dict

#is a dictionary with keys for all leaf nodes that are in the subtree rooted at q,
#where dict[k] is grad(q's params) p(k|X), which is a dictionary dict2 where
#dict2[p] is the gradient with respect to q's params_dict[p].
#- where f_dict[k] is a dict where dict[k_child] is (f_k(k_child|X),
#  grad f_k(k_child|X) )
def calc_grad_p_leaves_dict(q, p_dict, f_dict, X):
    out = {}
    for q_child in q._children:
        q_child_leaves = q_child._get_leaves()
        q_child_split = f_dict[q][q_child][0]
        grad_q_child_split = f_dict[q][q_child][1]

        for q_child_leaf in q_child_leaves:
            to_add = {}
            left_mul_val = numpy_helper.stable_divide(p_dict[q_child_leaf], q_child_split, 0)
            for param in grad_q_child_split:
                to_add[param] = numpy_helper.fast_multiply_along_first_axis(\
                    left_mul_val,\
                    grad_q_child_split[param])

            out[q_child_leaf] = to_add
    return out


def calc_gradient(q, p_dict, f_dict, p_sums_dict, p_sums_where_y_eq_ls_dict, X, where_y_eq_ls):
    grad_p_dict = calc_grad_p_leaves_dict(q, p_dict, f_dict, X)

    param_shape_dict = {p:q._model._params_dict[p].shape for p in q._model._params_dict}
    out = {p: np.zeros(q._model._params_dict[p].shape, dtype = q._model._params_dict[p].dtype) for p in q._model._params_dict}

    for k in q._get_leaves():
        grad_p_dict_sum_k = {}
        for param in grad_p_dict[k]:
            grad_p_dict_sum_k[param] = np.sum(grad_p_dict[k][param], axis = 0)

        grad_p_dict_sum_k_where_y_eq_ls = []
        for l_ind in range(len(where_y_eq_ls)):
            grad_p_dict_sum_k_where_y_eq_l_ind = {}
            for param in grad_p_dict[k]:
                grad_p_dict_sum_k_where_y_eq_l_ind[param] = np.sum(grad_p_dict[k][param][where_y_eq_ls[l_ind]], axis = 0)
            grad_p_dict_sum_k_where_y_eq_ls.append(grad_p_dict_sum_k_where_y_eq_l_ind)

        u_k = __u(p_sums_dict[k])
        v_k = __v(p_sums_where_y_eq_ls_dict[k])
        grad_u_k = __grad_u(p_sums_dict[k], grad_p_dict_sum_k)
        grad_v_k = __grad_v(param_shape_dict,p_sums_where_y_eq_ls_dict[k], grad_p_dict_sum_k_where_y_eq_ls)

        for param in out:
            out[param] += v_k*grad_u_k[param] + u_k*grad_v_k[param]

    for param in out:
        out[param] *= -1.0/float(X.shape[0])
    return out

def __u(p_dict_sum_k):
    return 1.0/p_dict_sum_k

def __v(p_dict_sum_k_where_y_eq_ls):
    out = 0
    for l_ind in range(len(p_dict_sum_k_where_y_eq_ls)):
        sqrt_out_plus = p_dict_sum_k_where_y_eq_ls[l_ind]
        out += sqrt_out_plus*sqrt_out_plus
    return out

def __grad_u(p_dict_sum_k, grad_p_dict_sum_k):
    denominator = p_dict_sum_k*p_dict_sum_k
    out = {}
    for param in grad_p_dict_sum_k:
        out[param] = -grad_p_dict_sum_k[param]/denominator
    return out

def __grad_v(param_shape_dict, p_dict_sum_k_where_y_eq_ls, grad_p_dict_sum_k_where_y_eq_ls):
    out = {p:np.zeros(param_shape_dict[p]) for p in param_shape_dict}
    for l_ind in range(len(p_dict_sum_k_where_y_eq_ls)):
        p_where_y_eq_l_sum = p_dict_sum_k_where_y_eq_ls[l_ind]

        for param in grad_p_dict_sum_k_where_y_eq_ls[l_ind]:
            out[param] += 2.0*p_where_y_eq_l_sum * \
                grad_p_dict_sum_k_where_y_eq_ls[l_ind][param]
    return out
