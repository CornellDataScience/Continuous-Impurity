import numpy as np
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
from performance.stopwatch_profiler import StopwatchProfiler

class GlobalImpurityModelTree2:

    def __init__(self, head):
        self.__head = head


    #is a dictionary where dict[node] is a vector v where v[i] is p(node|X[i])
    def calc_p_dict(self, X):
        def fill_p_dict(node, p_dict):
            if not node._is_root():
                parent_probs = p_dict[node._parent]
                split_to_node_probs = node._parent._f(node, X)
                p_dict[node] = parent_probs*split_to_node_probs
            for node_child in node._children:
                fill_p_dict(node_child, p_dict)
        p_dict = {self.__head: np.ones(X.shape[0], dtype = X.dtype)}
        fill_p_dict(self.__head, p_dict)
        return p_dict

    #is a dictionary with keys for all leaf nodes that are in the subtree rooted at q,
    #where dict[k] is grad(q's params) p(k|X), which is a dictionary dict2 where
    #dict2[p] is the gradient with respect to q's params_dict[p]
    def calc_grad_p_leaves_dict(self, q, p_dict, X):
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



    def __print_performance(self, p_dict, X, y):
        leaves = self.__head._get_leaves()
        subset_assign_probs = np.zeros((X.shape[0], len(leaves)))
        for i in range(len(leaves)):
            subset_assign_probs[:,i] = p_dict[leaves[i]]
        print("EXPECTED GINI: ", impurity.expected_gini(subset_assign_probs, y))



    def train(self, X, y, iters, learn_rate):
        unique_labels = np.unique(y)
        nonleaves = self.__head._get_nonleaves()
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l))
        for iter in range(iters):
            p_dict = self.calc_p_dict(X)
            for node in nonleaves:
                node._step_params(self.__calc_gradient(node, p_dict, X, y, unique_labels, where_y_eq_ls), learn_rate)
            if iter % 100 == 0:
                print("iter: ", iter)
                self.__print_performance( p_dict, X, y)
                print("------------------------------------------")

    #lots summing over p(k|X) happens a lot -- maybe pass in a dict with these
    #values so they are precalculated?
    def __calc_gradient(self, q, p_dict, X, y, unique_labels, where_y_eq_ls):
        #check how expensive this is
        grad_p_dict = self.calc_grad_p_leaves_dict(q, p_dict, X)

        param_shape_dict = {p:q._model._params_dict[p].shape for p in q._model._params_dict}
        out = {p: np.zeros(q._model._params_dict[p].shape, dtype = q._model._params_dict[p].dtype) for p in q._model._params_dict}
        profiler = StopwatchProfiler()
        for k in q._get_leaves():
            profiler.start()


            p_dict_sum_k = np.sum(p_dict[k], axis = 0)

            grad_p_dict_sum_k = {}
            for param in grad_p_dict[k]:
                grad_p_dict_sum_k[param] = np.sum(grad_p_dict[k][param], axis = 0)

            p_dict_sum_k_where_y_eq_ls = []
            grad_p_dict_sum_k_where_y_eq_ls = []
            for l_ind in range(len(where_y_eq_ls)):
                p_dict_sum_k_where_y_eq_ls.append(np.sum(p_dict[k][where_y_eq_ls[l_ind]]))

                grad_p_dict_sum_k_where_y_eq_l_ind = {}
                for param in grad_p_dict[k]:
                    grad_p_dict_sum_k_where_y_eq_l_ind[param] = np.sum(grad_p_dict[k][param][where_y_eq_ls[l_ind]], axis = 0)
                grad_p_dict_sum_k_where_y_eq_ls.append(grad_p_dict_sum_k_where_y_eq_l_ind)



            profiler.lap("precalculation finished")




            u_k = self.__u(p_dict_sum_k)
            profiler.lap("u_k calculated")

            v_k = self.__v(p_dict_sum_k_where_y_eq_ls)
            profiler.lap("v_k calculated")

            grad_u_k = self.__grad_u(p_dict_sum_k, grad_p_dict_sum_k)
            profiler.lap("grad_u_k calculated")



            grad_v_k = self.__grad_v(param_shape_dict, p_dict_sum_k_where_y_eq_ls, grad_p_dict_sum_k_where_y_eq_ls)
            profiler.lap("grad_v_k calculated")

            for param in out:
                out[param] += v_k*grad_u_k[param] + u_k*grad_v_k[param]

            profiler.lap("params stepped")
            profiler.stop()

            #print("Relative profiles: ", profiler.relative_lap_deltas())
            #print("----------------------")

            profiler.reset()
        for param in out:
            out[param] *= -1.0/float(X.shape[0])
        return out


    def __u(self, p_dict_sum_k):
        return 1.0/p_dict_sum_k

    def __v(self, p_dict_sum_k_where_y_eq_ls):
        out = 0
        for l_ind in range(len(p_dict_sum_k_where_y_eq_ls)):
            sqrt_out_plus = p_dict_sum_k_where_y_eq_ls[l_ind]
            out += sqrt_out_plus*sqrt_out_plus
        return out

    def __grad_u(self, p_dict_sum_k, grad_p_dict_sum_k):
        denominator = np.square(p_dict_sum_k)
        out = {}
        for param in grad_p_dict_sum_k:
            out[param] = -grad_p_dict_sum_k[param]/denominator
        return out

    def __grad_v(self, param_shape_dict, p_dict_sum_k_where_y_eq_ls, grad_p_dict_sum_k_where_y_eq_ls):
        out = {p:np.zeros(param_shape_dict[p]) for p in param_shape_dict}
        for l_ind in range(len(p_dict_sum_k_where_y_eq_ls)):
            p_where_y_eq_l_sum = p_dict_sum_k_where_y_eq_ls[l_ind]

            for param in grad_p_dict_sum_k_where_y_eq_ls[l_ind]:
                out[param] += 2.0*p_where_y_eq_l_sum * \
                    grad_p_dict_sum_k_where_y_eq_ls[l_ind][param]
        return out
