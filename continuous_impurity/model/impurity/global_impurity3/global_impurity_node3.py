import numpy as np
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import gc



class GlobalImpurityNode3:

    def __init__(self, parent, model, children = []):
        self.__parent = parent
        self.__children = children
        self.__model = model
        self._ID = None


    def to_list_and_set_IDs(head):
        nodes = GlobalImpurityNode3.__to_list(head)
        GlobalImpurityNode3.__set_IDs(nodes)
        return nodes

    def __to_list(head):
        def f(node, acc):
            acc.append(node)
            for child in node.__children:
                f(child, acc)
        out = []
        f(head, out)
        return out

    def get_leaves(nodes):
        out = []
        for node in nodes:
            if node.__is_leaf():
                out.append(node)
        return out

    def __set_IDs(nodes):
        for i in range(len(nodes)):
            nodes[i]._ID = i


    def train(self, X, y, learn_rate, n_iters, print_progress_iters = 25, GC_frequency = None):
        nodes = GlobalImpurityNode3.to_list_and_set_IDs(self)
        leaves = GlobalImpurityNode3.get_leaves(nodes)
        unique_labels = np.unique(y)
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l))

        for iter in range(n_iters):
            if GC_frequency is not None and iter%GC_frequency == 0:
                gc.collect()

            f_arr = GlobalImpurityNode3.calc_f_arr(nodes, X)
            grad_f_arr = GlobalImpurityNode3.calc_grad_f_arr(nodes, X, f_arr)
            p_arr = GlobalImpurityNode3.calc_p_arr(nodes, X, f_arr)
            grad_EG = GlobalImpurityNode3.calc_grad(X, y, unique_labels, where_y_eq_ls, nodes, leaves, f_arr, p_arr, grad_f_arr)

            for node_ID in range(len(grad_EG)):
                if grad_EG[node_ID] is not None:
                    for param_ind in range(len(grad_EG[node_ID])):
                        nodes[node_ID].__model._params[param_ind] -= learn_rate*grad_EG[node_ID][param_ind]


            if iter%print_progress_iters == 0:
                print("EXPECTED GINI: ", GlobalImpurityNode3.__expected_GINI(leaves, p_arr, y))
                print("----------------------------------")





    def __expected_GINI(leaves, p_arr, y):
        subset_assign_probs = np.zeros((y.shape[0], len(leaves)))
        for leaf_ind in range(len(leaves)):
            subset_assign_probs[:,leaf_ind] = p_arr[leaves[leaf_ind]._ID]
        return impurity.expected_gini(subset_assign_probs, y)

    #returns grad expected impurity of the whole tree w.r.t. all parameters of
    #node q
    def calc_grad(X, y, unique_labels, where_y_eq_ls, nodes, leaves, f_arr, p_arr, grad_f_arr):
        grad_EG = [None for i in range(len(nodes))]
        for q in nodes:
            grad_EG[q._ID] = None if q.__model is None else \
                [np.zeros(q_param.shape, dtype = q_param.dtype) for q_param in q.__model._params]
        grad_EG = tuple(grad_EG)




        for k in leaves:
            k_ID = k._ID
            assert(k_ID is not None)
            p_ks = p_arr[k_ID]
            grad_p_ks = GlobalImpurityNode3.calc_grad_p_arr(nodes, k_ID, p_arr, f_arr, grad_f_arr)


            p_sums_where_y_eq_ls = np.zeros(len(where_y_eq_ls))
            for y_eq_l_ind in range(len(where_y_eq_ls)):
                p_sums_where_y_eq_ls[y_eq_l_ind] = np.sum(p_ks[where_y_eq_ls[y_eq_l_ind]])

            p_sum = np.sum(p_sums_where_y_eq_ls)




            u_k = GlobalImpurityNode3.__calc_u(p_sum)
            v_k = GlobalImpurityNode3.__calc_v(p_sums_where_y_eq_ls)



            for (q_ID, grad_q) in grad_p_ks:

                grad_q_sums = []
                grad_q_sums_where_y_eq_ls = []
                for param_ind in range(len(grad_q)):
                    iter_add = np.zeros((len(where_y_eq_ls),) + grad_q[param_ind].shape[1:])
                    for l_ind in range(len(where_y_eq_ls)):
                        iter_add[l_ind] = np.sum(grad_q[param_ind][where_y_eq_ls[l_ind]], axis = 0)


                    grad_q_sums.append(np.sum(iter_add, axis = 0))
                    grad_q_sums_where_y_eq_ls.append(iter_add)





                grad_u_k = GlobalImpurityNode3.__calc_grad_u(p_sum, grad_q_sums)
                grad_v_k = GlobalImpurityNode3.__calc_grad_v(p_sums_where_y_eq_ls, grad_q_sums_where_y_eq_ls)#GlobalImpurityNode3.__calc_grad_v(p_sums_where_y_eq_ls, grad_q, y, unique_labels)#

                for param_ind in range(len(grad_EG[q_ID])):
                    grad_EG[q_ID][param_ind] -=  (v_k*grad_u_k[param_ind] + \
                        u_k*grad_v_k[param_ind])/float(X.shape[0])

        return grad_EG


    def __calc_u(p_k_sum):
        return 1.0/p_k_sum

    def __calc_v(p_sums_where_y_eq_ls):
        return np.sum(np.square(p_sums_where_y_eq_ls))


    def __calc_grad_u(p_sum, grad_p_sums):
        out = []
        denominator = p_sum*p_sum
        for param_grad_sum in grad_p_sums:
            numerator = -param_grad_sum
            out.append(numerator/denominator)
        return out




    def __calc_grad_v(p_sums_where_y_eq_ls, grad_p_sums_where_y_eq_ls):
        out = []

        for param_grad_sums in grad_p_sums_where_y_eq_ls:
            v_param = 0
            for l_ind in range(len(param_grad_sums)):
                #l = unique_labels[l_ind]
                #where_y_eq_l = np.where(y == l)
                left = p_sums_where_y_eq_ls[l_ind]#np.sum(p_ks[where_y_eq_l])
                right = param_grad_sums[l_ind]#np.sum(param_grad[where_y_eq_l], axis = 0)
                v_param += left*right

            v_param *= 2.0
            out.append(v_param)
        return out


    def f(self, X):
        return self.__model._f(X)

    def grad_f(self, X, f_outs):
        return self.__model._grad_f(X, f_outs)

    def calc_f_arr(nodes, X):
        out = [None for i in range(len(nodes))]
        for node in nodes:
            if not node.__is_leaf():
                out[node._ID] = node.f(X)
        return out

    def calc_grad_f_arr(nodes, X, f_arr):
        out = [None for i in range(len(nodes))]
        for node in nodes:
            if not node.__is_leaf():
                out[node._ID] = node.grad_f(X, f_arr[node._ID])
        return out


    def calc_p_arr(nodes, X, f_arr):
        head = nodes[0]
        out = np.zeros((len(nodes), X.shape[0]))
        out[head._ID] = np.ones(X.shape[0])

        #assumes out[node._ID] is already calculated
        def p(node):
            ps = out[node._ID]
            node_f_outs = f_arr[node._ID]
            for child_ind in range(len(node.__children)):
                child = node.__children[child_ind]
                child_ID = child._ID
                out[child_ID] = node_f_outs[child_ind]*ps
                p(child)
            return None
        p(head)
        return out




    def calc_grad_p_arr(nodes, leaf_ID, p_arr, f_arr, grad_f_arr):
        #returns an array, L, s.t. L[i] contains information about the ith
        #relevant node to the path to leaf_ID from the head. L[i] is (id, gradient),
        #where gradient is a list of gradients for each param in nodes[id]

        assert(nodes[leaf_ID].__is_leaf()), "nodes[leaf_ID] is not a leaf..."
        out = []
        p_leaf = p_arr[leaf_ID]
        q = nodes[leaf_ID].__parent
        q_c = nodes[leaf_ID]
        while q is not None:
            q_c_ind = q.__child_ind(q_c)
            f_q_c = f_arr[q._ID][q_c_ind]
            grad_f_q = grad_f_arr[q._ID]
            iter_out = []
            for param_ind in range(len(grad_f_q)):
                grad_f_param_q_c = grad_f_q[param_ind][q_c_ind]
                grad_leaf_wrt_q = numpy_helper.stable_divide(p_leaf, f_q_c, 0)[:,np.newaxis] * grad_f_param_q_c


                iter_out.append(grad_leaf_wrt_q)
            out.append((q._ID, iter_out))
            q_c = q
            q = q.__parent

        return out





    def __child_ind(self, child):
        return self.__children.index(child)

    def _add_children(self, new_children):
        if not isinstance(new_children, list):
            new_children = [new_children]
        assert(len(new_children) + len(self.__children) <= 2), ("children would've been: " + str(len(new_children) + len(self.__children)))

        self.__children.extend(new_children)

    def __is_root(self):
        return self.__parent is None


    def __is_leaf(self):
        is_leaf = len(self.__children) == 0
        if is_leaf:
            assert(self.__model is None)
        return is_leaf




#functions below this line are not technicaly optimal time (e.g. depth not being constant time, etc.)

    def __depth(self):
        if self.__is_root():
            return 0
        return 1 + self.__parent.__depth()


    def __repr__(self):
        return "ID: " + str(self._ID)