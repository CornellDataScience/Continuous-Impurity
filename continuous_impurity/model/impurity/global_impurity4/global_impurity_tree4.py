import numpy as np
import function.stable_func as stable_func
import function.impurity as impurity
import model.impurity.global_impurity4.array_binary_tree as arr_tree
import toolbox.numpy_helper as numpy_helper
from performance.stopwatch_profiler import StopwatchProfiler

class GlobalImpurityTree4:


    def __init__(self, params_tree, split_func, d_split_func):
        self.__params_tree = params_tree
        self.__split_func = split_func
        self.__d_split_func = d_split_func
        self.__params_depth = arr_tree.depth_from(self.__params_tree, arr_tree.root())
        self.__leaf_root_paths = self.__calc_leaf_paths_to_root(self.__params_tree)
        self.__where_rights = self.__calc_where_rights()



    def __calc_leaf_paths_to_root(self, tree):
        leaf_ind_range = arr_tree.node_at_depth_range(self.__params_depth+1)
        out = np.zeros((leaf_ind_range[1]-leaf_ind_range[0], self.__params_depth + 1))
        out[:,out.shape[1]-1] = np.arange(leaf_ind_range[0], leaf_ind_range[1], 1)
        for d_from_leaf in range(1, self.__params_depth + 1):
            out[:,out.shape[1] - 1 - d_from_leaf] = arr_tree.parent(out[:,out.shape[1] - d_from_leaf])
        return out.astype(np.int)

    #returns a list A s.t. A[d] is the numpy array of all nodes at that depth
    def __calc_where_rights(self):
        out = []
        for d in range(self.__params_depth):
            qs = self.__leaf_root_paths[:,d]
            q_cs = self.__leaf_root_paths[:,d+1]
            q_c_child_nums = arr_tree.child_num(qs, q_cs)

            #pretty simple pattern here if can think of a fast way to do that instead
            #of using np.where() (print q_c_child_nums to see)
            out.append(np.where(q_c_child_nums == 1))
        return out

    def calc_split_tree(self, X):
        return self.__split_func(np.dot(self.__params_tree, X.T))

    def calc_grad_split_tree(self, X, split_tree, slow_assert = True):
        out = np.apply_along_axis(lambda f_outs: self.__d_split_func(X, f_outs), 0, split_tree)[:,:,np.newaxis]*X
        if slow_assert:
            np.testing.assert_array_almost_equal(out, self.__slow_calc_grad_split_tree(X, split_tree))
        return out

    def __slow_calc_grad_split_tree(self, X, split_tree):
        out = np.zeros(split_tree.shape + (X.shape[1],), dtype = split_tree.dtype)
        for i in range(out.shape[0]):
            out[i] = self.__d_split_func(X, split_tree[i])[:,np.newaxis]*X
        return out

    def calc_p_leaves(self, split_tree, slow_assert = True):
        num_X = split_tree[arr_tree.root()].shape[0]
        p_d_parent = np.ones(num_X, dtype = np.float32)
        for d in range(1, self.__params_depth + 1):
            d_parent_node_bounds = arr_tree.node_at_depth_range(d)
            left_splits_d_parent = split_tree[d_parent_node_bounds[0]:d_parent_node_bounds[1]]
            right_splits_d_parent = 1-left_splits_d_parent

            d_node_bounds = arr_tree.node_at_depth_range(d+1)
            p_d = np.zeros((d_node_bounds[1]-d_node_bounds[0], num_X))
            p_d[np.arange(0,p_d.shape[0],2)] = left_splits_d_parent*p_d_parent
            p_d[np.arange(1,p_d.shape[0],2)] = right_splits_d_parent*p_d_parent
            p_d_parent = p_d
        if slow_assert:
            np.testing.assert_array_almost_equal(p_d_parent, self.__slow_calc_p_leaves(split_tree))
        return p_d_parent

    def __slow_calc_p_leaves(self, split_tree):
        num_X = split_tree[arr_tree.root()].shape[0]
        p_tree = np.zeros((self.__params_tree.shape[0] + 2**(self.__params_depth), num_X), dtype = np.float32)
        p_tree[arr_tree.root()] = np.ones(num_X, dtype = np.float32)
        def f(tree, node, acc):
            if node == arr_tree.root():
                return None
            node_parent = arr_tree.parent(node)
            p_tree[node] = (split_tree[node_parent] if arr_tree.child_num(node_parent, node) == 0 else 1-split_tree[node_parent]) * \
                p_tree[node_parent]
        arr_tree.fold(p_tree, f, None)
        leaf_bounds = arr_tree.node_at_depth_range(self.__params_depth + 1)
        return p_tree[leaf_bounds[0]:leaf_bounds[1]]



    def calc_grad_p_leaves(self, split_tree, grad_split_tree, p_leaves, slow_assert = True):
        #For further improvement: Need to be careful when using by-entire-depth "speedups",
        #the last one I did actually did O(nodes^2) calculations since a lot of slice of
        #self.__leaf_root_paths[:,d] were repeated at higher depths!

        #indexed by (k is leaf, q is a reachable node from k)
        #out[k, depth(q), i] = grad p(k|X[i]) w.r.t. theta_q
        out = np.zeros((p_leaves.shape[0], self.__params_depth) + grad_split_tree.shape[1:], dtype = np.float32)

        for k in range(out.shape[0]):
            for d in range(self.__leaf_root_paths.shape[1] - 1):
                q = self.__leaf_root_paths[k,d]
                q_c = self.__leaf_root_paths[k,d+1]
                if arr_tree.child_num(q, q_c) == 0:
                    d_split = split_tree[q]
                    d_grad_split = grad_split_tree[q]
                else:
                    d_split = 1.0-split_tree[q]
                    d_grad_split = -grad_split_tree[q]

                out[k,d] = numpy_helper.stable_divide(p_leaves[k], d_split, 0)[:,np.newaxis] * d_grad_split
        return out


    def train(self, X, y, n_iters, learn_rate):
        unique_labels = np.unique(y)
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l)[0])

        for iter in range(n_iters):

            grad_gini = self.calc_expected_gini_gradient(X, where_y_eq_ls)#self.calc_expected_gini_gradient_slow(p_leaves, grad_p_leaves, y, unique_labels)
            self.__params_tree -= learn_rate*grad_gini
            if iter%10 == 0:
                print("Iter: ", iter)
                split_tree = self.calc_split_tree(X)
                p_leaves = self.calc_p_leaves(split_tree)
                print("EXPECTED GINI: ", impurity.expected_gini(p_leaves.T, y))
                print("----------------------------------------------")


    def calc_expected_gini_gradient(self, X, where_y_eq_ls, slow_assert = True):
        stopwatch = StopwatchProfiler()
        stopwatch.start()

        leaf_range = arr_tree.node_at_depth_range(self.__params_depth+1)
        splits = self.calc_split_tree(X)
        stopwatch.lap("spilts calc'd")

        grad_splits = self.calc_grad_split_tree(X, splits, slow_assert = False)
        stopwatch.lap("grad_splits calc'd")

        p_leaves = self.calc_p_leaves(splits, slow_assert = False)
        stopwatch.lap("p_leaves calc'd")

        grad_p_leaves = self.calc_grad_p_leaves(splits, grad_splits, p_leaves, slow_assert = False)
        stopwatch.lap("grad_p_leaves calc'd")

        out = np.zeros(self.__params_tree.shape, dtype = self.__params_tree.dtype)

        label_p_leaf_sums = np.zeros((len(where_y_eq_ls), p_leaves.shape[0]))
        label_grad_p_leaf_sums = np.zeros((len(where_y_eq_ls), grad_p_leaves.shape[0], grad_p_leaves.shape[1], grad_p_leaves.shape[3]))
        for l_ind in range(len(where_y_eq_ls)):
            where_y_eq_l = where_y_eq_ls[l_ind]
            label_p_leaf_sums[l_ind] = np.sum(p_leaves[:,where_y_eq_l], axis = 1)
            label_grad_p_leaf_sums[l_ind] = np.sum(grad_p_leaves[:,:,where_y_eq_l], axis = 2)


        p_leaf_sums = np.sum(label_p_leaf_sums, axis = 0)
        grad_p_leaf_sums = np.sum(label_grad_p_leaf_sums, axis = 0)
        stopwatch.lap("sums calc'd")

        u = self.__calc_u(p_leaf_sums)
        v = self.__calc_v(label_p_leaf_sums)
        #not sure of how I can prevent the bad duplicates problem...
        for q_d in range(self.__params_depth):
            grad_u_wrt_q = self.__calc_grad_u(q_d, p_leaf_sums, grad_p_leaf_sums)
            grad_v_wrt_q = self.__calc_grad_v(q_d, label_p_leaf_sums, label_grad_p_leaf_sums)

            nodes_at_q_d = self.__leaf_root_paths[:,q_d]
            unq_nodes_at_q_d = np.unique(nodes_at_q_d)

            #grad_summands = u[:,np.newaxis]*grad_v_wrt_q + v[:,np.newaxis]*grad_u_wrt_q

            #WORKS
            for k in range(0, leaf_range[1]-leaf_range[0]):
                out[nodes_at_q_d[k]] += u[k]*grad_v_wrt_q[k] + v[k]*grad_u_wrt_q[k]#grad_summands[k]#


            #DOESN'T WORK (because nodes_at_q_d often contains duplicates and it seems to only do one assign
            #for duplicate indices)
            '''nodes_at_q_d = self.__leaf_root_paths[:,q_d]
            #print("nodes at q_d: ", nodes_at_q_d)
            #out[nodes_at_q_d] += u[:,np.newaxis]*grad_v_wrt_q + v[:,np.newaxis]*grad_u_wrt_q'''

        out *= -1.0/float(p_leaves.shape[1])
        stopwatch.lap("out calc'd")
        stopwatch.stop()
        print("relative laps; ", stopwatch.relative_lap_deltas())

        stopwatch.reset()
        if slow_assert:
            #TODO
            return out
        return out


    def __calc_u(self, p_leaf_sums):
        return 1.0/p_leaf_sums

    def __calc_grad_u(self, q_d, p_leaf_sums, grad_p_leaf_sums):
        numerator = -grad_p_leaf_sums[:,q_d]
        denominator = np.square(p_leaf_sums)
        return numerator/denominator[:,np.newaxis]

    def __calc_v(self, label_p_leaf_sums):
        return np.sum(np.square(label_p_leaf_sums), axis = 0)

    def __calc_grad_v(self, q_d, label_p_leaf_sums, label_grad_p_leaf_sums):
        return 2.0*np.sum(label_p_leaf_sums[:,:,np.newaxis] * label_grad_p_leaf_sums[:,:,q_d], axis = 0)


    def __calc_expected_gini_gradient_slow(self, p_leaves, grad_p_leaves, y, unique_labels):
        out = np.zeros(self.__params_tree.shape, dtype = self.__params_tree.dtype)
        leaf_range = arr_tree.node_at_depth_range(self.__params_depth + 1)
        for k in range(0, leaf_range[1]-leaf_range[0]):
            u_k = self.__calc_u_k_slow(p_leaves[k])
            v_k = self.__calc_v_k_slow(p_leaves[k], y, unique_labels)

            for q_d in range(self.__params_depth):
                grad_u_k = self.__calc_grad_u_k_slow(p_leaves[k], grad_p_leaves[k,q_d])
                grad_v_k = self.__calc_grad_v_k_slow(p_leaves[k], grad_p_leaves[k,q_d], y, unique_labels)
                out[self.__leaf_root_paths[k,q_d]] +=  u_k*grad_v_k + v_k*grad_u_k
        out *= -1.0/float(p_leaves.shape[1])
        return out

    def __calc_u_k_slow(self, p_leaves_k):
        return 1.0/np.sum(p_leaves_k, axis = 0)

    def __calc_v_k_slow(self, p_leaves_k, y, unique_labels):
        out = 0
        for l in unique_labels:
            out += np.square(np.sum(p_leaves_k[np.where(y == l)], axis = 0))
        return out

    def __calc_grad_u_k_slow(self, p_leaves_k, grad_p_leaves_k_q):
        numerator = -np.sum(grad_p_leaves_k_q, axis = 0)
        denominator = np.square(np.sum(p_leaves_k, axis = 0))
        return numerator/denominator

    def __calc_grad_v_k_slow(self, p_leaves_k, grad_p_leaves_k_q, y, unique_labels):
        out = np.zeros(grad_p_leaves_k_q.shape[1:], dtype = self.__params_tree.dtype)
        for l in unique_labels:
            where_y_eq_l = np.where(y == l)
            left = np.sum(p_leaves_k[where_y_eq_l], axis = 0)
            right = np.sum(grad_p_leaves_k_q[where_y_eq_l], axis = 0)
            out += left*right
        return 2.0*out
