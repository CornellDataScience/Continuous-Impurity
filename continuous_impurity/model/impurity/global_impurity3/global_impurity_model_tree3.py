import numpy as np
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import model.impurity.global_impurity.global_impurity_tree_math2 as global_impurity_tree_math2
from performance.stopwatch_profiler import StopwatchProfiler
from model.impurity.global_impurity3.computation_node3 import ComputationNode3

class GlobalImpurityModelTree3:
    #TODO: make this take the model_at_depth_func and construct/prune
    #TODO: allow for abstract extension for a custom pruning implementation?
    def __init__(self, model_at_depth_func):
        self.__model_at_depth_func = model_at_depth_func
        self.__head = GlobalImpurityNode3(None, model_at_depth_func(0))
        self.__head._add_children(GlobalImpurityNode3(self.__head, None))
        self.__head._add_children(GlobalImpurityNode3(self.__head, None))

    def train(self, X, y, learn_rate, n_iters, min_depth, max_depth, \
        min_gini_to_grow, max_gini_to_prune, min_data_to_split, \
        print_progress_iters = 25, iters_per_prune = 1000):

        unique_labels = np.unique(y)
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l))

        stopwatch = StopwatchProfiler()

        for iter in range(n_iters):

            stopwatch.start()


            nodes, nonleaves, leaves = self.__head.to_list()
            stopwatch.lap("node, nonleaves, leaves calc'd")
            '''
            f_arr = self.__calc_f_arr(nodes, X)
            stopwatch.lap("f_arr calc'd")
            grad_f_arr = self.__calc_grad_f_arr(nodes, nonleaves, X, f_arr)
            stopwatch.lap("grad_f_arr calc'd")
            p_arr = self.__calc_p_arr(nodes, X, f_arr)
            stopwatch.lap("p_arr calc'd")'''
            comp_head = self.__build_computation_tree(ComputationNode3(None), \
                self.__gradient_computation_tree_assigner, X)
            f_arr, grad_f_arr, p_arr = self.__extract_f_grad_f_p(comp_head)
            stopwatch.lap("f_arr, grad_f_arr, p_arr calcd")
            grad_EG = self.__calc_grad(X, y, unique_labels, where_y_eq_ls, nodes, leaves, f_arr, p_arr, grad_f_arr)
            stopwatch.lap("grad_EG calc'd")
            for node_ID in range(len(grad_EG)):
                if grad_EG[node_ID] is not None:
                    for param_ind in range(len(grad_EG[node_ID])):
                        nodes[node_ID]._model._params[param_ind] -= learn_rate*grad_EG[node_ID][param_ind]
            stopwatch.lap("tree stepped using grad")
            if iter%iters_per_prune == 0:
                self.__prune(X, y, unique_labels, min_depth, max_depth, \
                    min_gini_to_grow, max_gini_to_prune, min_data_to_split)
            stopwatch.lap("tree pruned")
            stopwatch.stop()
            print("absolute lap times: ", stopwatch.lap_deltas())
            print("-----------------------------------------------------")
            print("rel lap times: ", stopwatch.relative_lap_deltas())
            print("")
            print("")
            stopwatch.reset()
            if iter%print_progress_iters == 0:
                print("iter: ", iter)
                self.__print_progress(X, y, unique_labels)


        return None

    #returns a list where list[id] is the array of ints s.t. X[i] passes through the node with id id
    def __get_node_label_subsets(self, X):
        out = []
        self.__head._set_node_inds(X, np.arange(0,X.shape[0],1), out)
        return out


    def __prune(self, X, y, unique_labels, min_depth, max_depth, min_gini_to_grow, max_gini_to_prune, min_data_to_split):
        def grow_leaf(node, node_depth):
            assert(node._is_leaf())
            node._model = self.__model_at_depth_func(node_depth)
            node._add_children([GlobalImpurityNode3(node, None), GlobalImpurityNode3(node, None)])
            assert(not node._is_leaf())

        def prune_nonleaf(node):
            assert(not node._is_leaf())
            node._model = None
            #just in case pointers aren't completely detached from the tree somehow
            for child in node._children:
                child._parent = None
            node._children = []
            assert(node._is_leaf())

        node_label_subsets = self.__get_node_label_subsets(X)
        def dfs_prune(node, node_depth):
            node_labels = node_label_subsets[node._ID]
            #not sure how to handle when len(node_labels) == 0... TODO: IMPORTANT
            if node_labels.shape[0] <= min_data_to_split:
                #case 2
                if not node._is_leaf():
                    prune_nonleaf(node)
                    return None
                #not sure what to do when node_labels.shape[0] < min_data_to_split but node is leaf
            else:
                node_gini = impurity.gini(node_labels)
                if node._is_leaf():
                    #case 3)
                    if node_depth < max_depth and (node_gini >= min_gini_to_grow or node_depth < min_depth):
                        grow_leaf(node, node_depth)
                    return None
                #case 1)
                elif node_gini <= max_gini_to_prune:
                    #prune by case 1
                    prune_nonleaf(node)
                    return None

            #no pruning required. Prune children.
            for child in node._children:
                dfs_prune(child, node_depth + 1)
        dfs_prune(self.__head, 0)

        nodes, _, leaves = self.__head.to_list()
        f_arr = self.__calc_f_arr(nodes, X)
        p_arr = self.__calc_p_arr(nodes, X, f_arr)
        self.__set_leaf_predicts(leaves, p_arr, y, unique_labels)


    #returns grad expected impurity of the whole tree w.r.t. all parameters of
    #node q
    def __calc_grad(self, X, y, unique_labels, where_y_eq_ls, nodes, leaves, f_arr, p_arr, grad_f_arr):
        def calc_p_and_p_sums(p_ks):
            p_sums_where_y_eq_ls = np.zeros(len(where_y_eq_ls))
            for y_eq_l_ind in range(len(where_y_eq_ls)):
                p_sums_where_y_eq_ls[y_eq_l_ind] = np.sum(p_ks[where_y_eq_ls[y_eq_l_ind]])
            p_sum = np.sum(p_sums_where_y_eq_ls)
            return p_sum, p_sums_where_y_eq_ls

        def calc_grad_q_and_grad_q_sums(grad_q):
            grad_q_sums = []
            grad_q_sums_where_y_eq_ls = []
            for param_ind in range(len(grad_q)):
                iter_add = np.zeros((len(where_y_eq_ls),) + grad_q[param_ind].shape[1:])
                for l_ind in range(len(where_y_eq_ls)):
                    iter_add[l_ind] = np.sum(grad_q[param_ind][where_y_eq_ls[l_ind]], axis = 0)
                grad_q_sums.append(np.sum(iter_add, axis = 0))
                grad_q_sums_where_y_eq_ls.append(iter_add)
            return grad_q_sums, grad_q_sums_where_y_eq_ls

        def init_grad_EG():
            grad_EG = [None for i in range(len(nodes))]
            for q in nodes:
                grad_EG[q._ID] = None if q._model is None else \
                    [np.zeros(q_param.shape, dtype = q_param.dtype) for q_param in q._model._params]
            return tuple(grad_EG)

        grad_EG = init_grad_EG()

        for k in leaves:
            k_ID = k._ID
            assert(k_ID is not None)
            p_ks = p_arr[k_ID]
            grad_p_ks = self.__calc_grad_p_arr(nodes, k_ID, p_arr, f_arr, grad_f_arr)

            p_sum, p_sums_where_y_eq_ls = calc_p_and_p_sums(p_ks)

            u_k = self.__calc_u(p_sum)
            v_k = self.__calc_v(p_sums_where_y_eq_ls)



            for (q_ID, grad_q) in grad_p_ks:
                grad_q_sums, grad_q_sums_where_y_eq_ls = calc_grad_q_and_grad_q_sums(grad_q)
                grad_u_k = self.__calc_grad_u(p_sum, grad_q_sums)
                grad_v_k = self.__calc_grad_v(p_sums_where_y_eq_ls, grad_q_sums_where_y_eq_ls)

                for param_ind in range(len(grad_EG[q_ID])):
                    grad_EG[q_ID][param_ind] -=  (v_k*grad_u_k[param_ind] + \
                        u_k*grad_v_k[param_ind])/float(X.shape[0])

        return grad_EG


    def __calc_u(self, p_k_sum):
        return 1.0/p_k_sum

    def __calc_v(self, p_sums_where_y_eq_ls):
        return np.sum(np.square(p_sums_where_y_eq_ls))

    def __calc_grad_u(self, p_sum, grad_p_sums):
        out = []
        denominator = p_sum*p_sum
        for param_grad_sum in grad_p_sums:
            numerator = -param_grad_sum
            out.append(numerator/denominator)
        return out

    def __calc_grad_v(self, p_sums_where_y_eq_ls, grad_p_sums_where_y_eq_ls):
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




    def __calc_f_arr(self, nodes, X):
        out = [None for i in range(len(nodes))]
        def f(node):
            if not node._is_leaf():
                out[node._ID] = node.f(X)
        self.__head.fold_in_place(f)
        return out

    def __calc_grad_f_arr(self, nodes, nonleaves, X, f_arr):
        out = [None for i in range(len(nodes))]
        for nonleaf_node in nonleaves:
            out[nonleaf_node._ID] = nonleaf_node.grad_f(X, f_arr[nonleaf_node._ID])
        return out

    def __calc_p_arr(self, nodes, X, f_arr):
        head = nodes[0]
        out = np.zeros((len(nodes), X.shape[0]))
        out[head._ID] = np.ones(X.shape[0])

        #assumes out[node._ID] is already calculated
        def p(node):
            ps = out[node._ID]
            node_f_outs = f_arr[node._ID]
            for child_ind in range(len(node._children)):
                child = node._children[child_ind]
                child_ID = child._ID
                out[child_ID] = node_f_outs[child_ind]*ps
                p(child)
            return None
        p(head)
        return out

    def __calc_grad_p_arr(self, nodes, leaf_ID, p_arr, f_arr, grad_f_arr):
        #returns an array, L, s.t. L[i] contains information about the ith
        #relevant node to the path to leaf_ID from the head. L[i] is (id, gradient),
        #where gradient is a list of gradients for each param in nodes[id]

        assert(nodes[leaf_ID]._is_leaf()), "nodes[leaf_ID] is not a leaf..."
        out = []
        p_leaf = p_arr[leaf_ID]
        q = nodes[leaf_ID]._parent
        q_c = nodes[leaf_ID]
        while q is not None:
            q_c_ind = q._child_ind(q_c)
            f_q_c = f_arr[q._ID][q_c_ind]
            grad_f_q = grad_f_arr[q._ID]
            iter_out = []
            for param_ind in range(len(grad_f_q)):
                grad_f_param_q_c = grad_f_q[param_ind][q_c_ind]
                grad_leaf_wrt_q = numpy_helper.stable_divide(p_leaf, f_q_c, 0)[:,np.newaxis] * grad_f_param_q_c


                iter_out.append(grad_leaf_wrt_q)
            out.append((q._ID, iter_out))
            q_c = q
            q = q._parent

        return out


    def __gradient_computation_tree_assigner(self, node, comp_node, X):
        #setting id just for troubleshooting.
        comp_node._ID = node._ID

        #for calc'ing all p's
        if node._is_root():
            comp_node._p = np.ones(X.shape[0])
        else:
            comp_node_child_ind = comp_node._parent._child_ind(comp_node)
            comp_node._p = comp_node._parent._p * comp_node._parent._f[comp_node_child_ind]

        #for calc'ing all f's and grad f's
        if not node._is_leaf():
            comp_node._f = node.f(X)
            comp_node._grad_f = node.grad_f(X, comp_node._f)
        else:
            comp_node._f = None
            comp_node._grad_f = None

    def __build_computation_tree(self, computation_head, computation_assigner_func, X):
        def f(node, comp_node):
            assert(comp_node._is_leaf())

            computation_assigner_func(node, comp_node, X)

            for child in node._children:
                comp_node_child = ComputationNode3(comp_node)
                comp_node._add_children(comp_node_child)
                f(child, comp_node_child)
        f(self.__head, computation_head)
        return computation_head

    def __extract_f_grad_f_p(self, compute_head):
        f_arr = []
        grad_f_arr = []
        p_arr = []
        def f(comp_node):
            f_arr.append(comp_node._f)
            grad_f_arr.append(comp_node._grad_f)
            p_arr.append(comp_node._p)
        compute_head.fold_in_place(f)
        return f_arr, grad_f_arr, p_arr


    def __print_progress(self, X, y, unique_labels):
        nodes,_,leaves = self.__head.to_list()
        f_arr = self.__calc_f_arr(nodes, X)
        p_arr = self.__calc_p_arr(nodes, X, f_arr)
        self.__set_leaf_predicts(leaves, p_arr, y, unique_labels)
        predictions = self.__head.predict(X)
        unq, counts = np.unique(predictions, return_counts = True)
        print("label distribution: ", [(unq[i], counts[i]) for i in range(len(unq))])
        print("ACCURACY: ", 100.0*np.sum(predictions == y)/float(y.shape[0]))
        print("NODES: ", len(nodes))
        print("LEAVES: ", len(leaves))
        print("EXPECTED GINI: ", self.__expected_GINI(leaves, p_arr, y))
        print("----------------------------------")



    def __expected_GINI(self, leaves, p_arr, y):
        subset_assign_probs = np.zeros((y.shape[0], len(leaves)))
        for leaf_ind in range(len(leaves)):
            subset_assign_probs[:,leaf_ind] = p_arr[leaves[leaf_ind]._ID]
        return impurity.expected_gini(subset_assign_probs, y)


    def __set_leaf_predicts(self, leaves, p_arr, y, unique_labels):
        #can rewrite this with fold
        for leaf in leaves:
            p_leaf = p_arr[leaf._ID]
            l_scores = np.zeros(unique_labels.shape[0])
            for l_ind in range(len(unique_labels)):
                where_y_eq_l = np.where(y == unique_labels[l_ind])
                l_scores[l_ind] = np.sum(p_leaf[where_y_eq_l])
            leaf._leaf_predict = unique_labels[np.argmax(l_scores)]
