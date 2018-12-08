import numpy as np
from model.impurity.global_impurity3.global_impurity_node3 import GlobalImpurityNode3
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import model.impurity.global_impurity.global_impurity_tree_math2 as global_impurity_tree_math2
from performance.stopwatch_profiler import StopwatchProfiler

class GlobalImpurityModelTree3:

    def __init__(self, head):#(self, model_at_depth_func):
        #self.__model_at_depth_func = model_at_depth_func
        self.__head = head

    def train(self, X, y, learn_rate, n_iters, print_progress_iters = 25, GC_frequency = None):
        unique_labels = np.unique(y)
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l))

        for iter in range(n_iters):
            if GC_frequency is not None and iter%GC_frequency == 0:
                gc.collect()

            nodes = GlobalImpurityNode3.to_list_and_set_IDs(self.__head)
            leaves = GlobalImpurityNode3.get_leaves(nodes)

            f_arr = GlobalImpurityNode3.calc_f_arr(nodes, X)
            grad_f_arr = GlobalImpurityNode3.calc_grad_f_arr(nodes, X, f_arr)
            p_arr = GlobalImpurityNode3.calc_p_arr(nodes, X, f_arr)
            grad_EG = GlobalImpurityNode3.calc_grad(X, y, unique_labels, where_y_eq_ls, nodes, leaves, f_arr, p_arr, grad_f_arr)

            for node_ID in range(len(grad_EG)):
                if grad_EG[node_ID] is not None:
                    for param_ind in range(len(grad_EG[node_ID])):
                        nodes[node_ID]._model._params[param_ind] -= learn_rate*grad_EG[node_ID][param_ind]


            if iter%print_progress_iters == 0:
                print("iter: ", iter)
                self.__print_progress(leaves, p_arr, X, y, unique_labels)
                
        #TODO: need to set leaf predicts after training

    def __print_progress(self, leaves, p_arr, X, y, unique_labels):
        self.__set_leaf_predicts(leaves, p_arr, y, unique_labels)
        predictions = self.__head.predict(X)
        unq, counts = np.unique(predictions, return_counts = True)
        print("label distribution: ", [(unq[i], counts[i]) for i in range(len(unq))])
        print("ACCURACY: ", 100.0*np.sum(predictions == y)/float(y.shape[0]))
        print("EXPECTED GINI: ", self.__expected_GINI(leaves, p_arr, y))
        print("----------------------------------")



    def __expected_GINI(self, leaves, p_arr, y):
        subset_assign_probs = np.zeros((y.shape[0], len(leaves)))
        for leaf_ind in range(len(leaves)):
            subset_assign_probs[:,leaf_ind] = p_arr[leaves[leaf_ind]._ID]
        return impurity.expected_gini(subset_assign_probs, y)


    def __set_leaf_predicts(self, leaves, p_arr, y, unique_labels):
        for leaf in leaves:
            p_leaf = p_arr[leaf._ID]
            l_scores = np.zeros(unique_labels.shape[0])
            for l_ind in range(len(unique_labels)):
                where_y_eq_l = np.where(y == unique_labels[l_ind])
                l_scores[l_ind] = np.sum(p_leaf[where_y_eq_l])
            leaf._leaf_predict = unique_labels[np.argmax(l_scores)]
