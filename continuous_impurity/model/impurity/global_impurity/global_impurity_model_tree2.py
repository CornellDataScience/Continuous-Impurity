import numpy as np
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import model.impurity.global_impurity.global_impurity_tree_math2 as global_impurity_tree_math2
from performance.stopwatch_profiler import StopwatchProfiler



'''
HEUREISTIC THAT MIGHT HELP PREVENT MODEL FROM BASICALLY TRAINING GREEDILY:
Don't let nodes assign data extremely high values for the split probabilities.
This prevents a vanishing gradient problem (or helps to). Not sure how to
accomplish this...

'''
class GlobalImpurityModelTree2:
    #could switch all dicts to an array indexed by node ID or a speedup?

    def __init__(self, model_at_depth_func):
        self.__model_at_depth_func = model_at_depth_func
        self.__head = GlobalImpurityNode2(None, self.__model_at_depth_func(0))
        self.__head.add_children(GlobalImpurityNode2(self.__head, None), \
            GlobalImpurityNode2(self.__head, None))


    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        inds = np.arange(0,X.shape[0], 1)
        self.__head._predict(X, inds, predictions)
        return predictions


    #TODO: add min_data_to_split to prevent overfitting
    def train(self, X, y, iters, learn_rate, min_depth = 1, max_depth = 10, min_gini_to_grow = 0.02, max_gini_to_prune = 0.02, print_progress_iters = 100):
        unique_labels = np.unique(y)
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l))

        for iter in range(iters):
            nonleaves = self.__head._get_nonleaves()
            leaves = self.__head._get_leaves()
            global_impurity_tree_math2.take_gradient_descent_step(self.__head, X, y, learn_rate, unique_labels, where_y_eq_ls, leaves, nonleaves)

            self.__prune(X, y, unique_labels, self.__get_node_label_subsets(X,y), min_depth, max_depth, \
                min_gini_to_grow, max_gini_to_prune)

            if iter % print_progress_iters == 0:
                self.__print_performance(iter, X, y, unique_labels)






    def __get_node_label_subsets(self, X, y):
        inds_dict = {}
        self.__head._fill_node_ind_dict(inds_dict, X, np.arange(0,X.shape[0],1).astype(np.int))
        out = {}
        for node in self.__head._to_list():
            out[node] = y[inds_dict[node]]
        return out



    '''
    Prune semantics:
        The following are addressed in the order of depth first search:

        - 1) If a non-leaf has perfect (or almost perfect) GINI (by some threshold),
          its children should be eliminated and it should become a leaf, so long
          as the depth of this leaf satisfies some minimum depth threshold.



        - 3) (Might not be a good idea, since at the beginning of training, GINI
          anywhere is going to stink. This will grow the tree up to max_depth,
          which might be just fine since the other rules can help shrink the
          tree down later. May want to remove this case if causing trouble/very
          big trees): If a leaf has poor GINI (by some threshold), then it should have two
          children added, so long as the depth of these new nodes do not exceed
          some maximum depth threshold.

        - self.__head should never be removed for any reason. Head may not be a leaf.

        - Originally had: '- 2) If a any node receives zero (or a miniscule number by some threshold)
          of data to a child, that node should become a leaf, so long as the depth
          of this leaf satisfies some minimum depth threshold.' But realized this case
          may happen a lot especially with random initializations, and doens't make sense.

        - may want to see if pruning using actual GINI or expected GINI is better
    '''
    def __prune(self, X, y, unique_labels, node_label_subsets, min_depth, max_depth, min_gini_to_grow, max_gini_to_prune):
        def grow_leaf(node, node_depth):
            assert(node._is_leaf())
            node._model = self.__model_at_depth_func(node_depth)
            node.add_children(GlobalImpurityNode2(node, None), GlobalImpurityNode2(node, None))
            assert(not node._is_leaf())

        def prune_nonleaf(node):
            assert(not node._is_leaf())
            node._model = None
            #just in case pointers aren't completely detached from the tree somehow
            for child in node._children:
                child._parent = None
            node._children = []
            assert(node._is_leaf())

        def dfs_prune(node, node_depth):
            node_labels = node_label_subsets[node]
            #not sure how to handle when len(node_labels) == 0... TODO: IMPORTANT
            if len(node_labels) != 0:
                node_gini = impurity.gini(node_labels)
                if node._is_leaf():
                    #case 3)
                    if node_depth < max_depth and (node_gini > min_gini_to_grow or node_depth < min_depth):
                        grow_leaf(node, node_depth)
                    return None
                #case 1)
                elif node_gini < max_gini_to_prune:
                    #prune by case 1
                    prune_nonleaf(node)
                    return None

            #no pruning required. Prune children.
            for child in node._children:
                dfs_prune(child, node_depth + 1)

        dfs_prune(self.__head, 0)




        #needs to reassign leaves classes since may have pruned grown a leaf
        #or added new children
        self.__assign_leaves_classes(X, y, unique_labels, True)



    def __assign_leaves_classes(self, X, y, unique_labels, probabilistically):
        if probabilistically:
            p_dict = global_impurity_tree_math2.calc_p_dict(self.__head, X)
            leaves = self.__head._get_leaves()
            leaf_label_scores = np.zeros((len(unique_labels), len(leaves)))
            for label_ind in range(len(unique_labels)):
                for leaf_ind in range(len(leaves)):
                    leaf_label_scores[label_ind, leaf_ind] = np.sum(p_dict[leaves[leaf_ind]][np.where(y == unique_labels[label_ind])])
            max_leaf_label_scores = np.argmax(leaf_label_scores, axis = 0)
            for leaf_ind in range(len(leaves)):
                leaves[leaf_ind]._leaf_predict = unique_labels[max_leaf_label_scores[leaf_ind]]

        else:
            #TODO: Fix non-probabilistic leaf assigns now that pruning should be working
            leaves = self.__head._get_leaves()
            leaf_predicts = self.predict_leaves(X)
            for leaf in leaves:
                y_where_leaf_predicts_eq_id = y[np.where(leaf_predicts == leaf._leaf_id)]
                unq, counts = np.unique(y_where_leaf_predicts_eq_id, return_counts = True)
                leaf._leaf_predict = -1 if len(unq) == 0 else unq[np.argmax(counts)]


    def __print_performance(self, iter, X, y, unique_labels):
        self.__assign_leaves_classes(X, y, unique_labels, True)
        print("iter: ", iter)
        p_dict = global_impurity_tree_math2.calc_p_dict(self.__head, X)
        leaves = self.__head._get_leaves()
        max_leaf_depth = None
        min_leaf_depth = None
        for leaf in leaves:
            depth = leaf._depth()
            if max_leaf_depth is None or depth > max_leaf_depth:
                max_leaf_depth = depth
            if min_leaf_depth is None or depth < min_leaf_depth:
                min_leaf_depth = depth



        subset_assign_probs = np.zeros((X.shape[0], len(leaves)))
        for i in range(len(leaves)):
            subset_assign_probs[:,i] = p_dict[leaves[i]]
        print("- Expected GINI: ", impurity.expected_gini(subset_assign_probs, y))
        print("- # nodes: ", len(self.__head._to_list()))
        print("- # leaves: ", len(self.__head._get_leaves()))
        print("- Min/max leaf depth: ", (min_leaf_depth, max_leaf_depth))
        predictions = self.predict(X)
        _, counts = np.unique(predictions, return_counts = True)
        print("- Prediction distribution: ",counts)
        print("- Train accuracy: ", 100.0*np.sum(y == predictions)/float(y.shape[0]))
        print("------------------------------------------")
