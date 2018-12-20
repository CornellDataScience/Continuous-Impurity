import numpy as np
import function.stable_func as stable_func
import model.impurity.global_impurity4.array_binary_tree as arr_tree

class GlobalImpurityTree4:


    def __init__(self, params_tree, split_func, d_split_func):
        self.__params_tree = params_tree
        self.__split_func = split_func
        self.__d_split_func = d_split_func
        self.__depth = arr_tree.depth_from(self.__params_tree, arr_tree.root())
        self.__leaf_root_paths = self.__calc_leaf_paths_to_root(self.__params_tree)

    def __calc_leaf_paths_to_root(self, tree):
        leaf_ind_range = arr_tree.node_at_depth_range(self.__depth - 1)
        out = np.zeros((leaf_ind_range[1] - leaf_ind_range[0], self.__depth), dtype = np.int)
        out[:,0] = np.arange(leaf_ind_range[0], leaf_ind_range[1], 1)
        for d_from_leaf in range(1, out.shape[1]):
            out[:,d_from_leaf] = arr_tree.parent(out[:,d_from_leaf-1])
        return out


    def calc_split_tree(self, X):
        return self.__split_func(np.dot(self.__params_tree, X.T))

    def calc_grad_split_tree(self, X, logit_tree):
        d_splits = self.__d_split_func(X, logit_tree)
        return d_splits[:,:,np.newaxis]*X

    #may want to transition all calculations over to be by layer for memory
    #reasons. Not sure how taking gradients would play into it since it may
    #need more info
    def calc_p_tree(self, split_tree):
        #if memory problems, can easily just have it spit out leaf probs,
        #only keeping track of parent probs, rather than assigning them to
        #all for tree
        num_X = split_tree[arr_tree.root()].shape[0]
        p_tree = np.zeros(split_tree.shape)
        p_tree[arr_tree.root()] = np.ones(num_X)
        parent_range = np.array([0], dtype = np.int)

        for d in range(1, self.__depth):

            d_bounds = arr_tree.node_at_depth_range(d)

            parent_splits = split_tree[parent_range]
            parent_ps = p_tree[parent_range]
            p_tree[np.arange(d_bounds[0], d_bounds[1], 2)] = parent_splits*parent_ps
            p_tree[np.arange(d_bounds[0]+1, d_bounds[1], 2)] = (1.0-parent_splits)*parent_ps
            parent_range = np.arange(d_bounds[0], d_bounds[1], 1)
        return p_tree

    def calc_p_leaves(self, split_tree):
        num_X = split_tree[arr_tree.root()].shape[0]
        parent_ps = np.ones(num_X)
        parent_range = np.array([0], dtype = np.int)

        for d in range(1, self.__depth):

            d_bounds = arr_tree.node_at_depth_range(d)
            parent_splits = split_tree[parent_range]
            ps_d = np.zeros((d_bounds[1] - d_bounds[0], num_X), dtype = np.float32)
            ps_d[np.arange(0, ps_d.shape[0], 2)] = parent_splits*parent_ps
            ps_d[np.arange(1, ps_d.shape[0], 2)] = (1.0-parent_splits)*parent_ps
            parent_ps = ps_d
            parent_range = np.arange(d_bounds[0], d_bounds[1], 1)
        return parent_ps

    #since the p(k|X) for all leaves is such a large array it caused memory problems,
    #this is to mitigate this factor since the gradient of E(G) needs these sums anyway.
    #returns an array A s.t. A[l,k,d_from_leaf] = the sum over grad p(k|X) w.r.t. theta[d_from_leaf parents from k] over indices such that y_i = l
    def calc_label_sum_grad_p_leaves(self, leaf_probs, split_tree, grad_split_tree, y, unique_labels):
        grad_split_tree_elem_shape = grad_split_tree[arr_tree.root()].shape
        out = np.zeros((len(unique_labels), leaf_probs.shape[0], self.__depth) + grad_split_tree_elem_shape[1:], dtype = np.float32)
        print("leaf probs shape: ", leaf_probs.shape)
        print("leaf probs dtype: ", leaf_probs.dtype)
        print("split tree shape: ", split_tree.shape)
        print("split tree dtype: ", split_tree.dtype)
        print("grad split tree shape: ", grad_split_tree.shape)
        print("grad split tree dtype: ", grad_split_tree.dtype)
        print("out shape: ", out.shape)
        print("out dtype: ", out.dtype)
        where_y_eq_ls = []
        for l in unique_labels:
            where_y_eq_ls.append(np.where(y == l)[0])

        for d_from_leaf in range(1, self.__depth):
            d_from_leaf_nodes = self.__leaf_root_paths[:, d_from_leaf]
            d_from_leaf_children = self.__leaf_root_paths[:, d_from_leaf-1]
            d_from_leaf_children_inds = arr_tree.child_num(d_from_leaf_nodes, d_from_leaf_children)
            #the below yields a fairly obvious, but kind of annoying pattern to code quickly, for
            #a possibly faster calculation of d_from_leaf_children_inds. Change later for a possible
            #speedup
            #print("d_from_leaf: ", d_from_leaf)
            #print("d_from_leaf_children_inds: ", d_from_leaf_children_inds)
            #print("------------------------")
            where_left = np.where(d_from_leaf_children_inds == 0)
            where_right = np.where(d_from_leaf_children_inds == 1)





            grad_p_d_from_leaf_minus_1 = np.zeros((leaf_probs.shape[0],) + grad_split_tree_elem_shape, dtype = np.float32)
            print("left mul: ", (leaf_probs[where_left]/split_tree[d_from_leaf_nodes[where_left]]).shape)
            print("right mul: ", grad_split_tree[d_from_leaf_nodes[where_left]].shape)
            grad_p_d_from_leaf_minus_1[where_left] = (leaf_probs[where_left]/split_tree[d_from_leaf_nodes[where_left]])[:,:,np.newaxis] * grad_split_tree[d_from_leaf_nodes[where_left]]
            print("left*right: ", grad_p_d_from_leaf_minus_1[where_left].shape)
            grad_p_d_from_leaf_minus_1[where_right] = (leaf_probs[where_right]/split_tree[d_from_leaf_nodes[where_right]])[:,:,np.newaxis] * grad_split_tree[d_from_leaf_nodes[where_right]]
            for l_ind in range(len(where_y_eq_ls)):
                where_y_eq_l = where_y_eq_ls[l_ind]
                out[l_ind, :, d_from_leaf-1] = np.sum(grad_p_d_from_leaf_minus_1[:,where_y_eq_l], axis = 1)


    def calc_sum_grad_p_leaves(self, label_sum_grad_p_leaves):
        return np.sum(label_sum_grad_p_leaves, axis = 0)

    '''
    returns an array, A, s.t. A[k,q] is grad p(k|X) w.r.t. the qth node starting from
    the first parent of k
    '''
    '''
    def calc_grad_p_leaves(self, leaf_probs, split_tree, grad_split_tree):
        grad_split_tree_elem_shape = grad_split_tree[arr_tree.root()].shape
        #GET VALUEERROR: TOO BIG HERE IF LARGE ENOUGH DEPTH
        #NOT SURE HOW TO FIX...
        #print("grad_p_leaves shape: ", (leaf_probs.shape[0], self.__depth) + grad_split_tree_elem_shape)
        out = np.zeros((leaf_probs.shape[0], self.__depth) + grad_split_tree_elem_shape)
        for d_from_leaf in range(1, self.__depth):
            d_from_leaf_nodes = self.__leaf_root_paths[:, d_from_leaf]
            d_from_leaf_children = self.__leaf_root_paths[:, d_from_leaf - 1]
            d_from_leaf_children_inds = arr_tree.child_num(d_from_leaf_nodes, d_from_leaf_children)
            #the below yields a fairly obvious, but kind of annoying pattern to code quickly, for
            #a possibly faster calculation of d_from_leaf_children_inds. Change later for a possible
            #speedup
            #print("d_from_leaf: ", d_from_leaf)
            #print("d_from_leaf_children_inds: ", d_from_leaf_children_inds)
            #print("------------------------")
            where_left = np.where(d_from_leaf_children_inds == 0)
            where_right = np.where(d_from_leaf_children_inds == 1)

            out[:, d_from_leaf-1] = np.zeros((leaf_probs.shape[0],) + grad_split_tree_elem_shape)
            print("depth assign shape; ", out[:,d_from_leaf-1].shape)
            out[where_left, d_from_leaf-1] = (leaf_probs[where_left]/split_tree[d_from_leaf_nodes[where_left]])[:,:,np.newaxis] * grad_split_tree[d_from_leaf_nodes[where_left]]
            out[where_right, d_from_leaf-1] = (leaf_probs[where_right]/split_tree[d_from_leaf_nodes[where_right]])[:,:,np.newaxis] * grad_split_tree[d_from_leaf_nodes[where_right]]
        return out
    '''
