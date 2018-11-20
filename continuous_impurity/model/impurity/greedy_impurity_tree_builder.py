from model.impurity.logistic_impurity_model import LogisticImpurityModel
from model.impurity.greedy_impurity_model_tree import GreedyImpurityModelTree
from model.impurity.greedy_impurity_tree_leaf import GreedyImpurityTreeLeaf
from model.impurity.matrix_activation_logistic_impurity import MatrixActivationLogisticImpurity


def build_logistic_impurity_tree(x_length, max_depth):
    def model_creator_func(depth):
        return LogisticImpurityModel(x_length)
    return build_binary_tree(model_creator_func, max_depth)

def build_matrix_activation_logistic_impurity_tree(act_funcs, transform_x_lengths, x_length):
    assert len(act_funcs) == len(transform_x_lengths)
    max_depth = len(act_funcs)
    def model_creator_func(depth):
        act_func = act_funcs[depth]
        transform_x_length = transform_x_lengths[depth]
        return MatrixActivationLogisticImpurity(act_func, x_length, transform_x_length)
    return build_binary_tree(model_creator_func, max_depth)

'''
where model_creator_func returns a model given an input depth
'''
def build_binary_tree(model_creator_func, max_depth):
    head = GreedyImpurityModelTree(model_creator_func(0))
    __rec_build_binary_tree(head, model_creator_func, 1, max_depth)
    return head

def __rec_build_binary_tree(head, model_creator_func, curr_depth, max_depth):
    if curr_depth == max_depth:
        head_children = [GreedyImpurityTreeLeaf(), GreedyImpurityTreeLeaf()]
        head.add_children(head_children)
        return None
    head_children = [GreedyImpurityModelTree(model_creator_func(curr_depth)),\
        GreedyImpurityModelTree(model_creator_func(curr_depth))]
    head.add_children(head_children)
    for head_child in head_children:
        __rec_build_binary_tree(head_child, model_creator_func, curr_depth+1, max_depth)
