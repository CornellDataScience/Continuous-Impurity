import numpy as np
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
from model.impurity.global_impurity.node_model2 import NodeModel2
import toolbox.data_helper as data_helper
import function.stable_func as stable_func

def construct_tree(depth, model_with_depth_fn):
    def build(node, remaining_depth):
        if remaining_depth == 0:
            return None
        for i in range(0, 2):
            add_child = GlobalImpurityNode2(node, model_with_depth_fn(depth - remaining_depth)) \
                if remaining_depth != 1 else \
                GlobalImpurityNode2(node, None)
            node.add_children(add_child)
        for child in node._children:
            build(child, remaining_depth - 1)


    head = GlobalImpurityNode2(None, model_with_depth_fn(0))
    build(head, depth - 1)
    return head

#TODO: Replace all sigmoids with stable sigmoids

def construct_logistic_tree(depth, x_shape):
    def create_logistic_node_model(d):
        #TODO: Change function names
        def dud_func(params_dict, k, X):
            X_affine = data_helper.affine_X(X)
            k_eq_0_out = stable_func.sigmoid(np.dot(X_affine, params_dict["theta"]))
            return k_eq_0_out if k == 0 else 1-k_eq_0_out

        def dud_grad_func(params_dict, k, X):
            k_eq_0_out = dud_func(params_dict, 0, X)
            X_affine = data_helper.affine_X(X)
            grad_k_eq_0_out = (k_eq_0_out*(1-k_eq_0_out))[:,np.newaxis] * X_affine
            return {"theta":grad_k_eq_0_out} if k == 0 else {"theta":-grad_k_eq_0_out}

        params_dict = {"theta": 0.000001*(np.random.rand((x_shape + 1))-0.5)}
        return NodeModel2(params_dict, dud_func, dud_grad_func)
    return construct_tree(depth, create_logistic_node_model)

def construct_matrix_activation_logistic_tree(x_shape, act_func,\
    transform_shape_with_depth_list):

    def create_model(d):
        BIAS_DAMPEN_FACTOR = 1
        TRANSFORM_BIAS_DAMPEN_FACTOR = 1#.000001
        A_GRAD_DAMPEN_FACTOR = 1
        THETA_GRAD_DAMPEN_FACTOR = 1
        A_rows = transform_shape_with_depth_list[d]

        def t(params_dict, X):
            X_affine = data_helper.affine_X(X)
            A_Xs = np.dot(params_dict["A"], X_affine.T).T
            assert(A_Xs.shape[0] == X_affine.shape[0])
            return data_helper.affine_X(act_func.act(A_Xs))

        def f(params_dict, k, X):
            t_X = t(params_dict, X)
            k_0_out = stable_func.sigmoid(np.dot(t_X, params_dict["theta"]))
            return k_0_out if k == 0 else 1 - k_0_out


        def grad_f_A(params_dict, k, X, t_X, f_0_out):
            X_affine = data_helper.affine_X(X)
            d_t_X = act_func.derivative_wrt_activation(t_X)#d_t_X[i,j] is the derivative of act_func w.r.t. t_X[i,j]
            k_0_out = np.zeros((X.shape[0], ) + params_dict["A"].shape)
            for k in range(k_0_out.shape[1]):
                for r in range(k_0_out.shape[2]):
                    k_0_out[:,k,r] = params_dict["theta"][k] * \
                        X_affine[:,r] * \
                        (f_0_out*(1-f_0_out)) * \
                        d_t_X[:,k]

            k_0_out[:,:,k_0_out.shape[2]-1] *= TRANSFORM_BIAS_DAMPEN_FACTOR
            k_0_out *= A_GRAD_DAMPEN_FACTOR
            return k_0_out if k == 0 else -k_0_out



        def grad_f_theta(params_dict, k, X, t_X, f_0_out):
            k_0_out = (f_0_out*(1-f_0_out))[:,np.newaxis]*t_X

            k_0_out[:,k_0_out.shape[1]-1] *= BIAS_DAMPEN_FACTOR
            k_0_out *= THETA_GRAD_DAMPEN_FACTOR
            return k_0_out if k == 0 else -k_0_out

        def grad_f(params_dict, k, X):
            t_X = t(params_dict, X)
            f_0_out = f(params_dict, 0, X)
            return {"A": grad_f_A(params_dict, k, X, t_X, f_0_out), "theta": grad_f_theta(params_dict, k, X, t_X, f_0_out)}


        A = 0.01*(np.random.random((A_rows, x_shape+1))-.5)


        for i in range(min(A.shape[0], A.shape[1]-1)):
            print("A shape: ", A.shape)
            print("i: ", i)
            A[i,i] += 1
        
        A[:,A.shape[1]-1] = 0

        theta = 0.0001*(np.random.random(A_rows+1)-.5)
        return NodeModel2({"A": A, "theta": theta}, f, grad_f)

    return construct_tree(len(transform_shape_with_depth_list), create_model)
