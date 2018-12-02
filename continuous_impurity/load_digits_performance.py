import sklearn.datasets as datasets
from model.impurity.global_impurity.global_impurity_model_tree2 import GlobalImpurityModelTree2
import model.impurity.global_impurity.node_model2_maker as node_model2_maker
from model.impurity.global_impurity.node_model2 import NodeModel2
import toolbox.data_helper as data_helper
import function.stable_func as stable_func
import numpy as np
from sklearn import tree

MYMODEL_HYPERPARAMS = {"iters": 100000, \
    "learn_rate": 10, \
    "min_depth": 1, \
    "max_depth": 5, \
    "print_progress_iters": 25, \
    "min_gini_to_grow": .02, \
    "max_gini_to_prune":.02}

DECISIONTREE_HYPERPARAMS = {"max_depth": 10}

X,y = datasets.load_digits(return_X_y = True)
X = X.astype(np.float64)
X /= 16.0

(X_train, y_train), (X_test, y_test) = data_helper.train_test_split(X, y, 0.8, seed = 42)

mymodel = GlobalImpurityModelTree2(node_model2_maker.logistic_model_at_depth(X_train, y_train))
try:
    mymodel.train(X_train, \
        y_train, \
        MYMODEL_HYPERPARAMS["iters"], \
        MYMODEL_HYPERPARAMS["learn_rate"], \
        min_depth = MYMODEL_HYPERPARAMS["min_depth"], \
        max_depth = MYMODEL_HYPERPARAMS["max_depth"], \
        print_progress_iters = MYMODEL_HYPERPARAMS["print_progress_iters"], \
        min_gini_to_grow = MYMODEL_HYPERPARAMS["min_gini_to_grow"], \
        max_gini_to_prune = MYMODEL_HYPERPARAMS["max_gini_to_prune"])
except KeyboardInterrupt:
    print("training halted.")



print("----------------------------------------")
print("My model train accuracy: ", 100.0*data_helper.evaluate_accuracy(mymodel, X_train, y_train))
print("My model test accuracy: ", 100.0*data_helper.evaluate_accuracy(mymodel, X_test, y_test))

print("----------------------------------------")
decision_tree = tree.DecisionTreeClassifier(max_depth = DECISIONTREE_HYPERPARAMS["max_depth"])
decision_tree.fit(X_train, y_train)
print("Decision tree train accuracy: ", 100.0*data_helper.evaluate_accuracy(decision_tree, X_train, y_train))
print("Decision tree test accuracy: ", 100.0*data_helper.evaluate_accuracy(decision_tree, X_test, y_test))
