import sklearn.datasets as datasets
from model.impurity.global_impurity.global_impurity_model_tree2 import GlobalImpurityModelTree2
import model.impurity.global_impurity.node_model2_maker as node_model2_maker
from model.impurity.global_impurity.node_model2 import NodeModel2
import toolbox.data_helper as data_helper
import function.stable_func as stable_func
import numpy as np
from sklearn import tree
import plot.decision_bound_plotter as decision_bound_plotter
from function.activation.tanh import TanH
from function.activation.identity import Identity
import matplotlib.pyplot as plt

def plot(predict_func, X, y, step):
    if predict_func is not None:
        ax = plt.gca()
        decision_bound_plotter.plot_contours(X, predict_func, ax, step, alpha = 0.75)
    if X is not None:
        plt.scatter(X[:,0], X[:,1], c = y, edgecolors = "black")




DECISIONTREE_HYPERPARAMS = {"max_depth": 3}

#X,y =  datasets.make_classification(n_samples = 200, n_features=2, n_redundant=0, n_informative=2,random_state=2, n_clusters_per_class=2)#
X,y = datasets.make_moons()#
#X,y = datasets.load_iris(return_X_y = True)
FEATURES = [0,1]
X = X[:,FEATURES]
assert(X.shape[1] == 2), "X NEEDS SHAPE 2 IN ORDER TO BE PLOTTED"
X = X.astype(np.float64)
X = data_helper.unit_square_normalize(X)
X = data_helper.mean_center(X)

(X_train, y_train), (X_test, y_test) = data_helper.train_test_split(X, y, 0.8, seed = 42)


mymodel = GlobalImpurityModelTree2(node_model2_maker.logistic_model_at_depth(X_train.shape[1]))
#mymodel = GlobalImpurityModelTree2(node_model2_maker.matrix_activation_logistic_impurity_model_at_depth(\
#    X.shape[1], lambda x: TanH(), lambda x: 2))

MYMODEL_HYPERPARAMS = {"iters": 50000, \
    "learn_rate": 10.0, \
    "probabilistic_leaf": True,\
    "min_depth": 1, \
    "max_depth": 3, \
    "print_progress_iters": 250, \
    "min_gini_to_grow": .02, \
    "max_gini_to_prune":.02}

NUM_TRAIN_STOPS = 10
DECISION_BOUND_GRAPH_STEP = 0.005

try:
    for train_stop in range(NUM_TRAIN_STOPS):
        try:
            mymodel.train(X_train, \
                y_train, \
                MYMODEL_HYPERPARAMS["iters"], \
                MYMODEL_HYPERPARAMS["learn_rate"], \
                probabilistic_leaf = MYMODEL_HYPERPARAMS["probabilistic_leaf"],\
                min_depth = MYMODEL_HYPERPARAMS["min_depth"], \
                max_depth = MYMODEL_HYPERPARAMS["max_depth"], \
                print_progress_iters = MYMODEL_HYPERPARAMS["print_progress_iters"], \
                min_gini_to_grow = MYMODEL_HYPERPARAMS["min_gini_to_grow"], \
                max_gini_to_prune = MYMODEL_HYPERPARAMS["max_gini_to_prune"])
        except KeyboardInterrupt:
            print("Training halted.")

        print("Completed train stop: " + str(train_stop) + " of " + str(NUM_TRAIN_STOPS))
        plot(mymodel.leaf_predict, X_train, y_train, DECISION_BOUND_GRAPH_STEP)
        plt.show()
        plot(mymodel.predict, X_train, y_train, DECISION_BOUND_GRAPH_STEP)
        plt.show()
except KeyboardInterrupt:
    print("TRAINING STOPPED")



FINAL_DISPLAY_DECISION_BOUND_GRAPH_STEP = 0.0025

print("----------------------------------------")
plot(mymodel.predict, X, y, FINAL_DISPLAY_DECISION_BOUND_GRAPH_STEP)
plt.show()
print("My model train accuracy: ", 100.0*data_helper.evaluate_accuracy(mymodel, X_train, y_train))
print("My model test accuracy: ", 100.0*data_helper.evaluate_accuracy(mymodel, X_test, y_test))

print("----------------------------------------")
decision_tree = tree.DecisionTreeClassifier(max_depth = DECISIONTREE_HYPERPARAMS["max_depth"])
decision_tree.fit(X_train, y_train)
plot(decision_tree.predict, X, y, FINAL_DISPLAY_DECISION_BOUND_GRAPH_STEP)
plt.show()
print("Decision tree train accuracy: ", 100.0*data_helper.evaluate_accuracy(decision_tree, X_train, y_train))
print("Decision tree test accuracy: ", 100.0*data_helper.evaluate_accuracy(decision_tree, X_test, y_test))
