import numpy as np

class GreedyImpurityTreeLeaf:

    def train(self, parent, X, y, min_data_to_split, sgd_iters, sgd_step_sizes):
        unq, counts = np.unique(y, return_counts = True)
        self.__leaf_predict = unq[np.argmax(counts)]

    def _predict(self, X, inds, predicts):
        predicts[inds] = self.__leaf_predict
