import numpy as np
from model.impurity.global_impurity3.node3 import Node3
#ideally all nodes involved in global impurity 3 would
#be an abstract extension so wouldn't have to duplicate folding
class ComputationNode3(Node3):

    def __init__(self, parent):
        Node3.__init__(self, [])
        self._parent = parent


    def _is_leaf(self):
        return len(self._children) == 0

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)
