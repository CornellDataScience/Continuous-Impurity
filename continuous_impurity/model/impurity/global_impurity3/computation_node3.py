import numpy as np
from model.impurity.global_impurity3.node3 import Node3
#ideally all nodes involved in global impurity 3 would
#be an abstract extension so wouldn't have to duplicate folding
class ComputationNode3(Node3):

    def __init__(self, parent):
        Node3.__init__(self, [])
        self._parent = parent


    def extract_field(self, field_name):
        out = []
        def f(node):
            out.append(getattr(node, field_name))
        self.fold_in_place(f)
        return out

    def _is_leaf(self):
        return len(self._children) == 0

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)
