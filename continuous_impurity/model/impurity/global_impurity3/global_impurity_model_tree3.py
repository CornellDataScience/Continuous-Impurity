import numpy as np
from model.impurity.global_impurity.global_impurity_node2 import GlobalImpurityNode2
import toolbox.numpy_helper as numpy_helper
import function.impurity as impurity
import model.impurity.global_impurity.global_impurity_tree_math2 as global_impurity_tree_math2
from performance.stopwatch_profiler import StopwatchProfiler

class GlobalImpurityModelTree3:

    def __init__(self, head):
        self.__head = head


    
