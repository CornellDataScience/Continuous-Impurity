import numpy as np
'''
Rep invariant: assumes a tree has length equal to f(d) = sum from i = 0 to d (2^d) for
some value of d (i.e. all leaves have the same depth d and there are 2^d of them)
'''
def root():
    return 0

def is_root(n):
    return n == 0

def is_leaf(tree, n):
    return left(n) >= len(tree) or right(n) >= len(tree)

'''
returns -1 if q is not a child of n
returns 0 if q is the left child of n
returns 1 if q is the right child of n.

if n and q are vectors, is done elementwise
'''
def child_num(n, q):
    if isinstance(n, np.ndarray):
        assert(isinstance(q, np.ndarray))
        assert(n.shape == q.shape)
        out = np.zeros(n.shape)
        out[np.where(parent(q) != n)] = -1
        out[np.where(right(n) == q)] = 1
        return out
    if parent(q) != n:
        return None
    if left(n) == q:
        return 0
    return 1


def left(n):
    return 2*n + 1

def right(n):
    return 2*n + 2

def parent(n):
    return np.floor_divide(n-1, 2)#(n-1)//2

def node_at_depth_range(depth):
    start_ind = np.sum(np.power(2, np.arange(0,depth,1)))
    return (start_ind, start_ind + 2**depth)

def depth_from(tree, node):
    def f(node, val, acc):
        curr_depth, depth_counts = (acc[0], acc[1])
        if (depth_counts + 1)%(2**curr_depth) == 0:
            return (curr_depth+1, 0)
        return (curr_depth, depth_counts+1)
    return fold(tree, f, (0,0), start_node = node)[0]

#folds in order of: node, left, right
def fold(tree, f, acc, start_node = root()):
    def traverse(node, acc):
        acc = f(node, tree[node], acc)
        r = right(node)
        if r < len(tree):
            acc = traverse(left(node), acc)
            acc = traverse(r, acc)
        return acc
    return traverse(start_node, acc)
