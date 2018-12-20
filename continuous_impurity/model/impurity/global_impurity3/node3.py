
class Node3:

    def __init__(self, children):
        self._children = children


    def fold(self, f, acc):
        def traverse(node, acc):
            acc = f(node, acc)
            if node._is_leaf():
                return acc
            for child in node._children:
                acc = traverse(child, acc)
            return acc

        return traverse(self, acc)

    def fold_in_place(self, f):
        def traverse(node):
            f(node)
            for child in node._children:
                traverse(child)
        traverse(self)

    def _child_ind(self, child):
        return self._children.index(child)

    def _add_children(self, new_children):
        if not isinstance(new_children, list):
            new_children = [new_children]
        assert(len(new_children) + len(self._children) <= 2), ("children would've been: " + str(len(new_children) + len(self._children)))
        self._children.extend(new_children)

    def _is_leaf(self):
        return len(self._children) == 0

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)
