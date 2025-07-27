from graphviz import Digraph

from engine import Value


def trace(root: Value) -> tuple[set, set]:
    # Builds a set of all nodes and edges in the graph
    nodes, edges = set(), set()
    def build(v: Value):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                # This is important: first the child and then the resulting value
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges


def draw_trace(root: Value):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # For any value in the graph, create a rectanctular ('record') node for it
        dot.node(name=uid, label="{ %s | data  %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # If this value is a result of some operation, create an op node for it
            dot.node(name=uid+n._op, label=n._op)
            # and conect this node to it
            dot.edge(uid+n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot