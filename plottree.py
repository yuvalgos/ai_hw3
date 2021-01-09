from graphviz import Digraph


def plot_tree(tree):
    node_style = {'style': 'filled', 'fillcolor': '#DAE8FC', 'color': '#6C8EBF', 'penwidth': '3'}
    edge_style = {'penwidth': '2'}

    def add_node_to_tree(node, index):
        if node.is_leaf:
            tree_graph.node(index, f'{int(node.prediction)}', shape='plaintext')
        else:
            node_text = f'{x_fields[node.field_num]} >= {int(node.threshold)}'
            tree_graph.node(index, node_text, **node_style)
            tree_graph.edge(index, index + '_0', 'No', **edge_style)
            add_node_to_tree(node.node_false, index + '_0')
            tree_graph.edge(index, index + '_1', 'Yes', **edge_style)
            add_node_to_tree(node.node_true, index + '_1')

    tree_graph = Digraph(comment='Tree')
    tree_graph.format = 'png'
    tree_graph.node('root', '<<I><B>x</B></I>>', shape='plaintext')
    tree_graph.edge('root', 'node', **edge_style)
    add_node_to_tree(tree, 'node')

    return tree_graph