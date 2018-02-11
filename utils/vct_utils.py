import os
try:
    import pydot_ng as pydot
except ImportError:
    try:
        import pydotplus as pydot
    except ImportError:
        try:
            import pydot
        except ImportError:
            pydot = None

try:
    pydot.Dot.create(pydot.Dot())
except Exception:
    raise ImportError('Failed to import pydot. You must install pydot'
                      ' and graphviz for `pydotprint` to work.')

def plot(root, to_file='vct_tree.png'):
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    root.id = 0
    label = 'root\nproof:{:d} disproof:{:d}'.format(root.proof, root.disproof)
    dot.add_node(pydot.Node(root.id, label=label))
    node_stack = []
    node_stack.append(root)
    count = 0
    while len(node_stack):
        node = node_stack.pop()
        if node.parent is not None:
            parent_id = node.parent.id
            node_id = node.id
            dot.add_edge(pydot.Edge(parent_id, node_id))
        for position, child_node in node.children.items():
            count += 1
            child_node.id = count
            label = str(position) + '\nproof:{:d} disproof:{:d}'.format(child_node.proof, child_node.disproof)
            dot.add_node(pydot.Node(child_node.id, label=label))
            node_stack.append(child_node)

    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
