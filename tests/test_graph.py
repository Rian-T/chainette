from chainette.core.graph import GraphNode, Graph

def test_graph_traversal_and_validation():
    # Build small diamond graph: a -> b, a -> c, b -> d, c -> d
    a = GraphNode(id="a", ref="A")
    b = GraphNode(id="b", ref="B")
    c = GraphNode(id="c", ref="C")
    d = GraphNode(id="d", ref="D")

    a.connect(b, c)
    b.connect(d)
    c.connect(d)

    g = Graph([a])

    # Nodes returned in depth-first order without duplicates
    ids = [n.id for n in g.nodes()]
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c", "d"}

    # DAG validation should not raise
    g.validate_dag() 