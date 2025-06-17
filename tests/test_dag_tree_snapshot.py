def test_dag_tree_snapshot(capsys):
    from chainette.utils.logging import show_dag_tree

    step_ids = ["a", "b", "parallel_branches", "c"]
    show_dag_tree(step_ids)

    captured = capsys.readouterr()
    assert "Execution DAG" in captured.out
    assert "parallel_branches" in captured.out 