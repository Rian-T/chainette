from importlib import reload

from chainette.utils.events import publish, StepTotalItems, BatchFinished
import chainette.utils.logging as log


def test_progress_events_monotonic():
    import chainette.utils.events as ev
    ev._REGISTRY.clear()

    reload(log)
    log.stop()

    # Simulate events
    publish(StepTotalItems(step_id="s1", total=10))
    publish(BatchFinished(step_id="s1", batch_no=0, count=3))
    publish(BatchFinished(step_id="s1", batch_no=1, count=4))

    prog = log._ensure_progress()
    task_id = log._tasks["s1"]
    task = prog.tasks[task_id]

    assert task.total == 10
    # Completed batches (count=1) should equal number of events
    assert task.completed == 7

    log.stop()


def test_progress_midpoint():
    from importlib import reload as _r
    import chainette.utils.events as ev
    ev._REGISTRY.clear()
    _r(log)
    log.stop()

    publish(StepTotalItems(step_id="s2", total=10))
    for i in range(5):
        publish(BatchFinished(step_id="s2", batch_no=i, count=2))

    prog = log._ensure_progress()
    task = prog.tasks[log._tasks["s2"]]
    assert task.completed == 10
    assert task.percentage == 100

    log.stop() 