from importlib import reload

from chainette.utils.events import publish, StepTotalItems, BatchFinished
import chainette.utils.logging_v3 as log


def test_progress_events_monotonic():
    # Reload logger to reset internal state
    reload(log)

    # Ensure clean state
    log.stop()

    # Simulate events
    publish(StepTotalItems(step_id="s1", total=10))
    publish(BatchFinished(step_id="s1", batch_no=0, count=1))
    publish(BatchFinished(step_id="s1", batch_no=1, count=1))

    prog = log._ensure_progress()
    task_id = log._tasks["s1"]
    task = prog.tasks[task_id]

    assert task.total == 10
    # Completed batches (count=1) should equal number of events
    assert task.completed >= 2

    log.stop() 