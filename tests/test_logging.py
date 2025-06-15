from chainette.utils.logging import get

def test_logger_levels():
    logger = get("debug")
    assert logger.level == 10  # DEBUG
    logger = get("error")
    assert logger.level == 40  # ERROR 