import logging


def setup_logger(name=__name__, file_level="debug", stream_level="info"):

    level_dict = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    file_level = level_dict[file_level]
    stream_level = level_dict[stream_level]
    base_level = min(file_level, stream_level)

    # Initialize logger
    # Set Logger level
    logger = logging.getLogger(name)
    logger.setLevel(base_level)

    #  Create file handler
    handler = logging.FileHandler("log/{{ cookiecutter.repo_name }}.log")
    handler.setLevel(file_level)

    # Create stream handler
    console = logging.StreamHandler()
    console.setLevel(stream_level)

    #  Set log format
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    #  Add handler to logger
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger
