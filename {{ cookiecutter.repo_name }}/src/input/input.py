from src.utils.logger import setup_logger

__all__ = "Input"
logger = setup_logger(__all__)


def make_list_leveled(l: list):
    '''To make a list of lists leveled, which means to make ALL the lists within having the same length

    Args:
        l (list): a list of lists which have different length

    Returns:
        A list of lists having same length with "" as replacement of missing
    '''
    max_level = max([len(e) for e in l])
    l_leveled = [e + [""] * (max_level - len(e)) for e in l]
    logger.info(f'Leveled the provided lists')
    return l_leveled