import pandas as pd
import regex as re
from src.utils.logger import setup_logger

__all__ = "Build_features"
logger = setup_logger(__all__)


def seg_word(text: pd.Series, stopword_path: str):
    '''To first segment the document with Ckiptagger and then exclude stopwords.
        Finally combine the segmented words into space separated string.

    Args:
        text (pd.Series): cleaned document as a Series
        stopwords (str): path for the stopwords

    Returns:
        a space separated string of the document without stopwords
    '''
    # import required packages and turn off warning log
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from ckiptagger import WS
    # instantiate a WS object for word segmentation
    ws = WS("./models/ckiptagger/data")
    logger.info(f"Instantiated a WS object for word segmentation")
    # load the stopwords and turn it into a list
    stopwords = pd.read_csv(stopword_path).stop_word.tolist()
    logger.info(f"Loaded stopwords from {stopword_path}")
    # segment the document as the text provided
    seg_text = ws(text)
    logger.info(f"Segmented document from the Series provided")
    # exclude stopwords from the document
    text = [
        ' '.join([
            re.sub(' ', '', word) for word in doc
            if re.sub(' ', '', word) not in stopwords
        ]) for doc in seg_text
    ]
    logger.info(f"Constructed list of the space separated document")
    return text


def hier_target(srs: pd.Series):
    '''To convert a series of hierarchical targets into list of lists

    Args:
        srs (pd.Series): a series containing the columns of hierarchical targets

    Returns:
        List of lists as hierarchy sequence
    '''
    # level = df.columns
    hier = []
    for x in srs:
        lvl = 5 - len(re.search('((00){0,4})1$', x).group(1)) // 2
        hier.append([
            x[0:2 + 2 * (l + 1)].ljust(13, '0') if l + 1 < lvl else x
            for l in range(lvl)
        ])
    logger.info(f"Constructed list of the category in hierarchy sequence")
    return hier