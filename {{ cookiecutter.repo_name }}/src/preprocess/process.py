import regex as re
from src.utils.logger import setup_logger

__all__ = "Preprocess"
logger = setup_logger(__all__)


def clean_text(text: str):
    '''To clean the document as text provided

    Args:
        text (str): a string of document of a SKU

    Returns:
        a cleaned text

    Example:

    '''
    PUNC_REGEX = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^\p{Han}0-9a-z #+_]')

    text = text.lower()
    text = PUNC_REGEX.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)

    return text