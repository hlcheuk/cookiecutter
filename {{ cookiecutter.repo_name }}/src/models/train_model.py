from hiclass import LocalClassifierPerParentNode
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger

__all__ = "Train_model"
logger = setup_logger(__all__)


def train_model_hiclass(clf, X, y):
    '''

    Args:
        clf (object): an object for the specified classifier
        X (list): feature data for training in the format of list of leveled lists
        y (list): label data for training in the format of list of leveled lists
    '''

    # instantiate a Local Classifier Per Node (LCPN) with a specified classifier
    lcpn = LocalClassifierPerParentNode(local_classifier=clf)
    logger.info(f'Instantiated a LCPN with a specified classifier')

    # setup the pipeline steps
    steps = [
        ('vec', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', lcpn),
    ]

    # construct pipeline
    pipeline = Pipeline(steps)

    # fit the pipeline to the training set
    hiclass = pipeline.fit(X, y)
    logger.info(
        f'Trained model with a specified classifier under a coherent hierarchy'
    )

    return hiclass