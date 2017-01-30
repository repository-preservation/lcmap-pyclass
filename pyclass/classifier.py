from pyclass import app
log = app.logging.getLogger(__name__)


def rf_predict(forest, X):
    proba = forest.predict_proba(X)
