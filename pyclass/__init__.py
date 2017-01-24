"""
Main lead in methods
"""
from pyclass import app, stats
log = app.logging.getLogger(__name__)


def train(trends, coefs, rmse, dem, aspect, slope, posidex, mpw, qa):
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)


def classify(coefs, rmse, dem, aspect, slope, posidex, mpw, qa):
    cloud_prob, snow_prob, water_prob = stats.quality_stats(qa)
