"""toxicity_prediction package for drug safety analysis."""

__version__ = "0.1.0"
__all__ = ['featurize', 'predict_toxicity']

from .drug_toxicity_prediction import featurize, predict_toxicity

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)