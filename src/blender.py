import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from typing import List, Dict

class WeightedBlender:
    """
    Finds optimal weights for blending multiple models using SciPy minimize.
    """
    def __init__(self):
        self.weights = None

    def _loss_func(self, weights, oof_preds: List[np.ndarray], y_true: np.ndarray):
        # Normalize weights to sum to 1
        normalized_weights = weights / np.sum(weights)
        
        final_pred = np.zeros_like(oof_preds[0])
        for w, pred in zip(normalized_weights, oof_preds):
            final_pred += w * pred
            
        return log_loss(y_true, final_pred)

    def fit(self, oof_preds: List[np.ndarray], y_true: np.ndarray):
        n_models = len(oof_preds)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1)] * n_models
        
        result = minimize(
            self._loss_func, 
            initial_weights, 
            args=(oof_preds, y_true),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.weights = result.x / np.sum(result.x)
        print(f"Optimal weights: {self.weights}")
        return self.weights

    def predict(self, test_preds: List[np.ndarray]) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Blender not fitted.")
            
        final_pred = np.zeros_like(test_preds[0])
        for w, pred in zip(self.weights, test_preds):
            final_pred += w * pred
        return final_pred
