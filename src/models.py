import pandas as pd
import numpy as np
import xgboost as xgb
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from typing import Dict, Any, Optional
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import optuna

class LamaModel:
    """Wrapper for LightAutoML (LAMA) baseline model."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config['model_params']['lama']
        self.task = Task('multiclass')
        self.model = None

    def train(self, train_df: pd.DataFrame, roles: Dict[str, Any]) -> None:
        self.model = TabularAutoML(
            task=self.task,
            timeout=self.config.get('timeout', 1200),
            cpu_limit=self.config.get('cpu_limit', 4),
            reader_params={'n_folds': self.config.get('n_folds', 5), 'cv': 5, 'random_state': 42},
            general_params={'use_algos': [['linear_l2', 'lgb', 'cb']], 'lgb_params': {'device': 'gpu'}}
        )
        self.oof_pred = self.model.fit_predict(train_df, roles=roles, verbose=1)

    def predict(self, test_df: pd.DataFrame) -> Any:
        if self.model is None: raise ValueError("Model not trained")
        test_pred = self.model.predict(test_df)
        return test_pred.data

class CustomXGBoostModel:
    """
    XGBoost model with Optuna hyperparameter optimization.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config['model_params']['custom']
        self.best_params = None
        self.model = None

    def optimize(self, X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> Dict[str, Any]:
        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'verbosity': 0,
                'seed': 42,
                'device': 'cuda',
                'tree_method': 'hist',
                'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            }
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
                dval = xgb.DMatrix(X[val_idx], label=y[val_idx])
                
                model = xgb.train(params, dtrain, num_boost_round=1000, 
                                  evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
                cv_scores.append(model.best_score)
            
            return np.mean(cv_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        return self.best_params

    def train(self, X: np.ndarray, y: np.ndarray, params: Dict[str, Any]) -> None:
        dtrain = xgb.DMatrix(X, label=y)
        # Add static params
        params.update({
            'objective': 'multi:softprob', 
            'num_class': 3, 
            'eval_metric': 'mlogloss',
            'device': 'cuda',
            'tree_method': 'hist'
        })
        self.model = xgb.train(params, dtrain, num_boost_round=500)

    def predict(self, X: np.ndarray) -> np.ndarray:
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
