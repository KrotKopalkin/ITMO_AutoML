import pandas as pd
import numpy as np
import os
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from src.data_loader import load_config, load_data
from src.features import TextStatsExtractor, TextCleaner
from src.models import CustomXGBoostModel
from src.utils import set_seed, TensorboardLogger

def main():
    config = load_config()
    set_seed(config.get('seed', 42))
    
    train_df, test_df = load_data(config)
    
    # 1. Prepare Target
    le = LabelEncoder()
    y = le.fit_transform(train_df['author'])
    
    # 2. Define Preprocessing Pipeline
    # We combine stylometric features with NLP (TF-IDF + SVD)
    # TF-IDF can produce sparse matrix which TruncatedSVD handles natively.
    stylometry_pipe = Pipeline([
        ('stats', TextStatsExtractor(text_col='text'))
    ])
    
    nlp_pipe = Pipeline([
        ('clean', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 3))),
        ('svd', TruncatedSVD(n_components=200, random_state=42))
    ])
    
    preprocessor = ColumnTransformer([
        ('sty', stylometry_pipe, ['text']),
        ('nlp', nlp_pipe, 'text')
    ])
    
    print("Preprocessing data (Stylometry + NLP SVD)...")
    print("Fitting preprocessor...")
    X_transformed = preprocessor.fit_transform(train_df)
    print(f"X_transformed shape: {X_transformed.shape}")
    print("Transforming test data...")
    X_test_transformed = preprocessor.transform(test_df)
    print("Preprocessing finished.")
    
    # 3. Optimize XGBoost with Optuna
    logger = TensorboardLogger(config['paths']['log_dir'] + "custom_xgboost")
    print("Starting Optuna Hyperparameter Optimization...")
    
    xgb_model = CustomXGBoostModel(config)
    best_params = xgb_model.optimize(X_transformed, y, n_trials=50)
    
    print(f"Best Params: {best_params}")
    
    # 4. Final Training with Cross-Validation to get OOF score
    print("Final training with 5-fold CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(y), 3))
    test_preds = np.zeros((len(test_df), 3))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_transformed, y)):
        print(f"Fold {fold+1}...")
        X_tr, X_val = X_transformed[train_idx], X_transformed[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # Train fold model
        fold_model = CustomXGBoostModel(config)
        fold_model.train(X_tr, y_tr, best_params.copy())
        
        oof_preds[val_idx] = fold_model.predict(X_val)
        test_preds += fold_model.predict(X_test_transformed) / 5.0
        
        fold_score = log_loss(y_val, oof_preds[val_idx])
        logger.log_scalar(f"Fold/LogLoss", fold_score, fold)

    final_oof_score = log_loss(y, oof_preds)
    print(f"\nCustom XGBoost OOF LogLoss: {final_oof_score:.5f}")
    
    # Save OOF for blending
    oof_df = pd.DataFrame(oof_preds, columns=le.classes_)
    oof_df.insert(0, 'id', train_df['id'])
    oof_path = os.path.join(config['paths']['output_dir'], "oof_custom_xgb.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")
    
    logger.log_scalar("Metric/Final_OOF_LogLoss", final_oof_score, 0)
    logger.close()
    
    # 5. Prepare submission
    submission = pd.DataFrame(test_preds, columns=le.classes_)
    submission.insert(0, 'id', test_df['id'])
    
    output_path = os.path.join(config['paths']['output_dir'], "submission_custom_xgb.csv")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    main()
