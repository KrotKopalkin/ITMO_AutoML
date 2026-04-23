import pandas as pd
import numpy as np
import os
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_loader import load_config, load_data
from src.features import TextStatsExtractor, TextCleaner
from src.models import LamaModel
from src.utils import set_seed, TensorboardLogger

def main():
    config = load_config()
    set_seed(config.get('seed', 42))
    
    train_df, test_df = load_data(config)
    
    print("Preprocessing data for LAMA...")
    # 1. Extract Stats
    stats_extractor = TextStatsExtractor(text_col='text')
    train_stats = stats_extractor.fit_transform(train_df.drop(columns=['author']))
    test_stats = stats_extractor.transform(test_df)
    
    # 2. TF-IDF
    cleaner = TextCleaner()
    train_clean = cleaner.transform(train_df['text'])
    test_clean = cleaner.transform(test_df['text'])
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    train_tfidf = tfidf.fit_transform(train_clean).toarray()
    test_tfidf = tfidf.transform(test_clean).toarray()
    
    # Convert TF-IDF to DataFrame
    tfidf_cols = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    train_tfidf_df = pd.DataFrame(train_tfidf, columns=tfidf_cols)
    test_tfidf_df = pd.DataFrame(test_tfidf, columns=tfidf_cols)
    
    # 3. Combine
    X_train = pd.concat([train_stats, train_tfidf_df], axis=1)
    X_train['author'] = train_df['author']
    
    X_test = pd.concat([test_stats, test_tfidf_df], axis=1)
    
    # Encode Target
    le = LabelEncoder()
    X_train['author'] = le.fit_transform(X_train['author'])
    
    # Define roles
    roles = {
        'target': 'author',
        'drop': ['id']
    }
    
    # Initialize logger
    logger = TensorboardLogger(config['paths']['log_dir'] + "lama_baseline_v2")
    
    # Train Lama
    print("Starting LAMA Training on tabular data...")
    lama = LamaModel(config)
    lama.train(X_train, roles)
    
    # Calculate OOF metric
    # If some folds were not calculated due to timeout, lama might have NaNs
    oof_data = lama.oof_pred.data
    if np.isnan(oof_data).any():
        print("Warning: NaNs found in OOF predictions. Filling with mean probabilities...")
        # Fill NaNs with uniform probability or mean of non-NaNs
        mask = np.isnan(oof_data)
        oof_data[mask] = 1.0 / oof_data.shape[1]

    oof_score = log_loss(X_train['author'], oof_data)
    print(f"\nLAMA OOF LogLoss: {oof_score:.5f}")
    
    # Log to Tensorboard
    logger.log_scalar("Metric/OOF_LogLoss", oof_score, 0)
    logger.close()
    
    # Prediction
    print("Predicting on test set...")
    test_preds = lama.predict(X_test)
    
    # Prepare submission
    submission = pd.DataFrame(test_preds, columns=le.classes_)
    submission.insert(0, 'id', test_df['id'])
    
    output_path = os.path.join(config['paths']['output_dir'], "submission_lama_v2.csv")
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    # Save OOF for blending
    oof_df = pd.DataFrame(oof_data, columns=le.classes_)
    oof_df.insert(0, 'id', train_df['id'])
    oof_path = os.path.join(config['paths']['output_dir'], "oof_lama.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"OOF predictions saved to {oof_path}")

if __name__ == "__main__":
    main()
