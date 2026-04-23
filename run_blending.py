import pandas as pd
import numpy as np
import os
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from src.data_loader import load_config, load_data
from src.blender import WeightedBlender
from src.utils import set_seed

def main():
    config = load_config()
    set_seed(config.get('seed', 42))
    
    train_df, _ = load_data(config)
    le = LabelEncoder()
    y_true = le.fit_transform(train_df['author'])
    
    # Paths to OOF files
    oof_lama_path = os.path.join(config['paths']['output_dir'], "oof_lama.csv")
    oof_custom_path = os.path.join(config['paths']['output_dir'], "oof_custom_xgb.csv")
    
    if not os.path.exists(oof_lama_path) or not os.path.exists(oof_custom_path):
        print(f"Error: OOF files not found. Ensure both LAMA and Custom models are trained.")
        return

    print("Loading OOF predictions for blending...")
    oof_lama = pd.read_csv(oof_lama_path).drop(columns=['id']).values
    oof_custom = pd.read_csv(oof_custom_path).drop(columns=['id']).values
    
    # Models to blend
    oof_list = [oof_lama, oof_custom]
    
    # Find optimal weights
    blender = WeightedBlender()
    print("Finding optimal blending weights...")
    weights = blender.fit(oof_list, y_true)
    
    # Calculate final OOF score
    final_oof_score = log_loss(y_true, blender.predict(oof_list))
    print(f"\nFinal Blended OOF LogLoss: {final_oof_score:.5f}")
    
    # Load Test predictions
    sub_lama_path = os.path.join(config['paths']['output_dir'], "submission_lama_v2.csv")
    sub_custom_path = os.path.join(config['paths']['output_dir'], "submission_custom_xgb.csv")
    
    sub_lama = pd.read_csv(sub_lama_path)
    sub_custom = pd.read_csv(sub_custom_path)
    
    test_ids = sub_lama['id']
    test_preds_lama = sub_lama.drop(columns=['id']).values
    test_preds_custom = sub_custom.drop(columns=['id']).values
    
    # Apply weights
    test_preds_list = [test_preds_lama, test_preds_custom]
    final_test_preds = blender.predict(test_preds_list)
    
    # Prepare final submission
    submission = pd.DataFrame(final_test_preds, columns=le.classes_)
    submission.insert(0, 'id', test_ids)
    
    output_path = os.path.join(config['paths']['output_dir'], "submission_final_blend.csv")
    submission.to_csv(output_path, index=False)
    print(f"Final blended submission saved to {output_path}")

if __name__ == "__main__":
    main()
