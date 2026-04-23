from src.data_loader import load_config, load_data
from src.utils import set_seed
from src.eda import analyze_target, analyze_text_stats, analyze_common_words, analyze_handcrafted_features

def main():
    config = load_config()
    set_seed(config.get('seed', 42))
    
    train_df, test_df = load_data(config)
    
    print(f"Data Loaded: Train shape {train_df.shape}, Test shape {test_df.shape}")
    
    # Run EDA
    analyze_target(train_df)
    analyze_text_stats(train_df)
    analyze_common_words(train_df)
    analyze_handcrafted_features(train_df)
    
    print("\nEDA finished. Results saved to logs/eda/")

if __name__ == "__main__":
    main()
