import pandas as pd
import yaml
from typing import Dict, Any, Tuple

def load_config(config_path: str = "configs/base_config.yaml") -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test dataframes based on config paths.
    
    Args:
        config (Dict[str, Any]): Project configuration.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Train and test dataframes.
    """
    train_path = config['paths']['train_data']
    test_path = config['paths']['test_data']
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df
