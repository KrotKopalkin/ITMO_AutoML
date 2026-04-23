import pandas as pd
import numpy as np
import re
import string
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional

class TextStatsExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts numerical features from text for stylometric analysis.
    Optimized with vectorized pandas operations.
    """
    def __init__(self, text_col: str = 'text'):
        self.text_col = text_col
        self._stopwords = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # If X is a Series (from ColumnTransformer), convert to DataFrame
        if isinstance(X, pd.Series):
            df = X.to_frame(name=self.text_col)
        else:
            df = X.copy()
            
        texts = df[self.text_col].astype(str)

        # Basic counts (Vectorized)
        df['char_count'] = texts.str.len()
        df['word_count'] = texts.str.split().str.len()
        # Count punctuation
        df['punctuation_count'] = texts.apply(lambda x: len([c for c in x if c in string.punctuation]))
        
        # Stylometric features
        df['avg_word_len'] = df['char_count'] / (df['word_count'] + 1)
        
        # Upper ratio
        df['upper_ratio'] = texts.apply(lambda x: sum(1 for c in x if c.isupper())) / (df['char_count'] + 1)
        
        # Specific punctuation
        df['exclamation_count'] = texts.str.count('!')
        df['question_count'] = texts.str.count(r'\?')
        df['comma_count'] = texts.str.count(',')
        
        # Stopwords and unique words (still using apply but can be faster)
        # Note: NLTK stopwords need to be pre-loaded
        if self._stopwords is None:
            from nltk.corpus import stopwords
            self._stopwords = set(stopwords.words('english'))
            
        df['stopword_count'] = texts.apply(lambda x: len([w for w in x.lower().split() if w in self._stopwords]))
        df['unique_word_ratio'] = texts.apply(lambda x: len(set(x.lower().split()))) / (df['word_count'] + 1)

        # Return only the new features
        return df.drop(columns=[self.text_col])

class TextCleaner(BaseEstimator, TransformerMixin):
    """Cleans text: lowercase, remove punctuation, etc."""
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return X.astype(str).apply(self._clean)

    def _clean(self, text: str) -> str:
        text = text.lower()
        # Remove punctuation except spaces
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()
        return text
