import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from src.features import TextStatsExtractor

# Download stopwords
nltk.download('stopwords', quiet=True)

def analyze_target(df: pd.DataFrame, output_dir: str = "logs/eda/") -> None:
    """Performs target variable analysis and saves plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    counts = df['author'].value_counts()
    percentages = df['author'].value_counts(normalize=True) * 100
    
    print("\n=== Target Distribution ===")
    for author, count in counts.items():
        print(f"{author}: {count} ({percentages[author]:.2f}%)")
        
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='author', hue='author', palette='viridis', legend=False)
    plt.title("Distribution of Authors (Classes)")
    plt.xlabel("Author")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "01_author_distribution.png"))
    plt.close()

def analyze_text_stats(df: pd.DataFrame, output_dir: str = "logs/eda/") -> None:
    """Saves boxplots for character and word counts."""
    df['char_count'] = df['text'].apply(len)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df, x='author', y='char_count', hue='author', palette='Set2', legend=False)
    plt.title("Character Count Distribution")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='author', y='word_count', hue='author', palette='Set2', legend=False)
    plt.title("Word Count Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_text_statistics.png"))
    plt.close()

def analyze_common_words(df: pd.DataFrame, output_dir: str = "logs/eda/", top_n: int = 15) -> None:
    """Visualizes top words excluding stopwords for each author."""
    authors = df['author'].unique()
    stop_words = set(stopwords.words('english'))
    
    fig, axes = plt.subplots(1, len(authors), figsize=(20, 8))
    
    for i, author in enumerate(authors):
        text = " ".join(df[df['author'] == author]['text']).lower()
        words = [word for word in text.split() if word.isalpha() and word not in stop_words]
        common = Counter(words).most_common(top_n)
        
        words_list, counts = zip(*common)
        sns.barplot(x=list(counts), y=list(words_list), ax=axes[i], hue=list(words_list), palette='magma', legend=False)
        axes[i].set_title(f"Top {top_n} Words: {author}")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_top_words_per_author.png"))
    plt.close()

def analyze_handcrafted_features(df: pd.DataFrame, output_dir: str = "logs/eda/") -> None:
    """Analyzes stylometric features and their correlation with the author."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract features
    extractor = TextStatsExtractor(text_col='text')
    features_df = extractor.transform(df)
    
    # Encode target
    le = LabelEncoder()
    target_encoded = le.fit_transform(df['author'])
    
    # Combine for correlation
    analysis_df = features_df.select_dtypes(include=[np.number]).copy()
    analysis_df['target'] = target_encoded
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    corr = analysis_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Correlation of Handcrafted Features with Author")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_feature_correlations.png"))
    plt.close()
    
    # 2. Key feature distributions
    key_features = ['punctuation_count', 'stopword_count', 'avg_word_len', 'upper_ratio']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    plot_df = features_df.copy()
    plot_df['author'] = df['author']
    
    for i, col in enumerate(key_features):
        if col in plot_df.columns:
            sns.boxplot(data=plot_df, x='author', y=col, ax=axes[i], hue='author', palette='Set3', legend=False)
            axes[i].set_title(f"Distribution of {col}")
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_handcrafted_distributions.png"))
    plt.close()
    
    print(f"Handcrafted feature analysis saved to {output_dir}")
