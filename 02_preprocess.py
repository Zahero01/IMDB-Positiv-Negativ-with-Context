import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    # Strip HTML tags, lowercase, remove non-alphabetic characters, and collapse whitespace.
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_data(file_path):
    print("Loading and cleaning data...")
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_text)
    
    # Perform an 80/20 stratified train/test split with a fixed random seed of 42.
    # Stratified preserves the 50/50 class ratio in both splits.
    df_train, df_test = train_test_split(df, test_size=0.20, stratify=df['sentiment'], random_state=42)
    
    # Save to train.csv (40,000 reviews) and test.csv (10,000 reviews).
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    print("Saved train.csv and test.csv")

if __name__ == "__main__":
    preprocess_data('movie_data.csv')