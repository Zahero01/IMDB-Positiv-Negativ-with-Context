import os
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'train.csv')

    print("Loading dataset...")
    df = pd.read_csv(train_path).dropna()

    print("Loading pre-trained Transformer encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating dense semantic context vectors for training data...")
    # Encodes review texts into 384-dimensional contextual vectors
    X_train = encoder.encode(df['review'].tolist(), show_progress_bar=True, batch_size=64)
    y_train = df['sentiment']

    print("Training Logistic Regression on context embeddings...")
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    # Save trained model artifact
    model_save_path = os.path.join(base_dir, 'context_sentiment_model.pkl')
    joblib.dump(model, model_save_path)
    print(f"\nSUCCESS: Model saved as '{model_save_path}'!")

if __name__ == "__main__":
    main()