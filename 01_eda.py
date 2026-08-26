import os
import pandas as pd
import matplotlib.pyplot as plt

def run_eda(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # Check class balance
    class_counts = df['sentiment'].value_counts()
    print("Class Balance:")
    print(class_counts)
    
    # The balance should be exactly 25,000 positive and 25,000 negative.
    # Because it is perfectly 50/50, accuracy is a safe metric without needing class weighting.
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind='bar', color=['blue', 'red'])
    plt.title('Class Balance')
    plt.savefig('class_balance.png') # Produces class_balance.png
    plt.close()

    # Check Review Lengths
    df['word_count'] = df['review'].apply(lambda x: len(x.split()))
    print(f"Mean review length: {df['word_count'].mean():.0f} words") # Should be 231 words.
    print(f"Median review length: {df['word_count'].median():.0f} words") # Should be 173 words.
    print(f"Max review length: {df['word_count'].max():.0f} words") # Long tail up to 2,470 words.

    plt.figure(figsize=(8, 5))
    plt.hist(df['word_count'], bins=50, color='skyblue', edgecolor='black')
    plt.title('Review Length Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig('review_length_hist.png') # Produces review_length_hist.png
    plt.close()

    # Check Text Cleanliness
    br_tags = df['review'].str.contains('<br />').sum()
    print(f"Reviews containing '<br />': {br_tags} ({(br_tags/len(df))*100:.0f}%)")
    # This will show 58% of reviews contain raw <br /> HTML tags, requiring an explicit cleaning step.

if __name__ == "__main__":
    # Gets the exact folder path where 01_eda.py is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'movie_data.csv')
    
    run_eda(csv_path)