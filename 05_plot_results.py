import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison():
    # The final accuracy results from your scripts
    models = ['Logistic Regression\n(TF-IDF)', 'Linear SVM\n(TF-IDF)', 'LSTM\n(From Scratch)']
    accuracies = [91.36, 91.29, 81.51]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create colors: Classic ML models in one color, Deep Learning in another
    colors = ['#1f77b4', '#1f77b4', '#ff7f0e']
    
    # Plot the bars
    bars = ax.bar(models, accuracies, color=colors, width=0.6)

    # Add text labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", 
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Formatting
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('IMDB Sentiment Analysis: Model Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 105) # Set y-axis to 105 to leave room for labels
    
    # Add a horizontal grid for easier reading
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Save the figure
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    print("Plot successfully saved as 'model_comparison.png'")
    
    # Display the plot window
    plt.show()

if __name__ == "__main__":
    plot_model_comparison()