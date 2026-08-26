from transformers import pipeline

def main():
    print("Loading DeBERTa-v3 Zero-Shot NLI Model...")
    # NLI Transformer trained on logical premise-hypothesis entailment
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    print("Model ready!\n")

    test_reviews = [
        "Rdj really was made for this role I could never imagine Tom Cruise doing all of this",
        "Whoever dislikes this movie is not welcome in this household",
        "My favourite american military propaganda movie I dont care this shit gets five stars from me",
        "Still waiting for the story to begin tbh"
    ]

    # Star rating hypotheses for the NLI engine to evaluate logically
    labels = ["5 stars positive review", "1 star negative review"]

    print("=" * 60)
    print("ZERO-SHOT NLI LOGICAL EVALUATION")
    print("=" * 60)

    for review in test_reviews:
        result = classifier(review, candidate_labels=labels)
        top_label = result['labels'][0]
        confidence = result['scores'][0]

        # Formatting label for output
        star_prediction = "5 STARS" if "5 stars" in top_label else "1 STAR"

        print(f"\nReview: \"{review}\"")
        print(f"--> Predicted Rating: {star_prediction}")
        print(f"--> Confidence: {confidence:.1%}")
        print("-" * 60)

if __name__ == "__main__":
    main()