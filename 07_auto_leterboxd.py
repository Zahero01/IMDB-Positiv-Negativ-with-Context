import os
import re
import warnings
warnings.filterwarnings("ignore")

from bs4 import BeautifulSoup
from fpdf import FPDF
from curl_cffi import requests
from transformers import pipeline

# ==================== NLI BINARY SENTIMENT LOGIC ====================
def predict_binary_sentiment(classifier, review_text):
    """
    Evaluates review context using Zero-Shot NLI logic.
    Outputs strictly POSITIVE or NEGATIVE without hardcoded rules.
    """
    candidate_labels = ["positive movie review", "negative movie review"]
    result = classifier(review_text, candidate_labels=candidate_labels)
    
    top_label = result['labels'][0]
    return "POSITIVE" if "positive" in top_label else "NEGATIVE"


# ==================== PDF GENERATOR ====================
def generate_pdf(reviewer_name, review_history):
    if not review_history:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    clean_name = reviewer_name.encode('latin-1', 'ignore').decode('latin-1')

    # Banner Header
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(0, 0, 210, 35, 'F')
    
    pdf.set_font("Arial", 'B', 18)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 8)
    pdf.cell(190, 10, txt="AI Context Sentiment Report", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 11)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(190, 6, txt=f"Letterboxd Profile: {clean_name}", ln=True, align='C')

    pdf.set_y(42)

    for i, (movie, review, sentiment_str) in enumerate(review_history, 1):
        clean_movie = movie.encode('latin-1', 'ignore').decode('latin-1')
        clean_review = review.encode('latin-1', 'ignore').decode('latin-1')

        if pdf.get_y() > 220:
            pdf.add_page()
            pdf.set_y(20)

        # Entry Header
        pdf.set_font("Arial", 'B', 10)
        pdf.set_fill_color(241, 245, 249)
        pdf.set_draw_color(203, 213, 225)
        pdf.set_text_color(15, 23, 42)
        
        pdf.cell(25, 8, txt=f" Entry #{i}", border=1, fill=True)
        pdf.cell(165, 8, txt=f" Movie: {clean_movie}", border=1, ln=True, fill=True)

        # Review Text Row
        pdf.set_font("Arial", '', 10)
        pdf.set_fill_color(255, 255, 255)
        pdf.multi_cell(190, 7, txt=f' Review: "{clean_review}"', border='LR')

        # Sentiment Badge Row (Green for Positive, Red for Negative)
        pdf.set_font("Arial", 'B', 10)
        if sentiment_str == "POSITIVE":
            pdf.set_fill_color(220, 252, 231)  # Soft Green
            pdf.set_text_color(22, 101, 52)    # Dark Green text
        else:
            pdf.set_fill_color(254, 226, 226)  # Soft Red
            pdf.set_text_color(153, 27, 27)    # Dark Red text

        pdf.cell(190, 9, txt=f" The better than Claude Z-AI says this review is: {sentiment_str}", border=1, ln=True, fill=True)
        
        pdf.set_text_color(0, 0, 0)
        pdf.ln(6)

    pdf_filename = f"{clean_name.replace(' ', '_')}_sentiment_report.pdf"
    pdf.output(pdf_filename)
    return pdf_filename


# ==================== SCRAPER ====================
def extract_username(user_input):
    cleaned = user_input.strip().rstrip('/')
    if "letterboxd.com/" in cleaned:
        parts = cleaned.split("letterboxd.com/")[-1].split("/")
        return parts[0]
    return cleaned

def fetch_letterboxd_reviews(user_input, max_reviews=50):
    username = extract_username(user_input)
    session = requests.Session(impersonate="chrome")
    
    reviews_data = []
    seen_movies = set() 

    rss_url = f"https://letterboxd.com/{username}/rss/"
    
    try:
        response = session.get(rss_url, timeout=10)
        if response.status_code == 200:
            try:
                soup = BeautifulSoup(response.content, 'xml')
            except Exception:
                soup = BeautifulSoup(response.content, 'html.parser')
                
            items = soup.find_all('item')
            for item in items:
                if len(reviews_data) >= max_reviews:
                    break
                link_tag = item.find('link')
                if link_tag and '/list/' in link_tag.text.lower():
                    continue

                film_title_tag = item.find(lambda tag: 'filmtitle' in tag.name.lower())
                if film_title_tag:
                    title = film_title_tag.text.strip()
                else:
                    title_tag = item.find('title')
                    if title_tag:
                        raw_title = title_tag.text
                        if ' - ' in raw_title:
                            title = raw_title.split(' - ')[0].split(',')[0].strip()
                        else:
                            continue
                    else:
                        continue

                desc_tag = item.find('description')
                if desc_tag and title:
                    desc_soup = BeautifulSoup(desc_tag.text, 'html.parser')
                    for img in desc_soup.find_all('img'):
                        img.decompose()
                        
                    raw_text = desc_soup.get_text(separator=' ').strip()
                    raw_text = re.sub(r'\s+', ' ', raw_text)

                    if "view the full list on letterboxd" in raw_text.lower():
                        continue

                    clean_review = re.sub(
                        r'^Watched on [A-Za-z]+ [A-Za-z]+ \d{1,2}, \d{4}\.?', 
                        '', raw_text, flags=re.IGNORECASE
                    ).strip()
                    clean_review = re.sub(r'^[★½\s]+', '', clean_review).strip()

                    if clean_review and len(clean_review) > 2 and title not in seen_movies:
                        reviews_data.append((title, clean_review))
                        seen_movies.add(title)
                        
    except Exception:
        pass

    return username, reviews_data


# ==================== MAIN EXECUTION ====================
def main():
    print("Loading Context Sentiment Model...")
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    print("Model ready!")

    print("\n" + "=" * 60)
    user_input = input("Paste Letterboxd Username or Profile URL: ")
    limit_input = input("How many written reviews to scrape? (Press Enter for 50): ")
    print("=" * 60)

    try:
        max_reviews = int(limit_input) if limit_input.strip() else 50
    except ValueError:
        max_reviews = 50

    print(f"\nFetching reviews from Letterboxd...")
    username, scraped_reviews = fetch_letterboxd_reviews(user_input, max_reviews=max_reviews)

    if not scraped_reviews:
        print(f"ERROR: Could not find written reviews for '{username}'.")
        return

    print(f"Acquired {len(scraped_reviews)} written reviews for '{username}'.")
    print("Analyzing sentiment with context model...")

    review_history = []
    for movie, review in scraped_reviews:
        sentiment_str = predict_binary_sentiment(classifier, review)
        review_history.append((movie, review, sentiment_str))

    print("Generating PDF report...")
    pdf_filename = generate_pdf(username, review_history)

    if pdf_filename:
        print("\n" + "=" * 60)
        print(f"SUCCESS: Generated PDF report -> '{pdf_filename}'")
        print("=" * 60)

if __name__ == "__main__":
    main()