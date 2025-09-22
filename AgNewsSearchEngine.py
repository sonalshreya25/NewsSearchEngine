# News Search Engine
# Imports
import nltk
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download nltk resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("punkt_tab")

#Loading  AG News dataset
print("Loading AG News dataset...")
dataset = load_dataset("ag_news", split="train[:1000]")  # small subset for demo
articles = [{"label": item["label"], "title": item["text"], "content": item["text"]} for item in dataset]

# Label mapping from AG News (0=World, 1=Sports, 2=Business, 3=Sci/Tech)
label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Preprocessing function

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation/numbers
    text = re.sub(r"[^a-z\s]", "", text)
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords + lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess titles for embedding
processed_titles = [preprocess(article["title"]) for article in articles]

# Step 3: Load SentenceTransformer model
print("Loading sentence-transformers/multi-qa-MiniLM-L6-cos-v1...")
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# Precompute embeddings for dataset
article_embeddings = model.encode(processed_titles, convert_to_tensor=True, show_progress_bar=True)

# Function for Interactive Search

def news_search():
    # Ask user for query
    query = input("\n💡 What type of news are you interested in reading today?\n👉 Example: economy, football, politics, technology\n\nYour query: ")

    # Preprocess query
    query_processed = preprocess(query)

    # Encoding query
    query_embedding = model.encode(query_processed, convert_to_tensor=True)

    # Computing similarity scores
    scores = util.pytorch_cos_sim(query_embedding, article_embeddings)[0]

    # Getting top 10 results
    top_results = np.argsort(-scores)[:10]

    # Displaying top results
    print(f"\n🔎 Top {len(top_results)} articles related to '{query}':\n")
    for idx, result in enumerate(top_results, start=1):
        print(f"{idx}. {articles[result]['title']}  [{label_map[articles[result]['label']]}]")

    # Asking user to select one
    try:
        choice = int(input("\nEnter the number of the article you'd like to read: "))
        if 1 <= choice <= len(top_results):
            selected_idx = top_results[choice - 1]
            print("\n📰 Full Article:\n")
            print(articles[selected_idx]["content"])
        else:
            print("Invalid choice. Please run again.")
    except ValueError:
        print("Invalid input. Please enter a number.")


if __name__ == "__main__":
# Run search
    news_search()

