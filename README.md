# NewsSearchEngine
News Search engine using NLP 

## Project Overview

In today’s digital world, people are overwhelmed by the massive volume of news published every second. Finding relevant information quickly is both a **business problem** and a **user experience challenge**. Traditional keyword search systems often fail to capture context and meaning — for example, searching for “football” might miss articles titled “soccer” or “World Cup.”

This project addresses that problem by building a **semantic news search engine** using the **AG News dataset**. Instead of relying only on keywords, the system leverages **modern NLP embeddings** to understand the semantic meaning of user queries and news articles.  

The key goals of the project are:
- Provide users with **relevant news articles** based on natural language queries.  
- Combine **classical NLP preprocessing** (tokenization, stopword removal, lemmatization) with **modern embedding techniques** for robust search.  
- Demonstrate the use of **state-of-the-art sentence transformers** for semantic similarity.  
- Build an **interactive search system** where the user can:

  1. Enter a query (e.g., *economy, technology, sports, politics*).  
  2. View the **top 5–10 most relevant article titles** with categories (World, Sports, Business, Sci/Tech).  
  3. Select an article number from the list.  
  4. Read the **full article text** in the console.  

From an **academic and skill development perspective**, this project shows the integration of:
- Data loading and exploration using Hugging Face Datasets.  
- NLP text preprocessing with NLTK.  
- Feature extraction and vectorization using transformer models.  
- Information retrieval using cosine similarity.  
- End-to-end pipeline design for search.  

Ultimately, this system simulates how real-world applications like **Google News, Bing News, or AI-driven recommendation systems** retrieve and present relevant articles to their users.

This assignment demonstrates **Natural Language Processing (NLP)**, **text preprocessing**, and **semantic search using embeddings**.

---

##  Dataset
We use the **[AG News dataset](https://huggingface.co/datasets/ag_news)** from Hugging Face.  
- **Size:** ~120,000 news articles.  
- **Fields:**  
  - `label` → one of 4 categories:  
    - **0 → World** 
    - **1 → Sports**   
    - **2 → Business**   
    - **3 → Sci/Tech**   
  - `text` → full article text  

- **Processing:**  
  - I mapped numeric labels (0–3) to human-readable categories.  
  - Articles are cleaned and preprocessed before embedding.  

---

##  Workflow & Logic

### 1. Preprocessing
Applied classic NLP preprocessing to normalize input and dataset text:
- Lowercasing  
- Tokenization (splitting into words)  
- Removing punctuation  
- Removing stopwords (common words like *the, is, and*)  
- Lemmatization (reducing words to root form, e.g., *running → run*)  

This ensures clean and consistent input for semantic embedding.

---

### 2. Embedding
Used the **`multi-qa-MiniLM-L6-cos-v1`** model from **Sentence Transformers**:
- Converts text (queries and titles) into numerical vectors (embeddings).  
- Captures both **lexical** and **semantic** meaning.  

---

### 3. Semantic Search
- Compute embeddings for all article titles.  
- Encode the user’s query into an embedding.  
- Use **cosine similarity** to rank articles by closeness to the query.  
- Present the **top 10 results** with:
  - Title  
  - Category  

---

### 4. User Interaction
1. System asks:  What type of news are you interested in reading today? Example: economy, football, politics, technology

2. User enters query.  
3. System shows top results as a **numbered list**.  
4. User selects one by entering its number.  
5. System displays the **article selected**.  

---

##  Libraries & Techniques

### Libraries
- **[datasets](https://huggingface.co/docs/datasets)** → load AG News dataset.  
- **[sentence-transformers](https://www.sbert.net/)** → semantic embeddings & search.  
- **[torch](https://pytorch.org/)** → backend for embeddings.  
- **[nltk](https://www.nltk.org/)** → text preprocessing: stopwords, lemmatization, tokenization.  

### Techniques
- **Classic IR + Modern NLP**: combined preprocessing with semantic embeddings.  
- **Cosine Similarity**: measures similarity between query and article vectors.  
- **Interactive CLI Program**: simple user input/output for easy use.  

---

##  How to Run Locally

### 1. Install dependencies
Run the following command in your terminal or Anaconda Prompt to install required libraries:

```bash
pip install datasets sentence-transformers torch nltk
```
### 2.Run the script

Navigate to the folder containing your script and run:

python AgNewsSearchEngine.py



## Conclusion

This project successfully demonstrates how **text preprocessing**, **semantic embeddings**, and **cosine similarity** can be combined to build a practical **news search engine**.  

By using the **AG News dataset** with the **`multi-qa-MiniLM-L6-cos-v1`** transformer model, the system retrieves relevant articles based on user queries and provides an interactive way to explore news content.  

The project highlights:
- How classical NLP techniques (tokenization, stopword removal, lemmatization) enhance data quality.  
- How modern transformer-based models capture semantic meaning beyond simple keyword matching.  
- How an end-to-end **information retrieval pipeline** can be designed and implemented in Python.  

Overall, the system is a simplified version of real-world **news recommendation and search engines**, demonstrating the **practical applications of NLP in solving information overload problems**.  

