# Research Notebook: Text Classification with Traditional ML and LLMs

## Overview
This research notebook explores text classification using both traditional machine learning models and a large language model (LLM). The dataset used is the `20 Newsgroups` dataset, with four selected categories: 
- `rec.motorcycles`
- `rec.sport.baseball`
- `rec.sport.hockey`
- `sci.electronics`

The core purpose of this research is to demonstrate that LLMs can achieve comparable performance to traditional ML models in text classification tasks by comparing their accuracy and evaluation metrics.

## Data Preprocessing
1. The `fetch_20newsgroups` dataset is loaded, and only relevant categories are selected.
2. The dataset is converted into a Pandas DataFrame with columns:
   - `text`: The raw text content of the documents.
   - `target`: Numerical label for each category.
   - `target_name`: Original category name.
3. The category names are mapped to simplified labels (`motorcycles`, `baseball`, `hockey`, `electronics`).
4. The dataset is shuffled and split into:
   - `df_train`: 80% for training.
   - `df_test`: 20% for testing.
5. Text data is transformed using `TfidfVectorizer`:
   - Removes English stop words.
   - Ignores words appearing in more than 50% of documents (`max_df=0.5`).
   - Ignores words appearing in fewer than 2 documents (`min_df=2`).

## Traditional Machine Learning Models
Four models are trained and tested on the dataset:
1. **Gradient Boosting Classifier** (100 estimators, `random_state=42`)
2. **Decision Tree Classifier** (`random_state=42`)
3. **Random Forest Classifier** (100 estimators, `random_state=42`)
4. **Multinomial Naïve Bayes**

Each model is trained on `X_train` (TF-IDF transformed text) and evaluated on `X_test` using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

The results are printed for comparison.

## LLM-based Classification
An alternative classification approach is tested using `GPT-4 Turbo` from OpenAI.

1. A prompt-based approach is used where the model is instructed to classify each text into one of the predefined categories.
2. Each test sample is passed through the LLM using `langchain.chat_models.ChatOpenAI`.
3. Predictions are stored in a new column, `predicted_category`.
4. Performance is evaluated using precision, recall, and F1-score.

## Comparison of Model Performance
| Model                  | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|-----------|--------|----------|
| Gradient Boosting      | 0.808    | 0.842      | 0.808   | 0.815    |
| Decision Tree         | 0.691     | 0.708      | 0.691   | 0.695     |
| Random Forest         | 0.844     | 0.854      | 0.844   | 0.846     |
| Naïve Bayes           | 0.900     | 0.904      | 0.900   | 0.901     |
| GPT-4 Turbo (LLM)     | 0.916     | 0.924      | 0.916   | 0.916     |

**Key Findings:**
- The LLM-based approach achieves comparable or even superior accuracy compared to traditional ML models.
- GPT-4 Turbo performs well without explicit feature engineering or training on the dataset.
- This suggests that LLMs can be viable alternatives for text classification tasks.

## Summary
This notebook provides insights into the effectiveness of:
- Traditional ML models trained on TF-IDF vectorized text.
- LLMs as zero-shot classifiers for text categorization.
- A direct comparison of model performance using accuracy and other metrics.

This is **not** a production-ready implementation but a research-focused experiment for understanding classification performance using different approaches.

## Requirements
- `sklearn`
- `pandas`
- `tqdm`
- `langchain`
- `openai`

## Usage
1. Ensure you have an OpenAI API key (`OPEN_AI_KEY`).
2. Run the notebook in an environment with the required dependencies installed.
3. Evaluate and compare model performance.

---
**Note:** The results may vary based on OpenAI's LLM responses and dataset randomness.

