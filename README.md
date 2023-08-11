# Suicide Sentiment Prediction using Machine Learning and Deep Learning

## Table of Contents

- [About the Project](#about-the-project)
- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
  - [Logistic Regression](#logistic-regression)
  - [LSTM using Word2Vec](#lstm-using-word2vec)
  - [BERT Model](#bert-model)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About the Project

This project focuses on predicting the probability of suicide sentiment using social media text. By employing various Machine Learning and Deep Learning techniques, we aim to provide a tool that can assist in identifying individuals at risk and potentially intervene to provide support.

## Project Overview

The project is structured as follows:

1. *Data Collection*: Gathering social media text data related to suicide sentiment.
2. *Data Preprocessing*: Cleaning and preparing the data for model training.
3. *Model Implementation*: Developing predictive models using different techniques.
4. *Usage*: Instructions on how to use the trained models for sentiment prediction.
5. *Results*: Presenting the performance and accuracy of each model.

## Data Collection
- The project employs the Suicide and Depression Detection dataset, sourced from Kaggle, containing posts extracted from Reddit.
- The dataset comprises two primary columns: "text" indicating the actual content of the posts, and "class" denoting the assigned label for each post.
- Data collection involved scraping a total of 232,074 posts from two distinct subreddits: "Suicide Watch" and "teenagers."
- Posts originating from the "Suicide Watch" subreddit were categorized as "suicidal," while those from the "teenagers" subreddit were labeled as "non-suicidal."
- The dataset's architecture is based on two columns: "text" holding the post content and "class" signifying whether the post is suicidal or non-suicidal.
- The distribution of classes in the original dataset is balanced, with an equal number of posts falling into each category.
- This balance is demonstrated by the fact that there are 116,037 posts (50% of the total dataset) in both the "suicidal" and "non-suicidal" categories.
- The dataset serves as the foundational resource for the project's objective of developing models capable of detecting signs of suicide and depression based on textual content.
- Machine learning models are trained using this dataset to effectively differentiate between posts that exhibit suicidal tendencies and those that do not.
## Data Preprocessing

Sure, here's the provided information summarized in points:

**Text Preprocessing:**

- Data preparation involves transforming the raw text data into a suitable format for building models, especially important for unstructured social media data.
- The following steps were taken in sequence for data cleaning and preprocessing (as shown in Figure 3):
  1. Remove accented characters to reduce vocabulary size and ensure uniformity.
  2. Expand contractions to original forms for standardization.
  3. Convert all text to lowercase to standardize case usage.
  4. Eliminate URLs, symbols, digits, special characters, and extra spaces as irrelevant.
  5. Address word lengthening where characters are wrongly repeated.
  6. Implement spelling correction using the Symspell algorithm for accurate words.
  7. Remove common stop words except "no" and "not" due to their context.
  8. Lemmatization to reduce words to their root form, retaining contextual meaning.

**Data Cleaning:**

- Initial exploration of most frequent words revealed an unusual term "filler" (55,442 occurrences) likely captured during data collection; it was removed.
- Empty rows without words after preprocessing were dropped.
- Word count distribution in preprocessed posts revealed outliers with excessively high word counts.
- Outliers were addressed by setting a threshold at the 75th percentile (62 words), removing posts exceeding this limit to enhance model training efficiency.
- The final cleaned dataset consists of 174,436 rows, containing processed text in the "clean_msg" column.
- The class distribution of the cleaned dataset is displayed in Figure 6, indicating a shift from the 5:5 suicidal-to-non-suicidal ratio in the original dataset (Figure 2) to an approximately 4:6 ratio. This change is due to the removal of longer suicidal posts during data cleaning.
- Despite a slight imbalance, the class distribution still allows for a normal classification problem, with model performance expected to remain unaffected.

## Model Implementation

### Logistic Regression

**Logistic Regression in NLP:**

- Logistic regression is a fundamental supervised algorithm for classification in Natural Language Processing (NLP).
- It has a close relationship with neural networks, as it can be seen as stacked logistic regression classifiers.

**Model Architecture:**

- Logistic regression consists of four main components: input feature representation, classification function for class estimation, objective function for learning, and algorithm for optimization.
- For the baseline logistic regression model, a validation dataset is not used, and the dataset is split into an 80-20 ratio for training and testing.
- The model learns weights and a bias term from the training dataset, where weights represent feature importance.
- The learned weights are then used to classify test data by multiplying features with their corresponding weights, adding bias, and passing through a sigmoid function for classification.

**Model Variants:**

- Two logistic regression model variants were experimented with:
  - Logit Model 1: Custom Word2Vec Embeddings (300 dimensions)
  - Logit Model 2: Pre-trained GloVe Embeddings (200 dimensions)

**Hyperparameters:**

- Default hyperparameters were used for the models.

**Model Performance:**

- Model performance for both variants is shown in Table 1.
- The best-performing model is Model 1 (Custom Word2Vec Embeddings), outperforming Model 2 (Pre-trained GloVe Embeddings) across all metrics.

**Comparison with Pre-trained Embeddings:**

- In regression and classification, pre-trained word embeddings often don't perform as well as embeddings learned from the original dataset.
- This could be due to the specialized nature of suicide detection and the influence of vocabulary differences.
- Pre-trained embeddings are more suited for low-resource scenarios.
- Custom trained word embeddings perform better, especially in niche use cases like suicide detection.

**Table 1: Logit Models Performance Comparison:**

| Logit Model Variants           | Accuracy | Recall | Precision | F1 Score |
|---------------------------------|----------|--------|-----------|----------|
| Custom Word2Vec Embedding       | 0.9111   | 0.8870 | 0.8832    | 0.8851   |
| Pre-trained GloVe Embedding     | 0.8774   | 0.8440 | 0.8394    | 0.8417   |

This summary captures the key points and findings from the provided information.

### LSTM using Word2Vec

**LSTM Model:**

- LSTM is a type of RNN that addresses vanishing gradient problems and retains memory of relevant information.
- It uses input, output, and forget gates to control information flow, allowing it to learn dependencies over time intervals.
- The architecture comprises 5 layers:
  1. Tokenization of words into integers.
  2. Embedding layer for word-to-vector conversion.
  3. LSTM layer with gates controlling information flow.
  4. Fully connected layer to map LSTM output.
  5. Sigmoid activation for classification.
- Model output reshaped to match batch size.

**Model Variants:**

- Three LSTM model variants experimented with embedding layers:
  - LSTM Model 1: Random Initialization (no pre-trained weights)
  - LSTM Model 2: Custom Word2Vec Embeddings (300 dimensions)
  - LSTM Model 3: Pre-trained GloVe Embeddings (200 dimensions)

**Hyperparameters:**

- Experimentation determined optimal hyperparameter values.
- Embedding size: 300, 128 hidden dimensions, 2 layers within LSTM.
- Dropout rate: 0.5.
- Adam optimizer with learning rate 0.00001.
- BCEWithLogitsLoss loss function.

**Model Training:**

- Training graph displayed stable train and validation accuracies across epochs for all model variants.

**Model Performance:**

- Model performance for LSTM variants presented in Table 3.
- Best performance achieved by Model 2 (Custom Word2Vec Embeddings) across all metrics.
- Consistent with findings from Logistic Regression and CNN models.

**Table 3: LSTM Models Performance Comparison:**

| LSTM Model Variants          | Accuracy | Recall | Precision | F1 Score |
|------------------------------|----------|--------|-----------|----------|
| Random Initialization        | 0.8724   | 0.7982 | 0.8611    | 0.8285   |
| Custom Word2Vec Embedding    | 0.9260   | 0.8649 | 0.9386    | 0.9003   |
| Pre-trained GloVe Embedding  | 0.8825   | 0.7613 | 0.9206    | 0.8334   |

This summary captures the key points and results from the provided information.

### BERT Model

**BERT (Bidirectional Encoder Representations from Transformers):**

- BERT, developed by Google in 2018, utilizes the transformer's encoder structure for language modeling.
- Pre-trained on two tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP).
- MLM involves masking words and training the model to predict masked tokens.
- NSP captures sentence-level features by predicting if a sentence follows another.

**Model Architecture:**

- BERT is pre-trained on large corpora, capturing bidirectional context.
- Figure 13 illustrates the Masked Language Model technique, where masked words are predicted.
- Figure 14 depicts Next Sentence Prediction, which captures sentence-level information.
- MLM replaces 15% of words with "[MASK]" token for training.
- NSP uses sentence pairs, predicting whether the second sentence follows the first.
- Tokens "[CLS]" and "[SEP]" indicate sentence boundaries.

**Implementation:**

- Hugging Face Transformers library used for BERT implementation.
- BERT base model chosen due to computational constraints.
- BERT fine-tuned on one epoch, batch size of 6, learning rate of 0.00001.

**Model Variants:**

- Two BERT model variants experimented with fine-tuning:
  - BERT Model 1: Pre-trained
  - BERT Model 2: Fine-tuned

**Model Performance:**

- Model performance for BERT variants presented in Table 4.
- Best performance achieved by Model 2 (Fine-tuned BERT) across all metrics.
- Pre-trained BERT exhibited high recall but subpar F1 score, indicating it predicts most inputs as positive.

**Table 4: BERT Models Performance Comparison:**

| BERT Model Variants      | Accuracy | Recall | Precision | F1 Score |
|--------------------------|----------|--------|-----------|----------|
| Pre-trained BERT         | 0.4681   | 0.9295 | 0.4156    | 0.5744   |
| Fine-tuned BERT          | 0.9757   | 0.9669 | 0.9701    | 0.9685   |

This summary captures the key points and results from the provided information.

## Usage
In the code files we will find the correct way to save the model which we have maded and after saving that model we can run it on any text as shown in the image 
![image](https://github.com/prashant67690/Suicide-Analyzer/assets/80661803/5972f6ab-6176-4c2c-b049-e71f555d811e)

## Results

**Summary of Model Results:**

| Best Models                     | Accuracy | Recall | Precision | F1 Score |
|---------------------------------|----------|--------|-----------|----------|
| Logistic Regression             | 0.9111   | 0.8870 | 0.8832    | 0.8851   |
| Convolutional Neural Network    | 0.9285   | 0.9013 | 0.9125    | 0.9069   |
| Long Short-Term Memory Network  | 0.9260   | 0.8649 | 0.9386    | 0.9003   |
| BERT                            | 0.9757   | 0.9669 | 0.9701    | 0.9685   |

**Model Comparison:**

- The best variations for Logistic Regression, CNN, and LSTM involve custom-trained Word2Vec embeddings.
- BERT, fine-tuned on the dataset, has shown outstanding performance across all metrics.
- BERT achieved the highest scores in Precision, F1 score, accuracy, and recall.
- The transformer models, particularly BERT, have outperformed the other models for all metrics.

**Final Model Selection:**

- BERT has demonstrated superior performance in all evaluated aspects.
- Therefore, BERT is chosen as the final model due to its exceptional accuracy and comprehensive evaluation across metrics.

This summary captures the key results and conclusions drawn from the provided information.

## Contributing

If you'd like to contribute to this project, please follow these steps:
1. Fork this repository.
2. Create a new branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m 'Add some feature'`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a pull request.

## Contact

Provide your contact information or ways for users to reach out with questions, concerns, or feedback.
