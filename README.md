# Offensive Tweet Detection and Categorization

This project focuses on building a machine learning model to classify tweets as offensive or non-offensive and categorize offensive tweets into specific types. 
The goal is to create a model that can identify offensive content on social media platforms and categorize it.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Usage](#usage)
- [Credits](#credits)

## Dataset

### Offensive Language Identification Dataset (OLID)
_v 1.0: March 15 2018
https://scholar.harvard.edu/malmasi/olid_
#
Levels **A** and **B** were my goals for this project.

**Offensive language identification (LEVEL A)**
- (NOT) Not Offensive - This post does not contain offense or profanity.
- (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

**Automatic categorization of offense types (LEVEL B)**
- (TIN) Targeted Insult and Threats - A post containing an insult or threat to an individual, a group, or others (see categories in sub-task C).
- (UNT) Untargeted - A post containing non-targeted profanity and swearing.
- (NULL) - For non-offensive posts.

**Offense target identification (LEVEL C)**
- (IND) Individual - The target of the offensive post is an individual: a famous person, a named individual or an unnamed person interacting in the conversation.
- (GRP) Group - The target of the offensive post is a group of people considered as a unity due to the same ethnicity, gender or sexual orientation, political affiliation, religious belief, or something else.
- (OTH) Other – The target of the offensive post does not belong to any of the previous two categories (e.g., an organization, a situation, an event, or an issue)

**Here are the possible label combinations in the dataset**
-	NOT NULL
-	OFF UNT
-	OFF TIN

[_THE DATASET IS CITED BELOW_](#citation)

## Preprocessing

The text data is preprocessed using the following steps:

- Column Removal:  Dropped the 'id' column from the dataset.
- Missing Value Handling: Filled NULL values in the `category` column with a default value `'NULL'` and removed rows with missing values in the `tweet` column.
- Lowercasing: Converted all text to lowercase.
- Mention Removal: Removed "@user" mentions from the text.
- Symbol Removal: Removed non-alphanumeric characters and symbols.
- Tokenization: Tokenized the text into words using NLTK's `word_tokenize()` function.
- Lemmatization: Performed lemmatization on the tokens using NLTK's WordNetLemmatizer
- Convert preprocessed text into TF-IDF features using the trained TF-IDF vectorizer..
- Vectorized the data using `TF-IDF` using `TfidfVectorizer` from `sklearn`.

## Model Training

Two instances of the logistic regression model is used for training the offensive detection and category detection. 

**Model: Logistic Regression**
- Parameters: max_iter=1000 This parameter ensures that the optimization algorithm has enough iterations to find a solution.


## Model Evaluation

The trained models are evaluated using classification reports and confusion matrices to measure their performance on test data. 
The classification report provides precision, recall, and F1-score for different classes. The confusion matrix gives insight into the true positives, true negatives, false positives, and false negatives.

![image](https://github.com/ShakeefAhmedRakin/Offensive-Tweet-Classifier/assets/112527326/b52a9ee0-5e69-4315-bfaf-ef43007d3e56)


## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/ShakeefAhmedRakin/Offensive-Tweet-Classifier.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r pandas scikit-learn nltk
   ```
   
3. Run the code to preprocess the data, train the models, and evaluate their performance.

4. Explore the sample tweets provided in the code to see how the models classify and categorize different types of content.

## Credits

### CITATION

Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019). Predicting the Type and Target of Offensive Posts in Social Media. In _Proceedings of NAACL_.
[Link](https://arxiv.org/abs/1902.09666)

Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019). SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval). In _Proceedings of The 13th International Workshop on Semantic Evaluation (SemEval)._


