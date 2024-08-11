import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from textblob import TextBlob  # Added missing import

# Download necessary NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Load the dataset
dataset = pd.read_csv("Musical_instruments_reviews.csv")

# Preprocessing
dataset.shape
dataset.isnull().sum()
dataset["reviewText"] = dataset["reviewText"].fillna(value="")
dataset["reviews"] = dataset["reviewText"] + " " + dataset["summary"]
dataset.drop(columns=["reviewText", "summary"], axis=1, inplace=True)

# Descriptive statistics
dataset.describe(include="all")

# Visualization: Distribution of Ratings
dataset.overall.value_counts().plot(kind="pie", legend=False, autopct="%1.2f%%", fontsize=10, figsize=(8,8))
plt.title("Percentages of Ratings Given From the Customers", loc="center")
plt.show()

# Labeling sentiments based on ratings
def Labelling(Rows):
    if(Rows["overall"] > 3.0):
        Label = "Positive"
    elif(Rows["overall"] < 3.0):
        Label = "Negative"
    else:
        Label = "Neutral"
    return Label

dataset["sentiment"] = dataset.apply(Labelling, axis=1)

# Visualization: Sentiment distribution
dataset["sentiment"].value_counts().plot(kind="bar", color="blue")
plt.title("Amount of Each Sentiment Based on Rating Given", loc="center", fontsize=15, color="red", pad=25)
plt.xlabel("Sentiments", color="green", fontsize=10, labelpad=15)
plt.xticks(rotation=0)
plt.ylabel("Amount of Sentiments", color="green", fontsize=10, labelpad=15)
plt.show()

# Text cleaning
def Text_Cleaning(Text):
    Text = Text.lower()
    punc = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    Text = Text.translate(punc)
    Text = re.sub(r'\d+', '', Text)
    Text = re.sub('https?://\S+|www\.\S+', '', Text)
    Text = re.sub('\n', '', Text)
    return Text

Stopwords = set(stopwords.words("english")) - set(["not"])

def Text_Processing(Text):
    Processed_Text = list()
    Lemmatizer = WordNetLemmatizer()
    Tokens = nltk.word_tokenize(Text)  # Fixed the variable name
    for word in Tokens:
        if word not in Stopwords:
            Processed_Text.append(Lemmatizer.lemmatize(word))
    return " ".join(Processed_Text)

dataset["reviews"] = dataset["reviews"].apply(lambda Text: Text_Cleaning(Text))
dataset["reviews"] = dataset["reviews"].apply(lambda Text: Text_Processing(Text))

# Sentiment Polarity and Length
dataset["polarity"] = dataset["reviews"].map(lambda Text: TextBlob(Text).sentiment.polarity)
dataset["polarity"].plot(kind="hist", bins=40, edgecolor="blue", linewidth=1, color="orange", figsize=(10,5))
plt.title("Polarity Score in Reviews", color="blue", pad=20)
plt.xlabel("Polarity", labelpad=15, color="red")
plt.ylabel("Amount of Reviews", labelpad=20, color="green")
plt.show()

dataset["length"] = dataset["reviews"].astype(str).apply(len)
dataset["length"].plot(kind="hist", bins=40, edgecolor="blue", linewidth=1, color="orange", figsize=(10,5))
plt.title("Length of Reviews", color="blue", pad=20)
plt.xlabel("Length", labelpad=15, color="red")
plt.ylabel("Amount of Reviews", labelpad=20, color="green")
plt.show()

# N-gram Analysis
def Gram_Analysis(Corpus, Gram, N):
    Vectorizer = CountVectorizer(stop_words=list(Stopwords), ngram_range=(Gram, Gram))
    ngrams = Vectorizer.fit_transform(Corpus)
    Count = ngrams.sum(axis=0)
    words = [(word, Count[0, idx]) for word, idx in Vectorizer.vocabulary_.items()]
    words = sorted(words, key=lambda x: x[1], reverse=True)
    return words[:N]

# Example usage of Gram_Analysis
Positive = dataset[dataset["sentiment"] == "Positive"].dropna()
Neutral = dataset[dataset["sentiment"] == "Neutral"].dropna()
Negative = dataset[dataset["sentiment"] == "Negative"].dropna()

# Unigram Visualization
words = Gram_Analysis(Positive["reviews"], 1, 20)
Unigram = pd.DataFrame(words, columns=["words", "Counts"])
Unigram.groupby("words").sum()["Counts"].sort_values().plot(kind="barh", color="green", figsize=(10,5))
plt.title("Unigram of Reviews with Positive Sentiments", loc="center", fontsize=15, color="blue", pad=25)
plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=15)
plt.xticks(rotation=0)
plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=15)
plt.show()

words = Gram_Analysis(Neutral["reviews"], 1, 20)
Unigram = pd.DataFrame(words, columns=["words", "Counts"])
Unigram.groupby("words").sum()["Counts"].sort_values().plot(kind="barh", color="red", figsize=(10,5))
plt.title("Unigram of Reviews with Neutral Sentiments", loc="center", fontsize=15, color="blue", pad=25)
plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=15)
plt.xticks(rotation=0)
plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=15)
plt.show()

words = Gram_Analysis(Negative["reviews"], 1, 20)
Unigram = pd.DataFrame(words, columns=["words", "Counts"])
Unigram.groupby("words").sum()["Counts"].sort_values().plot(kind="barh", color="orange", figsize=(10,5))
plt.title("Unigram of Reviews with Negative Sentiments", loc="center", fontsize=15, color="blue", pad=25)
plt.xlabel("Total Counts", color="magenta", fontsize=10, labelpad=15)
plt.xticks(rotation=0)
plt.ylabel("Top Words", color="cyan", fontsize=10, labelpad=15)
plt.show()

# WordCloud Visualization
wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=Stopwords).generate(str(Positive["reviews"]))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=Stopwords).generate(str(Neutral["reviews"]))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_words=50, width=3000, height=1500, stopwords=Stopwords).generate(str(Negative["reviews"]))
plt.figure(figsize=(15,15))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Preparing Data for Model Training
Columns = ["reviewerID", "asin", "reviewerName", "helpful", "unixReviewTime", "reviewTime", "polarity", "length", "overall"]
dataset.drop(columns=Columns, axis=1, inplace=True)
dataset.head()

Encoder = LabelEncoder()
dataset["sentiment"] = Encoder.fit_transform(dataset["sentiment"])
dataset["sentiment"].value_counts()

TF_IDF = TfidfVectorizer(max_features=5000, ngram_range=(2,2))
X = TF_IDF.fit_transform(dataset["reviews"])
X.shape

y = dataset["sentiment"]
Counter(y)

Balancer = SMOTE(random_state=42)
X_final, y_final = Balancer.fit_resample(X, y)
Counter(y_final)

x_train, x_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=42)

# Model Training and Evaluation
DTree = DecisionTreeClassifier()
LogReg = LogisticRegression()
SVC = SVC()
RForest = RandomForestClassifier()
Bayes = BernoulliNB()
KNN = KNeighborsClassifier()

Models = [DTree, LogReg, SVC, RForest, Bayes, KNN]

def Model_Evaluation(Models):
    for Model in Models:
        Model.fit(x_train, y_train)
        prediction = Model.predict(x_test)
        print(f"Confusion Matrix of {Model}:")
        print(confusion_matrix(y_test, prediction))
        print(f"Accuracy Score of {Model}:")
        print(accuracy_score(y_test, prediction))
        print(f"Classification Report of {Model}:")
        print(classification_report(y_test, prediction))

Model_Evaluation(Models)

# Displaying cross-validation scores
def Cross_Validation(Models):
    for Model in Models:
        accuracy = cross_val_score(Model, X_final, y_final, cv=5)
        print(f"{Model}'s Cross-Validation Score: {np.mean(accuracy)}")

Cross_Validation(Models)
