# coding: utf8

##############################
##### Project Information ####
##############################

# Seminar: Tatjana Scheffler, Classification of Social Media Text (2018)
# Project: Hate Speech Classifikation With Machine Learning and SpaCy
# Author: Olha Zolotarenko
# Mtr.Nr.: 787894
# Tutorial used: Text Classification With Machine Learning, SpaCy, Sklearn by Jcharis
# https://github.com/Jcharis/Natural-Language-Processing-Tutorials/blob/master/Text%20Classification%20With%20Machine%20Learning%2CSpaCy%2CSklearn(Sentiment%20Analysis)/Text%20Classification%20%26%20Sentiment%20Analysis%20with%20SpaCy%2CSklearn.ipynb
# To be compiled with: Python3

##########################
######### Part 0: ########
##### Getting Imports ####
##########################

# Import to keep track of running time
import time

# Import pandas
import pandas as pd
from pandas import read_table

# Import spacy and string for analysing the data
import spacy
import string

# SpaCy
# noinspection PyUnresolvedReferences
from spacy.lang.de.stop_words import STOP_WORDS
nlp = spacy.load('de')
# Load punctuations of string module
punctuations = string.punctuation

# Creating a Spacy Parser
from spacy.lang.de import German
parser = German()

# Import ML packages of sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn import metrics

# Plotting modules for ROC curves
import matplotlib.pyplot as plt

# Machine Learning With SKlearn
class Hate_Classifier(TransformerMixin):

    def __init__(self):

        # Getting list of stopwords from SpaCy
        self.stopwords = list(STOP_WORDS)

        ########################
        ######## Part 1: #######
        ###### Loading Data ####
        ########################

        # Load our GermEval dataset
        germeval = read_table('germeval2018.training_binary.txt')

        # The dataset is transferred into the list
        frames = [germeval]

        # Set column names for the dataset
        for name in frames:
            name.columns = ["Tweet", "Tag_Binary", "Tag_Multiple"]

        # Uncomment to see the dataset with colums
        # for colname in frames:
        #     print(colname.columns)

        # Concatenate pandas objects along a particular axis with optional set logic along the other axes.
        self.df = pd.concat(frames)

        # Drop the Multiple Column to word with binary classification
        self.df = self.df.drop('Tag_Multiple', 1)

        # Information about the dimensionality of our DataFrame.
        print("Rows: " + str(self.df.shape[0]))
        print("Columns: " + str(self.df.shape[1]))

        # Converting the DataFrame to csv format
        self.df.to_csv("sentiment_germeval_dataset.csv")

    #############################
    ########### Part 2: #########
    ##### Cleaning the Data #####
    #############################

    # Built function for parsing, lemmatization, and getting rid of stop words
    def spacy_tokenizer(self, sentence):
        # 1. Parse the sentence
        mytokens = parser(sentence)

        # 2. Delete pronouns and transform words in lowe case
        mytokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens]

        # 3. Remove stopwords and punctuation and hashtags
        mytokens = [word for word in mytokens if word not in self.stopwords and word not in punctuations
                    and '@' not in word]

        # --> Uncomment next line to see processed clear tweets
        # print(mytokens)
        return mytokens

    def num_there(self,s):
        return any(i.isdigit() for i in s)

    def transform(self, X, **transform_params):
        return [self.clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    # Basic function to clean the text
    def clean_text(self,text):
        return text.strip().lower()

if __name__ == "__main__":

    # Setting the start time of the script
    start_time = time.time()

    # Initializing the class
    predict = Hate_Classifier()

    # VECTORIZER
    # CountVectorizer
    vectorizer = CountVectorizer(analyzer='word', tokenizer=predict.spacy_tokenizer,
                                 ngram_range=(1,1), binary=True, min_df=1, max_df=1.0)
    # TfidfVectorizer
    # tfvectorizer = TfidfVectorizer(tokenizer=predict.spacy_tokenizer)

    # CLASSIFIER
    classifier = LogisticRegression()
    # classifier = SVC(probability=True)
    # classifier = MultinomialNB()
    # classifier = BernoulliNB()
    # classifier = MLPClassifier()
    # classifier = RandomForestClassifier()

    print("Classifier: " + str(classifier))
    print('Classifying the dataset...')

    # Features and Labels
    X = predict.df['Tweet']
    ylabels = predict.df['Tag_Binary']


    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

    # Create the  pipeline to clean-tokenize, vectorize, and classify
    pipe = Pipeline([("cleaner", Hate_Classifier()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Using Tfid Vectorizer
    # pipe_tfid = Pipeline([("cleaner", Senti_Classifier()),
    #                       ('vectorizer', tfvectorizer),
    #                       ('classifier', classifier)])

    # Fit our data
    pipe.fit(X_train, y_train)
    # pipe_tfid.fit(X_train, y_train)

    # Predicting with a test dataset
    sample_prediction1 = pipe.predict(X_test)
    # sample_prediction2 = pipe_tfid.predict(X_test)

    # Getting accuracy of the approach
    print("Accuracy on Train Data: ", pipe.score(X_train, y_train))
    print("Accuracy on Test Data: ", pipe.score(X_test, y_test))

    # print("Accuracy on Train Data: ", pipe_tfid.score(X_train, y_train))
    # print("Accuracy on Test Data: ", pipe_tfid.score(X_test, y_test))

    y_preds = pipe.predict_proba(X_test)
    # y_preds = pipe_tfid.predict_proba(X_test)

    # take the second column because the classifier outputs scores for
    # the 0 class as well
    preds = y_preds[:, 1]
    # For LinearSVC uncomment the following line
    # preds = pipe.predict(X_test)

    fpr, tpr, _ = metrics.roc_curve(y_test, preds)
    auc_score = metrics.auc(fpr, tpr)

    # Change the name of the plot here
    plt.title('ROC Curve - RandomForestClassifier')
    plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(auc_score))

    plt.plot([0, 1], [0, 1], 'r--')

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    plt.legend(loc='lower right')
    plt.show()

    # Prediction Results and writing to the text file tabular separated
    # Predictions: OFFENCE or OTHER
    # print("Writing the classifyed test dataset...")
    # with open("classified_germeval.txt", "w") as f:
    #     for (sample, prediction) in zip(X_test, sample_prediction2):
    #         f.write(sample + "\t" + prediction + "\n")
    # f.close()

    # Geting average runtime
    print("Runtime: " + "--- %s seconds ---" % (time.time() - start_time))

