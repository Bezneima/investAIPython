import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

commentInput = pd.read_csv("c:/temp/commentDB.txt")
priceInput = pd.read_csv("c:/temp/priceChangeDB.txt")
allComment = ""
for line in commentInput["comment"]:
    allComment += line

allCommentArray = [allComment]
vectorizer = TfidfVectorizer()
# Токенезирую
vectorizer.fit(allCommentArray)
vector = vectorizer.transform([allCommentArray[0]])
print(vector.toarray())
mnb = MultinomialNB()