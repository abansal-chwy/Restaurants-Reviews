import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk 
nltk.download('stopwords')

#quoting =3 implies ignore " " in the reviews
data= pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t",quoting=3)

#cleaning the data
#sub ensures what to remove and ^ ensures not to remove 
review = re.sub("[^a-zA-Z]"," ",data["Review"][0]) #removed character replaced by space
#converting into lowercase
review = review.lower()

#Remove stop words
review = review.split() #convert into a list
from nltk.corpus import stopwords
review = [word for word in review if word not in set(stopwords.words("english"))]

#Stemming - taking only roots
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
review = [ps.stem(word) for word in review if word not in set(stopwords.words("english"))]

#Reversing list back to string
review=" ".join(review)

#Clean whole dataset
AllReviews=[]
for i in range(0,len(data)):
    review = re.sub("[^a-zA-Z]"," ",data["Review"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [word for word in review if word not in set(stopwords.words("english"))]
    review=" ".join(review)
    AllReviews.append(review)
    
#Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)#filter non relevant words #keep max 1500
X=cv.fit_transform(AllReviews).toarray()

#Train the the classification model on bag of words

y=data.iloc[:,-1].values # assign dependent variable
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Accuracy = 69.5%


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm_RF = confusion_matrix(y_test, y_pred)

#Accuracy = 73%