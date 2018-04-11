# -*- coding: utf-8 -*-

# utilities


def get_required_data(dataset):
    return dataset.iloc[:,2:]

from textblob import TextBlob

def get_sentiment(text):
    '''
    Utility function to classify sentiment of passed tweet
    using textblob's sentiment method
    '''
    print (text)
    # create TextBlob object of passed tweet text
    analysis = TextBlob(text)
    # set sentiment
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

# Importing the libraries
        
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# step 1 -> Import data

dataset = pd.read_csv('feedback_data.csv')
X = dataset

# step 2 -> Preprocess

sentiment_index = [4,7,8,9,10,11]
categorical_coloum_index = [2,3,5,6] + sentiment_index


# 2.1 missing data

X.iloc[:,sentiment_index] = X.iloc[:,sentiment_index].replace(np.nan, '', regex=True)

# 2.2 Sentiment data -> Categories

for i in sentiment_index:
    X.iloc[:, i] = pd.DataFrame({X.columns[i]:list(map(get_sentiment,X.iloc[:, i]))})
# 2.3 encoding categorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

for i in categorical_coloum_index:
    labelencoder_X = LabelEncoder()
    X.iloc[:, i] = labelencoder_X.fit_transform(X.iloc[:, i])

X_values = X.iloc[:,2:]
onehotencoder = OneHotEncoder(categorical_features = sorted(list(map(lambda x: x-2,categorical_coloum_index))))
X_values = onehotencoder.fit_transform(X_values).toarray()

# Step 3 -> Clustering
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X_values)
y_kmeans = kmeans.predict(X_values)
# Step 4 -> Evaluating


# step 5 -> Visualizing

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

plt.scatter(X_values[:, 0], X_values[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);