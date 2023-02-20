from numpy.core.fromnumeric import ravel
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestRegressor

def tweets():
    df_training_tweets = pd.read_csv("TrainingDS.csv", usecols=["Text"])
    training_tweets = df_training_tweets["Text"].tolist()

    df_testing_tweets = pd.read_csv("TestingDS.csv", usecols=["Text"])
    testing_tweets = df_testing_tweets["Text"].tolist()

    return training_tweets, testing_tweets

def prepare_tf(training_tweets, testing_tweets):
    vector = CountVectorizer()

    training_bow = vector.fit_transform(training_tweets)
    tf_transformer = TfidfTransformer(use_idf=False)

    training_tf = tf_transformer.fit_transform(training_bow)
    
    testing_bow = vector.transform(testing_tweets)
    testing_tf = tf_transformer.transform(testing_bow)

    return training_tf, testing_tf


training_tweets, testing_tweets = tweets()
training_tf, testing_tf = prepare_tf(training_tweets, testing_tweets)


randomRegressor = RandomForestRegressor(n_estimators=60, max_features=0.67)

content = pd.read_csv("TrainingDS.csv", usecols=["Class"])
numpy_y = content.to_numpy()  
y = ravel(numpy_y) 

#Le cambio aquí dependiendo de que metodo quiera usar
prediction = randomRegressor.fit(training_tf, y)

bagOfWords=prediction.predict(testing_tf)

df1 = pd.DataFrame(bagOfWords)
df1.index += 1


df1.to_csv("randomForestRegressor.csv", index=True, index_label=["ID"], header=["Class"])







