# Xenophobic Tweet Detection
## Fernando Garrote De La Macorra

### Abstract
For this project I made a program that detects xenophobic tweets with the use of machine learning. The program was made with the use of a TestingDS and TrainingDS. The program was made using the bag of words method with Random Forest Regressor and it was made with the help of Scikit Learn. The program detects the tweets with an accuracy of 93.997%. There were many different versions of the program but the version that detected the tweets with the highest accuracy is the one with bag of words and Random Forest Regressor. Other methods that were used in the program were AdaBoost, Bagging, K Neighbors Classifier and even there was a program that had some words that were considered xenophobic and some that were not, and those words added or subtracted a percentage of xenophobia to each tweet.

### Introduction
Xenophobia is a problem that is present since the beginning of humanity. It means a fear and hatred of strangers or foreigners or of anything that is strange or foreign. Twitter was founded in 2006, it is a social networking service that lets its users to microblog. Since it was created it has had some xenophobic tweets and sometimes the application bans the users that do xenophobic tweets but there are still many xenophobic tweets. For this project, I had to do a program that uses machine learning to detect xenophobic tweets and there is a training set and testing set that were given prior to doing the project. The program was made with a model called bag of words and it uses Random Forest Regressor. 

### Proposal
The implemented method is bag of words with Random Forest Regressor. According to Towards Data Science, Bag of words is a representation that turns arbitrary text into fixed-length vectors by counting how many times each word appears. This process is often referred to as vectorization. According to Git Connected, Random Forest Regression is a supervised learning algorithm that uses ensemble learning method for regression. It is a technique that combines predictions from multiple machine learning algorithms to make a more accurate prediction than a single model. In order to use Random Forest Regressor there needs to be a number of estimators and max features. The number of estimators is a number that cannot be 0 and the max features is a number between 0 and 1. In the project the number of estimators was 60 while the max features was 0.67.

### Experimental results 
The program has a score of 0.93997. 

### Discussion Section
There were many different versions of the program that were tested in order to try to have a highest score. Out of all of the different versions, all of the versions that had a higher score than 80 were with bag of words. Bag of words was used with different classifiers such as K Neighbors, Adaboost, Bagging, Gradient Boosting, Random Forest, Extra Trees Classifier, Decision Tree and Random Forest Regressor. Out of those classifiers, the one that had the highest score with Bag of Words was Random Forest Regressor. Another version of the program made was a version that had 10 features of the tweets and a classifier was used for those features. There was a feature that had some words that were considered xenophobic and if a tweet had any of those words, then the tweet was given a percentage of xenophobia and there was another feature that had words that were considered not xenophobic and if a tweet had any of those then the tweet decreased the percentage of xenophobia.

###Conclusion
In conclusion, the best method used for the project is Bag of Words with Random Forest Regressor. Many other methods were used but none of them had a higher score than them. Even when bag of words was not being used, Random Forest Regressor was the method that had the highest score amongst the different classifiers. In order to improve the score, there are many different methods that can be used although the score that the program already have is very high.