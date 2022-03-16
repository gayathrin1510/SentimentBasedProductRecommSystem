# AUTHOR: GAYATHRI N
# DATE: 12- mARCH-2022
# Version: 1.5
# Importing required libraries

import pandas as pd
import numpy as np
import pickle

# Importing common_utils for loading required variables
import common_utils

# Loading Recommendation System
ProductRecommendationSystem = common_utils.load_from_pickle(common_utils.recomm_system_path)
# Loading ProductReviewMapping (TRAIN)
productMap = pd.read_csv(common_utils.prod_reviews_mapping_path)
productReviewsData = pd.read_csv(common_utils.processed_train_data_path)
# Loading Vectorizer
vectorizer = common_utils.load_from_pickle(common_utils.tfidf_vectorizer_path)
# Loading Sentiment Classifier
SentimentClassifier = common_utils.load_from_pickle(common_utils.sentiment_classifier_path)
# Loading the Probability Threshold
probThreshold = common_utils.prob_threshold_val

userList = list(ProductRecommendationSystem.index)

recommendationProductSentimentMap = {}


# Defining function for Sentiment based Product Recommendations
def get_sentimentBasedProductRecommendations(username):
    '''
    This function takes username as input and
    generates top 5 product recommendations based on their buying preferences.
    This further enhances the recommendations based on overall positive reviews for the products 
    -> Uses Item-based Recommendation System::
    '''
 
    # Fetching top 20 recommendations for user
    try:
        if len(username) < 1:
            raise KeyError
        # fetch the top 20 recommendations for the user
        topRecommendations_id = ProductRecommendationSystem.loc[username].sort_values(ascending=False)[:20]
        topRecommendations = pd.merge(left=productMap, right=topRecommendations_id, on='id')
        
    except KeyError:
        # If user doesn't exist print the error message
        errorMessage = "ERROR: Unable to find user '{}' in our database!\n\
            Please try again with a user from 'Available Users' list provided above. ".format(username)
        
        return errorMessage
    
    # Creating a Dictionary to store the percentage of positive sentiments
    recommendationsPositivePercentage = {}
    
    # Iterating over all the recommendations
    for product in topRecommendations.name:

        # extracting the reviews for every recommended product
        filteredReviews = (productReviewsData[productReviewsData.name == product].reviews_text)

        #pre_process and vectorize the text
        filteredReviews = filteredReviews.apply(common_utils.process_text)
        reviews_vectorized = vectorizer.transform(filteredReviews)

        # get the average sentiment for every product
        sentimentProbs = SentimentClassifier.predict_proba(reviews_vectorized).T[1]
        reviewsSentiment = [1 if sentiment>0.55 else 0 for sentiment in sentimentProbs]
        positvePercent = sum(reviewsSentiment)/len(reviewsSentiment)

        recommendationProductSentimentMap[product] = positvePercent

    # sort the dict with the decreasing order of average positive reviews
    recommendationProductSentimentMap_sorted = dict(sorted(recommendationProductSentimentMap.items(), key = lambda x: x[1], reverse = True))

    # retrieve the top 5 products based on positve sentiment
    finalRecommendations = list(recommendationProductSentimentMap_sorted.keys())[:5]
    
    return finalRecommendations
