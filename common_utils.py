# Constants for Sentiment Classifier
tfidf_vectorizer_path = './pickles/tfidf_word_vectorized.pkl'
sentiment_classifier_path = './pickles/Logistic_Reg_model.pkl'
prob_threshold_val = 0.55

# Constants for datapaths
prod_reviews_mapping_path = './datasets/product_map.csv'
processed_train_data_path =  './datasets/trainData.csv'

# Constants for Recommendation Engine
recomm_system_path = './pickles/item_item_based_recomm.pkl'

####### Utility Methods #######

import pickle
import os
import re
from nltk.corpus import stopwords

# model loader
def load_from_pickle(path):
    model_f = open(path, "rb")
    model = pickle.load(model_f)
    model_f.close()
    return model

# pickle dumper
def dump_as_pickle(model, name):
    try:
        os.listdir('model_pickles')
    except:
        os.mkdir('model_pickles')

    with open("model_pickles\\"+name+".pkl","wb") as save_classifier:
        pickle.dump(model, save_classifier) 
    print('successfully saved at: ',"model_pickles\\"+name+".pkl")

# text pre_processing
def pre_process_text(text):

    '''
        This method accepts a text and cleans and pre-processes the text for vectorizer to use. 
    '''
    # remove punctuations
    text = text.replace('[^\w\s]',' ')
    text = text.lower()

    # remove whitespace
    text = text.strip()
    text = re.sub(' +', ' ',text)

    # remove stopwords
    stop = stopwords.words('english')
    remove_stopwords = lambda x: ' '.join([word for word in x.split() if word not in (stop)])
    text = remove_stopwords(text)

    return text     


