#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 16:01:09 2024

@author: christopherjones
"""


#%% Loading Packages

import json
import pandas as pd
import time
import nltk as nl
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
# from transformers import pipeline

#%% Directories
dataDir = ''
busJsonName = ''
RevJsonName = ''
filtCSVName = ""
bizName = ""
#%% Function to load data

def load_yelp(path, fileName):
    data_list = []
    counter = 0
    t = time.time()
    
    # Open the file and load the JSON data
    with open(path + fileName, 'r') as file:
        # yelpData = json.load(file)
        for line in file:
            try:
                # Attempt to load each line as JSON
                data = json.loads(line)
                data_list.append(data)
                counter += 1
                print(str(counter) + " Records Completed \n")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    elapsed = time.time() - t
    print("Done Loading JSON. Process took " + str(elapsed/60) + " minutes. \n\n")
    print("Creating dataframe \n\n")
    t = time.time()
    df = pd.DataFrame(data_list)
    elapsed = time.time() - t
    print("Done Loading Dataframe. Process took " + str(elapsed/60) + " minutes. \n\n")
    return df

#%% Function to process review text and only include neutral reviews

def neutralize_words(orig_text):
    
    #removing punctuation
    orig_text = re.sub(r'[^\w\s]|(\d+)', '', orig_text)
    
    #tokenize & lower text. 
    tokens = nl.word_tokenize(orig_text.lower())
    
    # Lemmatize words. For example Changed, Changing, and Changes -> Change
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['J'] = wordnet.ADJ
    tag_map['V'] = wordnet.VERB
    tag_map['R'] = wordnet.ADV

    lemmatizer = WordNetLemmatizer()
    lematized_tokens = []
    for token, tag in nl.pos_tag(tokens):
        lemma = lemmatizer.lemmatize(token, tag_map[tag[0]])
        if token != lemma:
            # print(token, tag_map[tag[0]])
            # print(token, "=>", lemma)
            lematized_tokens.append(lemma)
    
    # Remove stop words, which are basically commonly used words in the english language with little informative value. 
    filtered_tokens = [token for token in lematized_tokens if token not in stopwords.words('english')]
    
    # Removing non english words (including mispelling)
    real_word_bool = [wordnet.synsets(i) for i in filtered_tokens]
    non_real_words = [i for (i, v) in zip(filtered_tokens, real_word_bool) if not v]
    
    for i in non_real_words:
        filtered_tokens.remove(i)
        # print(i)
  
    #getting word polarity scores and only keeping if neutral
    polarity_scores = dict()
    for i in filtered_tokens:
        score = SentimentIntensityAnalyzer().polarity_scores(i)
        polarity_scores[i] = score["neu"] #1 if neutral
        # print(i,"pos",score["pos"],"neu",score["neu"],"neg",score["neg"])
        
        
    
    neutral_words = [key for key, value in polarity_scores.items() if value == 1]
    return neutral_words
#%% Loading data

busData = load_yelp(dataDir, busJsonName)
revData = load_yelp(dataDir, RevJsonName)

#%% Merging business_id and raw review data
revData = pd.merge(revData, busData[['business_id', 'name']], on='business_id', how='inner')


#%% Preprocessing text and removing non neutral words. Note: Currently takes 41 days. Make faster. 
pd.set_option('display.max_colwidth', None)

# filt_data = revData[revData["name"] == "Pat's King of Steaks"].sort_values(by='stars', ascending=True).reset_index(drop = True)
filt_data = revData[revData["name"] == bizName].sort_values(by='stars', ascending=True).reset_index(drop = True)

neut_frame = pd.DataFrame(columns = ["review_id", "neut_text"])
t = time.time()
for i in filt_data.index:
    print("Now starting row " + str(i) + "\n")
    
    #getting neutral words
    text = neutralize_words(filt_data["text"].iloc[i])
    
    #removing duplicate words
    text = list(set(text))
    rev_ID = filt_data["review_id"].iloc[i]
    
    neut_frame = pd.concat([neut_frame, pd.DataFrame({
        "review_id":rev_ID,
        "neut_text":text
        })])

#adding neutral words to data in long format
filt_data = pd.merge(filt_data,neut_frame,on="review_id",how="inner")

#%% Saving Dataset
filt_data.to_csv(dataDir + filtCSVName, index=False)

elapsed = time.time() - t
print("Done Neutralizing Text.\n\n File saved as " 
      + dataDir + filtCSVName + "\n\n "+ "Process took " + str(elapsed/60) + " minutes. \n\n")


