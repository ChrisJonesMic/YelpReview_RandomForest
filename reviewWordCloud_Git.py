
"""
Created on Mon Mar 11 20:20:47 2024

@author: christopherjones
"""
#%% Creating Environment
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud as wc

#%% Data Directory
dataDir = ''
fileDir = ''
filtRevName = ''

#%% Load Data
revData = pd.read_csv(dataDir + filtRevName)

#%% Make Wordcloud
for i in revData['stars'].unique():
    wordList = revData["neut_text"][revData['stars'] == i].to_list()
    wordString = (" ").join(wordList)
    plt.figure(figsize=(12,8))
    word_cloud = wc.WordCloud(collocations = False, background_color = 'white', colormap = 'flare', scale = 4.5).generate(wordString)

    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(str(int(i)) + " star review", fontsize=13)
   
    # word_cloud.to_file(fileDir + str(i) + '_star_review.png')
    plt.savefig(fileDir + str(int(i)) + '_star_review.png')
    plt.show()
