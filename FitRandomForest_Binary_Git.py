
"""
Created on Sun Feb 18 15:01:23 2024

@author: christopherjones

Assumes ImportYelpData.py has completed. Otherwise data are not ready for random forest. 
"""
#%% Loading Packages
import pandas as pd
import time
import os.path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import  train_test_split
from sklearn.inspection import PartialDependenceDisplay
import seaborn as sns
import random

import scipy.stats as stats

from dtreeviz.trees import dtreeviz

#%% Function to balance data

def balance_data(df, verbose = False):
    
    label_list = df["stars"].unique().tolist()
    star_counts = df["stars"].value_counts()

    if verbose:
        print("Number of reviews per star",star_counts.min())
        
    sampleFrame = pd.DataFrame(columns = df.columns)
    for i in label_list: 
        sample = df[df['stars'] == i].sample(n=star_counts.min())
        
        if sampleFrame.empty:
            sampleFrame = sample
        else:
            sampleFrame = pd.concat([sampleFrame,sample])
            
    return df

#%% Function to re sample and split data 

def sample_and_split(df, verbose = False):

    #Function to balance data
    sampleFrame = balance_data(df, verbose)
            
    
    #Split data into features(x - words) and targets(y - number of stars)
    x = sampleFrame.drop(['stars','review_id'],axis = 1)
    y = sampleFrame[['stars']]
    
    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    

    return x_train, x_test, y_train, y_test

#%% Data Directories
dataDir = ''
filtRevName = ''
dummyRevName = 'dummy_' + filtRevName

#%% Load Data
revData = pd.read_csv(dataDir + filtRevName)

#%% Create Dummy Data
if os.path.isfile(dataDir + dummyRevName):
    dummyFrame = pd.read_csv(dataDir + dummyRevName)
    
else:

    text = revData["neut_text"].unique().tolist()
    revID = revData["review_id"].unique().tolist()
    rev_count = len(revID)
    
    myList = []
    
    dummyFrame = pd.DataFrame(columns=text)
    t = time.time()
    for i in revID: 
        print("Now starting review " + str(revID.index(i)) + " out of " + str(rev_count) + "\n")

        rev_words = revData["neut_text"].loc[revData["review_id"] == i].tolist()
        unused_words = list(set(text) - set(rev_words))
        
        #getting words in review and setting value to 1
        rev_dict = dict.fromkeys(rev_words, 1)
        
        #updating with words not in review and setting value to 0
        rev_dict.update(dict.fromkeys(unused_words, 0))
        
        #adding in review ID
        rev_dict["review_id"] = i
        
        dummyFrame = pd.concat([dummyFrame,pd.DataFrame([rev_dict])])
    
    dummyFrame.to_csv(dataDir + dummyRevName, index=False)
    elapsed = time.time() - t
    print("Done Loading Dataframe. Process took " + str(elapsed/60) + " minutes. \n\n")

#%% Add stars, merge, and delete unused data frames   
finalData = pd.merge(revData[['review_id','stars']].drop_duplicates(),dummyFrame, on='review_id',how='inner' )

del dummyFrame
del revData

#%% Removing all words used only 1 time and all reviews with no words after processing

#list comprehension to get columns with sum > 1.
drop_cols = [i for (i, v) in zip(finalData.select_dtypes(include='number').columns, list((finalData.sum(axis=0, numeric_only=True) <= 1))) if v]
finalData = finalData.drop(columns = drop_cols)

#removing any reviews that now have zero features after filters
finalData = finalData[finalData.sum(axis=1, numeric_only=True) > 1]

#spot removing words, such as restraunt name
finalData = finalData.drop(columns = [])

# del drop_cols

#%%Making Stars Binary, 1 = Good (4,5), 0 = Bad (1,2,3)

finalData['stars'] = finalData['stars'].apply(lambda x: 0 if x in [1, 2, 3] else (1 if x in [4, 5] else x))
#%% Fit Classifier
x_train, x_test, y_train, y_test = sample_and_split(finalData, verbose = True)

rf = RandomForestClassifier()
rf.fit(x_train, y_train.values.ravel())

y_pred = rf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("standard model accuracy:", accuracy)

#%% Get a sample for t test to see if its actually better
acc_list = []
for i in range(0,20):
    
    x_train, x_test, y_train, y_test = sample_and_split(finalData)
    rf.fit(x_train, y_train.values.ravel())
    y_pred = rf.predict(x_test)
    acc_list.append(accuracy_score(y_test, y_pred))
    print("accuracy of standard model train test iteration :",i,"=",acc_list[i])

#%% My Hyper Parameter Estimation
threshold = .63

t_acc_list = []
t_cm_list = []
roc_list = []
t_accuracy = 0
cnt = 0

while t_accuracy < threshold:
    cnt += 1
    
    x_train, x_test, y_train, y_test = sample_and_split(finalData)
    
    t_estimators = random.randint(200,500)
    t_depth = random.randint(5,100)
    # t_depth = 5
    t_samples = random.randint(2,10)
    
    t_rf = RandomForestClassifier(max_depth = t_depth, n_estimators = t_estimators, min_samples_leaf = t_samples, n_jobs = -1)
    t_rf.fit(x_train, y_train.values.ravel())
    y_pred = t_rf.predict(x_test)
    t_accuracy = accuracy_score(y_test, y_pred)
    
    print("identifying hyper params, current accuracy:", t_accuracy, "Iteration number", cnt)
    
    if t_accuracy > threshold:
        for i in range(0,20):
            
            #split
            x_train, x_test, y_train, y_test = sample_and_split(finalData)
            #train
            t_rf.fit(x_train, y_train.values.ravel())
            #test
            y_pred = t_rf.predict(x_test)
            #accuracy
            t_acc_list.append(accuracy_score(y_test, y_pred))
            #confusion matrix
            t_cm_list.append(confusion_matrix(y_test, y_pred))
            #test vals for ROC
            #prob for ROC
            score = t_rf.predict_proba(x_test)[:,1] #probability 1
            #arrays needed for roc curve
            fpr, tpr, thresholds = roc_curve(y_test, score)
            #adding dict to a list so I can get values later
            roc_list.append({"fpr":fpr,"tpr":tpr, "thresholds":thresholds, "auc": auc(fpr, tpr)})
            
            print("accuracy of best model train test iteration :",i,"=",t_acc_list[i])
            
    
print("Solution Found: n_estimatitors = ", t_estimators,
      "\nmax_depth =", t_depth,
      " \nmin_samples_leaf =", t_samples,"\n")

t_acc_array = np.array(t_acc_list)
acc_array = np.array(acc_list)

tstat, pval = stats.ttest_ind(a=t_acc_array, b=acc_array)

print("Two tailed t test to determine if my params harmed accuracy\n")
print("p value:", pval)
print("new model mean accuracy:",t_acc_array.mean(), "standard model mean accuracy:", acc_array.mean())

#%% Plot Aggregate Confusion For Best Model

# Create the confusion matrix
cm = np.sum(t_cm_list, axis = 0)

# ConfusionMatrixDisplay(confusion_matrix=cm, labels = t_rf.classes_).plot();
labels = [str(int(i)) for i in t_rf.classes_]
ax = sns.heatmap(cm, annot=True, fmt='g', yticklabels = labels, xticklabels = labels);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
plt.show()


#%% Plot ROC

auc_list = []
for i in roc_list:
    auc_list.append(i["auc"])
    
mean_auc = np.array([i["auc"] for i in roc_list]).mean()
for i in roc_list:
    # display = RocCurveDisplay(fpr=i["fpr"], tpr=i["tpr"], roc_auc=mean_auc,
    #                                   estimator_name='example estimator')
    plt.plot(i["fpr"], i["tpr"], color = np.array([.8,.2, .3]), alpha = .3)
    
plt.title('Receiver Operating Characteristic')

plt.plot(0, 0, color = np.array([.8,.2, .3]), label = 'Mean AUC = %0.2f' % mean_auc)
plt.legend(loc = 'lower right')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

    # display.plot()
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Show the plot
plt.show()

#%% Refitting model with full balanced dataset. 

trainingFrame = balance_data(finalData, verbose = True)
x_train = trainingFrame.drop(['stars','review_id'],axis = 1)
y_train = trainingFrame[['stars']]
t_rf.fit(x_train, y_train.values.ravel())


#%% plotting a single decision tree
plt.figure(figsize=(12,8))
plt.rcParams['figure.dpi'] = 600
tree.plot_tree(t_rf.estimators_[0])
plt.savefig(dataDir + "example_tree.png")

#%% Plotting feature importance

def plot_feature_importance(model, names, threshold = None):
    feature_importance_df = pd.DataFrame.from_dict({'feature_importance': model.feature_importances_,
                                                    'feature': names}).sort_values(by='feature_importance', ascending=False)
    
    if threshold is not None:
        feature_importance_df = feature_importance_df[feature_importance_df.feature_importance > threshold]
    plt.figure(figsize=(20,8))
    

    fig = sns.barplot(
        x='feature',
        y='feature_importance',
        data=feature_importance_df,
        hue = 'feature_importance',
        palette = "flare_r",
        legend = False
    )
    
    plt.title('Feature importance')
    plt.show()
    return feature_importance_df.reset_index(drop = True)
feature_df = plot_feature_importance(t_rf, x_train.columns, threshold = .01)

#%% Partial depence. Currently looking at both 1 and zero, need to look at 1. Maybe just convert to regressor. 
plt.figure(figsize=(15,10))
ax = plt.subplot()
PartialDependenceDisplay.from_estimator(t_rf, x_train, feature_df['feature'][:6], grid_resolution=5, ax = ax,line_kw={"color": np.array([.8,.2, .3])})

plt.show()

#%% #dtree vis plot
labels = [str(int(i)) for i in t_rf.classes_]

viz = dtreeviz(t_rf.estimators_[2],
                x_train,
                y_train.squeeze(),
                target_name='Positive',
                feature_names=x_train.columns,
                title="Example Decision Tree",
                class_names=labels,
                histtype='bar', # default 
                scale=1.2)
# viz.view()
viz.save(dataDir + 'binary_model.svg')

