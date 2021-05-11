# -*- coding: utf-8 -*-
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

# read data
df = pd.read_csv('../dataset/app_reviews_all_annotated2.csv')

# combine title and review wout nan values (maybe we can use later)
# data['title+review'] = df['title'].fillna('') + df['review'].fillna('')

# prepare binary labels for bert
argumentCats = ['Arg', 'Dec']
for ac in argumentCats:
    df[ac] = np.where((df['argument_cat'] == ac) | (df['argument_cat'] == 'Both'), 1, 0)
decisionCats = ['Acquiring','Requesting','Recommendation','Buying','Rating']
for dc in decisionCats:
    df[dc] = np.where(df['decision_cat'] == dc, 1, 0)
yt = df[argumentCats+decisionCats].values.tolist()

# split for train and test
x_train,x_test,y_train,y_test = train_test_split(df['review'].tolist(), yt, test_size=0.1, shuffle=True)
# split for train into training and validation
x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

