# -*- coding: utf-8 -*-
"""
Created on Sat May 13 10:20:26 2017

@author: tyasnuurk
"""

import pandas as pd
from nltk import RegexpTokenizer
from nltk import word_tokenize
from nltk import pos_tag
import numpy as np
seed = 2017
np.random.seed(seed)
df = pd.read_csv('data/training/data_train.csv',header=None)
#%%

#tokenizer = RegexpTokenizer(r'\w+')
texts = [word_tokenize(i) for i in df[2]]
#%%
from nltk.corpus import brown,treebank
pretrain = treebank.tagged_sents()

from nltk.tag import hmm
trainer = hmm.HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(pretrain)
#%%
tagging_text = [tagger.tag(text) for text in texts]
#%%
todata = np.array(tagging_text)
data = pd.DataFrame(todata)
data.to_csv('title_tagged_treebank.csv',header=False,index=True)

#%%
df_val = pd.read_csv('data/validation/data_valid.csv',header=None)
#tokenizer = RegexpTokenizer(r'\w+')
texts_val = [word_tokenize(i) for i in df_val[2]]
#%%
tagging_text_val = [pos_tag(text) for text in texts_val]
#%%
todata_val = np.array(tagging_text_val)
data_val = pd.DataFrame(todata_val)
data_val.to_csv('title_tagged_validation.csv',header=False,index=True)