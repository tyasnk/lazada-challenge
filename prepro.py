# -*- coding: utf-8 -*-
"""
Created on Mon May  1 12:33:40 2017

@author: tyasnuurk
"""

import pandas as pd
from bs4 import BeautifulSoup
import numpy as np
import re

df = pd.read_csv('data/training/data_train.csv',header=None)
#soup = BeautifulSoup(df['short_description'])

df[6] = df[6].fillna('empty')

df[6] = [BeautifulSoup(i).get_text() for i in df[6]]

df.loc[df[0] == 'ph',[7]] = df[7]*0.0282493928
df.loc[df[0] == 'my',[7]] = df[7]*0.32437692

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return 1
    return 0

df['title_len'] = [len(i) for i in df[2]]
df['desc_len'] = [len(i) for i in df[6]]
df['title_contain_number'] = [contains_number(i) for i in df[2]]
df['desc_contain_title'] = [contains_number(i) for i in df[6]]

df.drop([1,2,3,4,5,6],axis=1,inplace=True)
df.to_csv('train_prepro.csv',header=False,index=False)
#%%

df_val = pd.read_csv('data/validation/data_valid.csv',header=None)
#soup = BeautifulSoup(df['short_description'])

df_val[6] = df_val[6].fillna('empty')

df_val[6] = [BeautifulSoup(i).get_text() for i in df_val[6]]

df_val.loc[df_val[0] == 'ph',[7]] = df_val[7]*0.0282493928
df_val.loc[df_val[0] == 'my',[7]] = df_val[7]*0.32437692

def contains_number(s):
    regex = re.compile("\d")
    if regex.search(s):
        return 1
    return 0

df_val['title_len'] = [len(i) for i in df_val[2]]
df_val['desc_len'] = [len(i) for i in df_val[6]]
df_val['title_contain_number'] = [contains_number(i) for i in df_val[2]]
df_val['desc_contain_title'] = [contains_number(i) for i in df_val[6]]

df_val.drop([1,2,3,4,5,6],axis=1,inplace=True)
df_val.to_csv('val_prepro.csv',header=False,index=False)

