# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:19:59 2020

@author: Acer
"""
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Data updating
def update_data():
    git_pull = subprocess.Popen(' git pull',
                           cwd = os.path.dirname('../data/raw/COVID-19/'),
                           shell = True,
                           stdout = subprocess.PIPE,
                           stderr = subprocess.PIPE)
    (out, error) = git_pull.communicate()
    return(out, error)

update_data()

def pre_process():
    
    data_path=r'..\data\raw\COVID-19\csse_covid_19_data\csse_covid_19_time_series\time_series_covid19_confirmed_global.csv'
    pd_raw=pd.read_csv(data_path)


    pd_data_base = pd_raw.rename(columns={'Country/Region':'country',
                                      'Province/State': 'state'})
    pd_data_base['state']=pd_data_base['state'].fillna('no') 

                                                        # we do not need the lat-long columns as they are a master data,
                                                        #we need the transactional data
    pd_data_base = pd_data_base.drop(['Lat','Long'],axis=1) # axis=1 >> columns level


# we will now construct a primary key from the date, state, country
# we will push the state and country to an index by set_index command
# we will transpose 'T' the matrix to get the date in a column to represent the time dimention in our data

#check what kind od data structure we have 
#test_pd.columns
# 'MultiIndex' it is a two dimentional index column (country,state)
# we want to merge them in row base like pivot tables in exel with ''stack date'' command
# then we reset the index

    pd_relational_model = pd_data_base.set_index (['state','country']) \
                                              .T                   \
                                              .stack(level=[0,1])  \
                                              .reset_index()       \
                                              .rename(columns={'level_0':'date',
                                                                0:'confirmed'})



# convert the date fron str to date object data type
# useing ''astype'' command

    pd_relational_model['date']=pd_relational_model.date.astype('datetime64[ns]')
#pd_relational_model.dtypes

#store our model
    pd_relational_model.to_csv('../data/processed/COVID_relational_confirmed.csv',sep=';')
#print(pd_relational_model[pd_relational_model['country']=='Germany'].tail())

pre_process()