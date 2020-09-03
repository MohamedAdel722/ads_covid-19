# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 23:23:20 2020

@author: Acer
"""
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import signal
from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)



# calculating slop 
# get_rate_via_regression
def get_doubling_time_via_regression(in_array):# doubling rate continuasly changes over time
    ''' Use a linear regression to approximate the doubling rate''' # we have to cout out a window for x days and calculate
   # print(in_array) 
   # print(type(in_array))  
   # y = np.array(in_array['confirmed']) as the input is data frame so we have to slice the correct column                        
    y = np.array(in_array)               # target vector 
    X = np.arange(-1,2).reshape(-1, 1)  # entry matrix, but we go with one array so we have to reshape it (-1,1)
                                        # (-1,2) piecewise for only 3 data points 
    assert len(in_array)==3             # window size is called a parameter(hyber parameter)
    reg.fit(X,y)
    intercept=reg.intercept_
    slope=reg.coef_

    return intercept/slope  # derivative calc
test_data_reg=np.array([2,4,6])
result=get_doubling_time_via_regression(test_data_reg)
print('the test slope is: '+str(result))


def savgol_filter (df_input,column='confirmed',window=5): # before calculating regression we have to apply the filter first
    ''' Savgol Filter which can be used in groupby apply function #to clean the data so we do not distort the trend
        it ensures that the data structure is kept'''
    window=5, #based on a polynomial fun
    degree=1  # polynomial order 
    df_result=df_input
    
    filter_in=df_input[column].fillna(0) # attention with the neutral element here
    
    result=signal.savgol_filter(np.array(filter_in),
                           5, # window size used for filtering & try diff days(3,4)
                           1)
    df_result[column+'_filtered']=result
    return df_result

def rolling_reg(df_input,col='confirmed'):
    ''' input has to be a data frame''' # wrapper around reg.fit function
    ''' return is single series (mandatory for group by apply)'''
    days_back=3
    result=df_input[col].rolling( #rolling command is moving a window across a time series
                window=days_back,
                min_periods=days_back).apply(get_doubling_time_via_regression,raw=False)
    return result

def calc_filtered_data(df_input,filter_on='confirmed'):
    '''  Calculate savgol filter and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'

    df_output=df_input.copy() # we need a copy here otherwise the filter_on column will be overwritten

    pd_filtered_result=df_output[['state','country',filter_on]].groupby(['state','country']).apply(savgol_filter)#.reset_index()

    # left merge is more safer than outer merge, as we can control our result
    df_output=pd.merge(df_output,pd_filtered_result[[str(filter_on+'_filtered')]],left_index=True,right_index=True,how='left')
    return df_output.copy()

def calc_doubling_rate(df_input,filter_on='confirmed'):
    ''' Calculate approximated doubling rate and return merged data frame

        Parameters:
        ----------
        df_input: pd.DataFrame
        filter_on: str
            defines the used column
        Returns:
        ----------
        df_output: pd.DataFrame
            the result will be joined as a new column on the input data frame
    '''

    must_contain=set(['state','country',filter_on])# set expression???
    assert must_contain.issubset(set(df_input.columns)), ' Erro in calc_filtered_data not all columns in data frame'


    pd_DR_result= df_input.groupby(['state','country']).apply(rolling_reg,filter_on).reset_index()

    pd_DR_result=pd_DR_result.rename(columns={filter_on:filter_on+'_DR',
                             'level_2':'index'})

    #we do the merge on the index of our big table and on the index column after groupby
    df_output=pd.merge(df_input,pd_DR_result[['index',str(filter_on+'_DR')]],left_index=True,right_on=['index'],how='left')
    df_output=df_output.drop(columns=['index'])


    return df_output

def post_process():
    #test_data_reg=np.array([2,4,6])
   # result=get_doubling_time_via_regression(test_data_reg)
   # print('the test slope is: '+str(result))
    
    

    pd_JH_data=pd.read_csv('../data/processed/COVID_relational_confirmed.csv',sep=';',parse_dates=[0])
    pd_JH_data = pd_JH_data.drop(['Unnamed: 0'],axis=1)
    pd_JH_data=pd_JH_data.sort_values('date',ascending=True).copy()

    #test_structure=pd_JH_data[((pd_JH_data['country']=='US')|
    #                  (pd_JH_data['country']=='Germany'))]
    
    pd_result_larg=calc_filtered_data(pd_JH_data)
    print("25 %")
    pd_result_larg=calc_doubling_rate(pd_result_larg)
    print("75 %")
    pd_result_larg=calc_doubling_rate(pd_result_larg,'confirmed_filtered')
    print("100 %")

    mask=pd_result_larg['confirmed']>100
    pd_result_larg['confirmed_filtered_DR']=pd_result_larg['confirmed_filtered_DR'].where(mask, other=np.NaN)#where command to filter according to the mask
    pd_result_larg.to_csv('../data/processed/COVID_final_set.csv',sep=';',index=False)
    print("Data saved in data/processed/COVID_final_set.csv")
    return pd_result_larg

post_process().head()