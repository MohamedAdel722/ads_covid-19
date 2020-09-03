# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:42:52 2020

@author: Acer
"""
#from model import model
#import SIR_functions
import subprocess
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal

from sklearn import linear_model
reg = linear_model.LinearRegression(fit_intercept=True)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,State
import plotly.graph_objects as go
from scipy import optimize
from scipy import integrate

from app import app
#############
def SIR_model(SIR,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return([dS_dt,dI_dt,dR_dt])

def SIR_model_t(SIR,t,beta,gamma):
    ''' Simple SIR model
        S: susceptible population
        t: time step, mandatory for integral.odeint
        I: infected people
        R: recovered people
        beta: 
        
        overall condition is that the sum of changes (differnces) sum up to 0
        dS+dI+dR=0
        S+I+R= N (constant size of population)
    
    '''
    
    S,I,R=SIR
    dS_dt=-beta*S*I/N0          #S*I is the 
    dI_dt=beta*S*I/N0-gamma*I
    dR_dt=gamma*I
    return dS_dt,dI_dt,dR_dt #([dS_dt,dI_dt,dR_dt])

def fit_odeint(x, beta, gamma):
    '''
    helper function for the integration
    '''
    return integrate.odeint(SIR_model_t, (S0, I0, R0), t, args=(beta, gamma))[:,1] # we only would like to get dI

#############
df_input_large=pd.read_csv(r'C:\Users\Acer\Desktop\Python in 100 Minutes Package\Intro to Data science\ads_covid-19\data\processed/COVID_final_set.csv',sep=';')
#df_input_large.head()
country_list = 'Germany'
df_plot=df_input_large[df_input_large['country']==country_list]

ydata = np.array(df_plot.confirmed[35:]) 
t=np.arange(len(ydata))
N0=1000000 #max susceptible population
I0=ydata[0] 
S0=N0-I0
R0=0

beta=0.4   # infection spread dynamics
gamma=0.1  # recovery rate
t_initial=20
t_intro_measures=14
t_hold=41
t_relax=80

beta_max=0.4
beta_min=0.11
#############
fig = go.Figure()

app = dash.Dash()
app.layout = html.Div([
    html.H3('App 2'),

    dcc.Markdown('''
    #  Applied Data Science on COVID-19 data

    This dashboard shows the confirmed infected COVID-19 cases with the static and dynamic fitting of the SIR _ model.

    '''),

    dcc.Markdown('''
    ## Multi-Select Country for visualization
    '''),


    dcc.Dropdown(
        id='country_drop_down',
        options=[ {'label': each,'value':each} for each in df_input_large['country'].unique()],
        value='Germany', # which is pre-selected
        multi=False
    ),

    dcc.Markdown('''
        ## SIR static oor Dynamic Beta
        '''),


    dcc.Dropdown(
    id='beta',
    options=[
        {'label': 'Static KK ', 'value': 'static'},
        {'label': 'Dynamic KK', 'value': 'dynamic'},
    ],
    value='static',
    multi=False
    ),

    dcc.Graph(figure=fig, id='main_window_slope')
])



@app.callback(
    Output('main_window_slope', 'figure'),
    [Input('country_drop_down', 'value'),
    Input('beta', 'value')])

def update_figure(country_list,show_beta):
    df_plot=df_input_large[df_input_large['country']==country_list]
    ydata = np.array(df_plot.confirmed[35:]) 
    t=np.arange(len(ydata))
    I0=ydata[0] 
    S0=N0-I0
    R0=0
    print('shape of array (ydata) :', ydata.shape)
    print('shape of array (t) :', t.shape)
    #print('shape of Egypt array :', df_input_large['country']['Egypt'].shape)
    
    
    my_yaxis={'type':"log",
                  'title':'Confirmed infected people (Model, log-scale)'
              }
    

    traces = []

    if show_beta=='static':
        
        SIR=np.array([S0,I0,R0])
        propagation_rates=pd.DataFrame(columns={'susceptible':S0,
                                        'infected':I0,
                                        'recoverd':R0})

        for each_t in np.arange(100):
   
            new_delta_vec=SIR_model(SIR,beta,gamma)
   
            SIR=SIR+new_delta_vec
    
            propagation_rates=propagation_rates.append({'susceptible':SIR[0],
                                                'infected':SIR[1],
                                                'recovered':SIR[2]}, ignore_index=True)
        popt, pcov = optimize.curve_fit(fit_odeint, t, ydata)
        perr = np.sqrt(np.diag(pcov))
    
        print('standard deviation errors : ',str(perr), ' start infect:',ydata[0])
        print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

        fitted=fit_odeint(t, *popt)
        x1=t
        y1=fitted
        traces.append(dict(x=x1,
                                y=y1,
                                mode='markers+lines',
                                opacity=0.9,
                                name=country_list+" Model"
                        )
                )
        traces.append(dict(x=t,
                                y=ydata,
                                mode='markers+lines',
                                opacity=0.9,
                                name=country_list+" Confirmed"
                        )
                    )
        
    else:
        pd_beta=np.concatenate((np.array(t_initial*[beta_max]),
                       np.linspace(beta_max,beta_min,t_intro_measures),
                       np.array(t_hold*[beta_min]),
                        np.linspace(beta_min,beta_max,t_relax),
                       ))
        SIR=np.array([S0,I0,R0])
        propagation_rates=pd.DataFrame(columns={'susceptible':S0,
                                        'infected':I0,
                                        'recoverd':R0})



        for each_beta in pd_beta:
   
                new_delta_vec=SIR_model(SIR,each_beta,gamma)
   
                SIR=SIR+new_delta_vec
    
                propagation_rates=propagation_rates.append({'susceptible':SIR[0],
                                                'infected':SIR[1],
                                                'recovered':SIR[2]}, ignore_index=True)

        x1=propagation_rates.index
        y1=propagation_rates.infected

        traces.append(dict(x=x1,
                                y=y1,
                                mode='markers+lines',
                                opacity=0.9,
                                name=country_list+" Model"
                        )
                )
        traces.append(dict(x=t,
                                y=ydata,
                                mode='markers+lines',
                                opacity=0.9,
                                name=country_list+" Confirmed"
                        )
                    )

        

    return {
            'data': traces,
            'layout': dict (
                width=1280,
                height=720,

                xaxis={'title':'Timeline',
                        'tickangle':-45,
                        'nticks':20,
                        'tickfont':dict(size=14,color="#7f7f7f"),
                      },

                yaxis=my_yaxis
        )
    }


app.run_server(debug=True, use_reloader=False)



