# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 21:43:19 2020

@author: Acer
"""
import dash

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server