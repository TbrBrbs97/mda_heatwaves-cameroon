#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages

import numpy as np
import pandas as pd


# In[2]:


# Import data

emdat_data = pd.read_excel('Data/HeatWaves_Europe_1900-2020.xlsx',header=6)
oecd_data = pd.read_excel('Data/Annual Heating Degree Days.xls.xlsx',header=4)


# In[3]:


# Cleaning EM-DAT Data

emdat_selection = emdat_data[['Year','Location','Start Month','End Month','Start Day','End Day','Total Deaths']]
emdat_selection.dropna(subset=['Location', 'Total Deaths'])
emdat_selection = emdat_selection[emdat_selection['Year']>=2000]
emdat_selection


# In[5]:


# Cleaning OECD Data

oecd_selection = oecd_data.loc[:, ~oecd_data.columns.str.contains('^Unnamed')]
oecd_selection = oecd_selection[oecd_selection['2000']!='..']
oecd_selection.dropna()


# In[19]:


# Make dictionairies
emdat_cities = emdat_selection['Location']
emdat_cities.to_dict

oecd_cities = oecd_selection['Year']
oecd_cities.to_dict

#for key in emdat_cities.keys():
#    emdat_cities[key]


# In[20]:


# Matching procedure
matching_dict = dict()

for emdat_key in emdat_cities.keys():
    emdat_str = emdat_cities[emdat_key]
    for oecd_key in oecd_cities.keys():
        oecd_str = oecd_cities[oecd_key]
        if any(x in emdat_str for x in oecd_str):
            matching_dict[emdat_key] = oecd_key
          
        

