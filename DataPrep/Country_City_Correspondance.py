#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages

import numpy as np
import pandas as pd
import wbgapi as wb
import sklearn.preprocessing
import seaborn as sns
from pandas import DataFrame


# In[ ]:


#cities_set = set(oecd_data1['METRO_ID'])
#country_dict = country_codes[['CODE','Country']].to_dict('records')

#function that selects first few letters of a string

def first_letters(string):
    output = ''
    for char in range(len(string)):
        if ord(str(string[char])) >= 65:
            output += str(string[char])
    return output

#Built dictionairy which relates every city to the country it belongs to

def country_cities_correspondance(country_dict,cities_set):

    #country_dict = country_codes[['CODE','Country']].to_dict('records')
    
    dict_country_city = dict.fromkeys(cities_set,0)

    for city in dict_country_city.keys():
        for i in range(len(country_dict)):
            if len(first_letters(city)) > 2:
                if city[:3] == country_dict[i]['CODE'][:3]:
                    dict_country_city[city] = country_dict[i]['Country']
            else:
                if city[:2] == country_dict[i]['CODE'][:2]:
                    dict_country_city[city] = country_dict[i]['Country']

    #exception handling
    for entry in dict_country_city.keys():
        if first_letters(entry) == 'UK':
            dict_country_city[entry] = 'United Kingdom'
        elif first_letters(entry) == 'CL':
            dict_country_city[entry] = 'Chile'
        elif first_letters(entry) == 'SI':
            dict_country_city[entry] = 'Slovenia'
        elif first_letters(entry) == 'LV':
            dict_country_city[entry] = 'Latvia'
        elif first_letters(entry) == 'SK':
            dict_country_city[entry] = 'Slovak Republic'
        elif first_letters(entry) == 'PT':
            dict_country_city[entry] = 'Portugal'
        elif first_letters(entry) == 'EE':
            dict_country_city[entry] = 'Estonia'
        elif first_letters(entry) == 'IE':
            dict_country_city[entry] = 'Ireland'
        elif first_letters(entry) == 'EL':
            dict_country_city[entry] = 'Greece'
        elif first_letters(entry) == 'DK':
            dict_country_city[entry] = 'Denmark'
        elif first_letters(entry) == 'CZ':
            dict_country_city[entry] = 'Czech Republic'
        elif first_letters(entry) == 'PL':
            dict_country_city[entry] = 'Poland'
        elif first_letters(entry) == 'AT':
            dict_country_city[entry] = 'Austria'
        elif first_letters(entry) == 'KOR':
            dict_country_city[entry] = 'South Korea'
        elif first_letters(entry) == 'LT':
            dict_country_city[entry] = 'South Korea'
        elif first_letters(entry) == 'BE':
            dict_country_city[entry] = 'Belgium'


    dict_country_city

    #Create dataframe for country-metro area correspondance
    df_country_city = pd.DataFrame.from_dict(dict_country_city, orient='index')
    df_country_city.rename(columns = {0:'Country'}, inplace = True)
    
    return df_country_city

