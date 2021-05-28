#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from pandas import DataFrame
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim


# In[2]:


# Import OECD-data from CSV-files

#Metropolitan-based data
oecd_data1 = pd.read_csv('Data/CITIES_POPULATION.csv', sep="|", header=0)
oecd_data2 = pd.read_csv('Data/CITIES_AGE.csv', sep="|", header=0)
oecd_data3 = pd.read_csv('Data/CITIES_ECONOMY.csv', sep="|", header=0)
oecd_data4 = pd.read_csv('Data/CITIES_LABOUR.csv', sep="|", header=0)
oecd_data5 = pd.read_csv('Data/CITIES_TERRITORY.csv', sep="|", header=0)
oecd_data6 = pd.read_csv('DATA/CITIES_ENVIRONMENT.csv', sep="|", header=0)


# In[3]:


# Put all OECD-data into dataframe and append all dataframes
oecd_data_df = [oecd_data1, oecd_data2, oecd_data3, oecd_data4, oecd_data5, oecd_data6]
# Call concat method
oecd_df = pd.concat(oecd_data_df)


# In[6]:


# create a dictionary where key = old name and value = new name
dict = {'METRO_ID': 'metroId',
        'Metropolitan areas': 'metropolitanAreas',
        'VAR': 'var',
       'Variables' : 'variables',
        'TIME' : 'time',
        'Year' : 'year',
        'Unit Code': 'unitCode',
        'Unit' : 'unit',
        'PowerCode Code' : 'powerCodeCode',
        'PowerCode': 'powerCode',
        'Reference Period Code' : 'referencePeriodCode',
        'Reference Period' : 'referencePeriod',
        'Value':'value',
        'Flag Codes' : 'flagCodes',
        'Flags': 'flags'
       }
  
# call rename () method
oecd_df.rename(columns=dict,
          inplace=True)


# In[7]:


# The length of the list is determined first, to determine which function to apply
len(oecd_df['metropolitanAreas'].unique().tolist())


# In[8]:


# Assign the list to a variable
metro_list= oecd_df['metropolitanAreas'].unique().tolist()
# Convert the list into a dataframe
df_metropolitan=DataFrame (metro_list, columns=['metropolitanAreas'] )
# Check the dataframe
df_metropolitan


# In[9]:


# use function to find the coordinate of a given metropolitanArea
def findGeocode(city):

    # try and catch is used to overcome the exception thrown by geolocator using geocodertimedout
    try:
        # Specify the user_agent as app name: this is my (Arabella) local address
        geolocator = Nominatim(user_agent="127.0.0.1:8888") 
        return geolocator.geocode(city)
    except GeocoderTimedOut:
        return findGeocode(city)
    except:
        return findGeocode(city)


# In[ ]:


# declare an empty list to store latitude and longitude of values of the metropolitanAreas column
longitude = []
latitude = []

# each value from city column will be fetched and sent to function find_geocode
for i in (df_metropolitan["metropolitanAreas"]):

    if findGeocode(i) != None:

        loc = findGeocode(i)

        # coordinates returned from function is stored into two separate list
        latitude.append(loc.latitude)
        longitude.append(loc.longitude)

    # if coordinate for a city not found, insert "NaN" indicating missing value
    else:
        latitude.append(np.nan)
        longitude.append(np.nan)


# In[ ]:


# now add each column to dataframe
df_metropolitan["longitude"] = longitude
df_metropolitan["latitude"] = latitude


# In[ ]:


df_metropolitan.to_csv('Data/CITIES_COORD.csv', header="True", sep="|", doublequote=True)

