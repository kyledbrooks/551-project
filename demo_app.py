import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import folium
import geopy
import ast
import warnings
from streamlit_folium import folium_static
from sklearn import preprocessing
import pickle 
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

import boto
import boto.s3.connection
from io import StringIO
import boto3
import pandas as pd
import sys
from opencage.geocoder import OpenCageGeocode
from sklearn.metrics.pairwise import haversine_distances
from math import radians


st.set_page_config(layout="wide")

row1_1, row1_4 = st.columns(2)

with row1_1:

    st.title("SafePath Los Angeles")
    
row1_2, row1_3 = st.columns((1, 0.75))

with row1_2:
    st.write(
        """
    ##
    Examining risk level of street roads in Los Angeles. 
    """
    )
    hour_selected = st.slider("Select hour of day", 0, 23, 4)
    day_selected = st.selectbox("Pick a day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                           'Saturday', 'Sunday'])
    user_input = st.text_input("Enter street address here", 'FIGUEROA ST')

color = "risk_level"
popup = ["address","traffic_vol", "acc_count"]

address = user_input + ', Los Angeles'

map_df = pd.read_csv('data/map_df.csv')

location_lat = list(map_df.loc[(map_df['address'] == address) & (map_df['day_week']==day_selected) & (map_df['hour']==hour_selected), 'latitude'])[0]
location_long = list(map_df.loc[(map_df['address'] == address) & (map_df['day_week']==day_selected) & (map_df['hour']==hour_selected), 'longitude'])[0]

copy = map_df.copy()

lst_colors=["red","green","orange"]
lst_elements = sorted(list(map_df[color].unique()))
copy["color"] = copy[color].apply(lambda x: 
                lst_colors[lst_elements.index(x)])

map_ = folium.Map(location=[location_lat, location_long], 
                      tiles='cartodbpositron', zoom_start=20)
folium.Marker(location=[location_lat, location_long],popup=["address","traffic_vol", "acc_count"],tooltip='Click here to see Popup').add_to(map_)

required_df = copy[(copy['day_week']==day_selected) & (copy['hour']==hour_selected)]

size = []

for risk in required_df["risk_level"]:

    if risk == "low":
        size.append(10)
    elif risk == "medium":
        size.append(20)
    elif risk == "high":
        size.append(30)

required_df["size"] = size


# In[77]:


required_df.apply(lambda row: folium.CircleMarker(
           location=[row["latitude"], row["longitude"]], popup=row[popup],
           color=row["color"], fill=True,
           radius=row["size"]).add_to(map_), axis=1)

with row1_3:

    st.subheader(
        f"""**Risk Level for {user_input} at {hour_selected}:00 on {day_selected}**"""
    )
    folium_static(map_)




# In[71]:





# In[10]:

# In[82]:

    
    
    

# In[80]:


row2_1, row2_2 = st.columns((1, 0.75))
row3_1, row3_2 = st.columns((1,0.75))

with row2_1:

    st.subheader(
        f"""**Traffic Volume on {day_selected} for {user_input}**"""
    )
with row2_2:

    st.subheader(
        f"""**Accident Count on {day_selected} for {user_input}**"""
    )



with row3_1:

    chart_data = copy[(copy['address']==address)]
    chart_data= chart_data[(chart_data['day_week']==day_selected)][['hour', 'traffic_vol']]
    chart_data = pd.DataFrame(chart_data.groupby(by=['hour'])['traffic_vol'].sum(numeric_only=False))
    chart_data.reset_index(inplace=True)

    st.bar_chart(chart_data, width=500, height=500, use_container_width=True)
    


# In[81]:


with row3_2:
    chart_data_acc = copy[(copy['address']==address)]
    chart_data_acc = chart_data_acc[(chart_data_acc['day_week']==day_selected)][['hour', 'acc_count']]
    chart_data_acc = pd.DataFrame(chart_data_acc.groupby(by=['hour'])['acc_count'].sum(numeric_only=False))
    chart_data_acc.reset_index(inplace=True)
    
    st.bar_chart(chart_data_acc, width=500, height=500, use_container_width=True)


