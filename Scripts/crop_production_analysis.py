import pandas as pd
import os
import numpy as np
from requests import get
import requests
from bs4 import BeautifulSoup
import re
import json
import folium
import pickle
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar,LogColorMapper, LogTicker
from bokeh.palettes import brewer
import geopandas as gpd


def get_country_id(string_or,countries):
    
    countries.Name=countries.Name.apply(lambda x:x.lower())
    string=string_or.lower()
    
    if string in countries.Name:
        temp=countries.loc[countries.Name==string,'ID'].tolist
        return temp
    
    strs=string.split('-')
    if len(strs)>1:
        string= ' '.join(strs)

    if string=="côte d'ivoire":
        return ['CIV']
    if string=="united states of america":
        return ['USA']
    if string=="vietnam" or string=="viet nam":
        return ["VNM"]
    if string=="south africa":
        return ['ZAF']
    if string=='north korea' or string=="democratic people's republic of korea":
        return["PRK"]
    if string=="russian federation":
        return ["RUS"]
    if string=="cyprus":
        return ["CYP"]
    if string=="ussr":
        return ['RUS']
    delimiters = " and "," ",","
    pattern = '|'.join(map(re.escape, delimiters))
    strings=list(filter(None,re.split(pattern,string)))
    for s in strings:
        if s=="french":
            return ["FRA"]
        if s=="lao":
            return ["LAO"]
        if s in ['southern','islands','south','democratic','the','arab','united','central','states','republic','of','new','africa','america']:
            continue
        for name in countries.Name:
            if s in name.split():
                temp=countries.loc[countries.Name==name,'ID'].tolist()
                return temp
    return [np.nan]

def split_names(x):
    x=x.lower()
    if x!="guinea-bissau" and len(x.split('-'))>1:
        return x.split("-")
    if len(x.split(' & '))>1:
        return x.split(' & ')
    return x

def get_df_with_ids(df):
    
    df=df.copy()
    world_countries_file="../Data/world-countries.json"
    geo_json_data=json.load(open(world_countries_file))
    names=[c['properties']['name'] for c in geo_json_data['features']]
    ids=[c['id'] for c in geo_json_data['features']]
    country_df=pd.DataFrame()
    country_df['Name']=names
    country_df['ID']=ids
    
    print("Exploding dataframe :")
    df.Area=df.Area.apply(lambda x: split_names(x))
    df=df.explode('Area')
    df.Area=df.Area.apply(lambda x:x.lower())
    
    print("Getting IDs :")
    
    countries_df=pd.DataFrame(df.Area.unique()).rename(columns={0:'Area'})
    
    countries_df['ID']=countries_df.Area.apply(lambda x: get_country_id(x,country_df)[0])
    countries_df.ID=countries_df.ID.replace("-99",np.nan)
    print(len(countries_df[countries_df.ID.isna()])," Countries without IDs")
    df=df.merge(countries_df,how='left')
    
    return df

def get_food_crops():

#Return a list of crops categorized as food crops https://world-crops.com/food-crops/

    url="https://world-crops.com/food-crops/"
    r=requests.get(url,headers={"User-Agent": "XY"})
    soup=BeautifulSoup(r.text,'html.parser')
    elements_temp=soup.find_all('a',href=re.compile("^../"))
    elements=[el.text for el in elements_temp]
    
    #only 40 elements are displayed on each page->iterating on the total list
    for i in range(40,401,40):
        url_i=url+"?ss="+str(i)
        r=requests.get(url_i,headers={"User-Agent":"XY"})
        soup=BeautifulSoup(r.text,'html.parser')
        new_elements=soup.find_all('a',href=re.compile("^../"))
        elements+=[el.text for el in new_elements]
    return elements


def inclusive_search(string,elements):

#returns true if the string can be found in elements. The search removes special characters from string in order to include more positive results

    string=string.lower()
    delimiters = ",", "(","&",")"," and "," "
    pattern = '|'.join(map(re.escape, delimiters))
    strings=list(filter(None,re.split(pattern,string)))
    found=False
    for s in strings:
        if s=="nes":
            continue
        for el in elements:
            found=(s in el.split())
            if found==False and s[-1]=="s":
                found=s[:-1] in el.split()
            if found==False and s[-2:]=="es":
                found=s[:-2] in el.split()
            if found==False and s[-3:]=="ies":
                found=s[:-3]+"y" in el.split()
            if found==True:
                return found
    return found


def get_food_crop_data(df):
    
    #extract the food crop data, returns 4 df: Area,Production,Seed and yield
    
    df=df.copy()

    food_crops=list(map(lambda x: x.lower(),get_food_crops()))
                    
    crop_types_df=df[['Item','Value']].groupby('Item').sum()
    crop_types_df=crop_types_df[list(map(lambda x : inclusive_search(x,food_crops) , crop_types_df.index ))]
                    
    food_crop_df=df[df.Item.apply(lambda x: x in crop_types_df.index)]

                    
    return (food_crop_df[food_crop_df.Element=='Area harvested'],
            food_crop_df[food_crop_df.Element=='Production'],
            food_crop_df[food_crop_df.Element=='Seed'],
            food_crop_df[food_crop_df.Element=='Yield'])
    



def visualise_world_data_folium(df,year):
    world_countries_file="../Data/world-countries.json"
    geo_json_data=json.load(open(world_countries_file))
    to_plot=df[df.Year==year]
    to_plot=(to_plot[['ID','Value']]
             .groupby('ID')
             .sum()
             .reset_index()
             .dropna())
    to_plot.Value=np.log10(to_plot.Value)
    to_plot=to_plot.sort_values('Value',ascending=False)
    m=folium.Map(titles="test",location=[40,-10],zoom_start=1.6)
    plot1=folium.Choropleth(geo_data=geo_json_data,data=to_plot,
            columns=['ID','Value'],
            key_on='feature.id',
            fill_color='GnBu',fill_opacity=0.7,line_opacity=0.2,nan_fill_opacity=0.0)
    plot1.add_to(m)
    return m
    


def visualise_world_data_bokeh(df):
    #return a map plot of the data in df using bokeh and geopandas
    
    
    #import geodata
    geo_file = '../Data/geoData/ne_110m_admin_0_countries.shp'
    gdf = gpd.read_file(geo_file)[['ADMIN', 'ADM0_A3', 'geometry']]
    gdf.columns = ['Country', 'ID', 'geometry']
    
    #merging df
    merged=gpd.GeoDataFrame(gdf.merge(df,left_on='ID',right_on='ID',how='right'))
    
    #exporting to JSON
    merged=merged.sort_values('Value',ascending=False)
    json_data = merged.to_json()
    
    #plotting
    geosource = GeoJSONDataSource(geojson = json_data)

    #defining the color palette
    palette = brewer['YlGnBu'][7]
    palette = palette[::-1]

    #creating a color mapper assigning color to values
    minima=int(merged.Value.to_list()[-1])-1
    maxima=int(merged.Value.to_list()[0])+1
    step=int((maxima-minima)/8)
    color_mapper = LogColorMapper(palette="Viridis256", low=minima, high=maxima)

    #creating label for color bar legend
    tick_labels = {minima: str(minima), minima+step: str(minima+step), minima+2*step:str(minima+2*step), minima+3*step:str(minima+3*step), minima+4*step:str(minima+4*step),
                   minima+5*step:str(minima+5*step), minima+6*step:str(minima+6*step),minima+7*step:str(minima+7*step), maxima: str(maxima)}

    #Creating color bar. 
    color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                     label_standoff=12, border_line_color=None,location=(0,0))

    #creating the figure
    p = figure(title = 'Food crop production (log)', plot_height = 600 , plot_width = 950, toolbar_location = 'left')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.axis.visible=False

    #adding patches=heatmap overlay to the figure 
    p.patches('xs','ys', source = geosource, fill_color = {'field' :'Value', 'transform' : color_mapper},
              line_color = 'black', line_width = 0.2, fill_alpha = 1)

    #Specify figure layout.
    p.add_layout(color_bar, 'right')
    
    output_notebook()
    return p
