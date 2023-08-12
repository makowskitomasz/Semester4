#!/usr/bin/env python
# coding: utf-8

# In[11]:


import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import pyrosm


# In[12]:


parameters = json.load(open('proj4_params.json', 'r'))


# In[13]:


points = gpd.read_file('proj4_points.geojson')
identification_column = parameters['id_column']
points = points.to_crs(epsg=2180)
points['buffer'] = points.geometry.buffer(100)
points['count'] = points.apply(lambda row: points.within(row['buffer']).sum(), axis=1)

df_ex01 = points[[identification_column, 'count']]
df_ex01.to_csv('proj4_ex01_counts.csv', index=False)


# In[14]:


points = points.to_crs(epsg=4326)
points['lat'] = points.geometry.y
points['lon'] = points.geometry.x
df_ex_01_1 = points[[identification_column, 'lat', 'lon']]
df_ex_01_1.to_csv('proj4_ex01_coords.csv', index=False)


# In[ ]:


fp = pyrosm.get_data(parameters['city'])
osm = pyrosm.OSM(fp)
gdf_driving = osm.get_network(network_type='driving')
roads_osm =gdf_driving[gdf_driving['highway'] == 'primary']
roads_osm['osm_id'] = roads_osm.index
ex_02 = roads_osm[['osm_id', 'name', 'geometry']]
ex_02.reset_index(drop=True, inplace=True)
ex_02.to_file('proj4_ex02_primary_roads.geojson', driver = 'GeoJSON')


# In[ ]:


points = points.to_crs(epsg = 2180)
ex_02['buffer'] = ex_02.geometry.buffer(50)
ex_02['point_count'] = ex_02.apply(lambda row: points.within(row['buffer']).sum(), axis = 1)
ex_02 = ex_02.groupby('name')['point_count'].sum().reset_index()
ex_02 = ex_02.sort_values(by = 'name')

ex_02[['name', 'point_count']].to_csv('proj4_ex03_streets_points.csv', index = False)


# In[5]:


countries_gdf = gpd.read_file('proj4_countries.geojson')

countries_gdf.plot(facecolor='none')
countries_gdf.to_pickle('proj4_ex04_gdf.pkl')

for index, row in countries_gdf.iterrows():
    name = row['name'].lower()
    fig, ax = plt.subplots()
    geometry = gpd.GeoSeries(row.geometry)
    geometry.plot(facecolor='none', ax=ax)
    crs = countries_gdf.crs.to_string()
    ctx.add_basemap(ax, crs=crs)
    fig.savefig(f'proj4_ex04_{name}.png')

