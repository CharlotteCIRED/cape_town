# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:25:33 2020

@author: Charlotte Liotta
"""

import fiona 
import geopandas as gpd

# Get all the layers from the .gdb file 
layers = fiona.listlayers(gdb_file)

gdf = gpd.read_file('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb')
