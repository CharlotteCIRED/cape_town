# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 12:04:53 2020

@author: Charlotte Liotta
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
import math
import copy
import scipy.io
import pickle
import os
from scipy.interpolate import interp1d

class FloodData:
    
    def __init__(self):
        
        self
        
    def import_floods_data(self):
    
        #Import floods data
        floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
        path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
    
        #Hypotheses
        structural_damages_small_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0479, 0.1312, 0.1795, 0.3591, 1, 1])
        structural_damages_medium_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.083, 0.2273, 0.3083, 0.62, 1, 1])
        structural_damages_large_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0799, 0.2198, 0.2997, 0.5994, 1, 1])
        content_damages = interp1d([0, 0.1, 0.3, 0.6, 1.2, 1.5, 2.4, 10], [0, 0.06, 0.15, 0.35, 0.77, 0.95, 1, 1])
        structural_damages = structural_damages_medium_houses
    
        d = {}
        for flood in floods:
            type_flood = copy.deepcopy(flood)
            d[flood] = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))

        interval0 = 1 - (1/5)    
        interval1 = (1/5) - (1/10)
        interval2 = (1/10) - (1/20)
        interval3 = (1/20) - (1/50)
        interval4 = (1/50) - (1/75)
        interval5 = (1/75) - (1/100)
        interval6 = (1/100) - (1/200)
        interval7 = (1/200) - (1/250)
        interval8 = (1/250) - (1/500)
        interval9 = (1/500) - (1/1000)
        interval10 = (1/1000)

        damages0 = (d['FD_5yr'].prop_flood_prone * structural_damages(d['FD_5yr'].flood_depth)) + (d['FD_5yr'].prop_flood_prone * structural_damages(d['FD_10yr'].flood_depth))
        damages1 = (d['FD_5yr'].prop_flood_prone * structural_damages(d['FD_5yr'].flood_depth)) + (d['FD_10yr'].prop_flood_prone * structural_damages(d['FD_10yr'].flood_depth))
        damages2 = (d['FD_10yr'].prop_flood_prone * structural_damages(d['FD_10yr'].flood_depth)) + (d['FD_20yr'].prop_flood_prone * structural_damages(d['FD_20yr'].flood_depth))
        damages3 = (d['FD_20yr'].prop_flood_prone * structural_damages(d['FD_20yr'].flood_depth)) + (d['FD_50yr'].prop_flood_prone * structural_damages(d['FD_50yr'].flood_depth))
        damages4 = (d['FD_50yr'].prop_flood_prone * structural_damages(d['FD_50yr'].flood_depth)) + (d['FD_75yr'].prop_flood_prone * structural_damages(d['FD_75yr'].flood_depth))
        damages5 = (d['FD_75yr'].prop_flood_prone * structural_damages(d['FD_75yr'].flood_depth)) + (d['FD_100yr'].prop_flood_prone * structural_damages(d['FD_100yr'].flood_depth))
        damages6 = (d['FD_100yr'].prop_flood_prone * structural_damages(d['FD_100yr'].flood_depth)) + (d['FD_200yr'].prop_flood_prone * structural_damages(d['FD_200yr'].flood_depth))
        damages7 = (d['FD_200yr'].prop_flood_prone * structural_damages(d['FD_200yr'].flood_depth)) + (d['FD_250yr'].prop_flood_prone * structural_damages(d['FD_250yr'].flood_depth))
        damages8 = (d['FD_250yr'].prop_flood_prone * structural_damages(d['FD_250yr'].flood_depth)) + (d['FD_500yr'].prop_flood_prone * structural_damages(d['FD_500yr'].flood_depth))
        damages9 = (d['FD_500yr'].prop_flood_prone * structural_damages(d['FD_500yr'].flood_depth)) + (d['FD_1000yr'].prop_flood_prone * structural_damages(d['FD_1000yr'].flood_depth))
        damages10 = (d['FD_1000yr'].prop_flood_prone * structural_damages(d['FD_1000yr'].flood_depth)) + (d['FD_1000yr'].prop_flood_prone * structural_damages(d['FD_1000yr'].flood_depth))
        
        damages_contents0 = (d['FD_5yr'].prop_flood_prone * content_damages(d['FD_5yr'].flood_depth)) + (d['FD_5yr'].prop_flood_prone * content_damages(d['FD_10yr'].flood_depth))
        damages_contents1 = (d['FD_5yr'].prop_flood_prone * content_damages(d['FD_5yr'].flood_depth)) + (d['FD_10yr'].prop_flood_prone * content_damages(d['FD_10yr'].flood_depth))
        damages_contents2 = (d['FD_10yr'].prop_flood_prone * content_damages(d['FD_10yr'].flood_depth)) + (d['FD_20yr'].prop_flood_prone * content_damages(d['FD_20yr'].flood_depth))
        damages_contents3 = (d['FD_20yr'].prop_flood_prone * content_damages(d['FD_20yr'].flood_depth)) + (d['FD_50yr'].prop_flood_prone * content_damages(d['FD_50yr'].flood_depth))
        damages_contents4 = (d['FD_50yr'].prop_flood_prone * content_damages(d['FD_50yr'].flood_depth)) + (d['FD_75yr'].prop_flood_prone * content_damages(d['FD_75yr'].flood_depth))
        damages_contents5 = (d['FD_75yr'].prop_flood_prone * content_damages(d['FD_75yr'].flood_depth)) + (d['FD_100yr'].prop_flood_prone * content_damages(d['FD_100yr'].flood_depth))
        damages_contents6 = (d['FD_100yr'].prop_flood_prone * content_damages(d['FD_100yr'].flood_depth)) + (d['FD_200yr'].prop_flood_prone * content_damages(d['FD_200yr'].flood_depth))
        damages_contents7 = (d['FD_200yr'].prop_flood_prone * content_damages(d['FD_200yr'].flood_depth)) + (d['FD_250yr'].prop_flood_prone * content_damages(d['FD_250yr'].flood_depth))
        damages_contents8 = (d['FD_250yr'].prop_flood_prone * content_damages(d['FD_250yr'].flood_depth)) + (d['FD_500yr'].prop_flood_prone * content_damages(d['FD_500yr'].flood_depth))
        damages_contents9 = (d['FD_500yr'].prop_flood_prone * content_damages(d['FD_500yr'].flood_depth)) + (d['FD_1000yr'].prop_flood_prone * content_damages(d['FD_1000yr'].flood_depth))
        damages_contents10 = (d['FD_1000yr'].prop_flood_prone * content_damages(d['FD_1000yr'].flood_depth)) + (d['FD_1000yr'].prop_flood_prone * content_damages(d['FD_1000yr'].flood_depth))
        

        d_structure = 0.5 * ((interval0 * damages0) + (interval1 * damages1) + (interval2 * damages2) + (interval3 * damages3) + (interval4 * damages4) + (interval5 * damages5) + (interval6 * damages6) + (interval7 * damages7) + (interval8 * damages8) + (interval9 * damages9) + (interval10 * damages10))
        d_contents = 0.5 * ((interval0 * damages_contents0) + (interval1 * damages_contents1) + (interval2 * damages_contents2) + (interval3 * damages_contents3) + (interval4 * damages_contents4) + (interval5 * damages_contents5) + (interval6 * damages_contents6) + (interval7 * damages_contents7) + (interval8 * damages_contents8) + (interval9 * damages_contents9) + (interval10 * damages_contents10))
        
        content_cost = 7395
        informal_structure_value = 4000
        
        self.d_structure = d_structure
        self.d_contents = d_contents
        self.content_cost = content_cost
        self.informal_structure_value = informal_structure_value