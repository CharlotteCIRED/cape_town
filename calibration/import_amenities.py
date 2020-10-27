# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 11:21:18 2020

@author: Charlotte Liotta
"""

import pandas as pd
import copy
import scipy.io
import numpy as np

def ImportAmenitiesSP():
    print('**************** NEDUM-Cape-Town - Import amenity data at the SP level ****************')

    #Import of the amenity files at the SP level
    amenity_data = pd.read_csv('./2. Data/Basile data/SP_amenities.csv', sep = ',')

    #Airport cones
    airportCone = copy.deepcopy(amenity_data.airport_cone)
    airportCone[airportCone==55] = 1
    airportCone[airportCone==60] = 1
    airportCone[airportCone==65] = 1    
    airportCone[airportCone==70] = 1
    airportCone[airportCone==75] = 1
    
    #Distance to RDP houses
    SP_distance_RDP = scipy.io.loadmat('./2. Data/Basile data/SPdistanceRDP.mat')["SP_distance_RDP"].squeeze()

    tableAmenities = pd.DataFrame(data = np.transpose(np.array([amenity_data.SP_CODE, amenity_data.distance_distr_parks < 2,
                                                                amenity_data.distance_ocean < 2, ((amenity_data.distance_ocean > 2) & (amenity_data.distance_ocean < 4)),
                                                                amenity_data.distance_world_herit < 2, ((amenity_data.distance_world_herit > 2) & (amenity_data.distance_world_herit < 4)),
                                                                amenity_data.distance_urban_herit < 2, amenity_data.distance_UCT < 2,
                                                                airportCone, ((amenity_data.slope > 1) & (amenity_data.slope < 5)), amenity_data.slope > 5, 
                                                                amenity_data.distance_train < 2, amenity_data.distance_protected_envir < 2, 
                                                                ((amenity_data.distance_protected_envir > 2) & (amenity_data.distance_protected_envir < 4)),
                                                                SP_distance_RDP, amenity_data.distance_power_station < 2, amenity_data.distance_biosphere_reserve < 2])), columns = ['SP_CODE', 'distance_distr_parks', 'distance_ocean', 'distance_ocean_2_4', 'distance_world_herit', 'distance_world_herit_2_4', 'distance_urban_herit', 'distance_UCT', 'airport_cone2', 'slope_1_5', 'slope_5', 'distance_train', 'distance_protected_envir', 'distance_protected_envir_2_4', 'RDP_proximity', 'distance_power_station', 'distance_biosphere_reserve'])

    return tableAmenities

#reorder tableAmenities to match order from data.XXX
#[~,idx] = ismember(data.spCode, amenity_data.SP_CODE)
#tableAmenities = tableAmenities(idx,:);

def ImportAmenitiesGrid():

    #Import of the amenity files at the grid level for the extrapolation
    amenity_data = pd.read_csv('./2. Data/Basile data/grid_amenities.csv', sep = ',')

    #Airport cones
    airportCone = copy.deepcopy(amenity_data.airport_cone)
    airportCone[airportCone==55] = 1
    airportCone[airportCone==60] = 1
    airportCone[airportCone==65] = 1    
    airportCone[airportCone==70] = 1
    airportCone[airportCone==75] = 1

    #Distance to RDP houses
    grid_distance_RDP = scipy.io.loadmat('./2. Data/Basile data/gridDistanceRDP.mat')["grid_distance_RDP"].squeeze()

    #Output as a table
    
    tableAmenities = pd.DataFrame(data = np.transpose(np.array([amenity_data.distance_distr_parks < 2,
                                                                amenity_data.distance_ocean < 2, ((amenity_data.distance_ocean > 2) & (amenity_data.distance_ocean < 4)),
                                                                amenity_data.distance_world_herit < 2, ((amenity_data.distance_world_herit > 2) & (amenity_data.distance_world_herit < 4)),
                                                                amenity_data.distance_urban_herit < 2, amenity_data.distance_UCT < 2,
                                                                airportCone, ((amenity_data.slope > 1) & (amenity_data.slope < 5)), amenity_data.slope > 5, 
                                                                amenity_data.distance_train < 2, amenity_data.distance_protected_envir < 2, 
                                                                ((amenity_data.distance_protected_envir > 2) & (amenity_data.distance_protected_envir < 4)),
                                                                grid_distance_RDP, amenity_data.distance_power_station < 2, amenity_data.distance_biosphere_reserve < 2])), columns = ['distance_distr_parks', 'distance_ocean', 'distance_ocean_2_4', 'distance_world_herit', 'distance_world_herit_2_4', 'distance_urban_herit', 'distance_UCT', 'airport_cone2', 'slope_1_5', 'slope_5', 'distance_train', 'distance_protected_envir', 'distance_protected_envir_2_4', 'RDP_proximity', 'distance_power_station', 'distance_biosphere_reserve'])

    return tableAmenities



