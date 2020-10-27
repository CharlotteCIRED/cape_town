# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:25:19 2020

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

from data.functions_to_import_data import *
from data.grid import *
from data.flood import *
from plot_and_export_outputs.export_outputs_flood_damages import *

### COMPUTE OUTPUT ON FLOOD AND DAMAGES

def export_outputs_flood_damages(households_data, grid, name, initialState_householdsCenter, initialState_householdsHousingType, param, initialState_dwellingSize, initialState_rent, interest_rate, content_cost):
    
    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages')

    #Flood data
    floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
    path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
    flood = FloodData()
    flood.import_floods_data()
    
    #Data
    grid = SimulGrid()
    grid.create_grid()
    count_formal = households_data.formal_grid_2011 - households_data.GV_count_RDP
    count_formal[count_formal < 0] = 0
    
    #Basile's simulations
    mat1 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations scenarios - 201908.mat')
    mat2 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations - 201907.mat')
    simul1 = mat1["simulation_noUE"]
    simul2 = mat2["simulation_noUE"]
    simul1_householdsHousingType = simul1["householdsHousingType"][0][0]
    simul2_householdsHousingType = simul2["householdsHousingType"][0][0]
    simul1_householdsCenter = simul1["householdsCenter"][0][0]
    simul2_householdsCenter = simul2["householdsCenter"][0][0]
    simul1_housingSupply = simul1["housingSupply"][0][0]
    simul2_housingSupply = simul2["housingSupply"][0][0]
    simul1_rent = simul1["rent"][0][0]
    simul2_rent = simul2["rent"][0][0]
    simul1_dwellingSize = simul1["dwellingSize"][0][0]
    simul2_dwellingSize = simul2["dwellingSize"][0][0]

    #1. Compute stats on floods per housing type
    
    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_housing_types')
    
    #Data
    stats_per_housing_type = compute_stats_per_housing_type(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011)
    stats_per_housing_type.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_housing_types/data.xlsx')
    
    #Simul
    stats_per_housing_type = compute_stats_per_housing_type(floods, path_data, initialState_householdsHousingType[0, :], initialState_householdsHousingType[3, :], initialState_householdsHousingType[2, :], initialState_householdsHousingType[1, :])
    stats_per_housing_type.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_housing_types/simul.xlsx')

    #Data
    stats_per_housing_type = compute_stats_per_housing_type(floods, path_data, simul1_householdsHousingType[0, 0, :], simul1_householdsHousingType[0, 3, :], simul1_householdsHousingType[0, 2, :], simul1_householdsHousingType[0, 1, :])
    stats_per_housing_type.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_housing_types/Basile1.xlsx')

    #Data
    stats_per_housing_type = compute_stats_per_housing_type(floods, path_data, simul2_householdsHousingType[0, 0, :], simul2_householdsHousingType[0, 3, :], simul2_householdsHousingType[0, 2, :], simul2_householdsHousingType[0, 1, :])
    stats_per_housing_type.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_housing_types/Basile2.xlsx')

    #2. Compute stats on floods per income classes
    
    #Floods data
    
    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_income_class')

    #Income classes data
    grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP_intersect.csv', sep = ';')  
    income_class_grid = np.zeros((len(grid.dist), 4))  
    for index in range(0, len(grid.dist)): 
        intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])
        for i in range(0, len(intersect)): 
            if len(households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]]) != 0:  
                income_class_grid[index] = income_class_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]] / sum(grid_intersect.Area[grid_intersect.SP_CODE == intersect[i]])

    #Data
    stats_per_income_class = compute_stats_per_income_class(floods, path_data, income_class_grid)
    stats_per_income_class.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_income_class/data.xlsx')
    
    #Simul
    stats_per_income_class = compute_stats_per_income_class(floods, path_data, np.transpose(initialState_householdsCenter))
    stats_per_income_class.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_income_class/simul.xlsx')

    #Basile1
    stats_per_income_class = compute_stats_per_income_class(floods, path_data, np.transpose(simul1_householdsCenter[0, :, :]))
    stats_per_income_class.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_income_class/Basile1.xlsx')

    #Basile2
    stats_per_income_class = compute_stats_per_income_class(floods, path_data, np.transpose(simul2_householdsCenter[0, :, :]))
    stats_per_income_class.to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/stats_per_income_class/Basile2.xlsx')

    #3. Damages
    
    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/estimation_damages')
    
    #Data
    grid = SimulGrid()
    grid.create_grid()
    priceSimul = SP_to_grid_2011_1(households_data.sale_price_SP[2,:], households_data.Code_SP_2011, grid)
    formal_structure_cost  = priceSimul * (250000)  * land_UE.coeff_land[0, :] / count_formal
    formal_structure_cost[np.isinf(formal_structure_cost)] = np.empty(1)
    damages = compute_damages(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011, formal_structure_cost, content_cost, flood)
    damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/flood_damages/estimation_damages/data.xlsx')

    #Simul
    priceSimul = (initialState_rent[0, :] * param["coeff_A"] * param["coeff_b"]/ (interest_rate)) ** (1/param["coeff_a"])
    formal_structure_cost  = priceSimul * (250000)  * land_UE.coeff_land[0, :] / initialState_householdsHousingType[0, :]
    formal_structure_cost[np.isinf(formal_structure_cost)] = np.empty(1)
    damages = compute_damages(floods, path_data, initialState_householdsHousingType[0, :], initialState_householdsHousingType[3, :], initialState_householdsHousingType[2, :], initialState_householdsHousingType[1, :], formal_structure_cost, content_cost, flood)
    damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/flood_damages/estimation_damages/simul.xlsx')

    #Basile1
    priceSimul = (simul1_rent[0, 0, :] * param["coeff_A"] * param["coeff_b"]/ (interest_rate)) ** (1/param["coeff_a"])
    formal_structure_cost  = priceSimul * (250000)  * land_UE.coeff_land[0, :] / simul1_householdsHousingType[0, 0, :]
    formal_structure_cost[np.isinf(formal_structure_cost)] = np.empty(1)
    damages = compute_damages(floods, path_data, simul1_householdsHousingType[0, 0, :], simul1_householdsHousingType[0, 3, :], simul1_householdsHousingType[0, 2, :], simul1_householdsHousingType[0, 1, :], formal_structure_cost, content_cost, flood)
    damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/flood_damages/estimation_damages/basile1.xlsx')

    #Basile2
    priceSimul = (simul2_rent[0, 0, :] * param["coeff_A"] * param["coeff_b"]/ (interest_rate)) ** (1/param["coeff_a"])
    formal_structure_cost  = priceSimul * (250000)  * land_UE.coeff_land[0, :] / simul2_householdsHousingType[0, 0, :]
    formal_structure_cost[np.isinf(formal_structure_cost)] = np.empty(1)
    damages = compute_damages(floods, path_data, simul2_householdsHousingType[0, 0, :], simul2_householdsHousingType[0, 3, :], simul2_householdsHousingType[0, 2, :], simul2_householdsHousingType[0, 1, :], formal_structure_cost, content_cost, flood)
    damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/flood_damages/estimation_damages/basile2.xlsx')

    #4. Maps
    
    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/maps')

    data_5y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_5yr.xlsx"))
    data_20y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_20yr.xlsx"))
    data_50y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_50yr.xlsx"))
    data_100y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_100yr.xlsx"))
    data_1000y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_1000yr.xlsx"))

    data = pd.DataFrame([grid.horiz_coord, grid.vert_coord, data_100y.flood_depth])
    data = np.transpose(data)
    data_contour_100yr = pd.pivot(data, index='X', columns = "Y", values = 'flood_depth')

    #Data
    plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=count_formal,
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Formal housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 2)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=households_data.GV_count_RDP,
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Subsidized housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 3)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=households_data.informal_grid_2011,
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Informal settlements')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 4)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=households_data.backyard_grid_2011,
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Backyarding')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)   
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/maps/data.png')
    plt.close()
    
    #Simul
    plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=initialState_householdsHousingType[0, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Formal housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 2)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=initialState_householdsHousingType[3, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Subsidized housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 3)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=initialState_householdsHousingType[2, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Informal settlements')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 4)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=initialState_householdsHousingType[1, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Backyarding')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)   
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/maps/simul.png')
    plt.close()
    
    #Basile1
    plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul1_householdsHousingType[0, 0, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Formal housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 2)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul1_householdsHousingType[0, 3, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Subsidized housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 3)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul1_householdsHousingType[0, 2, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Informal settlements')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 4)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul1_householdsHousingType[0, 1, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Backyarding')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)   
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/maps/basile1.png')
    plt.close()
    
    #Basile2
    plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul2_householdsHousingType[0, 0, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Formal housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 2)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul2_householdsHousingType[0, 3, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Subsidized housing')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 3)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul2_householdsHousingType[0, 2, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Informal settlements')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
    plt.subplot(2, 2, 4)  # 1 ligne, 2 colonnes, sous-figure 2
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=simul2_householdsHousingType[0, 1, :],
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.title('Backyarding')
    plt.clim(0, 1000)
    plt.contour(data_contour_100yr.index, data_contour_100yr.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)   
    plt.savefig('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/flood_damages/maps/basile2.png')
    plt.close()

### Graph annualized damages
#damages =pd.DataFrame()
#damages["Structure"] = [13561467.56,	0,	227094.41,	128988.39]
#damages["Contents"] = [4205710.17,	271839.77,	657065.83,	325482.85]
#fig = plt.figure() # Create matplotlib figure
#ax = fig.add_subplot(111) # Create matplotlib axes
#width = 0.4
#damages.Structure.plot(kind='bar', color='red', ax=ax, width=width, position=1)
#damages.Contents.plot(kind='bar', color='blue', ax=ax, width=width, position=0)
#ax.set_ylabel('Annualized flood damages (R)')
#ax.legend(bbox_to_anchor=(0.4, 1))
#ax.set_xticks(np.arange(len(damages.Structure)))
#ax.set_xticklabels(["Formal", "Subsidized", "Informal", "Backyarding"], rotation = 0)
#plt.show()

def compute_stats_per_housing_type(floods, path_data, nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard):
    stats_per_housing_type = pd.DataFrame(columns = ['flood',
                                                     'fraction_formal_in_flood_prone_area', 'fraction_subsidized_in_flood_prone_area', 'fraction_informal_in_flood_prone_area', 'fraction_backyard_in_flood_prone_area',
                                                     'flood_depth_formal', 'flood_depth_subsidized', 'flood_depth_informal', 'flood_depth_backyard'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        stats_per_housing_type = stats_per_housing_type.append({'flood': type_flood, 
                                                                'fraction_formal_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_formal) / sum(nb_households_formal), 
                                                                'fraction_subsidized_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_subsidized) / sum(nb_households_subsidized),
                                                                'fraction_informal_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_informal) / sum(nb_households_informal), 
                                                                'fraction_backyard_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_backyard) / sum(nb_households_backyard),
                                                                'flood_depth_formal': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_formal)  / sum(flood['prop_flood_prone'] * nb_households_formal)),
                                                                'flood_depth_subsidized': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_subsidized)  / sum(flood['prop_flood_prone'] * nb_households_subsidized)),
                                                                'flood_depth_informal': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_informal)  / sum(flood['prop_flood_prone'] * nb_households_informal)),
                                                                'flood_depth_backyard': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_backyard)  / sum(flood['prop_flood_prone'] * nb_households_backyard))}, ignore_index = True)   
    return stats_per_housing_type

def compute_stats_per_income_class(floods, path_data, income_class_grid):
    stats_per_income_class = pd.DataFrame(columns = ['flood',
                                                     'fraction_class1_in_flood_prone_area', 'fraction_class2_in_flood_prone_area', 'fraction_class3_in_flood_prone_area', 'fraction_class4_in_flood_prone_area',
                                                     'flood_depth_class1', 'flood_depth_class2', 'flood_depth_class3', 'flood_depth_class4'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        stats_per_income_class = stats_per_income_class.append({'flood': type_flood, 
                                                                'fraction_class1_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,0]) / sum(income_class_grid[:,0]), 
                                                                'fraction_class2_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,1]) / sum(income_class_grid[:,1]),
                                                                'fraction_class3_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,2]) / sum(income_class_grid[:,2]), 
                                                                'fraction_class4_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,3]) / sum(income_class_grid[:,3]),
                                                                'flood_depth_class1': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,0])  / sum(flood['prop_flood_prone'] * income_class_grid[:,0])),
                                                                'flood_depth_class2': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,1])  / sum(flood['prop_flood_prone'] * income_class_grid[:,1])),
                                                                'flood_depth_class3': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,2])  / sum(flood['prop_flood_prone'] * income_class_grid[:,2])),
                                                                'flood_depth_class4': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,3])  / sum(flood['prop_flood_prone'] * income_class_grid[:,3]))}, ignore_index = True)   
    return stats_per_income_class

def compute_damages(floods, path_data,
                    nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard,
                    formal_structure_cost,
                    content_cost, flood):
    
    damages = pd.DataFrame(columns = ['flood',
                                      'formal_structure_damages',
                                      'subsidized_structure_damages',
                                      'informal_structure_damages',
                                      'backyard_structure_damages',
                                      'formal_content_damages',
                                      'subsidized_content_damages',
                                      'informal_content_damages',
                                      'backyard_content_damages'])
    for item in floods:
        type_flood = copy.deepcopy(item)
        data_flood = np.squeeze(pd.read_excel(path_data + item + ".xlsx"))
        formal_structure_damages = np.nansum(nb_households_formal * data_flood["prop_flood_prone"] * formal_structure_cost * flood.structural_damages(data_flood['flood_depth']))
        subsidized_structure_damages = 0
        informal_structure_damages = np.nansum(nb_households_informal * data_flood["prop_flood_prone"] * flood.informal_structure_value * flood.structural_damages(data_flood['flood_depth']))
        backyard_structure_damages = np.nansum(nb_households_backyard * data_flood["prop_flood_prone"] * flood.informal_structure_value * flood.structural_damages(data_flood['flood_depth']))
        formal_content_damages = np.nansum(nb_households_formal * data_flood["prop_flood_prone"] * content_cost[0, :] * flood.content_damages(data_flood['flood_depth']))
        subsidized_content_damages = np.nansum(nb_households_subsidized * data_flood["prop_flood_prone"] * content_cost[3, :] * flood.content_damages(data_flood['flood_depth']))
        informal_content_damages = np.nansum(nb_households_informal * data_flood["prop_flood_prone"] * content_cost[2, :] * flood.content_damages(data_flood['flood_depth']))
        backyard_content_damages = np.nansum(nb_households_backyard * data_flood["prop_flood_prone"] * content_cost[1, :] * flood.content_damages(data_flood['flood_depth']))
        damages = damages.append({'flood': type_flood,
                                  'formal_structure_damages': formal_structure_damages,
                                  'subsidized_structure_damages': subsidized_structure_damages,
                                  'informal_structure_damages': informal_structure_damages,
                                  'backyard_structure_damages': backyard_structure_damages,
                                  'formal_content_damages': formal_content_damages,
                                  'informal_content_damages': informal_content_damages,
                                  'backyard_content_damages': backyard_content_damages,
                                  'subsidized_content_damages': subsidized_content_damages}, ignore_index = True)
    
    return damages
