# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 09:10:28 2020

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
from plot_and_export_outputs.export_outputs import *
from data.grid import *

#### EXPORT MAPS ###

def export_density_rents_sizes(grid, name, households_data, initialState_householdsHousingType, initialState_dwellingSize, initialState_rent):

    #1. Prepare folder and Basile's data

    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)

    mat1 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations scenarios - 201908.mat')
    mat2 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations - 201907.mat')
    simul1 = mat1["simulation_noUE"]
    simul2 = mat2["simulation_noUE"]

    #2. Housing types

    simul1_householdsHousingType = simul1["householdsHousingType"][0][0]
    simul2_householdsHousingType = simul2["householdsHousingType"][0][0]

    count_formal = households_data.formal_grid_2011 - households_data.GV_count_RDP
    count_formal[count_formal < 0] = 0

    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types')

    #Formal
    error = (initialState_householdsHousingType[0, :] / count_formal - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_diff_with_data.png')  
    export_map(count_formal, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_data.png', 1200)
    export_map(initialState_householdsHousingType[0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_simul.png', 1200)
    export_map(simul1_householdsHousingType[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_Basile1.png', 1200)
    export_map(simul2_householdsHousingType[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/formal_Basile2.png', 1200)

    #Subsidized
    error = (initialState_householdsHousingType[3, :] / households_data.GV_count_RDP - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_diff_with_data.png')  
    export_map(households_data.GV_count_RDP, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_data.png', 1200)
    export_map(initialState_householdsHousingType[3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_simul.png', 1200)
    export_map(simul1_householdsHousingType[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_Basile1.png', 1200)
    export_map(simul2_householdsHousingType[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/subsidized_Basile2.png', 1200)
    
    #Informal
    error = (initialState_householdsHousingType[2, :] / households_data.informal_grid_2011 - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_diff_with_data.png')  
    export_map(households_data.informal_grid_2011, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_data.png', 800)
    export_map(initialState_householdsHousingType[2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_simul.png', 800)
    export_map(simul1_householdsHousingType[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_Basile1.png', 800)
    export_map(simul2_householdsHousingType[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/informal_Basile2.png', 800)

    #Backyard
    error = (initialState_householdsHousingType[1, :] / households_data.backyard_grid_2011 - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_diff_with_data.png')  
    export_map(households_data.backyard_grid_2011, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_data.png', 800)
    export_map(initialState_householdsHousingType[1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/housing_types/backyard_simul.png', 800)
    export_map(simul1_householdsHousingType[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_Basile1.png', 800)
    export_map(simul2_householdsHousingType[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '\housing_types/backyard_Basile2.png', 800)

    #3. Dwelling size
    
    simul1_dwellingSize = simul1["dwellingSize"][0][0]
    simul2_dwellingSize = simul2["dwellingSize"][0][0]

    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size')
    
    grid = SimulGrid()
    grid.create_grid()
    dwelling_size = SP_to_grid_2011_1(households_data.spDwellingSize, households_data.Code_SP_2011, grid)
    
    #Data
    export_map(dwelling_size, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/data.png', 300)
    
    #Class 1
    error = (initialState_dwellingSize[0, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_diff_with_data.png')  
    export_map(initialState_dwellingSize[0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_simul.png', 300)
    export_map(simul1_dwellingSize[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_Basile1.png', 300)
    export_map(simul2_dwellingSize[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class1_Basile2.png', 300)

    #Class 2
    error = (initialState_dwellingSize[1, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_diff_with_data.png')  
    export_map(initialState_dwellingSize[1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_simul.png', 200)
    export_map(simul1_dwellingSize[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_Basile1.png', 200)
    export_map(simul2_dwellingSize[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class2_Basile2.png', 200)

    #Class 3
    error = (initialState_dwellingSize[2, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_diff_with_data.png')  
    export_map(initialState_dwellingSize[2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_simul.png', 200)
    export_map(simul1_dwellingSize[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_Basile1.png', 200)
    export_map(simul2_dwellingSize[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class3_Basile2.png', 200)

    #Class 4
    error = (initialState_dwellingSize[3, :] / dwelling_size - 1) * 100
    error_map(error, grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_diff_with_data.png')  
    export_map(initialState_dwellingSize[3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_simul.png', 100)
    export_map(simul1_dwellingSize[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_Basile1.png', 100)
    export_map(simul2_dwellingSize[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/dwelling_size/class4_Basile2.png', 100)

    #4. Rents
    
    simul1_rent = simul1["rent"][0][0]
    simul2_rent = simul2["rent"][0][0]

    os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents')
    
    #Class 1
    export_map(initialState_rent[0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class1_simul.png', 800)
    export_map(simul1_rent[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class1_Basile1.png', 800)
    export_map(simul2_rent[0, 0, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class1_Basile2.png', 800)

    #Class 2
    export_map(initialState_rent[1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class2_simul.png', 700)
    export_map(simul1_rent[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class2_Basile1.png', 700)
    export_map(simul2_rent[0, 1, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class2_Basile2.png', 700)

    #Class 3
    export_map(initialState_rent[2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class3_simul.png', 600)
    export_map(simul1_rent[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class3_Basile1.png', 600)
    export_map(simul2_rent[0, 2, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class3_Basile2.png', 600)

    #Class 4
    export_map(initialState_rent[3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class4_simul.png', 500)
    export_map(simul1_rent[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class4_Basile1.png', 500)
    export_map(simul2_rent[0, 3, :], grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/rents/class4_Basile2.png', 500)

def error_map(error, grid, export_name):
    map = plt.scatter(grid.horiz_coord, 
            grid.vert_coord, 
            s=None,
            c=error,
            cmap = 'RdYlGn',
            marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.clim(-100, 100)
    plt.savefig(export_name)
    plt.close()
    
def export_map(value, grid, export_name, lim):
    map = plt.scatter(grid.horiz_coord, 
            grid.vert_coord, 
            s=None,
            c=value,
            cmap = 'Reds',
            marker='.')
    plt.colorbar(map)
    plt.axis('off')
    plt.clim(0, lim)
    plt.savefig(export_name)
    plt.close()
    
def export_utility_and_error(initialState_error, initialState_utility, initialState_householdsHousingType, name):
    mat1 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations scenarios - 201908.mat')
    mat2 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations - 201907.mat')
    simul1 = mat1["simulation_noUE"]
    simul2 = mat2["simulation_noUE"]
    simul1_error = simul1["error"][0][0]
    simul2_error = simul2["error"][0][0]
    pd.DataFrame([initialState_error, np.transpose(simul1_error[0]), np.transpose(simul2_error[0])]).to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/error.xlsx')
    simul1_utility = simul1["utility"][0][0]
    simul2_utility = simul2["utility"][0][0]
    pd.DataFrame([initialState_utility, np.transpose(simul1_utility[0]), np.transpose(simul2_utility[0])]).to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/utility.xlsx')
    
    simul1_householdsHousingType = simul1["householdsHousingType"][0][0]
    simul2_householdsHousingType = simul2["householdsHousingType"][0][0]
    pd.DataFrame([np.nansum(initialState_householdsHousingType, 1), np.nansum(simul2_householdsHousingType[0, :, :], 1), np.nansum(simul2_householdsHousingType[0, :, :], 1)]).to_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name + '/hh_per_housing_type.xlsx')
