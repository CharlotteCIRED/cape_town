# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:40:44 2020

@author: Charlotte Liotta
"""

#C'est le script qui permet de trouver Bib et Bis.

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
from sklearn.linear_model import LinearRegression

from parameters_and_options.parameters import *
from parameters_and_options.options import *
from data.data import *
from data.grille import *
from data.job import *
from data.land import *
from data.macro_data import *
from data.transport import *
from data.flood import *
from solver.solver import *
from solver.evolution import *
from plot_and_export_outputs.export_outputs import *
from plot_and_export_outputs.export_outputs_flood_damages import *


print('**************** NEDUM-Cape-Town - Calibration of the informal housing amenity parameters ****************')

# %% Choose parameters and options

print("\n*** Load parameters and options ***\n")

option = choice_options()

option["import_precalculated_parameters"] = 1
option["load_households_data"] = 0

#Floods
option["floods"] = 0
option["calibration_with_floods"] = 0
option["calibration_BIB_BIS_floods"] = 0

option["incur_formal_structure_damages"] = 'developers'
option["developers_pay_cost"] = 1

param = choice_param(option)

t = np.arange(0, 2)

# %% Import data for the simulation

print("\n*** Load data ***\n")

#Import grid
grid = SimulGrid()
grid.create_grid()

#Import macro data (population, inflation,...)
macro_data = MacroData()
macro_data.import_macro_data(param, option)

#Import data on households (Census, Housing, Income,...)
households_data = ImportHouseholdsData()
households_data.import_data(grid, param, option)

#Import employment data
job = ImportEmploymentData()
job.import_employment_data(grid, param, option, macro_data, t)

#Import land-use data
option["urban_edge"] = 1
land_UE = Land()
land_UE.import_land_use(grid, option, param, households_data, macro_data)

#Add construction parameters
param = add_construction_parameters(param, households_data, land_UE, grid)

#Floods data
flood = FloodData()
flood.import_floods_data()

# %% Run initial state for several values of amenities

Uo_init = np.array([1678, 3578, 16433, 76514]) #* 3

#List of parameters
listAmenityBackyard = np.arange(0.65, 0.81, 0.05)
listAmenitySettlement = np.arange(0.65, 0.81, 0.05)
housingTypeTotal = np.zeros((3, 4, len(listAmenityBackyard) * len(listAmenitySettlement)))

sumHousingTypes = lambda initialState_householdsHousingType : np.nansum(initialState_householdsHousingType, 1)
index = 0
for i in range(0, len(listAmenityBackyard)):
    for j in range(0, len(listAmenitySettlement)):
        grid = SimulGrid()
        grid.create_grid()
        param["amenity_backyard"] = listAmenityBackyard[i]
        param["amenity_settlement"] = listAmenitySettlement[j]
        initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(t[0], option, land_UE, grid, macro_data, param, job, Uo_init, flood)
        housingTypeTotal[0, : , index] = param["amenity_backyard"]
        housingTypeTotal[1, : , index] = param["amenity_settlement"]
        housingTypeTotal[2, : , index] = sumHousingTypes(initialState_householdsHousingType)
        Uo_init = initialState_utility
        index = index + 1

print('*** End of simulations for chosen parameters ***')

# %% Pick best solution

housingTypeData = np.array([np.nansum(households_data.formal_grid_2011) - np.nansum(initialState_householdsHousingType[3,:]), np.nansum(households_data.backyard_grid_2011), np.nansum(households_data.informal_grid_2011), np.nansum(households_data.formal_grid_2011 + households_data.backyard_grid_2011 + households_data.informal_grid_2011)])

distanceShare = np.abs(housingTypeTotal[2, 0:3, :] - housingTypeData[0:3, None])
distanceShareScore = distanceShare[0,:] + distanceShare[1,:]  + distanceShare[2,:]
which = np.argmin(distanceShareScore)
calibratedParamAmenities = housingTypeTotal[0:2, 0, which]

param["amenity_backyard"] = calibratedParamAmenities[0]
param["amenity_settlement"] = calibratedParamAmenities[1]
