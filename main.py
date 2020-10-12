# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:34:23 2020

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

print("\n*** NEDUM-Cape-Town - Polycentric Version - Formal and Informal housing ***\n")

# %% Choose parameters and options

print("\n*** Load parameters and options ***\n")

option = choice_options()

option["households_anticipate_floods"] = 0
option["import_precalculated_parameters"] = 1
option["load_households_data"] = 0

#Floods
option["floods"] = 0
option["calibration_with_floods"] = 0
option["calibration_BIB_BIS_floods"] = 0

param = choice_param(option)

t = np.arange(0, 30) #Years of the scenario

name = "12102020_without_floods"

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

#Transport data
#yearTraffic = np.arange(0, 29, 2)
#trans = TransportData()
#trans.import_transport_data(option, grid, macro_data, param, job, households_data, yearTraffic, 1)
#with open('C:/Users/Charlotte Liotta/Desktop/cape_town/3. Code/data/import_transport', 'rb') as config_dictionary_file:
    # Step 3
#    trans = pickle.load(config_dictionary_file)

#Floods data
flood = FloodData()
flood.import_floods_data()
    
# %% Initial state

print('*** Initial state ***')

if option["ownInitializationSolver"] == 1:
    mat = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedUtilities.mat')
    Uo_init = [500, 10000, float(mat["utilitiesCorrected"][[0]]), float(mat["utilitiesCorrected"][1])]
else:
    Uo_init = 10000

initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(t[0], option, land_UE, grid, macro_data, param, job, Uo_init, flood)

# %% Compute differences between our initial state, Basile's, and the data

export_density_rents_sizes(grid, name + "_initialState", households_data, initialState_householdsHousingType, initialState_dwellingSize, initialState_rent)
export_utility_and_error(initialState_error, initialState_utility, name + "_initialState")
export_outputs_flood_damages(households_data, grid, name + "_initialState", initialState_householdsCenter, initialState_householdsHousingType, param, initialState_dwellingSize, initialState_rent)
    
# %% Scenarios

option["urban_edge"] = 0
grid = SimulGrid()
grid.create_grid()
land_noUE = Land()
land_noUE.import_land_use(grid, option, param, households_data, macro_data)



iterCalcLite = 1 #One iteration every year
print('*** Nedum Cape Town lite: one iteration every year ***\n')

option["adjustHousingInit"] = copy.deepcopy(option["adjustHousingSupply"])
option["adjustHousingSupply"] = 0

#New time step
yearsSimulations = np.arange(t[0], t[len(t) - 1] + 1, (t[1] - t[0])/iterCalcLite)

option["ownInitializationSolver"] = 1

#Preallocating outputs
simulation_dwellingSize = np.zeros((len(t), initialState_housingSupply.shape[0], initialState_housingSupply.shape[1]))
simulation_rent = np.zeros((len(t), initialState_housingSupply.shape[0], initialState_housingSupply.shape[1]))
simulation_households = np.zeros((len(t), initialState_housingSupply.shape[0], initialState_housingSupply.shape[0], initialState_housingSupply.shape[1]))
simulation_housingSupply = np.zeros((len(t), initialState_housingSupply.shape[0], initialState_housingSupply.shape[1]))
simulation_householdsHousingType = np.zeros((len(t), initialState_housingSupply.shape[0], initialState_housingSupply.shape[1]))
simulation_householdsCenter = np.zeros((len(t), initialState_housingSupply.shape[0], initialState_housingSupply.shape[1]))
simulation_error = np.zeros((len(t), initialState_housingSupply.shape[0]))
simulation_utility = np.zeros((len(t), initialState_housingSupply.shape[0]))
simulation_derivHousing = np.zeros((len(t), initialState_housingSupply.shape[1]))
    
for indexIter in range(0, len(yearsSimulations)):
    print(indexIter)
    yearTemp = copy.deepcopy(yearsSimulations[indexIter])
    statTemp_utility = copy.deepcopy(initialState_utility)
    statTemp_incomeMatrix = copy.deepcopy(initialState_incomeMatrix)
    statTemp_housingSupply = copy.deepcopy(initialState_housingSupply)
    statTemp_rent = copy.deepcopy(initialState_rent)
        
    if indexIter > 1:
                
        if indexIter == len(t):
            print('stop')
            
        grid = SimulGrid()
        grid.create_grid()
        
        #Simulation with equilibrium housing stock
        print('Simulation without constraint')
        option["adjustHousingSupply"] = 1
        incomeTemp = InterpolateIncomeEvolution(macro_data, param, option, grid, job, yearTemp)
        incomeTemp = incomeTemp[0, :]
        Uo_unconstrained = (statTemp_utility) / statTemp_incomeMatrix[0, :] * incomeTemp
        tmpi_error, tmpi_simulatedJobs, tmpi_householdsHousingType, tmpi_householdsCenter, tmpi_households, tmpi_dwellingSize, tmpi_housingSupply, tmpi_rent, tmpi_rentMatrix, tmpi_capitalLand, tmpi_incomeMatrix, tmpi_limitCity, tmpi_utility, tmpi_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land_noUE, grid, macro_data, param, job, Uo_unconstrained)
        #tmpi_error, tmpi_simulatedJobs, tmpi_householdsHousingType, tmpi_householdsCenter, tmpi_households, tmpi_dwellingSize, tmpi_housingSupply, tmpi_rent, tmpi_rentMatrix, tmpi_capitalLand, tmpi_incomeMatrix, tmpi_limitCity, tmpi_utility, tmpi_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land_noUE, grid, macro_data, param, job, Uo_unconstrained)
            
        #Estimation of the derivation of housing supply between t and t+1
        derivHousingTemp = EvolutionHousingSupply(land_noUE, param, option, yearsSimulations[indexIter], yearsSimulations[indexIter - 1], tmpi_housingSupply[0, :], statTemp_housingSupply[0, :])
        param["housing_in"] = statTemp_housingSupply[0,:] + derivHousingTemp
        
        grid = SimulGrid()
        grid.create_grid()
        
        #Run a new simulation with fixed housing
        print('Simulation with constraint')
        option["adjustHousingSupply"] = 0   
        Uo_simulation = (tmpi_utility + Uo_unconstrained) / 2
        initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land_noUE, grid, macro_data, param, job, Uo_simulation)
        #initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land_noUE, grid, macro_data, param, job, Uo_simulation)
       
        #Ro de la simulation libre
        statTemp_utility = copy.deepcopy(tmpi_utility)
        statTemp_derivHousing = copy.deepcopy(derivHousingTemp)

    else:
        
        statTemp_derivHousing = np.zeros(len(statTemp_rent[0,:]))
            
    if (indexIter - 1) / iterCalcLite - np.floor((indexIter - 1) / iterCalcLite) == 0:

        simulation_householdsCenter[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_householdsCenter)
        simulation_householdsHousingType[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_householdsHousingType)
        simulation_dwellingSize[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_dwellingSize)
        simulation_rent[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_rent)
        simulation_households[int((indexIter - 1) / iterCalcLite + 1), :, :, :] = copy.deepcopy(initialState_households)
        simulation_error[int((indexIter - 1) / iterCalcLite + 1), :] = copy.deepcopy(initialState_error)
        simulation_housingSupply[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_housingSupply)
        simulation_utility[int((indexIter - 1) / iterCalcLite + 1), :] = copy.deepcopy(initialState_utility)
        simulation_derivHousing[int((indexIter - 1) / iterCalcLite + 1), :] = copy.deepcopy(statTemp_derivHousing)
            
if len(t) < len(yearsSimulations):
    T = copy.deepcopy(t)
else:
    T = copy.deepcopy(yearsSimulations)

simulation_T = copy.deepcopy(T)


#I set adjustHousing to its initial value
option["adjustHousingSupply"] = copy.deepcopy(option["adjustHousingInit"])



simulation_householdsCenter, simulation_householdsHousingType, simulation_dwellingSize, simulation_rent, simulation_households, simulation_error, simulation_housingSupply, simulation_utility, simulation_derivHousing, simulation_T = RunDynamicEvolutionNEDUM_LOGIT(t, initialState_utility, initialState_incomeMatrix, initialState_housingSupply, initialState_rent, initialState_householdsCenter, initialState_householdsHousingType, initialState_dwellingSize, initialState_households, initialState_error, trans, grid, land_noUE, job, param, macro_data, option)

mat1 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations scenarios - 201908.mat')
mat2 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations - 201907.mat')
simul1 = mat1["simulation_noUE"]
simul2 = mat2["simulation_noUE"]

simul1_dwellingSize = simul1["dwellingSize"][0][0]
simul1_rent = simul1["rent"][0][0]
simul1_households = simul1["households"][0][0]
simul1_housingSupply = simul1["housingSupply"][0][0]
simul1_householdsHousingType = simul1["householdsHousingType"][0][0]
simul1_householdsCenter = simul1["householdsCenter"][0][0]
simul1_error = simul1["error"][0][0]
simul1_utility = simul1["utility"][0][0]
simul1_derivHousing = simul1["derivHousing"][0][0]
simul1_T = simul1["T"][0][0]

simul2_dwellingSize = simul2["dwellingSize"][0][0]
simul2_rent = simul2["rent"][0][0]
simul2_households = simul2["households"][0][0]
simul2_housingSupply = simul2["housingSupply"][0][0]
simul2_householdsHousingType = simul2["householdsHousingType"][0][0]
simul2_householdsCenter = simul2["householdsCenter"][0][0]
simul2_error = simul2["error"][0][0]
simul2_utility = simul2["utility"][0][0]
simul2_derivHousing = simul2["derivHousing"][0][0]
simul2_T = simul2["T"][0][0]

formal_s1 = simul1_householdsHousingType[0, 0, :]
backyard_s1 = simul1_householdsHousingType[0, 1, :]
informal_s1 = simul1_householdsHousingType[0, 2, :]
subsidized_s1 = simul1_householdsHousingType[0, 3, :]

formal_s2 = simul2_householdsHousingType[0, 0, :]
backyard_s2 = simul2_householdsHousingType[0, 1, :]
informal_s2 = simul2_householdsHousingType[0, 2, :]
subsidized_s2 = simul2_householdsHousingType[0, 3, :]

formal_p1 = simul1_householdsHousingType[29, 0, :]
backyard_p1 = simul1_householdsHousingType[29, 1, :]
informal_p1 = simul1_householdsHousingType[29, 2, :]
subsidized_p1 = simul1_householdsHousingType[29, 3, :]

formal_p2 = simul2_householdsHousingType[28, 0, :]
backyard_p2 = simul2_householdsHousingType[28, 1, :]
informal_p2 = simul2_householdsHousingType[28, 2, :]
subsidized_p2 = simul2_householdsHousingType[28, 3, :]

households_data.formal_grid_2011
households_data.GV_count_RDP
households_data.informal_grid_2011
households_data.backyard_grid_2011


trees_shp = shapefile.Writer(shapefile.POINT)
Set the autoBalance to 1. This enforces that for every record there must be a corresponding geometry.

trees_shp.autoBalance = 1
Create the field names and data types for each.

trees_shp.field("TREE_ID", "C")
trees_shp.field("ADDRESS", "C")
trees_shp.field("TOWN", "C")
trees_shp.field("TREE_SPEC", "C")
trees_shp.field("SPEC_DESC", "C")
trees_shp.field("COMMONNAME", "C")
trees_shp.field("AGE_DESC", "C")
trees_shp.field("HEIGHT", "C")
trees_shp.field("SPREAD", "C")
trees_shp.field("TRUNK", "C")
trees_shp.field("TRUNK_ACTL", "C")
trees_shp.field("CONDITION", "C")