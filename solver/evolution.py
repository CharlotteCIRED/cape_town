# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 11:25:03 2020

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

from solver.useful_functions_solver import *
from data.functions_to_import_data import *
from solver.solver import *
from solver.compute_outputs_solver import *
from solver.evolution import *

def RunDynamicEvolutionNEDUM_LOGIT(t, initialState_utility, initialState_incomeMatrix, initialState_housingSupply, initialState_rent, initialState_householdsCenter, initialState_householdsHousingType, initialState_dwellingSize, initialState_households, initialState_error, trans, grid, land, job, param, macro_data, option):
    """Dynamic evolution of NEDUM.
    
    initialState_householdsCenter)
            simulation_householdsHousingType[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_householdsHousingType)
            simulation_dwellingSize[int((indexIter - 1) / iterCalcLite + 1), :, :] = copy.deepcopy(initialState_dwellingSize)
            simulation_households[int((indexIter - 1) / iterCalcLite + 1), :, :, :] = copy.deepcopy(initialState_households)
            simulation_error[int((indexIter - 1) / iterCalcLite + 1), :] = copy.deepcopy(initialState_error)
            simulation_derivHousing[int((indexIter - 1) / iterCalcLite + 1), :] = copy.deepcopy(statTemp_derivHousing)
            
    
    Computes equilibrium for each year, then adds inertia on the building stock
    """

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
    
        yearTemp = copy.deepcopy(yearsSimulations[indexIter])
        statTemp_utility = copy.deepcopy(initialState_utility)
        statTemp_incomeMatrix = copy.deepcopy(initialState_incomeMatrix)
        statTemp_housingSupply = copy.deepcopy(initialState_housingSupply)
        statTemp_rent = copy.deepcopy(initialState_rent)
        
        if indexIter > 1:
                
            if indexIter == len(t):
                print('stop')

        
            #Simulation with equilibrium housing stock
            print('Simulation without constraint')
            option["adjustHousingSupply"] = 1
            incomeTemp = InterpolateIncomeEvolution(macro_data, param, option, grid, job, yearTemp)
            incomeTemp = incomeTemp[0, :]
            Uo_unconstrained = (statTemp_utility) / statTemp_incomeMatrix[0, :] * incomeTemp
            tmpi_error, tmpi_simulatedJobs, tmpi_householdsHousingType, tmpi_householdsCenter, tmpi_households, tmpi_dwellingSize, tmpi_housingSupply, tmpi_rent, tmpi_rentMatrix, tmpi_capitalLand, tmpi_incomeMatrix, tmpi_limitCity, tmpi_utility, tmpi_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land, grid, macro_data, param, job, Uo_unconstrained)
            #tmpi_error, tmpi_simulatedJobs, tmpi_householdsHousingType, tmpi_householdsCenter, tmpi_households, tmpi_dwellingSize, tmpi_housingSupply, tmpi_rent, tmpi_rentMatrix, tmpi_capitalLand, tmpi_incomeMatrix, tmpi_limitCity, tmpi_utility, tmpi_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land_noUE, grid, macro_data, param, job, Uo_unconstrained)
            
            #Estimation of the derivation of housing supply between t and t+1
            derivHousingTemp = EvolutionHousingSupply(land, param, option, yearsSimulations[indexIter], yearsSimulations[indexIter - 1], tmpi_housingSupply[0, :], statTemp_housingSupply[0, :])
            param["housing_in"] = statTemp_housingSupply[0,:] + derivHousingTemp

            #Run a new simulation with fixed housing
            print('Simulation with constraint')
            option["adjustHousingSupply"] = 0   
            Uo_simulation = (tmpi_utility + Uo_unconstrained) / 2
            initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = RunEquilibriumSolverNEDUM_LOGIT(yearTemp, trans, option, land, grid, macro_data, param, job, Uo_simulation)
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

    return simulation_householdsCenter, simulation_householdsHousingType, simulation_dwellingSize, simulation_rent, simulation_households, simulation_error, simulation_housingSupply, simulation_utility, simulation_derivHousing, simulation_T

def EvolutionHousingSupply(land, param, option, t1, t0, housingSupply1, housingSupply0):
    
    T = copy.deepcopy(t1)

    #Interpolate for the simulation year
    housingLimitSimulation = InterpolateHousingLimitEvolution(land, option, param, T)

    #New housing supply (accounting for inertia and depreciation w/ time)
    if t1 - t0 > 0:
        diffHousing = (housingSupply1 - housingSupply0) * (housingSupply1 > housingSupply0) * (t1 - t0) / param["timeInvestHousing"] - housingSupply0 * (t1 - t0)  / param["timeDepreciationBuildings"]
    else:
        diffHousing = (housingSupply1 - housingSupply0) * (housingSupply1 < housingSupply0) * (t1 - t0) / param["timeInvestHousing"] - housingSupply0 * (t1 - t0)  / param["timeDepreciationBuildings"]

    housingSupplyTarget = housingSupply0 + diffHousing

    #Housing height is limited by potential regulations
    housingSupplyTarget = np.minimum(housingSupplyTarget, housingLimitSimulation)
    minimumHousingSupplyInterp = interp1d(np.array([2001, 2011, 2100]) - param["baseline_year"], np.transpose([np.zeros(len(param["minimumHousingSupply"])), param["minimumHousingSupply"], param["minimumHousingSupply"]]))
    minimumHousingSupplyInterp = minimumHousingSupplyInterp(t1)                                                                                       
    housingSupplyTarget = np.maximum(housingSupplyTarget, minimumHousingSupplyInterp)

    return housingSupplyTarget - housingSupply0



