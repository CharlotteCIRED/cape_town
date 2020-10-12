# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 12:35:45 2020

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

#def RunEquilibriumSolverNEDUM_LOGIT(yearEquilibrium, trans, option, land, grid, macro_data, param, job, Uo_init, param["housing_in"]):
    
def RunEquilibriumSolverNEDUM_LOGIT(yearEquilibrium, option, land, grid, macro_data, param, job, Uo_init, flood):
    
    #Income for the year (varies in time)
    incomeMatrix = InterpolateIncomeEvolution(macro_data, param, option, grid, job, yearEquilibrium)
    #Average income for the year of the simulation
    incomeAverage = macro_data.income(yearEquilibrium)

    #Interpolate income net of commuting costs
    incomeNetOfCommuting = np.load("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/precalculated_transport/year_" + str(yearEquilibrium) + ".npy")
    #incomeNetOfCommuting = InterpolateIncomeNetOfCommutingCostsEvolution(trans, param, yearEquilibrium)
    
    #Interpolate interest rates
    interestRate = InterpolateInterestRateEvolution(macro_data, yearEquilibrium)

    #Population
    population = InterpolatePopulationEvolution(macro_data, yearEquilibrium)
    totalRDP = macro_data.rdp(yearEquilibrium)

    #Construction coefficient
    constructionParam = InterpolateCoefficientConstruction(option, param, macro_data, incomeAverage)

    #Evolution of coeffLand
    land.coeffLand = InterpolateLandCoefficientEvolution(land, option, param, yearEquilibrium)
    land.numberPropertiesRDP = land.spline_estimate_RDP(yearEquilibrium)

    #Limit of housing construction
    housingLimit = InterpolateHousingLimitEvolution(land, option, param, yearEquilibrium);

    #Minimum housing supply
    housingMini2011 = param["minimumHousingSupply"]    
    inter_min_housing_supply = interp1d(np.array([2001, 2011, 2100]) - param["baseline_year"], np.transpose([np.zeros(len(grid.dist)), housingMini2011, housingMini2011]))
    param["minimumHousingSupply"] = inter_min_housing_supply(yearEquilibrium)

    #Transaction cost is the rent at the city limit (per year)
    agriculturalRent = InterpolateAgriculturalRentEvolution(option, param, macro_data, yearEquilibrium)
    rentReference = agriculturalRent

    #Tax outside the urban edge
    param["taxUrbanEdgeMat"] = np.zeros(len(grid.dist))
    if option["taxOutUrbanEdge"] == 1:
        param["taxUrbanEdgeMat"][land.urbanEdge == 0] = param["taxUrbanEdge"] * interestRate

    #Computation of the initial state
    if option["ownInitializationSolver"] == 1:
        initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = ComputeEquilibrium(option, land, grid, macro_data, param, yearEquilibrium, rentReference, housingLimit, incomeMatrix, incomeAverage, incomeNetOfCommuting, interestRate, population, agriculturalRent, constructionParam, job, Uo_init, totalRDP, flood);
    else:
        initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation = ComputeEquilibrium(option, land, grid, macro_data, param, yearEquilibrium, rentReference, housingLimit,incomeMatrix, incomeAverage, incomeNetOfCommuting, interestRate, population, agriculturalRent ,constructionParam, job, 1, totalRDP, flood);

    return initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation

def ComputeEquilibrium(option, land, grid, macro_data, param, yearEquilibrium, rentReference, housingLimit, incomeMatrix, incomeAverage, incomeNetOfCommuting, interestRate, population, agriculturalRent, constructionParam, job, Uo_init, totalRDP, flood):

    maxIteration = param["max_iter"]
    precision = param["precision"]

    # %% Preparation of the variables

    interestRate = interestRate + param["depreciation_rate"]

    #Income of each class
    averageIncome = interp1d(job.year, np.transpose(job.averageIncomeGroup[0:len(job.year), :]))
    averageIncome = averageIncome(yearEquilibrium + param["baseline_year"])
    job.incomeMult = averageIncome / macro_data.income(yearEquilibrium)
    param["incomeYearReference"] = macro_data.income_year_reference

    #Number of households of each class
    householdsGroup = interp1d(job.year, np.transpose(job.totalHouseholdsGroup[0:len(job.year), :]))
    householdsGroup = householdsGroup(yearEquilibrium + param["baseline_year"])

    #Ajust the population to remove the population in RDP
    ratio = population / sum(householdsGroup)
    householdsGroup = householdsGroup * ratio
    householdsGroup[0] = np.max(householdsGroup[0] - totalRDP, 0) #In case we have to much RDP
    employmentCenters = np.array([householdsGroup, np.matlib.repmat(grid.x_center,1,4).squeeze(), np.matlib.repmat(grid.y_center,1,4).squeeze()])

    #multiProbaGroup refers to fixed locations for income groups
    multiProbaGroup = param["multiProbaGroup"]

    # %% Amenities

    #Loading amenities
    amenities = land.amenities.squeeze()

    #We transform amenities in a matrix with as many lines as employment centers
    #amenities = np.matlib.repmat(amenities, incomeMatrix.shape[0], 1)

    # %% Pre-calculation of the utility / rent relationship

    #Precalculations for rents
    #uti = lambda Ro, revenu : ComputeUtilityFromRent(Ro, revenu, param["q0"], param) #EQUATION C.2

    #decompositionRent = np.concatenate(([10 ** (-9), 10 ** (-4), 10 ** (-3), 10 ** (-2)], np.arange(0.02, 0.081, 0.015), np.arange(0.1, 1.01, 0.02)))
    #decompositionIncome = np.concatenate(([10 ** (-9), 10 ** (-4), 10 ** (-3.5), 10 ** (-3), 10 ** (-2.5), 10 ** (-2), 0.03], np.arange(0.06, 1.01, 0.02)))

    #incomeVector = np.nanmax(incomeNetOfCommuting) * decompositionIncome
    #incomeMat = np.matlib.repmat(incomeVector, len(incomeVector), 1)
    
    #rentVector = incomeVector / param["q0"] #the maximum rent is the rent for which u = 0
    #rentMatrix = np.transpose(rentVector) * decompositionRent

    #XX = copy.deepcopy(incomeMat)
    #YY_R = uti(rentMatrix, incomeMat)
    #ZZ_R = copy.deepcopy(rentMatrix)
    #solus_R = lambda x, y : griddata((XX, YY), ZZ_R ** param["beta"], (x, y)) ** (1 / param["beta"])

    #Precalculations for dwelling sizes    
    utilitySize = lambda q, income : ComputeUtilityFromDwellingSize(q, income, param["q0"], param, flood, option) #EQUATION C.2

    decompositionQ = np.concatenate(([10 ** (-8), 10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1)], np.arange(0.11, 0.15, 0.01), np.arange(0.15, 1.15, 0.05), np.arange(1.2, 3.1, 0.1), np.arange(3.5, 13.1, 0.25), np.arange(15, 60, 0.5), np.arange(60, 100, 2.5), np.arange(110, 210, 10), [250, 300, 500, 1000, 2000, 200000, 1000000, 10 ** 12]))
    decompositionIncome = np.concatenate(([10 ** (-9), 10 ** (-4), 10 ** (-3.5), 10 ** (-3), 10 ** (-2.5), 10 ** (-2), 0.03], np.arange(0.06, 2.01, 0.01), np.arange(2.2, 2.7, 0.2), np.arange(3, 10, 1), [100, 10 ** 4]))

    incomeVector = np.nanmax(incomeNetOfCommuting) * decompositionIncome
    incomeMat = np.matlib.repmat(incomeVector, len(incomeVector), 1)
   
    dwellingSizeVector = param["q0"] + decompositionQ * 10
    dwellingSizeMatrix = np.transpose(np.matlib.repmat(dwellingSizeVector, len(incomeVector), 1))
    
    print(incomeMat.shape)
    print(sum(np.isnan(incomeMat)))
    print(dwellingSizeMatrix.shape)
    print(sum(np.isnan(dwellingSizeMatrix)))
    XX = copy.deepcopy(incomeMat)
    YY_Q = utilitySize(dwellingSizeMatrix, incomeMat)
    ZZ_Q = copy.deepcopy(dwellingSizeMatrix)
    param["max_U"] = np.nanmax(np.nanmax(YY_Q))
    param["max_q"] = np.max(dwellingSizeVector)
    solus_Q_temp = lambda x, y : griddata(points = (np.concatenate(XX), np.concatenate(YY_Q)), values = np.concatenate(ZZ_Q), xi = (x, np.fmin(y, param["max_U"])))

    #Redefine a grid (to use griddedInterpolant)
    #logUtilityVect = np.arange(-1, np.log(np.nanmax(np.nanmax(0.2 * incomeNetOfCommuting))) - 0.05, 0.05)
    #logIncome = np.arange(-1, np.log(np.nanmax(np.nanmax(incomeNetOfCommuting * 1.60))) + 0.1, 0.1)
    #logDwellingSize = np.log(solus_Q_temp(np.exp(logIncome), np.exp(logUtilityVect)))
    #solus_Q  = lambda income, utility : np.exp(griddata(points = (logUtilityVect, logIncome), values = logDwellingSize, xi = (np.log(utility), np.log(income))))
    solus_Q = solus_Q_temp
    
    #New dimensions to the grid (we remove the locations with coeffLand = 0)
    selectedPixels = (np.sum(land.coeffLand, 0) > 0.01).squeeze() & (np.nanmax(incomeNetOfCommuting, 0) > 0)
    land.coeffLand = land.coeffLand[:, selectedPixels]
    gridTemp = copy.deepcopy(grid)
    grid.dist = grid.dist[selectedPixels]
    housingLimit = housingLimit[selectedPixels]
    multiProbaGroup = multiProbaGroup[:, selectedPixels]
    incomeNetOfCommuting = incomeNetOfCommuting[:, selectedPixels]
    param_minimumHousingSupply = copy.deepcopy(param["minimumHousingSupply"][selectedPixels])
    param_housing_in = copy.deepcopy(param["housing_in"][selectedPixels])
    param_taxUrbanEdgeMat = copy.deepcopy(param["taxUrbanEdgeMat"][selectedPixels])
    incomeMatrix = incomeMatrix[selectedPixels, :]
    amenities = amenities[selectedPixels]

    #Income net of commuting
    transTemp_incomeNetOfCommuting = incomeNetOfCommuting 

    #Useful variables for the solver
    diffUtility = np.zeros((maxIteration, employmentCenters.shape[1]))
    simulatedPeopleHousingTypes = np.zeros((maxIteration,3,len(grid.dist))) #3 is because we have 3 types of housing in the solver
    simulatedPeople = np.zeros((3, 4, len(grid.dist)))
    simulatedJobs = np.zeros((maxIteration,3,employmentCenters.shape[1]))
    totalSimulatedJobs = np.zeros((maxIteration,employmentCenters.shape[1]))
    rentMatrix = np.zeros((maxIteration,3,len(grid.dist)))
    errorMaxAbs = np.zeros(maxIteration)
    errorMax = np.zeros(maxIteration)
    errorMean = np.zeros(maxIteration)
    numberError = np.zeros(maxIteration)
    error = np.zeros((maxIteration, employmentCenters.shape[1]))
    housingSupply = np.empty((3,len(grid.dist)))
    dwellingSize = np.empty((3,len(grid.dist)))
    R_mat = np.empty((3, 4, len(grid.dist)))
    
    #Utility for each center: variable we will adjust in the solver
    Uo = np.zeros((maxIteration,employmentCenters.shape[1]))
    
    #impossible_population = 1 if we cannot reach the objective population
    impossiblePopulation = np.zeros(employmentCenters.shape[1], 'bool') 
    numberImpossiblePopulation = 0
    
    #dummy that exits the solver if we cannot reach objective for the remaining centers
    conditionPossible = np.ones(1, 'bool')
    
    #Definition of Uo
    #if option["ownInitializationSolver"] == 0:
     #   Uo[0,:] = averageIncome * 0.2 #Initially, utility is set above the expected level
    #else:
    Uo[0,:] = Uo_init

    indexIteration = 0
    convergenceFactorInitial = 0.045 * (np.nanmean(averageIncome) / macro_data.income_year_reference) ** 0.4 #0.007;

    param["convergenceFactor"] = convergenceFactorInitial

    #Formal housing
    simulatedJobs[indexIteration,0,:],rentMatrix[indexIteration,0,:],simulatedPeopleHousingTypes[indexIteration,0,:],simulatedPeople[0,:,:],housingSupply[0,:],dwellingSize[0,:], R_mat[0,:,:] = ComputeNEDUMOutput_LOGIT(Uo[indexIteration,:],param,option,transTemp_incomeNetOfCommuting,grid,agriculturalRent,housingLimit,rentReference,constructionParam,interestRate,incomeMatrix,multiProbaGroup, 0, 0, land.coeffLand[0,:], job, amenities, solus_Q, 'formal', param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood)

    #Backyard housing
    simulatedJobs[indexIteration,1,:],rentMatrix[indexIteration,1,:],simulatedPeopleHousingTypes[indexIteration,1,:],simulatedPeople[1,:,:],housingSupply[1,:],dwellingSize[1,:], R_mat[1,:,:] = ComputeNEDUMOutput_LOGIT(Uo[indexIteration,:],param,option,transTemp_incomeNetOfCommuting,grid,agriculturalRent,housingLimit,rentReference,constructionParam,interestRate,incomeMatrix,multiProbaGroup, 0, 0, land.coeffLand[1,:], job, amenities, solus_Q, 'backyard', param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood)

    #Informal settlements
    simulatedJobs[indexIteration,2,:],rentMatrix[indexIteration,2,:],simulatedPeopleHousingTypes[indexIteration,2,:],simulatedPeople[2,:,:],housingSupply[2,:],dwellingSize[2,:], R_mat[2,:,:] = ComputeNEDUMOutput_LOGIT(Uo[indexIteration,:],param,option,transTemp_incomeNetOfCommuting,grid,agriculturalRent,housingLimit,rentReference,constructionParam,interestRate,incomeMatrix,multiProbaGroup, 0, 0, land.coeffLand[2,:], job, amenities, solus_Q, 'informal', param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood)

    #Total simulated population
    totalSimulatedJobs[indexIteration,:] = np.sum(simulatedJobs[indexIteration,:,:], 0)

    #deriv_U will be used to adjust the utility levels
    diffUtility[indexIteration,:] = np.log((totalSimulatedJobs[indexIteration, :] + 10) /(employmentCenters[0,:] + 10))
    diffUtility[indexIteration,:] = diffUtility[indexIteration,:] * param["convergenceFactor"]
    diffUtility[indexIteration, diffUtility[indexIteration, :]>0] = diffUtility[indexIteration, diffUtility[indexIteration, :] > 0] * 1.1

    #Difference with reality
    error[indexIteration, :] = (totalSimulatedJobs[indexIteration, :] / householdsGroup - 1) * 100
    errorMaxAbs[indexIteration] = np.nanmax(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1))
    errorMax[indexIteration] = -1
    errorMean[indexIteration] = np.nanmean(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] /(householdsGroup[employmentCenters[0, :] != 0] + 0.001) - 1))
    numberError[indexIteration] = np.nansum(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1) > precision)

    #Memory
    indexMemory = indexIteration
    simulatedPeopleMemory = simulatedPeople
    housingStockMemory = housingSupply
    dwellingSizeMemory = dwellingSize
    errorMeanMemory = numberError[indexMemory]


    while (indexIteration < maxIteration - 1) & (errorMaxAbs[indexIteration] > precision) & conditionPossible:
    
        #Iteration
        indexIteration = indexIteration + 1
    
        #Adjusting the level of utility
        Uo[indexIteration, :] = np.exp(np.log(Uo[indexIteration - 1, :]) + diffUtility[indexIteration - 1, :]) 
        #Minimum and maximum levels of utility
        Uo[indexIteration, Uo[indexIteration, :] < 0] = 10
        Uo[indexIteration, impossiblePopulation] = 10 #For the centers for which the objective cannot be attained (impossible_population = 1), utility level is set at an arbitrary low level
        Uo[indexIteration, employmentCenters[0, :]== 0] = 10000000
    
        #Adjusting param.factor_convergence
        param["convergenceFactor"] = convergenceFactorInitial / (1 + 0.5 * np.abs((totalSimulatedJobs[indexIteration, :] + 100) / (householdsGroup + 100)-1)) #.*(Jval./mean(Jval)).^0.3 %We adjust the parameter to how close we are from objective 
        param["convergenceFactor"] = param["convergenceFactor"] * (1 - 0.6 * indexIteration / maxIteration)
        
    
        #Formal housing
        simulatedJobs[indexIteration, 0, :], rentMatrix[indexIteration, 0, :], simulatedPeopleHousingTypes[indexIteration, 0, :], simulatedPeople[0, :, :], housingSupply[0,:], dwellingSize[0, :], R_mat[0, :, :] = ComputeNEDUMOutput_LOGIT(Uo[indexIteration, :], param, option, transTemp_incomeNetOfCommuting,grid,agriculturalRent, housingLimit, rentReference,constructionParam,interestRate,incomeMatrix,multiProbaGroup, 0, 0, land.coeffLand[0,:], job, amenities, solus_Q, 'formal', param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood)

        #Backyard housing
        simulatedJobs[indexIteration, 1, :], rentMatrix[indexIteration, 1, :], simulatedPeopleHousingTypes[indexIteration, 1, :], simulatedPeople[1, :, :], housingSupply[1, :], dwellingSize[1, :], R_mat[1,:,:] = ComputeNEDUMOutput_LOGIT(Uo[indexIteration, :], param, option, transTemp_incomeNetOfCommuting,grid,agriculturalRent,housingLimit, rentReference,constructionParam,interestRate,incomeMatrix, multiProbaGroup, 0, 0, land.coeffLand[1, :], job, amenities, solus_Q, 'backyard', param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood)
            
        #Informal settlements
        simulatedJobs[indexIteration, 2, :], rentMatrix[indexIteration, 2, :], simulatedPeopleHousingTypes[indexIteration, 2, :], simulatedPeople[2, :, :], housingSupply[2, :], dwellingSize[2, :], R_mat[2,:,:] = ComputeNEDUMOutput_LOGIT(Uo[indexIteration, :], param, option, transTemp_incomeNetOfCommuting,grid,agriculturalRent,housingLimit, rentReference,constructionParam,interestRate,incomeMatrix,multiProbaGroup,0,0, land.coeffLand[2, :], job, amenities, solus_Q, 'informal', param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood)

        #total simulated population
        totalSimulatedJobs[indexIteration,:] = np.sum(simulatedJobs[indexIteration, :, :], 0)
    
        
        #deriv_U will be used to adjust the utility levels
        diffUtility[indexIteration, :] = np.log((totalSimulatedJobs[indexIteration, :] + 10) / (employmentCenters[0, :] + 10))
        diffUtility[indexIteration, :] = diffUtility[indexIteration, :] * param["convergenceFactor"]
        diffUtility[indexIteration, diffUtility[indexIteration, :] > 0] = diffUtility[indexIteration, diffUtility[indexIteration, :] > 0] * 1.1
        
        #Variables to display
        error[indexIteration, :] = (totalSimulatedJobs[indexIteration, :] / householdsGroup - 1) * 100
        errorMaxAbs[indexIteration] = np.max(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1))
        m = np.argmax(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1))
        erreur_temp = (totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1)
        errorMax[indexIteration] = erreur_temp[m]
        errorMean[indexIteration] = np.mean(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / (householdsGroup[employmentCenters[0, :] != 0] + 0.001) - 1))
        numberError[indexIteration] = np.sum(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1) > precision)
    
    
        #In case, for one type of households, it is impossible to attain the objective population (basic need effect)
        if ((sum(Uo[indexIteration, :] < 1) > 0) & (np.max((totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1)) < precision)):
            impossiblePopulation[Uo[indexIteration, :] < 1] = np.ones(1, 'bool')
        if (sum(impossiblePopulation) + sum(np.abs(totalSimulatedJobs[indexIteration, employmentCenters[0, :] != 0] / householdsGroup[employmentCenters[0, :] != 0] - 1) < precision)) >= len(job.incomeMult): #If we have to stop the solver
            if sum(impossiblePopulation) == numberImpossiblePopulation:
                conditionPossible = np.zeros(1, 'bool') #We exit the solver
            else:
                numberImpossiblePopulation = sum(impossiblePopulation) #Gives the centers for which the model could not solve
        impossiblePopulation[totalSimulatedJobs[indexIteration, :] > (1 + precision) * householdsGroup] = 0 #In case there are problems with initialization
    
    
        #The best solution attained is stored in memory
        if numberError[indexIteration] <= errorMeanMemory:
            indexMemory = indexIteration
            simulatedPeopleMemory = simulatedPeople
            errorMeanMemory = numberError[indexMemory]
            housingStockMemory = housingSupply
            dwellingSizeMemory = dwellingSize
        
        print(error[indexIteration, :])

    indexIteration = indexMemory
    simulatedPeople = simulatedPeopleMemory
    housingSupply = housingStockMemory
    dwellingSize = dwellingSizeMemory
        
    #RDP houses 
    householdsRDP = land.numberPropertiesRDP * totalRDP / sum(land.numberPropertiesRDP)
    constructionRDP = np.matlib.repmat(param["RDP_size"] / (param["RDP_size"] + param["backyard_size"]), 1, len(gridTemp.dist)) * 1000000
    dwellingSizeRDP = np.matlib.repmat(param["RDP_size"], 1, len(gridTemp.dist))

    simulatedPeopleWithRDP = np.zeros((4, len(job.incomeMult), len(gridTemp.dist)))
    simulatedPeopleWithRDP[0, :, selectedPixels] = np.transpose(simulatedPeople[0, :, :,])
    simulatedPeopleWithRDP[1, :, selectedPixels] = np.transpose(simulatedPeople[1, :, :,])
    simulatedPeopleWithRDP[2, :, selectedPixels] = np.transpose(simulatedPeople[2, :, :,])    
    simulatedPeopleWithRDP[3, 0, :] = householdsRDP

    # %%Outputs of the solver 
    
    #Employment centers
    initialState_error = error[indexIteration, :]
    initialState_simulatedJobs = simulatedJobs[indexIteration, :, :]

    #Number of people
    initialState_householdsHousingType = np.sum(simulatedPeopleWithRDP, 1)
    initialState_householdsCenter = np.sum(simulatedPeopleWithRDP, 0)
    initialState_households = simulatedPeopleWithRDP

    #Housing stock and dwelling size
    housingSupplyExport = np.zeros((3, len(gridTemp.dist)))
    dwellingSizeExport = np.zeros((3, len(gridTemp.dist)))
    housingSupplyExport[:, selectedPixels] = housingSupply
    dwellingSizeExport[:, selectedPixels] = copy.deepcopy(dwellingSize)
    dwellingSizeExport[dwellingSizeExport<=0] = np.nan
    initialState_dwellingSize = np.vstack([dwellingSizeExport, dwellingSizeRDP])
    initialState_housingSupply = np.vstack([housingSupplyExport, constructionRDP])
    
    #Rents (hh in RDP pay a rent of 0)
    rentTemp = copy.deepcopy(rentMatrix[indexIteration, :, :])
    rentExport = np.zeros((3, len(gridTemp.dist)))
    rentExport[:, selectedPixels] = copy.deepcopy(rentTemp)
    rentExport[:, selectedPixels == 0] = np.nan
    initialState_rent = np.vstack([rentExport, np.zeros(len(gridTemp.dist))])
    rentMatrixExport = np.zeros((3, job.averageIncomeGroup.shape[1], len(gridTemp.dist)))
    rentMatrixExport[:,:,selectedPixels] = copy.deepcopy(R_mat)
    rentMatrixExport[:,:,selectedPixels == 0] = np.nan
    initialState_rentMatrix = copy.deepcopy(rentMatrixExport)
    
    #Other outputs
    initialState_capitalLand = (housingSupply / (param["coeff_A"])) ** (1 / param["coeff_b"])
    initialState_incomeMatrix = copy.deepcopy(incomeMatrix)
    initialState_limitCity = [initialState_households > 1]
    initialState_utility = Uo[indexIteration, :]
    initialState_impossiblePopulation = impossiblePopulation

    return initialState_error, initialState_simulatedJobs, initialState_householdsHousingType, initialState_householdsCenter, initialState_households, initialState_dwellingSize, initialState_housingSupply, initialState_rent, initialState_rentMatrix, initialState_capitalLand, initialState_incomeMatrix, initialState_limitCity, initialState_utility, initialState_impossiblePopulation


#def ComputeUtilityFromRent(Ro, income, basic_q, param):
    #if (basic_q != 0):
     #   utility = param["alpha"] ** param["alpha"] * param["beta"] ** param["beta"] * np.sign(income - basic_q *Ro) * np.abs(income - basic_q * Ro) / (Ro ** param["beta"])
     #   utility[(income - basic_q * Ro) < 0] = 0
    #else:
    #    utility = param["alpha"] ** param["alpha"] * param["beta"] ** param["beta"] * income / (Ro ** param["beta"])
    
    #utility[income==0] = 0
    #return utility

def ComputeUtilityFromDwellingSize(q, income, basic_q, param, flood, option):
    #if (basic_q != 0):
    if option["floods"] == 0:
        utility = (param["alpha"] * income) ** param["alpha"] * (q - basic_q) / (q - param["alpha"] * basic_q) ** param["alpha"]
    elif option["floods"] == 1:
        utility = (param["alpha"] * (income - (flood.d_contents * flood.content_cost))) ** param["alpha"] * (q - basic_q) / (q - param["alpha"] * basic_q) ** param["alpha"]
    utility[q < basic_q] = 0
    #else:
        #utility = (param["alpha"] * income) ** param["alpha"] * q ** param["beta"]
        
    utility[income==0] = 0
    return utility

