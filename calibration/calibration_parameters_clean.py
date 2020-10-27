# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:47:40 2020

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
from solver.useful_functions_solver import *
from calibration.import_amenities import *
from calibration.compute_income import *
from calibration.estimate_parameters_by_scanning import *
from calibration.estimate_parameters_by_optimization import * 

print('**************** NEDUM-Cape-Town - Calibration of the parameters ****************')

# %% Choose parameters and options

print("\n*** Load parameters and options ***\n")

option = choice_options()
option["floods"] = 1
option["load_households_data"] = 0
option["developers_pay_cost"] = 1
option["import_precalculated_parameters"] = 0
param = choice_param(option)

t = np.arange(0, 2) #Years of the scenario

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

#Ca va bugger parce qu'on doit importer les données transport au niveau SP

# %% Import data for the calibration

#Data coordinates (SP)
xData = copy.deepcopy(households_data.X_SP_2011)
yData = copy.deepcopy(households_data.Y_SP_2011)

#Data at the SP level
dataPrice = copy.deepcopy(households_data.sale_price_SP[2, :])
dataDwellingSize = copy.deepcopy(households_data.spDwellingSize)

#Income classes
dataIncomeGroup = np.zeros(len(households_data.income_SP_2011))
for j in range(0, 3):
    dataIncomeGroup[households_data.income_SP_2011 > households_data.income_groups_limits[j]] = j+1

#Import amenities at the SP level
amenities_sp = ImportAmenitiesSP()
variablesRegression = ['distance_ocean', 'distance_ocean_2_4', 'slope_1_5', 'slope_5', 'airport_cone2', 'distance_distr_parks', 'distance_biosphere_reserve', 'distance_train', 'distance_urban_herit']

# %% Estimation of coefficient of construction function

dataNumberFormal = (households_data.total_dwellings_SP_2011 - households_data.backyard_SP_2011 - households_data.informal_SP_2011)
dataDensity = dataNumberFormal / (households_data.spUnconstrainedArea * param["max_land_use"] / 1000000)

#other possibility
selectedDensity = (households_data.spUnconstrainedArea > 0.6 * 1000000 * households_data.area_SP_2011) & (dataIncomeGroup > 0) & (households_data.Mitchells_Plain_SP_2011 == 0) & (households_data.distance_SP_2011 < 40) & (dataPrice > np.nanquantile(dataPrice, 0.2)) & (households_data.spUnconstrainedArea < np.nanquantile(households_data.spUnconstrainedArea, 0.8))
X = np.transpose(np.array([np.log(dataPrice[selectedDensity]), np.log(param["max_land_use"] * households_data.spUnconstrainedArea[selectedDensity]), np.log(dataDwellingSize[selectedDensity])]))
y = np.log(dataNumberFormal[selectedDensity])
modelConstruction = LinearRegression().fit(X, y)
modelConstruction.score(X, y)
modelConstruction.coef_
modelConstruction.intercept_

#Export outputs
coeff_b = modelConstruction.coef_[0]
coeff_a = 1 - coeff_b
coeffKappa = (1 /coeff_b ** coeff_b) * np.exp(modelConstruction.intercept_)

#Correcting data for rents
dataRent = dataPrice ** (coeff_a) * (param["depreciation_rate"] + InterpolateInterestRateEvolution(macro_data, t[0])) / (coeffKappa * coeff_b ** coeff_b)
interestRate = (param["depreciation_rate"] + InterpolateInterestRateEvolution(macro_data, t[0]))

#Cobb-Douglas: 
#simulHousing_CD = coeffKappa.^(1/coeff_a)...
#        .*(coeff_b/interestRate).^(coeff_b/coeff_a)...
#        .*(dataRent).^(coeff_b/coeff_a);

#f1=fit(data.sp2011Distance(selectedDensity), data.spFormalDensityHFA(selectedDensity)','poly5');
#f2=fit(data.sp2011Distance(~isnan(simulHousing_CD)), simulHousing_CD(~isnan(simulHousing_CD))','poly5');

# %% Estimation of incomes and commuting parameters

listLambda = [4.27, 0]
#listLambda = 10 ** np.arange(0.6, 0.65, 0.01)
#listLambda = 10 ** np.arange(0.6, 0.61, 0.005)

timeOutput, distanceOutput, monetaryCost, costTime = import_transport_costs(option, grid, macro_data, param, job, households_data, [0, 1], 1)

modalShares, incomeCenters, timeDistribution, distanceDistribution = EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job, households_data, listLambda)

dataModalShares = np.array([7.8, 14.8, 39.5+0.7, 16, 8]) / (7.8+14.8+39.5+0.7+16+8) * 100
dataTimeDistribution = np.array([18.3, 32.7, 35.0, 10.5, 3.4]) / (18.3+32.7+35.0+10.5+3.4)
dataDistanceDistribution = np.array([45.6174222, 18.9010734, 14.9972971, 9.6725616, 5.9425438, 2.5368754, 0.9267125, 0.3591011, 1.0464129])

#Compute accessibility index
bhattacharyyaDistances = -np.log(np.nansum(np.sqrt(dataDistanceDistribution[:, None] /100 * distanceDistribution), 0))
whichLambda = np.argmin(bhattacharyyaDistances)

lambdaKeep = listLambda[whichLambda]
modalSharesKeep = modalShares[:, whichLambda]
timeDistributionKeep = timeDistribution[:, whichLambda]
distanceDistributionKeep = distanceDistribution[:, whichLambda]
incomeCentersKeep = incomeCenters[:,:,whichLambda]

incomeNetOfCommuting = ComputeIncomeNetOfCommuting(param, grid, job, households_data, lambdaKeep, incomeCentersKeep, timeOutput, distanceOutput, monetaryCost, costTime)
#On peut faire un graphe de contrôle ici.

# %% Estimation of housing demand parameters

#In which areas we actually measure the likelihood
#I remove the areas where there is informal housing, because dwelling size data is not reliable
selectedSPForEstimation = ((households_data.backyard_SP_2011 + households_data.informal_SP_2011) / households_data.total_dwellings_SP_2011 < 0.1) & (dataIncomeGroup > 0) 

#Coefficients of the model
listBeta = np.arange(0.1, 0.6, 0.2)
listBasicQ = np.arange(5, 16, 5)

#Utilities
utilityTarget = np.array([300, 1000, 3000, 10000])
listVariation = np.arange(0.5, 2.1, 0.3)
initUti2 = utilityTarget[1] 
listUti3 = utilityTarget[2] * listVariation
listUti4 = utilityTarget[3] * listVariation

parametersScan, scoreScan, parametersAmenitiesScan, modelAmenityScan, parametersHousing, X = EstimateParametersByScanning(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, dataDensity, selectedDensity, xData, yData, selectedSPForEstimation, tableAmenities, variablesRegression, 0, listBeta, listBasicQ, initUti2, listUti3, listUti4)

#Now run the optimization algo with identified value of the parameters
initBeta = parametersScan[0] 
initBasicQ = max(parametersScan[1], 5.1)

#Utilities
initUti3 = parametersScan[2]
initUti4 = parametersScan[3]

parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing, selectedSPRent = EstimateParametersByOptimization(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, dataDensity, selectedDensity, xData, yData, selectedSPForEstimation, tableAmenities, variablesRegression, listRho, initBeta, initBasicQ, initUti2, initUti3, initUti4)

#Generating the map of amenities
amenities_grid = ImportAmenitiesGrid()
amenities = np.exp(parametersAmenities[2:end] * table2array(tableAmenitiesGrid[:, variablesRegression]))

#Export
utilitiesCorrected = np.array([parameters[2], parameters[3]]) / np.exp(parametersAmenities[0])
calibratedUtility_beta = parameters[0]
calibratedUtility_q0 = parameters[1]







def ComputeIncomeNetOfCommuting(param, grid, job, households_data, param_lambda, incomeCenters, timeOutput, distanceOutput, monetaryCost, costTime):

    annualToHourly = 1 / (8 * 20 * 12)

    timeCost = copy.deepcopy(costTime)
    timeCost[np.isnan(timeCost)] = 10 ** 2
    monetaryCost = copy.deepcopy(monetaryCost) * annualToHourly
    monetaryCost[np.isnan(monetaryCost)] = 10 ** 3 * annualToHourly
    incomeCenters = incomeCenters * annualToHourly

    xInterp = households_data.X_SP_2011
    yInterp = households_data.Y_SP_2011
    
    incomeNetOfCommuting = np.zeros((4, timeCost.shape[1]))
    
    for j in range(0, 4):
        
        #Household size varies with transport costs
        householdSize = param["household_size"][j]
        whichCenters = incomeCenters[:,j] > -100000
        incomeCentersGroup = incomeCenters[whichCenters, j]
           
        #Transport costs and employment allocation
        transportCostModes = householdSize * monetaryCost[whichCenters,:,:] + timeCost[whichCenters,:,:] * incomeCentersGroup[:, None, None]
        
        #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
        valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500
        
        #Transport costs
        transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

        #minIncome is also to prevent diverging exponentials
        minIncome = np.nanmax(param_lambda * (incomeCentersGroup[:, None] - transportCost)) - 700

        #Income net of commuting (correct formula)
        incomeNetOfCommuting[j,:] = 1/param_lambda * (np.log(np.nansum(np.exp(param_lambda * (incomeCentersGroup[:, None] - transportCost) - minIncome), 0)) + minIncome)
             
    return incomeNetOfCommuting / annualToHourly



