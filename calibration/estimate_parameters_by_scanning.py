# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:50:37 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import math

from calibration.estimate_parameters_by_scanning import *
from calibration.loglikelihood import *

def EstimateParametersByScanning(incomeNetOfCommuting, dataRent, dataDwellingSize, dataIncomeGroup, dataHouseholdDensity, selectedDensity, xData, yData, selectedSP, tableAmenities, variablesRegression, initRho, listBeta, listBasicQ, initUti2, listUti3, listUti4):
    """ Automated estimation of the parameters of NEDUM by maximizing log likelihood 
    
    Here we scan a set of values for each parameters and determine the value
    of the log-likelihood (to see how the model behaves). 
    
    In EstimateParameters By Optimization, we use the minimization algorithm from Matlab to converge towards solution """ 

    #Data as matrices, where should we regress (remove where we have no data)
    #Where is which class
    incomeNetOfCommuting = incomeNetOfCommuting[1:4, :] #We remove income group 1
    groupLivingSpMatrix = (incomeNetOfCommuting > 0)
    for i in range(0, 3):
        groupLivingSpMatrix[i, dataIncomeGroup != i + 1] = False

    selectedTransportMatrix = (sum(groupLivingSpMatrix) == 1)
    incomeNetOfCommuting[incomeNetOfCommuting < 0] = np.nan

    selectedRents = ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
    selectedDwellingSize = ~np.isnan(dataDwellingSize) & ~np.isnan(dataRent) & selectedTransportMatrix & selectedSP
    selectedDensity = selectedDwellingSize & selectedDensity

    #For the regression of amenities
    tableRegression = tableAmenities[selectedRents, :]
    predictorsAmenitiesMatrix = table2array(tableRegression[:, variablesRegression])
    predictorsAmenitiesMatrix = [np.ones(size(predictorsAmenitiesMatrix,1),1), predictorsAmenitiesMatrix]
    modelAmenity = 0

    # %% Useful functions (precalculations for rents and dwelling sizes, likelihood function) 

    #Function for dwelling sizes
    #We estimate calcule_hous directly from data from rents (no extrapolation)
    CalculateDwellingSize = lambda beta, basic_q, incomeTemp, rentTemp : beta * incomeTemp / rentTemp + (1 - beta) * basic_q

    #Log likelihood for a lognormal law
    ComputeLogLikelihood = lambda sigma, error : np.nansum(- np.log(2 * math.pi * sigma ** 2) / 2 -  1 / (2 * sigma ** 2) * (error) ** 2)
    

    # %% Optimization algorithm

    #Function that will be minimized 
    optionRegression = 0

    #Initial value of parameters
    combinationInputs = combvec(listBeta, listBasicQ, listUti3, listUti4) #So far, no spatial autocorrelation

    #Scanning of the list
    scoreAmenities = - 10000 * np.ones(combinationInputs.shape[1])
    scoreDwellingSize = - 10000 * np.ones(combinationInputs.shape[1])
    scoreIncomeSorting = - 10000 * np.ones(combinationInputs.shape[1])
    scoreHousing = - 10000 * np.ones(combinationInputs.shape[1])
    iterPrint = np.floor(np.ones(combinationInputs.shape[1]) / 20)
    print('\nDone: ')
    for index in range(0, combinationInputs.shape[1]):
        X, scoreAmenities[index], scoreDwellingSize[index], scoreIncomeSorting[index], scoreHousing[index] = LogLikelihoodModel(combinationInputs[:,index], initUti2, incomeNetOfCommuting, groupLivingSpMatrix, dataDwellingSize, selectedDwellingSize, xData, yData, dataRent, selectedRents, dataHouseholdDensity, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variablesRegression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression)
        if floor(index / iterPrint) == index/iterPrint
            fprintf('%0.f%%  ', round(index / size(combinationInputs,2) .* 100));

    print('\nScanning complete')
    print('\n')

    scoreVect = scoreAmenities + scoreDwellingSize + scoreIncomeSorting + scoreHousing
    scoreTot = np.amax(scoreVect)
    which = np.argmax(scoreVect)
    parameters = combinationInputs[:, which]

    #Estimate the function to get the parameters for amenities
    optionRegression = 1
    [~, ~, ~, ~, ~, parametersAmenities, modelAmenity, parametersHousing] = LogLikelihoodModel(parameters, initUti2, incomeNetOfCommuting, groupLivingSpMatrix, dataDwellingSize, selectedDwellingSize, xData, yData, dataRent, selectedRents, dataHouseholdDensity, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variablesRegression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression);
    
    return parameters, scoreTot, parametersAmenities, modelAmenity, parametersHousing, selectedRents


def confidence_interval(indices_max, quoi_indices, compute_score):

    d_beta = 1.05

    print('\n')
    beta_interval = np.zeros(size(indices_max))
    for index in range (0, size(indices_max,1)):
        indices_ici = indices_max
        score_tmp = compute_score(indices_ici)
    
        indices_ici = indices_max
        score_tmp2 = compute_score(indices_ici)
    
        indices_ici = indices_max
        indices_ici[index] = indices_ici[index] - indices_ici[index] * (d_beta - 1)
        score_tmp3 = compute_score(indices_ici)
    
        indices_ici = indices_max
        dd_l_beta = -(score_tmp2 + score_tmp3 - 2 * score_tmp) / (indices_ici[index] * (d_beta - 1)) ** 2
        beta_interval[index] = 1.96 / (np.sqrt(np.abs(dd_l_beta)))
        fprintf('%s\t\t%g (%g ; %g)\n', quoi_indices{index}, indices_ici[index], indices_ici[index] - beta_interval[index], indices_ici[index] + beta_interval[index])

    return np.array([indices_ici - beta_interval, indices_ici + beta_interval])


def utilityFromRents(Ro, income, basic_q, beta):
    utility = (1 - beta) ** (1 - beta) * beta ** beta * np.sign(income - basic_q * Ro) * np.abs(income - basic_q * Ro) / (Ro ** beta)
    utility[(income - basic_q * Ro) < 0] = 0
    utility[income == 0] = 0
    return utility


def InterpolateRents(beta, basic_q, netIncome):
    """ Precalculations for rents, as a function
    
    The output of the function is a griddedInterpolant object, that gives the
    log of rents as a function of the log utility and the log income
    """

    #Decomposition for the interpolation (the more points, the slower the code)
    decompositionRent = [10.^([-9,-4,-3,-2]), 0.02:0.02:0.08, 0.1:0.05:1.4, 1.5:0.1:2.5, 100, 10^9]; 
    decompositionIncome = [10.^([-9,-4,-3,-2]), 0.02:0.02:0.08, 0.1:0.05:1.4, 1.5:0.1:2.5, 100, 10^9];

    #Min and Max values for the decomposition
    choiceIncome = 100000 .* decompositionIncome ;
    incomeMatrix = repmat(choiceIncome,length(choiceIncome),1)';
    if basic_q > 0.5:
        choiceRent = choiceIncome./basic_q; % the maximum rent is the rent for which u = 0
    else:
        choiceRent = 1000 .* decompositionRent;
        
    rentMatrix = choiceRent' * decompositionRent;

    utilityMatrix = utilityFromRents(incomeMatrix, rentMatrix, basic_q, beta);
    solusRentTemp = @(x,y) griddata(incomeMatrix,utilityMatrix,rentMatrix,x,y);
        
    #Redefine a grid (to use griddedInterpolant)
    utilityVectLog = -1:0.1:log(max(max(10.*netIncome)));
    incomeLog = (-1:0.2:log(max(max(10.*netIncome))))';
    rentLog = log(solusRentTemp(exp(incomeLog), exp(utilityVectLog)));
    griddedRents = griddedInterpolant({utilityVectLog, incomeLog}, rentLog, 'linear', 'none');
        
    return griddedRents
