# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 11:46:13 2020

@author: Charlotte Liotta
"""

import numpy as np

def LogLikelihoodModel(X0, Uo2, incomeNetOfCommuting, groupLivingMatrix, dataDwellingSize, selectedDwellingSize, xData, yData, dataRent, selectedRents, dataHouseholdDensity, selectedDensity, predictorsAmenitiesMatrix, tableRegression, variablesRegression, CalculateDwellingSize, ComputeLogLikelihood, optionRegression):
    """ Function to estimate the total likelihood of the model given the parameters """

    beta = X0[0]
    basicQ = X0[1]
    Uo = np.array([Uo2, X0[2], X0[3]])

    # %% Errors on the amenity

    #Calculate amenities as a residual
    residualAmenities = np.log(Uo) - np.log((1 - beta) ** (1 - beta) * beta ** beta * (incomeNetOfCommuting[:, selectedRents] - basicQ * dataRent[selectedRents]) / (dataRent[selectedRents] ** beta))
    residualAmenities = np.nansum(residualAmenities * groupLivingMatrix[:, selectedRents])
    residualAmenities[np.abs(residualAmenities.imag) > 0] = np.nan
    residualAmenities[residualAmenities == 0] = np.nan

    #residual for the regression of amenities follow a log-normal law
    if (optionRegression == 0):
        #Here regression as a matrix division (much faster)
        #parametersAmenities = predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :] \ (residualAmenities[~np.isnan(residualAmenities)]).real
        parametersAmenities = np.linalg.lstsq(predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :], (residualAmenities[~np.isnan(residualAmenities)]).real)
        errorAmenities = (residualAmenities[~np.isnan(residualAmenities)]).real - (predictorsAmenitiesMatrix[~np.isnan(residualAmenities), :] * parametersAmenities) 
    
    elif (optionRegression == 1):     
        #Compute regression with fitglm (longer)
        #Can only work if length(lists) = 1
        tableRegression.residu = residualAmenities.real
                                      parametersAmenities = predictorsAmenitiesMatrix(~isnan(residualAmenities'),:) \ real(residualAmenities(~isnan(residualAmenities))');
                                                                                      modelSpecification = ['residu ~ ', sprintf('%s + ', variablesRegression{1:end-1}), variablesRegression{end}];
                                                                                      modelAmenities = fitglm(tableRegression, modelSpecification);
        errorAmenities = modelAmenities.Residuals.Raw;
            
    # %% Error on allocation of income groups

    #log-likelihood of a logit model on the location of income groups
    griddedRents = InterpolateRents(beta, basicQ, incomeNetOfCommuting);
    bidRents = exp(griddedRents(log(Uo) - residualAmenities, log(incomeNetOfCommuting(:,selectedRents))));

    #Estimation of the scale parameter by maximization of the log-likelihood
    selectedBidRents = nansum(bidRents) > 0;
    incomeGroupSelectedRents = groupLivingMatrix(:,selectedRents);
    likelihoodIncomeSorting = @(scaleParam) - (nansum(nansum(bidRents(:,selectedBidRents)./scaleParam .* incomeGroupSelectedRents(:,selectedBidRents))) - nansum(log(nansum(exp(bidRents(:,selectedBidRents)./scaleParam),1))));
    scoreIncomeSorting = - likelihoodIncomeSorting(10000);
    
    # %% Errors on the dwelling sizes
    #simulated rent, real sorting
    simulatedRents = sum(bidRents(:,selectedDwellingSize(selectedRents)) .* groupLivingMatrix(:, selectedDwellingSize));
    dwellingSize = CalculateDwellingSize(beta, basicQ, sum(incomeNetOfCommuting(:, selectedDwellingSize) .* groupLivingMatrix(:, selectedDwellingSize)), simulatedRents);
    #Define errors
    errorDwellingSize = log(dwellingSize) - log(dataDwellingSize(selectedDwellingSize));
      
    # %% Errors on household density
    scoreHousing = 0;
    parametersHousing = 0;

    #Scores
    scoreDwellingSize = ComputeLogLikelihood(sqrt(nansum(errorDwellingSize.^2) ./ sum(~isnan(errorDwellingSize))), errorDwellingSize);
    scoreAmenities = ComputeLogLikelihood(sqrt(nansum(errorAmenities.^2) ./ sum(~isnan(errorAmenities))), errorAmenities);

    scoreTotal = scoreAmenities + scoreDwellingSize + scoreIncomeSorting

    return scoreTotal, scoreAmenities, scoreDwellingSize, scoreIncomeSorting, scoreHousing, parametersAmenities, modelAmenities, parametersHousing


def utilityFromRents(Ro, income, basic_q, beta):
    utility = (1 - beta) ** (1 - beta) * beta ** beta * (income - basic_q * Ro) / (Ro ** beta)
    utility[(income - basic_q * Ro) < 0] = 0
    utility[income == 0] = 0
    return utility

def InterpolateRents(beta, basic_q, netIncome)
    """ Precalculations for rents, as a function
    
    The output of the function is a griddedInterpolant object, that gives the
    log of rents as a function of the log utility and the log income
    """
    
    #Decomposition for the interpolation (the more points, the slower the code)
    decompositionIncome = [10.^([-9,-4:0.5:-2]), 0.03, 0.06:0.02:1.4,1.5:0.1:2.5, 4:2:10, 20, 10^9];
    decompositionRent = [10.^([-9,-4,-3,-2]), 0.02:0.01:0.79, 0.8:0.02:0.96, 0.98]; 
        
    #Min and Max values for the decomposition
    choiceIncome = 100000 .* decompositionIncome ;
    incomeMatrix = repmat(choiceIncome,length(choiceIncome),1)';
    choiceRent = choiceIncome./basic_q; % the maximum rent is the rent for which u = 0
    rentMatrix = choiceRent' * decompositionRent;

    utilityMatrix = utilityFromRents(rentMatrix, incomeMatrix, basic_q, beta);
    solusRentTemp = @(x,y) griddata(incomeMatrix, utilityMatrix, rentMatrix.^beta,x,y);
        
    #Redefine a grid (to use griddedInterpolant)
    utilityVectLog = -1:0.1:log(max(max(10.*netIncome)));
    incomeLog = (-1:0.2:log(max(max(10.*netIncome))))';
    rentLog = 1/beta.*log(solusRentTemp(exp(incomeLog), exp(utilityVectLog)));
    griddedRents = griddedInterpolant({utilityVectLog, incomeLog}, rent
                                      
    return griddedRents