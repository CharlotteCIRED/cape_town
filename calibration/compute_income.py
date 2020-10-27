# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:22:55 2020

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
import math
from sklearn.linear_model import LinearRegression

from calibration.compute_income import *

def EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job, households_data, listLambda):
    #Solve for income per employment centers for different values of lambda

    print('Estimation of local incomes, and lambda parameter')

    annualToHourly = 1 / (8*20*12)
    bracketsTime = np.array([0, 15, 30, 60, 90, np.nanmax(np.nanmax(np.nanmax(timeOutput)))])
    bracketsDistance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 200])

    timeCost = copy.deepcopy(costTime)
    timeCost[np.isnan(timeCost)] = 10 ** 2
    monetary_cost = monetaryCost * annualToHourly
    monetary_cost[np.isnan(monetary_cost)] = 10 ** 3 * annualToHourly
    transportTimes = timeOutput / 2
    transportDistances = distanceOutput[:, :, 0]

    modalSharesTot = np.zeros((5, len(listLambda)))
    incomeCentersSave = np.zeros((len(job.jobsCenters[:,0,0]), 4, len(listLambda)))
    timeDistribution = np.zeros((len(bracketsTime) - 1, len(listLambda)))
    distanceDistribution = np.zeros((len(bracketsDistance) - 1, len(listLambda)))

    for i in range(0, len(listLambda)):

        param_lambda = listLambda[i]
        
        print('Estimating for lambda = ', param_lambda)
        
        incomeCentersAll = -math.inf * np.ones((len(job.jobsCenters[:,0,0]), 4))
        modalSharesGroup = np.zeros((5, 4))
        timeDistributionGroup = np.zeros((len(bracketsTime) - 1, 4))
        distanceDistributionGroup = np.zeros((len(bracketsDistance) - 1, 4))

        for j in range(0, 4):
        
            #Household size varies with transport costs
            householdSize = param["household_size"][j]
            
            averageIncomeGroup = job.averageIncomeGroup[0, j] * annualToHourly
        
            print('incomes for group ', j)
        
            whichJobsCenters = job.jobsCenters[:, j, 0] > 600
            popCenters = job.jobsCenters[whichJobsCenters, j, 0]
            array1 = copy.deepcopy(households_data.total_number_per_income_class[1, j]) * sum(job.jobsCenters[whichJobsCenters, j, 0])
            popResidence = array1 / np.nansum(households_data.total_number_per_income_class[1, j])
                
            funSolve = lambda incomeCentersTemp: fun0(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)

            maxIter = 700
            tolerance = 0.001
            if j == 0:
                factorConvergenge = 0.008
            elif j == 1:
                factorConvergenge = 0.005
            else:
                factorConvergenge = 0.0005
        
            iter = 0
            error = np.zeros((len(popCenters), maxIter))
            scoreIter = np.zeros(maxIter)
            errorMax = 1
        
            #Initializing the solver
            incomeCenters = np.zeros((sum(whichJobsCenters), maxIter))
            incomeCenters[:, 0] =  averageIncomeGroup * (popCenters / np.nanmean(popCenters)) ** (0.1)
            error[:, 0] = funSolve(incomeCenters[:, 0])

        
            while ((iter <= maxIter) & (errorMax > tolerance)):
            
                iter = iter + 1
                incomeCenters[:,iter] = incomeCenters[:, max(iter-1, 0)] + factorConvergenge * averageIncomeGroup * error[:, max(iter - 1,0)] / popCenters
                error[:,iter] = funSolve(incomeCenters[:,iter])
                errorMax = np.nanmax(np.abs(error[:, iter] / popCenters))
                scoreIter[iter] = np.nanmean(np.abs(error[:, iter] / popCenters))
                print(np.nanmean(np.abs(error[:, iter])))
            
            if (iter > maxIter):
                scoreBest = np.amin(scoreIter)
                bestSolution = np.argmin(scoreIter)
                incomeCenters[:, iter] = incomeCenters[:, bestSolution]
                print(' - max iteration reached - mean error', scoreBest)
            else:
                print(' - computed - max error', errorMax)
        
        
            incomeCentersRescaled = incomeCenters[:, iter] * averageIncomeGroup / ((np.nansum(incomeCenters[:, iter] * popCenters) / np.nansum(popCenters)))
            modalSharesGroup[:,j] = modalShares(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)
            incomeCentersAll[whichJobsCenters,j] = incomeCentersRescaled
        
            timeDistributionGroup[:,j] = computeDistributionCommutingTimes(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportTimes[whichJobsCenters,:], bracketsTime, param_lambda)
            distanceDistributionGroup[:,j], X = computeDistributionCommutingDistances(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda)

        modalSharesTot[:,i] = np.nansum(modalSharesGroup, 1) / np.nansum(np.nansum(modalSharesGroup))
        incomeCentersSave[:,:,i] = incomeCentersAll / annualToHourly
        timeDistribution[:,i] = np.nansum(timeDistributionGroup, 1) / np.nansum(np.nansum(timeDistributionGroup))
        distanceDistribution[:,i] = np.nansum(distanceDistributionGroup, 1) / np.nansum(np.nansum(distanceDistributionGroup))

    return modalSharesTot, incomeCentersSave, timeDistribution, distanceDistribution

def fun0(incomeCenters, meanIncome, popCenters, popResidence, monetaryCost, timeCost, param_lambda):
    """ Computes error in employment allocation """

    #Control the average income of each group
    incomeCentersFull = incomeCenters * meanIncome / ((np.nansum(incomeCenters * popCenters) / np.nansum(popCenters)))

    #Transport costs and employment allocation
    transportCostModes = monetaryCost + timeCost * incomeCentersFull[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Transport costs
    transportCost = - 1 / param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCentersFull[:, None] - transportCost))) - 500

    #Differences in the number of jobs
    score = popCenters - np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome)) * popResidence, 1)

    return score


def modalShares(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, param_lambda):
    """ Computes total modal shares """

    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda  * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1 / param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500

    #Total modal shares
    modalSharesTot = np.nansum(np.nansum(modalSharesTemp * (np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)))[:, :, None] * popResidence, 1), 0)
    #modalSharesTot = np.tranpose(modalSharesTot, (2,0,1))

    return modalSharesTot


def computeDistributionCommutingTimes(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, transportTime, bracketsTime, param_lambda):

    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 600

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 600

    #Total distribution of times
    nbCommuters = np.zeros(len(bracketsTime) - 1)
    for k in range(0, len(bracketsTime)-1):
        which = (transportTime > bracketsTime[k]) & (transportTime <= bracketsTime[k + 1]) & (~np.isnan(transportTime))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which * modalSharesTemp * np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)[:, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)) * popResidence, 1)))

    return nbCommuters


def computeDistributionCommutingDistances(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, transportDistance, bracketsDistance, param_lambda):

    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1/param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500

    #Total distribution of times
    nbCommuters = np.zeros(len(bracketsDistance) - 1)
    for k in range(0, len(bracketsDistance)-1):
        which = (transportDistance > bracketsDistance[k]) & (transportDistance <= bracketsDistance[k + 1]) & (~np.isnan(transportDistance))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which[:, :, None] * modalSharesTemp * np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)[:, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)) * popResidence, 1)))

    return nbCommuters, np.nansum(np.nansum(np.nansum(modalSharesTemp * (np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)))[:, :, None] * popResidence, 1)))

