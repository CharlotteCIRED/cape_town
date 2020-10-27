# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:32:22 2020

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
from solver.compute_outputs_solver_v2 import *

def ComputeNEDUMOutput_LOGIT_v2(Uo, param, option, transTemp_incomeNetOfCommuting, grid, agriculturalRent, housingLimit, referenceRent, constructionParam, interestRate, income, multiProbaGroup, transportCost, transportCostRDP, coeffLand, job, amenities, solus_Q, typeHousing, param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood, selectedPixels, r_init):
    """ We suppose that households account for floods, and that the households
    living in formal housing incur the full cost of floods """
    
    if typeHousing == 'backyard':
        dwellingSize = param["shack_size"] * np.ones((len(job.incomeMult), len(grid.dist)))
        dwellingSize[job.backyard == 0, :] = np.nan
        amenities = copy.deepcopy(amenities) * param["amenity_backyard"]
        if option["floods"] == 0:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - (np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"]))
        elif option["floods"] == 1:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - ((np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"])) - (flood.d_contents[selectedPixels][None, :] * flood.content_cost) - (flood.d_structure[selectedPixels][None, :] * flood.informal_structure_value) - (interestRate * flood.informal_structure_value))
        R_mat[job.backyard == 0, :] = np.nan
    elif typeHousing == 'informal':
        dwellingSize = param["shack_size"] * np.ones((len(job.incomeMult), len(grid.dist)))
        dwellingSize[job.settlement == 0, :] = np.nan
        amenities = copy.deepcopy(amenities) * param["amenity_settlement"]
        R_mat = 1 / param["shack_size"] * (transTemp_incomeNetOfCommuting - (np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * (param["shack_size"] - param["q0"]) ** param["beta"])) ** (1 / param["alpha"]))
        R_mat[job.settlement == 0, :] = np.nan
    elif typeHousing == 'formal':       
        if option["floods"] == 1:
            income_temp = transTemp_incomeNetOfCommuting
            income_temp[income_temp < 0] = np.nan
            R_mat, diffUtility = SolveHouseholdsProgram(income_temp, param, flood, selectedPixels, constructionParam, interestRate, Uo, amenities, r_init)            
            R_mat[transTemp_incomeNetOfCommuting < 0] = 0
            R_mat[job.formal == 0, :] = np.nan
            #dwellingSize = (1 / R_mat) * param["beta"] * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost) - (flood.d_structure[selectedPixels][None, :] * ((param["coeff_b"] * param["coeff_A"] / interestRate) ** (1/param["coeff_a"])) * (R_mat ** (1/param["coeff_a"]))))
            dwellingSize = (1 / R_mat) * param["beta"] * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost))
            dwellingSize = np.maximum(dwellingSize, param["miniLotSize"])
            dwellingSize[job.formal == 0, :] = np.nan
        elif option["floods"] == 0:
            income_temp = transTemp_incomeNetOfCommuting
            income_temp[income_temp < 0] = np.nan
            dwellingSize = (Uo[:, None] / (((param["alpha"] * (income_temp)) ** param["alpha"])) * amenities[None, :]) ** (1/param["beta"])
            dwellingSize = np.maximum(dwellingSize, param["miniLotSize"])
            dwellingSize[job.formal == 0, :] = np.nan
            R_mat = param["beta"] * (transTemp_incomeNetOfCommuting) / (dwellingSize - param["alpha"] *param["q0"])
            R_mat[transTemp_incomeNetOfCommuting < 0] = 0
            R_mat[job.formal == 0, :] = np.nan
            
    R_mat[R_mat < 0] = 0
    R_mat[np.isnan(R_mat)] = 0

    #Income group in each location
    proba = (R_mat == np.nanmax(R_mat, axis = 0))
    limit = (transTemp_incomeNetOfCommuting > 0) & (proba > 0) & (~np.isnan(transTemp_incomeNetOfCommuting)) & (R_mat > 0)
    proba = proba * limit

    whichGroup = np.nanargmax(R_mat, 0)
    R = np.empty(len(whichGroup))
    dwellingSizeTemp = np.empty(len(whichGroup))
    for i in range(0, len(whichGroup)):
        R[i] = R_mat[int(whichGroup[i]), i]
        dwellingSizeTemp[i] = dwellingSize[int(whichGroup[i]), i]
    
    dwellingSize = dwellingSizeTemp

    #Housing Construction 

    if typeHousing == 'formal':
        if option["adjustHousingSupply"] == 1:
            #if option["floods"] == 0:
            housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate)) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
            #elif option["floods"] == 1:
                #housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate + flood.d_structure[selectedPixels][None, :])) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
            
            #Outside the agricultural rent, no housing (accounting for a tax)
            housingSupply = housingSupply.squeeze()
            housingSupply[R < agriculturalRent + param_taxUrbanEdgeMat] = 0
    
            housingSupply[np.isnan(housingSupply)] = 0
            #housingSupply(imag(housingSupply) ~= 0) = 0
            housingSupply[housingSupply < 0] = 0
            housingSupply = np.fmin(housingSupply, housingLimit)
    
            #To add the construction on Mitchells_Plain
            housingSupply = np.fmax(housingSupply, param_minimumHousingSupply * 1000000)
    
        else:   
            housingSupply = param_housing_in
    elif typeHousing == 'backyard':
        if option["floods"] == 0:
            housingSupply = param["alpha"] * (((param["RDP_size"] + param["backyard_size"]) / (param["backyard_size"]))) - (param["beta"] * ((income[:, 0] - transportCostRDP) /((param["backyard_size"]) * R)))
            housingSupply[income[:, 0] < transportCostRDP] = param["alpha"] * (param["RDP_size"] + param["backyard_size"]) / (param["backyard_size"]) - param["beta"] * (income[income[:, 0] < transportCostRDP, 0]) / ((param["backyard_size"]) * R[income[:, 0] < transportCostRDP])
        elif option["floods"] == 1:
            housingSupply = param["alpha"] * (((param["RDP_size"] + param["backyard_size"]) / (param["backyard_size"]))) - (param["beta"] * ((income[:, 0] - transportCostRDP - (flood.d_contents[selectedPixels] * flood.content_cost)) /((param["backyard_size"]) * R)))
            housingSupply[income[:, 0] < transportCostRDP] = (param["alpha"] * ((param["RDP_size"] + param["backyard_size"]) / (param["backyard_size"]))) - (param["beta"] * (income[income[:, 0] < transportCostRDP, 0] - (flood.d_contents[selectedPixels][income[:, 0] < transportCostRDP] * flood.content_cost)) / ((param["backyard_size"]) * R[income[:, 0] < transportCostRDP]))
        
        housingSupply[R == 0] = 0
        housingSupply = np.fmin(housingSupply, 1)
        housingSupply = np.fmax(housingSupply, 0)
        housingSupply = 1000000 * housingSupply

    elif typeHousing == 'informal':
        if param["double_storey_shacks"] == 0:
            housingSupply = 1000000 * np.ones(len(whichGroup))
            housingSupply[R == 0] = 0
        elif param["double_storey_shacks"] == 1:
            netIncome = sum(proba[job.classes == 0, :] * (income[job.classes == 0, :] - transportCost[job.classes == 0, :])) / sum(proba[job.classes == 0, :])
            housingSupply = 1 + param["alpha"] / param["coeff_mu"] - netIncome / R
            housingSupply = max(housingSupply, 1)
            housingSupply = min(housingSupply, 2)
            housingSupply[R == 0] = 0
            housingSupply = 1000000 * housingSupply
        

    peopleInit = housingSupply / dwellingSize * (np.sum(limit, 0) > 0)
    peopleInit[np.isnan(peopleInit)] = 0
    peopleInitLand = peopleInit * coeffLand * 0.5 ** 2

    peopleCenter = np.matlib.repmat(peopleInitLand, R_mat.shape[0], 1) * proba
    peopleCenter[np.isnan(peopleCenter)] = 0
    jobSimul = np.sum(peopleCenter, 1)
    
    if typeHousing == 'formal':
        R = np.fmax(R, agriculturalRent)

    return jobSimul, R, peopleInit, peopleCenter, housingSupply, dwellingSize, R_mat

def SolveHouseholdsProgram(income_temp, param, flood, selectedPixels, constructionParam, interestRate, Uo, amenities, r_init):
    maxIteration = 1999
    indexIteration = 0
    R_matTemp = np.ones((income_temp.shape[0], income_temp.shape[1], 1500))
    R_matTemp[:, :, 0] = r_init
    utilityTemp = np.ones((income_temp.shape[0], income_temp.shape[1], 1500))
    diffUtility = 400
    while (indexIteration < maxIteration - 1) & (diffUtility > 0.01):
        if indexIteration > 0:
            R_matTemp[:, :, indexIteration] = copy.deepcopy(R_matTemp[:, :, indexIteration - 1]) + 0.02*(utilityTemp[:, :, indexIteration - 1] - Uo[:, None])
            print(np.nanmean(R_matTemp[:, :, indexIteration]))
            print(sum(np.isnan(R_matTemp[:, :, indexIteration])))
        #utilityTemp[:, :, indexIteration] = (param["alpha"] ** param["alpha"]) * (param["beta"] ** param["beta"]) * (R_matTemp[:, :, indexIteration] ** (-param["beta"])) * amenities * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost) - (flood.d_structure[selectedPixels][None, :] * ((param["coeff_b"] * param["coeff_A"] / interestRate) ** (1/param["coeff_a"])) * (R_matTemp[:, :, indexIteration] ** (1/param["coeff_a"]))))
        utilityTemp[:, :, indexIteration] = (param["alpha"] ** param["alpha"]) * (param["beta"] ** param["beta"]) * (R_matTemp[:, :, indexIteration] ** (-param["beta"])) * amenities * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost))
        diffUtility = np.nansum(np.abs(utilityTemp[:, :, indexIteration] - Uo[:, None]))
        indexIteration = indexIteration + 1
        print(diffUtility)
        print(sum(np.isnan(utilityTemp[:, :, indexIteration - 1])))

    return R_matTemp[:, :, indexIteration - 2], diffUtility     