# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:08:04 2020

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
from solver.compute_outputs_solver import *

def ComputeNEDUMOutput_LOGIT(Uo, param, option, transTemp_incomeNetOfCommuting, grid, agriculturalRent, housingLimit, referenceRent, constructionParam, interestRate, income, multiProbaGroup, transportCost, transportCostRDP, coeffLand, job, amenities, solus_Q, typeHousing, param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood, selectedPixels):
    
    #basic_q_formal = param["q0"]
    basic_q_formal = 0
    
    content_cost = transTemp_incomeNetOfCommuting /3
    content_cost[np.isnan(content_cost)] = 1000
    content_cost[np.isinf(content_cost)] = 1000
    content_cost[content_cost < 1000] = 1000
    
    if typeHousing == 'backyard':
        dwellingSize = param["shack_size"] * np.ones((len(job.incomeMult), len(grid.dist)))
        dwellingSize[job.backyard == 0, :] = np.nan
        amenities = copy.deepcopy(amenities) * param["amenity_backyard"]
        if option["floods"] == 0:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - (np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"]))
        elif option["floods"] == 1:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - ((np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"])) - (flood.d_contents[selectedPixels][None, :] * content_cost) - (flood.d_structure[selectedPixels][None, :] * flood.informal_structure_value) - (interestRate * flood.informal_structure_value))
        R_mat[job.backyard == 0, :] = np.nan
    elif typeHousing == 'informal':
        dwellingSize = param["shack_size"] * np.ones((len(job.incomeMult), len(grid.dist)))
        dwellingSize[job.settlement == 0, :] = np.nan
        amenities = copy.deepcopy(amenities) * param["amenity_settlement"]
        if option["floods"] == 0:
            R_mat = 1 / param["shack_size"] * (transTemp_incomeNetOfCommuting - (np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * (param["shack_size"] - param["q0"]) ** param["beta"])) ** (1 / param["alpha"]))
        elif option["floods"] == 1:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - ((np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"])) - (flood.d_contents[selectedPixels][None, :] * content_cost) - (flood.d_structure[selectedPixels][None, :] * flood.informal_structure_value) - (interestRate * flood.informal_structure_value))
        R_mat[job.settlement == 0, :] = np.nan
    elif typeHousing == 'formal':  
        income_temp = transTemp_incomeNetOfCommuting
        income_temp[income_temp < 0] = np.nan
        if option["floods"] == 1:
            if option["incur_formal_structure_damages"] == 'households':           
                R_mat = SolveHouseholdsProgram(income_temp, param, flood, selectedPixels, constructionParam, interestRate, Uo, amenities, r_init)            
                R_mat[transTemp_incomeNetOfCommuting < 0] = 0
                R_mat[job.formal == 0, :] = np.nan
                dwellingSize = (1 / R_mat) * param["beta"] * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost) - (flood.d_structure[selectedPixels][None, :] * ((param["coeff_b"] * param["coeff_A"] / interestRate) ** (1/param["coeff_a"])) * (R_mat ** (1/param["coeff_a"]))))
                #dwellingSize = (1 / R_mat) * param["beta"] * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost))
                dwellingSize = np.maximum(dwellingSize, param["miniLotSize"])
                dwellingSize[job.formal == 0, :] = np.nan
            elif option["incur_formal_structure_damages"] == 'developers':
                dwellingSize = (Uo[:, None] / (((param["alpha"] * (income_temp - (flood.d_contents[selectedPixels][None, :] * content_cost))) ** param["alpha"])) * amenities[None, :]) ** (1/param["beta"])
                dwellingSize = np.maximum(dwellingSize, param["miniLotSize"])
                dwellingSize[job.formal == 0, :] = np.nan
                R_mat = param["beta"] * (transTemp_incomeNetOfCommuting - (flood.d_contents[selectedPixels][None, :] * content_cost)) / (dwellingSize - param["alpha"] * basic_q_formal)
                R_mat[transTemp_incomeNetOfCommuting < 0] = 0
                R_mat[job.formal == 0, :] = np.nan
        elif option["floods"] == 0:
            dwellingSize = (Uo[:, None] / (((param["alpha"] * (income_temp)) ** param["alpha"])) * amenities[None, :]) ** (1/param["beta"])
            dwellingSize = np.maximum(dwellingSize, param["miniLotSize"])
            dwellingSize[job.formal == 0, :] = np.nan
            R_mat = param["beta"] * (transTemp_incomeNetOfCommuting) / (dwellingSize - param["alpha"] * basic_q_formal)
            R_mat[transTemp_incomeNetOfCommuting < 0] = 0
            R_mat[job.formal == 0, :] = np.nan
    
    R_mat[R_mat < 0] = 0
    R_mat[np.isnan(R_mat)] = 0


    #Income group in each location
    proba = (R_mat == np.nanmax(R_mat, axis = 0))
    #proba[~np.isnan(multiProbaGroup)] = multiProbaGroup[~np.isnan(multiProbaGroup)]
    limit = (transTemp_incomeNetOfCommuting > 0) & (proba > 0) & (~np.isnan(transTemp_incomeNetOfCommuting)) & (R_mat > 0)
    proba = proba * limit

    whichGroup = np.nanargmax(R_mat, 0)
    #whichGroup[~np.isnan(multiProbaGroup[0, :])] = sum(np.matlib.repmat(np.arange(0, param["nb_of_income_classes"]), 1, sum(~np.isnan(multiProbaGroup[0, :]))) * proba[:, ~np.isnan(multiProbaGroup[0, :])))
    #temp = [0:transTemp_incomeNetOfCommuting.shape[1]-1] * transTemp_incomeNetOfCommuting.shape[0]
    #whichGroupTemp = whichGroup + temp

    
    R = np.empty(len(whichGroup))
    dwellingSizeTemp = np.empty(len(whichGroup))
    contentCostExport = np.empty(len(whichGroup))
    for i in range(0, len(whichGroup)):
        R[i] = R_mat[int(whichGroup[i]), i]
        dwellingSizeTemp[i] = dwellingSize[int(whichGroup[i]), i]
        contentCostExport[i] = content_cost[int(whichGroup[i]), i]
    
    dwellingSize = dwellingSizeTemp

    #Housing Construction 

    if typeHousing == 'formal':
        if option["adjustHousingSupply"] == 1:
            if option["floods"] == 0:
                housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate)) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
            elif option["floods"] == 1:
                if option["incur_formal_structure_damages"] == 'developers':
                    housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate + flood.d_structure[selectedPixels][None, :])) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
                elif option["incur_formal_structure_damages"] == 'households':
                    housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate)) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
            
            #Outside the agricultural rent, no housing (accounting for a tax)
            housingSupply = housingSupply.squeeze()
            housingSupply[R < agriculturalRent + param_taxUrbanEdgeMat] = 0
    
            housingSupply[np.isnan(housingSupply)] = 0
            housingSupply[housingSupply < 0] = 0
            housingSupply = np.fmin(housingSupply, housingLimit)
    
            #To add the construction on Mitchells_Plain
            housingSupply = np.fmax(housingSupply, param_minimumHousingSupply * 1000000)
    
        else:   
            housingSupply = param_housing_in
            
    elif typeHousing == 'backyard':
        if option["floods"] == 0:
            housingSupply = param["alpha"] * (((param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]))) - (param["beta"] * ((income[:, 0] - transportCostRDP) /((param["backyard_size"]) * R)))
            housingSupply[income[:, 0] < transportCostRDP] = param["alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["beta"] * (income[income[:, 0] < transportCostRDP, 0]) / ((param["backyard_size"]) * R[income[:, 0] < transportCostRDP])
        elif option["floods"] == 1:
            housingSupply = param["alpha"] * (((param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]))) - (param["beta"] * ((income[:, 0] - transportCostRDP - (flood.d_contents[selectedPixels] * contentCostExport)) /((param["backyard_size"]) * R)))
            housingSupply[income[:, 0] < transportCostRDP] = (param["alpha"] * ((param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]))) - (param["beta"] * (income[income[:, 0] < transportCostRDP, 0] - (flood.d_contents[selectedPixels][income[:, 0] < transportCostRDP] * (contentCostExport[income[:, 0] < transportCostRDP]))) / ((param["backyard_size"]) * R[income[:, 0] < transportCostRDP]))
        
        housingSupply[R == 0] = 0
        housingSupply = np.fmin(housingSupply, 1)
        housingSupply = np.fmax(housingSupply, 0)
        housingSupply = 1000000 * housingSupply

    elif typeHousing == 'informal':
        housingSupply = 1000000 * np.ones(len(whichGroup))
        housingSupply[R == 0] = 0
        
    peopleInit = housingSupply / dwellingSize * (np.sum(limit, 0) > 0)
    peopleInit[np.isnan(peopleInit)] = 0
    peopleInitLand = peopleInit * coeffLand * 0.5 ** 2

    peopleCenter = np.matlib.repmat(peopleInitLand, R_mat.shape[0], 1) * proba
    peopleCenter[np.isnan(peopleCenter)] = 0
        
    jobSimul = np.sum(peopleCenter, 1)
    
    if typeHousing == 'formal':
        R = np.fmax(R, agriculturalRent)

    return jobSimul, R, peopleInit, peopleCenter, housingSupply, dwellingSize, R_mat, contentCostExport


def SolveHouseholdsProgram(income_temp, param, flood, selectedPixels, constructionParam, interestRate, Uo, amenities, r_init):
    #option 1: own soler
    #maxIteration = 1999
    #indexIteration = 0
    #R_matTemp = np.ones((income_temp.shape[0], income_temp.shape[1], 1500))
    #R_matTemp[:, :, 0] = r_init
    #utilityTemp = np.ones((income_temp.shape[0], income_temp.shape[1], 1500))
    #diffUtility = 400
    #while (indexIteration < maxIteration - 1) & (diffUtility > 0.01):
    #    if indexIteration > 0:
    #        R_matTemp[:, :, indexIteration] = copy.deepcopy(R_matTemp[:, :, indexIteration - 1]) + 0.02*(utilityTemp[:, :, indexIteration - 1] - Uo[:, None])
    #        print(np.nanmean(R_matTemp[:, :, indexIteration]))
    #        print(sum(np.isnan(R_matTemp[:, :, indexIteration])))
    #    utilityTemp[:, :, indexIteration] = (param["alpha"] ** param["alpha"]) * (param["beta"] ** param["beta"]) * (R_matTemp[:, :, indexIteration] ** (-param["beta"])) * amenities * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost) - (flood.d_structure[selectedPixels][None, :] * ((param["coeff_b"] * param["coeff_A"] / interestRate) ** (1/param["coeff_a"])) * (R_matTemp[:, :, indexIteration] ** (1/param["coeff_a"]))))
        #utilityTemp[:, :, indexIteration] = (param["alpha"] ** param["alpha"]) * (param["beta"] ** param["beta"]) * (R_matTemp[:, :, indexIteration] ** (-param["beta"])) * amenities * (income_temp - (flood.d_contents[selectedPixels][None, :] * flood.content_cost))
    #    diffUtility = np.nansum(np.abs(utilityTemp[:, :, indexIteration] - Uo[:, None]))
    #    indexIteration = indexIteration + 1
    #    print(diffUtility)
    #    print(sum(np.isnan(utilityTemp[:, :, indexIteration - 1])))
    #return R_matTemp[:, :, indexIteration - 2]
        
    #option 2: minimize
    R_matTemp = np.ones((income_temp.shape[0], income_temp.shape[1]))
    for i in range(0, 4):
        print(i)
        for j in range(0, 4071):
            print(j)
            fun = lambda x: np.abs(((param["alpha"] ** param["alpha"]) * (param["beta"] ** param["beta"]) * (x ** (-param["beta"])) * amenities[j] * (income_temp[i, j] - (np.array(flood.d_contents[selectedPixels])[j] * flood.content_cost) - (np.array(flood.d_structure[selectedPixels])[j] * ((param["coeff_b"] * param["coeff_A"] / interestRate) ** (1/param["coeff_a"])) * (x ** (1/param["coeff_a"]))))) - Uo[i]) 
            res = optimize.minimize_scalar(fun)
            print(res)
            R_matTemp[i, j] = res.x
    return R_matTemp

