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

def ComputeNEDUMOutput_LOGIT(Uo, param, option, transTemp_incomeNetOfCommuting, grid, agriculturalRent, housingLimit, referenceRent, constructionParam, interestRate, income, multiProbaGroup, transportCost, transportCostRDP, coeffLand, job, amenities, solus_Q, typeHousing, param_minimumHousingSupply, param_housing_in, param_taxUrbanEdgeMat, flood):
    
    basic_q_formal = param["q0"]


    #Dwelling sizes
    if typeHousing == 'formal':       
        income_temp = transTemp_incomeNetOfCommuting
        income_temp[income_temp < 0] = np.nan
        dwellingSize = np.empty
        #dwellingSize = solus_Q_temp(income_temp, np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / np.matlib.repmat(amenities, income.shape[1], 1))
        dwellingSize = solus_Q(income_temp, np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / np.matlib.repmat(amenities, income.shape[1], 1))
        dwellingSize[np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / np.matlib.repmat(amenities, income.shape[1], 1) > param["max_U"]] = param["max_q"]
        dwellingSize = np.maximum(dwellingSize, param["miniLotSize"])
        dwellingSize[job.formal == 0, :] = np.nan
    elif typeHousing == 'backyard':
        dwellingSize = param["shack_size"] * np.ones((len(job.incomeMult), len(grid.dist)))
        dwellingSize[job.backyard == 0, :] = np.nan
    elif typeHousing == 'informal':
        dwellingSize = param["shack_size"] * np.ones((len(job.incomeMult), len(grid.dist)))
        dwellingSize[job.settlement == 0, :] = np.nan
    
    #Bid rents    
    if typeHousing == 'formal':
        if option["floods"] == 0:
            R_mat = param["beta"] * (transTemp_incomeNetOfCommuting) / (dwellingSize - param["alpha"] *param["q0"])
        elif option["floods"] == 1:
            R_mat = param["beta"] * (transTemp_incomeNetOfCommuting - (flood.d_contents * flood.content_cost)) / (dwellingSize - param["alpha"] *param["q0"])
        R_mat[transTemp_incomeNetOfCommuting < 0] = 0
        R_mat[job.formal == 0, :] = np.nan
    elif typeHousing == 'backyard':
        amenities = copy.deepcopy(amenities) * param["amenity_backyard"]
        if option["floods"] == 0:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - (np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"]))
        elif option["floods"] == 1:
            R_mat = (1 / param["shack_size"]) * (transTemp_incomeNetOfCommuting - ((np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * ((param["shack_size"] - param["q0"]) ** param["beta"]))) ** (1 / param["alpha"])) - (flood.d_contents * flood.content_cost) - (flood.d_structure * flood.informal_structure_value) - (interestRate * flood.informal_structure_value))
        R_mat[job.backyard == 0, :] = np.nan
    elif typeHousing == 'informal':
        amenities = copy.deepcopy(amenities) * param["amenity_settlement"]
        R_mat = 1 / param["shack_size"] * (transTemp_incomeNetOfCommuting - (np.transpose(np.matlib.repmat(Uo, income.shape[0], 1)) / ((np.matlib.repmat(amenities, income.shape[1], 1)) * (param["shack_size"] - param["q0"]) ** param["beta"])) ** (1 / param["alpha"]))
        R_mat[job.settlement == 0, :] = np.nan

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

    #R = R_mat[whichGroupTemp]
    R = np.empty(len(whichGroup))
    dwellingSizeTemp = np.empty(len(whichGroup))
    for i in range(0, len(whichGroup)):
        R[i] = R_mat[int(whichGroup[i]), i]
        #dwellingSize = dwellingSize[whichGroupTemp]
        dwellingSizeTemp[i] = dwellingSize[int(whichGroup[i]), i]
    
    dwellingSize = dwellingSizeTemp

    #Housing Construction 

    if typeHousing == 'formal':
        if option["adjustHousingSupply"] == 1:
            if option["floods"] == 0:
                housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate)) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
            elif option["floods"] == 1:
                housingSupply = 1000000 * constructionParam ** (1 / param["coeff_a"]) *(param["coeff_b"] / (interestRate + flood.d_structure)) ** (param["coeff_b"] /param["coeff_a"]) * R ** (param["coeff_b"] / param["coeff_a"])
            
            #Outside the agricultural rent, no housing (accounting for a tax)
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
            housingSupply = param["alpha"] * (((param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]))) - (param["beta"] * ((income[:, 0] - transportCostRDP) /((param["backyard_size"]) * R)))
            housingSupply[income[:, 0] < transportCostRDP] = param["alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["beta"] * (income[income[:, 0] < transportCostRDP, 0]) / ((param["backyard_size"]) * R[income[:, 0] < transportCostRDP])
        elif option["floods"] == 1:
            housingSupply = param["alpha"] * (((param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]))) - (param["beta"] * ((income[:, 0] - transportCostRDP - (flood.d_contents * flood.content_cost)) /((param["backyard_size"]) * R)))
            housingSupply[income[:, 0] < transportCostRDP] = (param["alpha"] * ((param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]))) - (param["beta"] * (income[income[:, 0] < transportCostRDP, 0] - (flood.d_contents * flood.content_cost)) / ((param["backyard_size"]) * R[income[:, 0] < transportCostRDP]))
        
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

