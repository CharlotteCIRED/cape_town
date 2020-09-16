# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:29:16 2020

@author: Charlotte Liotta
"""

import numpy as np
import scipy.io

def choice_param(option):
    
    #Year of the calibration
    param = {"baseline_year" : 2011}
    
    #Parameters of the utility fonction
    if option["import_precalculated_parameters"] == 1:
        if option["households_anticipate_floods"] == 0:
            param["beta"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedUtility_beta.mat')["calibratedUtility_beta"].squeeze()
            param["q0"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedUtility_q0.mat')["calibratedUtility_q0"].squeeze()
    param["alpha"] = 1 - param["beta"]
    
    #Parameters of the housing production fonction
    if option["import_precalculated_parameters"] == 1:
        if option["households_anticipate_floods"] == 0:
            param["coeff_b"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedHousing_b.mat')["coeff_b"].squeeze()
            param["coeff_A"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedHousing_kappa.mat')["coeffKappa"].squeeze()
    param["coeff_a"] = 1 - param["coeff_b"]
    param["depreciation_rate"] = 0.025
    param["interest_rate"] = 0.025    
    
    #Land-use, housing constraints and informal settlements
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.4
    param["minDensityUrban"] = 30000
    param["limitPeopleCityEdge"] = 40
    param["historicRadius"] = 100
    param["limitHeightCenter"] = 10 #very high => as if there were no limit
    param["limitHeightOut"] = 10
    param["shack_size"] = 20 #Size of a backyard shack (m2)
    param["RDP_size"] = 40 #Size of a RDP house (m2)
    param["backyard_size"] = 70 #size of the backyard of a RDP house (m2)
    if option["import_precalculated_parameters"] == 1:
        if option["households_anticipate_floods"] == 0:
            param["amenity_backyard"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedParamAmenities.mat')["calibratedParamAmenities"][0].squeeze()
            param["amenity_settlement"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedParamAmenities.mat')["calibratedParamAmenities"][1].squeeze()
    param["double_storey_shacks"] = 0
    param["coeffDoubleStorey"] = 0.02 #Random for now

    #Multiple income classes
    param["nb_of_income_classes"] = 4
    param["income_distribution"] = np.array([0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 4])
    param["thresholdJobs"] = 20000 #number of jobs above which we keep the employment center
    param["step"] = 2
    param["household_size"] = [1.14, 1.94, 1.94, 1.94] #Household size (accounting for unemployment rate)

    #Transportation
    param["waiting_time_metro"] = 10 #minutes
    param["walking_speed"] = 4 #km/h
    param["timeCost"] = 1
    param["timeCost2"] = 1
    if option["import_precalculated_parameters"] == 1:
        if option["households_anticipate_floods"] == 0:
            param["lambda"] = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/lambda.mat')["lambdaKeep"].squeeze()
    param["logitFactorMin"] = 6
    param["logitFactor"] = 3
    
    #Agricultural rent
    param["agriculturalRent2011"] = 807.2
    param["agriculturalRent2001"] = 70.7
    
    #Solver parameters
    param["max_iter"] = 1000
    param["precision"] = 0.025 #precision of the resolution: max 1% error
    param["iter_calc_lite"] = 1
    param["time_invest_h"] = 10
    param["timeInvestHousing"] = 3
    param["timeDepreciationBuildings"] = 100
    
    #Scenarios
    param["yearUrbanEdge"] = 2015 #in case option.urban_edge = 0, the year the constraint is removed
    param["taxOutUrbanEdge"] = 10000
    param["future_backyard_size"] = param["backyard_size"]
    param["futureRatePublicHousing"] = 5000
    
    #param["timeLimit"] = 20.2000
    #param["meanSlopeSmooth"] = 3

    return param

def add_construction_parameters(param, households_data, land, grid):
    
    param["housing_in"] = np.empty(len(households_data.gridFormalDensityHFA))
    param["housing_in"][land.coeff_land[0,:] != 0] = households_data.gridFormalDensityHFA[land.coeff_land[0,:] != 0] / land.coeff_land[0,:][land.coeff_land[0,:] != 0] * 1.1
    param["housing_in"][(land.coeff_land[0,:] == 0) | np.isnan(households_data.gridFormalDensityHFA)] = 0
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0
    
    #In Mitchells Plain, housing supply is given exogenously (planning), and household of group 2 live there (Coloured neighborhood). 
    param["minimumHousingSupply"] = np.zeros(len(grid.dist))
    param["minimumHousingSupply"][households_data.Mitchells_Plain_grid_2011] = households_data.gridFormalDensityHFA[households_data.Mitchells_Plain_grid_2011] / land.coeff_land[0, households_data.Mitchells_Plain_grid_2011]
    param["minimumHousingSupply"][(land.coeff_land[0,:] < 0.1) | (np.isnan(param["minimumHousingSupply"]))] = 0
    param["multiProbaGroup"] = np.empty((param["nb_of_income_classes"], len(grid.dist)))
    
    #Define minimum lot-size 
    param["miniLotSize"] = np.nanmin(households_data.spDwellingSize[households_data.total_dwellings_SP_2011 != 0][(households_data.informal_SP_2011[households_data.total_dwellings_SP_2011 != 0] + households_data.backyard_SP_2011[households_data.total_dwellings_SP_2011 != 0]) / households_data.total_dwellings_SP_2011[households_data.total_dwellings_SP_2011 != 0] < 0.1])

    return param

