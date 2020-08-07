# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:29:16 2020

@author: Charlotte Liotta
"""

import numpy as np

def choice_param():
    
    #Year of the calibration
    param = {"baseline_year" : 2011}
    
    #Parameters of the utility fonction
    param["coeff_beta"] = 0.25
    param["coeff_alpha"] = 1 - param["coeff_beta"]
    param["q0"] = 40
    
    #Parameters of the housing production fonction
    param["coeff_A"] = 0.69
    param["coeff_b"] = 0.55
    param["coeff_a"] = 1 - param["coeff_b"]
    param["depreciation_rate"] = 0.03
    param["interest_rate"] = 0.0250
    
    #Land-use, housing constraints and informal settlements
    param["max_land_use"] = 0.7
    param["max_land_use_backyard"] = 0.45
    param["max_land_use_settlement"] = 0.2
    param["max_density"] = 30000
    param["historic_center"] = 20 #Historical center radius
    param["housing_constraint_1"] = 0.5
    param["housing_constraint_2"] = 0.5
    param["shack_size"] = 20 #Size of a backyard shack (m2)
    param["rdp_size"] = 40 #Size of a RDP house (m2)
    param["backyard_size"] = 70 #size of the backyard of a RDP house (m2)
    param["future_backyard_size"] = param["backyard_size"]
    param["amenity_backyard"] = 0.38
    param["amenity_settlement"] = 0.37
    param["double_storey_shacks"] = 0

    #Multiple income classes
    param["nb_of_income_classes"] = 4
    param["income_distribution"] = np.array([0, 1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4])

    #Transportation
    param["waiting_time_metro"] = 10 #minutes
    param["walking_speed"] = 5 #km/h
    param["transaction_cost2011"] = 700 #Rands per m2 per year
    param["household_size"] = [1.14, 1.94, 1.94, 1.94] #Household size
    param["lambda"] = 1500
    
    #Solver parameters
    param["max_iter"] = 400
    param["precision"] = 0.025
    
    print('*** Parameters imported succesfully ***')
    
    #param["delay"] = 0
    
    #param["mini_lotsize"] = 1
    #param["coeff_mu"] = param["coeff_alpha"]
    
    #param["coeff_sigma"] = 1
    #param["ratio"] = 0
    #param["iter_calc_lite"] = 1

    #seuil du nombre d'emplois au-delà duquel on garde le centre d'emploi
    #param["seuil_emplois"] = 20000
    #param["pas"] = 2
        
    #Param for dynamic evolution
    #param["time_invest_h"] = 10
    #param["time_infra_km"] = 1
    
    #param["facteur_logit_min"] = 6
    #param["facteur_logit"] = 3
    #param["limite_temps"] = 20.2000;
    #param["moyenne_pente_smooth"] = 3;
    #param["prix_temps"] = 1
    #param["prix_temps2"] = param["prix_temps"] * 0.6
    #param["prix_tsport"] = 60
    
    return param

def add_construction_parameters(param, households_data, land, grid):
    param["housing_in"] = households_data.DENS_HFA_formal_grid / land.coeff_land[0,:] * 1.1
    param["housing_in"][~np.isfinite(param["housing_in"])] = 0
    param["housing_in"][param["housing_in"] > 2 * (10**6)] = 2 * (10**6)
    param["housing_in"][param["housing_in"] < 0] = 0
    param["housing_mini"] = np.zeros(len(grid.dist))
    param["housing_mini"][households_data.Mitchells_Plain_grid_2011] = households_data.DENS_HFA_formal_grid[households_data.Mitchells_Plain_grid_2011] / land.coeff_land[0, households_data.Mitchells_Plain_grid_2011]
    param["housing_mini"][(land.coeff_land[0,:] < 0.1) | (np.isnan(param["housing_mini"]))] = 0
    return param