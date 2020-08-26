# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:10:27 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class Land:
    
    def __init__(self):
        
        self

    def import_land_use(self, grille, option, param, households_data):

        area_pixel = (0.5 ** 2) * 1000000

        #Land Cover Data from our estimation (see R code for details)
        grid = pd.read_csv('./2. Data/Basile data/grid_NEDUM_Cape_Town_500.csv', sep = ';')
        urban = np.transpose(grid.urban) / area_pixel
        informal = np.transpose(grid.informal) / area_pixel
        coeff_land_no_urban_edge = (np.transpose(grid.unconstrained_out) + np.transpose(grid.unconstrained_UE)) / area_pixel
        coeff_land_urban_edge = np.transpose(grid.unconstrained_UE) / area_pixel
        
        #Number of RDP/BNG dwellings and area available for backyarding in each subplace
        RDP_houses_estimates = households_data.GV_count_RDP
        area_RDP = households_data.GV_area_RDP * param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
        area_backyard = households_data.GV_area_RDP * param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
        coeff_land_backyard = np.fmin(urban, area_backyard)

        method = 'linear'
        
        if option["future_construction_RDP"] == 1: 
            #if backyarding is possible in future RDP/BNG settlements

            construction_rdp = pd.read_csv('./2. Data/Basile data/grid_new_RDP_projects.csv')

            area_backyard_2025 = np.fmin(param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) * param["backyard_size"] / area_pixel)
            area_RDP_2025 = np.fmin(param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) * param["RDP_size"] / area_pixel)
            area_backyard_2040 = np.fmin(param["future_backyard_size"] / (param["future_backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT) * param["future_backyard_size"] / area_pixel)
            area_RDP_2040 = np.fmin(param["RDP_size"] / (param["future_backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT) * param["RDP_size"] / area_pixel)
            year_data_informal = np.transpose([1990, 2015, 2025, 2040]) - param["baseline_year"]
            spline_land_backyard = interp1d(year_data_informal, np.transpose([area_backyard, area_backyard, area_backyard_2025, area_backyard_2040]), method)
            spline_land_RDP = interp1d(year_data_informal, np.transpose([area_RDP, area_RDP, area_RDP_2025, area_RDP_2040]), method)
            spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates, RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) + np.transpose(construction_rdp.total_yield_DU_LT)]), method)

        elif option["future_construction_RDP"] == 0:
            #Scenario with no future construction of RDP

            year_data_informal = np.transpose([1990, 2040]) - param["baseline_year"]
            spline_land_backyard = interp1d(x = year_data_informal, y = np.transpose([area_backyard, area_backyard]), kind = method)
            spline_land_RDP = interp1d(year_data_informal,  np.transpose([area_RDP, area_RDP]), kind = method)
            spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates]), kind = method)

        
        coeff_land_private_urban_edge = (coeff_land_urban_edge - np.fmin(area_RDP + area_backyard, urban)) * param["max_land_use"]
        coeff_land_private_no_urban_edge = (coeff_land_no_urban_edge - informal - np.fmin(area_RDP + area_backyard, urban)) * param["max_land_use"]
        coeff_land_private_urban_edge[coeff_land_private_urban_edge < 0] = 0
        coeff_land_private_no_urban_edge[coeff_land_private_no_urban_edge < 0] = 0
        
        if option["urban_edge"] == 0:
            year_constraints = np.array([1990, 2015, 2016, 2040]) - param["baseline_year"]
            spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge, coeff_land_no_urban_edge, coeff_land_no_urban_edge])))
        else:
            year_constraints = np.array([1990, 2040]) - param["baseline_year"]
            spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge])))


        if option["urban_edge"] == 0:
            coeff_land_private = coeff_land_private_urban_edge
        else:
            coeff_land_private = coeff_land_private_no_urban_edge


        #Coeff_land for each housing type
        coeff_land_backyard = coeff_land_backyard * param["max_land_use_backyard"]
        coeff_land_backyard[coeff_land_backyard < 0] = 0
        coeff_land_settlement = informal * param["max_land_use_settlement"]
        coeff_land_RDP = np.ones(len(coeff_land_private))
        
        #Building limit
        interieur = (grille.dist <= param["historic_center"])
        exterieur = (grille.dist > param["historic_center"])
        housing_limit = param["housing_constraint_1"] * 1000000 * interieur + param["housing_constraint_2"] * 1000000 * exterieur
        
        self.urban = urban #Prop. urbanized
        self.informal = informal #Prop of the area occupied by informal dwellings
        self.coeff_land_urban_edge = coeff_land_urban_edge
        self.coeff_land_no_urban_edge = coeff_land_no_urban_edge
        self.RDP_houses_estimates = RDP_houses_estimates #Number of RDP houses
        
        self.area_RDP = area_RDP #Area of subsidized housing
        self.area_backyard = area_backyard #Area of backyard settlements
        self.area_backyard_2025 = area_backyard_2025 #Area of backayard settlements (2025)
        self.area_RDP_2025 = area_RDP_2025 #Area of subsidized housing (2025)
        self.area_backyard_2040 = area_backyard_2040 #Area of backyard settlements (2040)
        self.area_RDP_2040 = area_RDP_2040 #Area of subsidized housing (2040)
        
        self.spline_land_backyard = interp1d(year_data_informal, np.transpose([area_backyard, area_backyard, area_backyard_2025, area_backyard_2040]), method)
        self.spline_land_RDP = interp1d(year_data_informal, np.transpose([area_RDP, area_RDP, area_RDP_2025, area_RDP_2040]), method)
        self.spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates, RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) + np.transpose(construction_rdp.total_yield_DU_LT)]), method)
        
        self.coeff_land_private = coeff_land_private #Max proportion occupied by private housing
        self.coeff_land_backyard = coeff_land_backyard #Max proportion occupied by backyard settlements
        self.coeff_land_settlement = coeff_land_settlement #Max proportion occupied by informal settlements
        self.coeff_land_RDP = coeff_land_RDP #Max proportion occupied by subsidized housing
        self.coeff_land = np.array([coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP])
        
        self.housing_limit = housing_limit
        
        self.year_constraints = year_constraints
        self.spline_land_constraints = spline_land_constraints
        
        



