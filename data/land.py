# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:10:27 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy.io

class Land:
    
    def __init__(self):
        
        self

    def import_land_use(self, grille, option, param, households_data, macro_data):
        
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
        
        numberPropertiesRDP2000 = households_data.GV_count_RDP * (1 - grille.dist / max(grille.dist[households_data.GV_count_RDP > 0]))
        
        method = 'linear'
        
        if option["future_construction_RDP"] == 1: 
            #if backyarding is possible in future RDP/BNG settlements

            construction_rdp = pd.read_csv('./2. Data/Basile data/grid_new_RDP_projects.csv')
            
            yearBeginRDP = 2015
            yearRDP = np.arange(yearBeginRDP, 2040) - param["baseline_year"]
            numberRDP = macro_data.rdp(yearRDP)
            
            yearShortTerm = np.argmin(np.abs(sum(construction_rdp.total_yield_DU_ST) - (numberRDP - numberRDP[0])))
            yearLongTerm = np.argmin(np.abs(sum(construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) - (numberRDP - numberRDP[0])))

            areaRDPShortTerm = np.minimum(construction_rdp.area_ST, (param["backyard_size"] + param["RDP_size"]) * construction_rdp.total_yield_DU_ST)
            areaRDPLongTerm = np.minimum(np.minimum(construction_rdp.area_ST + construction_rdp.area_LT, (param["backyard_size"] + param["RDP_size"]) * (construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT)), area_pixel)

            #Share of pixel for RDP houses and backyards in ST and LT
            areaBackyardShortTerm = area_backyard + np.maximum(areaRDPShortTerm - construction_rdp.total_yield_DU_ST * param["RDP_size"], 0) / area_pixel
            areaRDPShortTerm = area_RDP + np.minimum(construction_rdp.total_yield_DU_ST * param["RDP_size"], construction_rdp.area_ST) / area_pixel
            areaBackyardShortTerm = np.minimum(areaBackyardShortTerm, param["max_land_use"] - areaRDPShortTerm)
            areaBackyardLongTerm = area_backyard + np.maximum(areaRDPLongTerm - (construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) * param["RDP_size"], 0) / area_pixel
            areaRDPLongTerm = area_RDP + np.minimum((construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) * param["RDP_size"], areaRDPLongTerm) / area_pixel
            areaBackyardLongTerm = np.minimum(areaBackyardLongTerm, param["max_land_use"] - areaRDPLongTerm)

            #area_backyard_2025 = np.fmin(param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) * param["backyard_size"] / area_pixel)
            #area_RDP_2025 = np.fmin(param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) * param["RDP_size"] / area_pixel)
            #area_backyard_2040 = np.fmin(param["future_backyard_size"] / (param["future_backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT) * param["future_backyard_size"] / area_pixel)
            #area_RDP_2040 = np.fmin(param["RDP_size"] / (param["future_backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT) * param["RDP_size"] / area_pixel)
           
            #year_data_informal = np.transpose([1990, 2015, 2025, 2040]) - param["baseline_year"]         
            #spline_land_backyard = interp1d(year_data_informal, np.transpose([area_backyard, area_backyard, area_backyard_2025, area_backyard_2040]), method)
            #spline_land_RDP = interp1d(year_data_informal, np.transpose([area_RDP, area_RDP, area_RDP_2025, area_RDP_2040]), method)
            #spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates, RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) + np.transpose(construction_rdp.total_yield_DU_LT)]), method)

            year_data_informal = [2000 - param["baseline_year"], yearBeginRDP - param["baseline_year"], yearShortTerm, yearLongTerm]
    
            spline_land_backyard = interp1d(year_data_informal,  np.transpose([area_backyard, area_backyard, areaBackyardShortTerm, areaBackyardLongTerm]), method)
            spline_land_RDP = interp1d(year_data_informal,  np.transpose([area_RDP, area_RDP, areaRDPShortTerm, areaRDPLongTerm]), method)
            spline_estimate_RDP = interp1d(year_data_informal, np.transpose([numberPropertiesRDP2000, RDP_houses_estimates, RDP_houses_estimates + construction_rdp.total_yield_DU_ST, RDP_houses_estimates + construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT]), method)

                                                        
        elif option["future_construction_RDP"] == 0:
            #Scenario with no future construction of RDP

            year_data_informal = np.transpose([1990, 2040]) - param["baseline_year"]
            spline_land_backyard = interp1d(x = year_data_informal, y = np.transpose([area_backyard, area_backyard]), kind = method)
            spline_land_RDP = interp1d(year_data_informal,  np.transpose([area_RDP, area_RDP]), kind = method)
            spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates]), kind = method)

        
        coeff_land_private_urban_edge = (coeff_land_urban_edge - informal - np.fmin(area_RDP + area_backyard, urban)) * param["max_land_use"]
        coeff_land_private_no_urban_edge = (coeff_land_no_urban_edge - informal - np.fmin(area_RDP + area_backyard, urban)) * param["max_land_use"]
        coeff_land_private_urban_edge[coeff_land_private_urban_edge < 0] = 0
        coeff_land_private_no_urban_edge[coeff_land_private_no_urban_edge < 0] = 0
        
        if option["urban_edge"] == 0:
            year_constraints = np.array([1990, param["yearUrbanEdge"] - 1, param["yearUrbanEdge"], 2040]) - param["baseline_year"]
            spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge, coeff_land_no_urban_edge, coeff_land_no_urban_edge])), method)
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
        centerRegulation = (grille.dist <= param["historicRadius"])
        outsideRegulation = (grille.dist > param["historicRadius"])
        housingLimit = param["limitHeightCenter"] * 1000000 * centerRegulation + param["limitHeightOut"] * 1000000 * outsideRegulation 









        self.urban = urban #Prop. urbanized
        self.informal = informal #Prop of the area occupied by informal dwellings
        self.coeff_land_urban_edge = coeff_land_urban_edge
        self.coeff_land_no_urban_edge = coeff_land_no_urban_edge
        self.RDP_houses_estimates = RDP_houses_estimates #Number of RDP houses
        self.area_RDP = area_RDP #Area of subsidized housing
        self.area_backyard = area_backyard #Area of backyard settlements
        self.housing_limit = housingLimit
        self.spline_estimate_RDP = spline_estimate_RDP
        self.spline_land_backyard = spline_land_backyard
        self.spline_land_RDP = spline_land_RDP
        self.coeff_land_private = coeff_land_private #Max proportion occupied by private housing
        self.coeff_land = np.array([coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP])
        self.coeff_land_RDP = coeff_land_RDP        
        self.coeff_land_backyard = coeff_land_backyard #Max proportion occupied by backyard settlements
        self.coeff_land_settlement = coeff_land_settlement #Max proportion occupied by informal settlements
        self.spline_land_constraints = spline_land_constraints
        self.coeff_land_private_urban_edge = coeff_land_private_urban_edge
        self.coeff_land_private_no_urban_edge = coeff_land_private_no_urban_edge
        precalculated_amenities = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/calibratedAmenities.mat')
        self.amenities = precalculated_amenities["amenities"] / np.nanmean(precalculated_amenities["amenities"])


