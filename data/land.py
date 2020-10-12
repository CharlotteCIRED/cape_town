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

    def import_land_use(self, grid, option, param, households_data, macro_data):
        
        area_pixel = (0.5 ** 2) * 1000000

        #0. Import Land Cover Data (see R code for details)
        land_use_data_old = pd.read_csv('./2. Data/Basile data/grid_NEDUM_Cape_Town_500.csv', sep = ';')
        informal_risks = pd.read_csv('./2. Data/Land occupation/informal_settlements_risk.csv', sep = ',')
        
        coeff_land_no_urban_edge = (np.transpose(land_use_data_old.unconstrained_out) + np.transpose(land_use_data_old.unconstrained_UE)) / area_pixel
        coeff_land_urban_edge = np.transpose(land_use_data_old.unconstrained_UE) / area_pixel
        informal = np.transpose(land_use_data_old.informal) / area_pixel
        urban = np.transpose(land_use_data_old.urban) / area_pixel
             
        #1. Informal
        
        coeff_land_settlement = (informal_risks.area / area_pixel) * param["max_land_use_settlement"]
        
        #2. RDP and backyard
        
        RDP_houses_estimates = households_data.GV_count_RDP #actual nb of RDP houses
        area_RDP = households_data.GV_area_RDP * param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
        area_backyard = households_data.GV_area_RDP * param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
        
        coeff_land_backyard = np.fmin(urban, area_backyard)
        actual_backyards = (households_data.backyard_grid_2011 / np.nanmax(households_data.backyard_grid_2011)) * np.max(coeff_land_backyard)
        coeff_land_backyard = np.fmax(coeff_land_backyard, actual_backyards)
        
        numberPropertiesRDP2000 = households_data.GV_count_RDP * (1 - grid.dist / max(grid.dist[households_data.GV_count_RDP > 0]))
        method = 'linear'        
        construction_rdp = pd.read_csv('./2. Data/Basile data/grid_new_RDP_projects.csv')            
        yearBeginRDP = 2015
        yearRDP = np.arange(yearBeginRDP, 2040) - param["baseline_year"]
        numberRDP = macro_data.rdp(yearRDP)           
        yearShortTerm = np.argmin(np.abs(sum(construction_rdp.total_yield_DU_ST) - (numberRDP - numberRDP[0])))
        yearLongTerm = np.argmin(np.abs(sum(construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) - (numberRDP - numberRDP[0])))
        areaRDPShortTerm = np.minimum(construction_rdp.area_ST, (param["backyard_size"] + param["RDP_size"]) * construction_rdp.total_yield_DU_ST)
        areaRDPLongTerm = np.minimum(np.minimum(construction_rdp.area_ST + construction_rdp.area_LT, (param["backyard_size"] + param["RDP_size"]) * (construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT)), area_pixel)

        areaBackyardShortTerm = area_backyard + np.maximum(areaRDPShortTerm - construction_rdp.total_yield_DU_ST * param["RDP_size"], 0) / area_pixel
        areaRDPShortTerm = area_RDP + np.minimum(construction_rdp.total_yield_DU_ST * param["RDP_size"], construction_rdp.area_ST) / area_pixel
        areaBackyardShortTerm = np.minimum(areaBackyardShortTerm, param["max_land_use"] - areaRDPShortTerm)
        areaBackyardLongTerm = area_backyard + np.maximum(areaRDPLongTerm - (construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) * param["RDP_size"], 0) / area_pixel
        areaRDPLongTerm = area_RDP + np.minimum((construction_rdp.total_yield_DU_LT + construction_rdp.total_yield_DU_ST) * param["RDP_size"], areaRDPLongTerm) / area_pixel
        areaBackyardLongTerm = np.minimum(areaBackyardLongTerm, param["max_land_use"] - areaRDPLongTerm)

        year_data_informal = [2000 - param["baseline_year"], yearBeginRDP - param["baseline_year"], yearShortTerm, yearLongTerm]
        spline_land_backyard = interp1d(year_data_informal,  np.transpose([np.fmax(area_backyard, actual_backyards), np.fmax(area_backyard, actual_backyards), np.fmax(areaBackyardShortTerm, actual_backyards), np.fmax(areaBackyardLongTerm, actual_backyards)]), method)
        
        spline_land_RDP = interp1d(year_data_informal,  np.transpose([area_RDP, area_RDP, areaRDPShortTerm, areaRDPLongTerm]), method)
        spline_estimate_RDP = interp1d(year_data_informal, np.transpose([numberPropertiesRDP2000, RDP_houses_estimates, RDP_houses_estimates + construction_rdp.total_yield_DU_ST, RDP_houses_estimates + construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT]), method)


        #3. Formal
       
        coeff_land_private_urban_edge = (coeff_land_urban_edge - informal - area_RDP - area_backyard) * param["max_land_use"]
        coeff_land_private_no_urban_edge = (coeff_land_no_urban_edge - informal - area_RDP - area_backyard) * param["max_land_use"]
        coeff_land_private_urban_edge[coeff_land_private_urban_edge < 0] = 0
        coeff_land_private_no_urban_edge[coeff_land_private_no_urban_edge < 0] = 0
        
        if option["urban_edge"] == 0:
            coeff_land_private = coeff_land_private_urban_edge
        else:
            coeff_land_private = coeff_land_private_no_urban_edge
                       
        #4. Constraints
        
        if option["urban_edge"] == 0:
            year_constraints = np.array([1990, param["yearUrbanEdge"] - 1, param["yearUrbanEdge"], 2040]) - param["baseline_year"]
            spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge, coeff_land_no_urban_edge, coeff_land_no_urban_edge])), method)
        else:
            year_constraints = np.array([1990, 2040]) - param["baseline_year"]
            spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge])))

        #Coeff_land for each housing type
        coeff_land_backyard = coeff_land_backyard * param["max_land_use_backyard"]
        coeff_land_backyard[coeff_land_backyard < 0] = 0
        coeff_land_RDP = np.ones(len(coeff_land_private))
        
        #Building limit
        centerRegulation = (grid.dist <= param["historicRadius"])
        outsideRegulation = (grid.dist > param["historicRadius"])
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


