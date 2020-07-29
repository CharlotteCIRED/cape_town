# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:10:27 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class Land:
    """ Class definig a grid with:
        - ID
        - coord_horiz
        - coord_vert
        - xcentre, ycentre
        - dist """
    
    def __init__(self):
        
        self

    def import_coeff_land_CAPE_TOWN2(self, grille, option, param, data_courbe):

        area_pixel = (0.5 ** 2) * 1000000

        #Land Cover Data from our estimation (see R code for details)
        grid = pd.read_csv('./2. Data/Basile data/grid_NEDUM_Cape_Town_500.csv', sep = ';')
        urbanise = np.transpose(grid.urban)/area_pixel
        informal = np.transpose(grid.informal)/area_pixel
        coeff_land_no_urban_edge = (np.transpose(grid.unconstrained_out) + np.transpose(grid.unconstrained_UE))/area_pixel
        coeff_land_urban_edge = np.transpose(grid.unconstrained_UE)/area_pixel
 
        #Here we estimate the number of RDP/BNG dwellings and the area available
        #for backyarding in each subplace using GV2012 data
        RDP_houses_estimates = data_courbe.GV_count_RDP
        area_RDP = data_courbe.GV_area_RDP * param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
        area_backyard = data_courbe.GV_area_RDP * param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
        coeff_land_backyard = np.fmin(urbanise, area_backyard)

        method = 'linear'
        
        if option["future_construction_RDP"] == 1: 
            #if backyarding is possible in future RDP/BNG settlements

            construction_rdp = pd.read_csv('./2. Data/Basile data/grid_new_RDP_projects.csv')

            area_backyard_2025 = np.fmin(param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) * param["backyard_size"] / area_pixel)
            area_RDP_2025 = np.fmin(param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) * param["RDP_size"] / area_pixel)
            area_backyard_2040 = np.fmin(param["backyard_size_future"] / (param["backyard_size_future"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT) * param["backyard_size_future"] / area_pixel)
            area_RDP_2040 = np.fmin(param["RDP_size"] / (param["backyard_size_future"] + param["RDP_size"]), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST + construction_rdp.total_yield_DU_LT) * param["RDP_size"] / area_pixel)
            year_data_informal = np.transpose([1990, 2015, 2025, 2040]) - param["year_begin"]
            spline_land_backyard = interp1d(year_data_informal, np.transpose([area_backyard, area_backyard, area_backyard_2025, area_backyard_2040]), method)
            spline_land_RDP = interp1d(year_data_informal, np.transpose([area_RDP, area_RDP, area_RDP_2025, area_RDP_2040]), method)
            spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates, RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) + np.transpose(construction_rdp.total_yield_DU_LT)]), method)

        elif option["future_construction_RDP"] == 0:
            #Scenario with no future construction of RDP

            year_data_informal = np.transpose([1990, 2040]) - param["year_begin"]
            spline_land_backyard = interp1d(x = year_data_informal, y = np.transpose([area_backyard, area_backyard]), kind = method)
            spline_land_RDP = interp1d(year_data_informal,  np.transpose([area_RDP, area_RDP]), kind = method)
            spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates]), kind = method)

        #Coeff_land for each housing type
        coeff_land_private_urban_edge = (coeff_land_urban_edge - np.fmin(area_RDP + area_backyard, urbanise)) * param["coeff_landmax"]
        coeff_land_private_no_urban_edge = (coeff_land_no_urban_edge - informal - np.fmin(area_RDP + area_backyard, urbanise)) * param["coeff_landmax"]
        coeff_land_private_urban_edge[coeff_land_private_urban_edge < 0] = 0
        coeff_land_private_no_urban_edge[coeff_land_private_no_urban_edge < 0] = 0
        
        #Evolution of constraints
        #if option["urban_edge"] == 0:
        #year_constraints = np.transpose([1990, param["annee_urban_edge"] - 1, param["annee_urban_edge"], 2040]) - param["year_begin"]
        #land_spline_land_constraints = interp1d(year_constraints, [coeff_land_urban_edge, coeff_land_urban_edge, coeff_land_no_urban_edge, coeff_land_no_urban_edge])
        #else:
        year_constraints = np.transpose([1990, 2040]) - param["year_begin"]
        land_spline_land_constraints = interp1d(year_constraints, np.transpose(np.array([coeff_land_urban_edge, coeff_land_urban_edge])))

        #For the initial state
        #if option["urban_edge"] == 1:
        #    coeff_land_private = coeff_land_private_urban_edge
        #    self.coeff_land_private = coeff_land_private
        #else:
        coeff_land_private = coeff_land_private_no_urban_edge
        coeff_land_backyard = coeff_land_backyard * param["coeff_landmax_backyard"]
        coeff_land_backyard[coeff_land_backyard < 0] = 0
        coeff_land_settlement = informal * param["coeff_landmax_settlement"]
        coeff_land_RDP = np.ones(len(coeff_land_private))
        
        #Building limit
        interieur = (grille.dist <= param["rayon_historique"])
        exterieur = (grille.dist > param["rayon_historique"])
        housing_limite = param["taille_limite1"] * 1000000 * interieur + param["taille_limite2"] * 1000000 * exterieur
        housing_limite_politique = param["taille_limite1"] * 1000000 * interieur + param["taille_limite2"] * 1000000 * exterieur

        self.coeff_landmax = param["coeff_landmax"]
        self.coeff_landmax_backyard = param["coeff_landmax_backyard"]
        self.coeff_landmax_settlement = param["coeff_landmax_settlement"]
        self.urbanise = urbanise
        self.informal = informal
        self.coeff_land_no_urban_edge = coeff_land_no_urban_edge
        self.coeff_land_urban_edge = coeff_land_urban_edge
        self.RDP_houses_estimates = RDP_houses_estimates
        self.area_RDP = area_RDP
        self.area_backyard = area_backyard
        self.area_backyard_2025 = area_backyard_2025
        self.area_RDP_2025 = area_RDP_2025
        self.area_backyard_2040 = area_backyard_2040
        self.area_RDP_2040 = area_RDP_2040
        self.spline_land_backyard = interp1d(year_data_informal, np.transpose([area_backyard, area_backyard, area_backyard_2025, area_backyard_2040]), method)
        self.spline_land_RDP = interp1d(year_data_informal, np.transpose([area_RDP, area_RDP, area_RDP_2025, area_RDP_2040]), method)
        self.spline_estimate_RDP = interp1d(year_data_informal, np.transpose([RDP_houses_estimates, RDP_houses_estimates, RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST), RDP_houses_estimates + np.transpose(construction_rdp.total_yield_DU_ST) + np.transpose(construction_rdp.total_yield_DU_LT)]), method)
        self.coeff_land_private_urban_edge = coeff_land_private_urban_edge
        self.coeff_land_private_no_urban_edge = coeff_land_private_no_urban_edge
        self.coeff_land_backyard = coeff_land_backyard
        self.coeff_land_settlement = coeff_land_settlement
        self.coeff_land_RDP = coeff_land_RDP
        self.coeff_land = np.array([coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP])
        self.housing_limite = housing_limite
        self.housing_limite_politique = housing_limite_politique
        self.spline_land_constraints = land_spline_land_constraints