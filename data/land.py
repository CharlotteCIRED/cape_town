# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:10:27 2020

@author: Charlotte Liotta
"""

from pandas import pd

def import_coeff_land_CAPE_TOWN2(grille, option, param, data_courbe):

    #Coeff max. urbanization
    land.coeff_landmax = param["coeff_landmax"]
    land.coeff_landmax_backyard = param["coeff_landmax_backyard"]
    land.coeff_landmax_settlement = param["coeff_landmax_settlement"]
    area_pixel = (0.5 ** 2) * 1000000

    #Land Cover Data from our estimation (see R code for details)
    grid = pd.read_csv('grid_decoupee_500.csv')
    land.urbanise = np.transpose(urban)/area_pixel
    land.informal = np.transpose(informal)/area_pixel
    land.coeff_land_no_urban_edge = (np.transpose(unconstrained_out) + np.transpose(unconstrained_UE))/area_pixel
    land.coeff_land_urban_edge = np.transpose(unconstrained_UE)/area_pixel
 
    #Here we estimate the number of RDP/BNG dwellings and the area available
    #for backyarding in each subplace using GV2012 data
    land.RDP_houses_estimates = data_courbe.GV_count_RDP
    land.area_RDP = data_courbe.GV_area_RDP * param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
    land.area_backyard = data_courbe.GV_area_RDP * param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]) / area_pixel
    land.coeff_land_backyard = min(land.urbanise, land.area_backyard)

    method = 'linear'

    if (option.future_construction_RDP == 1) 
        #if backyarding is possible in future RDP/BNG settlements

        construction_rdp = pd.read_csv('grid_new_RDP_projects.csv')

        land.area_backyard_2025 = min(param["backyard_size"] / (param["backyard_size"] + param["RDP_size"]), land.RDP_houses_estimates + np.transpose(total_yield_DU_ST) * param["backyard_size"] /area_pixel)
        land.area_RDP_2025 = min(param["RDP_size"] / (param["backyard_size"] + param["RDP_size"]), land.RDP_houses_estimates + np.transpose(total_yield_DU_ST) * param["RDP_size"] / area_pixel)
        land.area_backyard_2040 = min(param["backyard_size_future"] / (param["backyard_size_future"] + param["RDP_size"]), land.RDP_houses_estimates + np.transpose(total_yield_DU_ST + total_yield_DU_LT) .* param["backyard_size_future"] / area_pixel)
        land.area_RDP_2040 = min(param["RDP_size"] / (param["backyard_size_future"] + param["RDP_size"]), land.RDP_houses_estimates + (total_yield_DU_ST + total_yield_DU_LT)' .* param.RDP_size./area_pixel)
        year_data_informal = np.transpose([1990, 2015, 2025, 2040]) - param["year_begin"]
        land.spline_land_backyard = interp1d(year_data_informal,  [land.area_backyard, land.area_backyard, land.area_backyard_2025, land.area_backyard_2040], method, 'pp')
        land.spline_land_RDP = interp1d(year_data_informal,  [land.area_RDP, land.area_RDP, land.area_RDP_2025, land.area_RDP_2040], method, 'pp')
        land.spline_estimate_RDP = interp1d(year_data_informal, [land.RDP_houses_estimates, land.RDP_houses_estimates, land.RDP_houses_estimates + np.transpose(total_yield_DU_ST), land.RDP_houses_estimates + np.transpose(total_yield_DU_ST) + np.transpose(total_yield_DU_LT)], method, 'pp')

    elif option.future_construction_RDP == 1
        #Scenario with no future construction of RDP

        year_data_informal = [1990; 2040]' - param.year_begin;
        land.spline_land_backyard = interp1(year_data_informal,  [land.area_backyard; land.area_backyard], method, 'pp');
        land.spline_land_RDP = interp1(year_data_informal,  [land.area_RDP; land.area_RDP], method, 'pp');
        land.spline_estimate_RDP = interp1(year_data_informal, [land.RDP_houses_estimates; land.RDP_houses_estimates], method, 'pp');


    #Coeff_land for each housing type
    land.coeff_land_private_urban_edge = (land.coeff_land_urban_edge - min(land.area_RDP + land.area_backyard, land.urbanise)) * land.coeff_landmax
    land.coeff_land_private_no_urban_edge = (land.coeff_land_no_urban_edge - land.informal - min(land.area_RDP + land.area_backyard, land.urbanise)) * land.coeff_landmax
    land.coeff_land_private_urban_edge(land.coeff_land_private_urban_edge < 0) = 0
    land.coeff_land_private_no_urban_edge(land.coeff_land_private_no_urban_edge < 0) = 0

    #Evolution of constraints
    if option.urban_edge == 0:
        year_constraints = np.transpose([1990, param.annee_urban_edge - 1, param.annee_urban_edge 2040]) - param["year_begin"]
        land.spline_land_constraints = interp1d(year_constraints, [land.coeff_land_urban_edge, land.coeff_land_urban_edge, land.coeff_land_no_urban_edge, land.coeff_land_no_urban_edge], method, 'pp')
    else:
        year_constraints = np.transpose([1990, 2040]) - param["year_begin"]
        land.spline_land_constraints = interp1d(year_constraints, [land.coeff_land_urban_edge, land.coeff_land_urban_edge], method, 'pp')

    #For the initial state
    if option.urban_edge == 1:
        land.coeff_land_private = land.coeff_land_private_urban_edge
    else 
        land.coeff_land_private = land.coeff_land_private_no_urban_edge

    land.coeff_land_backyard = land.coeff_land_backyard * land.coeff_landmax_backyard
    land.coeff_land_backyard[land.coeff_land_backyard < 0] = 0
    land.coeff_land_settlement = land.informal * land.coeff_landmax_settlement
    land.coeff_land_RDP = np.ones(size(land.coeff_land_private))
    land.coeff_land = [land.coeff_land_private; land.coeff_land_backyard; land.coeff_land_settlement; land.coeff_land_RDP];

    #Building limit
    interieur = (grille.dist <= param["rayon_historique"])
    exterieur = (grille.dist > param["rayon_historique"])
    land.housing_limite = param["taille_limite1"] * 1000000 * interieur + param["taille_limite2"] * 1000000 * exterieur
    land.housing_limite_politique = land.housing_limite

    return land
