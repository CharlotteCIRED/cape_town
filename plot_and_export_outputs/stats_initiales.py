# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:34:53 2020

@author: Charlotte Liotta
"""

import copy
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def compute_stats_initial_state(trans, land, grid, macro_data, param, option, etat_initial, job, t_ici):

    population = etat_initial_people_housing_type
    total_population = sum(sum(population))

    moy_initial_polycentrique = lambda entree : moy_polycentrique(etat_initial_people_center, entree)
    moy_initial = lambda entree : moy(grid, param, np.ones(len(np.sum(etat_initial_people_housing_type, 1))), np.sum(etat_initial_people_housing_type, 1), np.sum(land.coeff_land, 1), total_population, entree)
    moy_initial_formel = lambda entree : moy(grid, param, np.ones(size(etat_initial_people_housing_type[0, :])), etat_initial_people_housing_type[0,:], land.coeff_land[0,:], total_population, entree)
    
    #cout_monetaire = macro_data.spline_fuel(t_ici) + param["taxe"]
    cout_monetaire = macro_data.spline_fuel(t_ici)
    coeff_land = land.coeff_land

    income = revenu2_polycentrique(macro_data, param, option, grid, job, t_ici)
    mean_income = np.sum(np.transpose(np.matlib.repmat(income, 24014, 1)) * etat_initial_people_center, 1) / np.sum(etat_initial_people_center, 1)
    people_income_group = np.zeros((param["nb_of_income_classes"], len(grid.dist)))
    for i in range (0, param["nb_of_income_classes"]):
        people_income_group[i,:] = np.sum(etat_initial_people_center[job.classes == i, :], 0)

    modal_share = np.empty(5)
    for i in range(0, 5):
        modal_share[i] = np.nansum(etat_initial_people_center[trans.quel[:,:,0] == i]) / total_population

    #distance moyenne effectu?e par mode par m?nage pour un aller
    distance_temp = np.empty((trans.quel.shape[0], trans.quel.shape[1]))
    for i in range(0, trans.quel.shape[0]):
        for j in range(0, trans.quel.shape[1]):
            print(i)
            print(j)
            distance_temp[i, j] = trans.distance_sortie[i, j, int(trans.quel[i, j, 1])]
    distance = sum(distance_temp * etat_initial_people_center / total_population)

    #urbanized_area = np.sum((sum((etat_initial_housing1 > param["max_density"]) * land.coeff_land, 0) / param["max_land_use"]) * 0.25)
    #floor_space = np.sum(sum(etat_initial_housing1 * land.coeff_land, 1) / param["max_land_use"]) * 0.25, 1)
    #floor_space_formal = np.sum(sum(etat_initial_housing1[0, :] * (etat_initial_housing1[0, :] > param["max_density"]) * land.coeff_land[0, :] / param["max_land_use"]) * 0.25, 1)

    return stat_initiales

def moy_polycentrique(people,entree):
    return np.nansum(entree * people, 1) / np.nansum(people, 1)

def moy(grid, param, limite, people, coeff_land_ici, population, entree):
    filtre = (~np.isnan(population)) & (~np.isnan(entree))
    if len(grid.delta_d) == 1:
        sortie = sum(entree[filtre] * limite[filtre] * people[filtre] * coeff_land_ici[filtre] * 0.5 * 0.5 , 2) / population
    else:
        sortie = sum(entree[filtre] * limite[filtre] * population[filtre] * coeff_land_ici[filtre] * 0.5 * 0.5, 2) / population

    return sortie

def somme_sur_ville(grid,param,entree):
    tps = np.ones(size(entree, 1), 1)
    return sum(entree * 0.5 * 0.5, 2)

def revenu2(macro,t):
    return ppval(macro.spline_revenu,t)

def transaction_cost(param,macro,revenu):
    return (revenu / macro.revenu_ref) * param["transaction_cost2011"]

def revenu2_polycentrique(macro, param, option, grid, job, T):
    revenu_tmp = interp1d(np.array(job.annee) - param["baseline_year"], np.transpose(job.avg_inc))#Evolution du revenu
    revenu = revenu_tmp(T)
    return revenu