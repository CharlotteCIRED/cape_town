# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:34:53 2020

@author: Charlotte Liotta
"""

import copy
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

def compute_stat_initiales_formal_informal(trans, land, grille, macro, param, option, etat_initial, poly, t_ici):

    stat_initiales.people1vrai = etat_initial.people_housing_type  #.* land.coeff_land;
    stat_initiales.population = sum(sum(stat_initiales.people1vrai))
    population = copy.deepcopy(stat_initiales.population)

    moy_initial_polycentrique = lambda entree : moy_polycentrique(etat_initial.people_center, entree)
    moy_initial = lambda entree : moy(grille, param, np.ones(size(sum(etat_initial.people_housing_type,1))), sum(etat_initial.people_housing_type,1), sum(land.coeff_land,1), population, entree)
    moy_initial_formel = lambda entree : moy(grille, param, np.ones(size(etat_initial.people_housing_type[1, :])), etat_initial.people_housing_type[1,:], land.coeff_land[1,:], population, entree)
    
    cout_monetaire = ppval(macro.spline_carburant,t_ici) + param["taxe"]
    stat_initiales.coeff_land1 = land.coeff_land

    revenu1 = revenu2_polycentrique(macro,param,option,grille, poly,t_ici)
    stat_initiales.revenu_moy = sum(revenu1 * etat_initial.people_center, 1) / sum(etat_initial.people_center,1)
    stat_initiales.people_income_group = np.zeros(param["multiple_class"], len(grille.dist))
    for i in range (0, param["multiple_class"] - 1):
        stat_initiales.people_income_group[i,:] = sum(etat_initial.people_center(np.transpose(poly.class[1,:]) == i, :), 1)


    for i = 1:trans.nbre_modes
        stat_initiales.frac_mode(i) = sum(nansum(trans.quel(:,:,i) .* etat_initial.people_center)) ./ sum(nansum(etat_initial.people_center));

    #distance moyenne effectu?e par mode par m?nage pour un aller
    stat_initiales.distance = moy_initial(moy_polycentrique(etat_initial.people_center, sum(trans.quel(:,:,:,1).*repmat(trans.distance_sortie(1,:,:,:),size(trans.quel,1),1),3)));

    #distance moyenne domicile travail (vol d'oiseau) paetout sur la carte
    stat_initiales.commuting_distance = sum(etat_initial.people_center .* trans.distance_sortie(:,:,1), 1) ./ sum(etat_initial.people_center, 1);

    stat_initiales.surf_urba1 = somme_sur_ville(grille,param,sum((etat_initial.housing1>param.borne).*land.coeff_land,1)./land.coeff_landmax);

    #Surfaces planchers construites
    stat_initiales.floor_space1 = somme_sur_ville(grille,param,sum(etat_initial.housing1.*land.coeff_land,1)./land.coeff_landmax);
    #Surfaces planchers formelles
    stat_initiales.floor_space_formal1 = somme_sur_ville(grille,param, etat_initial.housing1(1,:).*(etat_initial.housing1(1,:)>param.borne).*land.coeff_land(1,:)./land.coeff_landmax);

    return stat_initiales

def moy_polycentrique(people,entree):
    return np.nansum(entree * people, 1) / np.nansum(people, 1)

def moy(grille, param, limite, people, coeff_land_ici, population, entree):
    filtre = (~np.isnan(people)) & (~np.isnan(entree))
    if len(grille.delta_d) == 1:
        sortie = sum(entree[filtre] * limite[filtre] * people[filtre] * coeff_land_ici[filtre] * 0.5 * 0.5 , 2) / population
    else:
        sortie = sum(entree[filtre] * limite[filtre] * people[filtre] * coeff_land_ici[filtre] * 0.5 * 0.5, 2) / population

    return sortie

def somme_sur_ville(grille,param,entree):
    tps = np.ones(size(entree, 1), 1)
    return sum(entree * 0.5 * 0.5,2)

def revenu2(macro,t):
    return ppval(macro.spline_revenu,t)

def transaction_cost(param,macro,revenu):
    return (revenu / macro.revenu_ref) * param["transaction_cost2011"]

def revenu2_polycentrique(macro, param, option, grille, poly, T):
    t = np.tranpose(T)
    revenu_tmp = np.transpose(interp1(poly.annee - param["year_begin"], poly.avg_inc, T)) #Evolution du revenu
    revenu = np.zeros([len(poly.Jx), len(grille.dist), len(T)])
    for index in range (0, len(T)):
        revenu[]:, :, index] = revenu_tmp[:, index] * np.ones(size(grille.dist))
    
    return revenu