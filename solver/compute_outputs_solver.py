# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:11:18 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
import numpy as np
import pandas as pd

from solver.useful_functions_solver import *

def coeur_poly2(Uo, param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, 
                construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, coeff_land_ici, 
                coeff_landmax, job, amenite, solus, uti, type_housing, selected_pixels):
    
    """ Works both for formal or informal housing """
    
    Ro = solus((income1[:,0]), (np.transpose(Uo))) #D = 18 #n = 55 #XX et YY doivent être (55, 18)
    Ro[np.isnan(Ro)] = 0
    Ro[Ro < 0] = 0

    basic_q_formal = param["q0"]
    if (type_housing == 'backyard') | (type_housing == 'informal'):
        param["q0"] = 0   

    #Estimate bid rents using precalculate matrix
    if type_housing == 'formal':
        R_mat = solus(income1 - trans_tmp_cout_generalise, (np.transpose(np.matlib.repmat(Uo, n = 1, m = income1.shape[1]))/ (amenite)))
        R_mat[job.formal == 0, :] = 0
    elif type_housing == 'backyard':
        amenite = amenite * param["amenity_backyard"]
        #R_mat = definit_R_informal(Uo, param, trans_tmp_cout_generalise, income1, amenite)
        R_mat = 1 / param["shack_size"] * (income1 - trans_tmp_cout_generalise - (np.transpose(np.matlib.repmat(np.transpose(Uo), n = 1, m=income1.shape[1]))/(amenite * (param["shack_size"] - param["q0"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"])) 
        R_mat[job.backyard == 0,:] = 0
    elif type_housing == 'informal':
        amenite = amenite * param["amenity_settlement"]
        R_mat = 1 / param["shack_size"] * (income1 - trans_tmp_cout_generalise - (np.transpose(np.matlib.repmat(np.transpose(Uo), n = 1, m=income1.shape[1]))/(amenite * (param["shack_size"] - param["q0"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"]))        
        R_mat[job.settlement == 0,:] = 0

    #R_mat = single(R_mat)
    R_mat[R_mat < 0] = 0
    R_mat[np.isnan(R_mat)] = 0

    #Estimate rents
    R = np.nanmax(R_mat, 0) #quel is the type of households with the highest bid-rent in each location
    quel = np.argmax(R_mat, 0) 
    
    #Estimate dwelling size
    quel_mat = np.matlib.repmat(np.zeros(18, 'bool'), n = 1, m = price_trans.shape[1])
    for i in range(0, sum(selected_pixels)):
        for j in range(0, 18):
            if quel[i] == j:
                quel_mat[i,j] = np.ones(1, 'bool')
                
    quel_mat = np.transpose(quel_mat)
    
    #Estimate housing
    if type_housing == 'formal':
        #hous = param["coeff_beta"] * (income1[quel_mat] - price_trans[quel_mat]) / R + param["coeff_alpha"] * param["basic_q"]
        hous = param["coeff_beta"] * ((income1[quel_mat] - price_trans[quel_mat]) / R) + param["coeff_alpha"] * param["q0"] #Demande de logements correspondant au loyer
    elif type_housing == 'backyard':
        hous = param["shack_size"] * np.ones(sum(selected_pixels))
    elif type_housing == 'informal':
        hous = param["shack_size"] * np.ones(sum(selected_pixels))

    hous[hous <  0] = 0
    hous[np.isinf(hous)] = np.nan
    #hous_formal_mat = np.ones(np.transpose(Ro).shape) * hous
    #depense_mat = np.ones(np.transpose(Ro).shape) * (hous * R)
    hous_formal_mat = hous
    depense_mat = (hous * R)
    Z = (income1) - (trans_tmp_cout_generalise) - (depense_mat) #Dépenses en bien composite
    Z[Z<=0] = 0
    utility = utilite_amenite(Z, hous_formal_mat, param, amenite, income1, 0) #On n'a que des nan parce que les logements sont tous plus petits que le basic_Q
    utility_max = np.transpose(Uo) #l'utilit? "max" est constante par centre
    utility_max_mat = np.matlib.repmat(utility_max, m = len(R), n = 1)
    utility = (np.abs(utility)) ** 0.01
    utility_max_mat = (np.abs(utility_max_mat)) ** 0.01 #1000 si 0.01%2000 si 0.005;11000 si 0.001;110000 si 0.0001
    param_lambda = param["lambda"] * np.ones(utility.shape) #lambda(1:20,:)=lambda(1:20,:)*0.96;%pour 0.01
    proba_log = -(utility_max_mat / np.transpose(utility) - 1) * np.transpose(param_lambda)
    
    lieu_zero = np.isnan(proba_log) | np.isinf(proba_log)
    proba_log[lieu_zero] = -100000

    medi1 = np.max(proba_log, 0)
    medi2 = np.max(proba_log, 1)
    medi1[np.isnan(medi1)] = 0
    medi1[np.isinf(medi1)] = 0
    medi1 = np.matlib.repmat(medi1, R_mat.shape[1], 1)
    proba_log1 = proba_log - medi1 #Probabilité que les gens d'un centre d'emploi veuillent habiter à tel endroit
    medi2[np.isnan(medi2)] = 0
    medi2[np.isinf(medi2)] = 0
    medi2 = np.matlib.repmat(medi2, R_mat.shape[0], 1)
    proba_log2 = proba_log - np.transpose(medi2)
    
    #Number of jobs
    proba_log1 = proba_log1 + np.log(np.transpose(multi_proba))
    proba_log2 = proba_log2 + np.log(np.transpose(multi_proba))
    
    if sum(sum(~np.isreal(proba_log2))) >= 1:
        print('nombres complexes dans proba!! pause !')

    #Exponential form
    proba1 = np.exp(proba_log1)
    proba2 = np.exp(proba_log2)
    proba1[np.transpose(Z) <= 0] = 0
    proba2[np.transpose(Z) <= 0] = 0

    proba1[lieu_zero] = 0
    proba1[np.transpose(R_mat) <= 0] = 0
    proba2[lieu_zero] = 0
    proba2[np.transpose(R_mat) <= 0] = 0

    #Normalization of the proba
    proba_1 = np.nansum(proba1,0)
    proba1 = proba1 / np.matlib.repmat(proba_1, (R_mat).shape[1], 1)
    proba1[np.matlib.repmat(proba_1, (R_mat).shape[1], 1) == 0] = 0 #Probabilité que les employés de chaque zone d'emploi habitent dans chaque cellule de la grille

    proba_2 = np.nansum(proba2,1)
    proba2 = proba2 / np.transpose(np.matlib.repmat(proba_2, (R_mat).shape[0], 1))
    proba2[np.transpose(np.matlib.repmat(proba_2, (R_mat).shape[0], 1) == 0)] = 0 #Probabilité que les habitants de chaque cellue travaillent dans chaque zone d'emploi

    #Housing construction
    if type_housing == 'formal':
        housing = housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in, rent_reference,interest_rate1)
    elif type_housing == 'backyard':
        housing = 1000000 * housing_backyard(R, grid, param, basic_q_formal, income1, price_trans_RDP)
    elif type_housing == 'informal':
        if param["double_storey_shacks"] == 0:
            housing = 1000000 * np.ones(sum(selected_pixels))
            housing[R == 0] = 0
        elif param["double_storey_shacks"] == 1:
            housing = 1000000 * housing_informal(R, grid, param, job, income1, price_trans, proba)

    #limite1 = (income1 > price_trans) & (np.transpose(proba1) > 0) & (~np.isnan(price_trans)) & (R_mat > 0)
    #proba1 = np.transpose(proba1) * limite1
    
    limite2 = (income1 > price_trans) & (np.transpose(proba2) > 0) & (~np.isnan(price_trans)) & (R_mat > 0)
    proba2 = np.transpose(proba2) * limite2

    #people_init1 = housing / hous * (np.sum(limite1,0)>0)
    #people_init1[np.isnan(people_init1)] = 0
    #people_init_vrai1 = people_init1 * coeff_land_ici * 0.5 ** 2
    
    people_init2 = housing / hous * (np.sum(limite2,0)>0)
    people_init2[np.isnan(people_init2)] = 0
    people_init_vrai2 = people_init2 * coeff_land_ici * 0.5 ** 2

    #people_travaille1 = people_init_vrai1 * proba1
    #people_travaille1[np.isnan(people_travaille1)] = 0
    #job_simul1 = np.sum(people_travaille1, axis = 1) #Entre people_init_vrai et job_simul, il y a des gens qui se sont perdus ^^'
    
    #people_travaille2 = people_init_vrai2 * proba2
    people_travaille2 = np.matlib.repmat(people_init_vrai2, 18, 1) * proba2
    people_travaille2[np.isnan(people_travaille2)] = 0
    job_simul2 = np.sum(people_travaille2, axis = 1)
    
    if type_housing == 'formal':
        R = np.maximum(R, transaction_cost_in)

    return job_simul2, R, people_init2, people_travaille2, housing, hous, R_mat
    #job_simul: nb of persons per employment center and income group. Most important variable.
    #R: rents (highest bid rentà).
    #R_mat: rents (for each income group and employment center)
    #hous: dwelling sizes.
    #housing: housing supply.
    #people_init: nb of persons per grid cell
    #people_travaille: nb of perso per grid cell working in each employment center
    
    
