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
                coeff_landmax, poly, amenite, solus, uti, type_housing):
    
    """ Works both for formal or informal housing """
    
    Ro = solus((income1[:,0]), (np.transpose(Uo))) #D = 18 #n = 55 #XX et YY doivent être (55, 18)
    Ro[Ro < 0] = 0

    basic_q_formal = param["basic_q"]
    if (type_housing == 'backyard') | (type_housing == 'informal'):
        param["basic_q"] = 0   

    #Estimate bid rents using precalculate matrix
    if type_housing == 'formal':
        R_mat = solus(income1 - trans_tmp_cout_generalise[:,:,0], (np.transpose(np.matlib.repmat(Uo, n = 1, m = income1.shape[1]))/ (amenite)))
        R_mat[poly.formal == 0, :] = 0
    elif type_housing == 'backyard':
        amenite = amenite * param["amenite_backyard"]
        #R_mat = definit_R_informal(Uo, param, trans_tmp_cout_generalise, income1, amenite)
        R_mat = 1 / param["size_shack"] * (income1 - trans_tmp_cout_generalise[:,:,0] - (np.transpose(np.matlib.repmat(np.transpose(Uo), n = 1, m=income1.shape[1]))/(amenite * (param["size_shack"] - param["basic_q"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"])) 
        R_mat[poly.backyard == 0,:] = 0
    elif type_housing == 'informal':
        amenite = amenite * param["amenite_settlement"]
        R_mat = 1 / param["size_shack"] * (income1 - trans_tmp_cout_generalise[:,:,0] - (np.transpose(np.matlib.repmat(np.transpose(Uo), n = 1, m=income1.shape[1]))/(amenite * (param["size_shack"] - param["basic_q"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"]))        
        R_mat[poly.settlement == 0,:] = 0

    #R_mat = single(R_mat)
    R_mat[R_mat < 0] = 0

    #Estimate rents
    R = np.nanmax(R_mat, 0) #quel is the type of households with the highest bid-rent in each location
    quel = np.argmax(R_mat, 0) 
    
    #Estimate dwelling size
    quel_mat = np.matlib.repmat(np.zeros(18, 'bool'), n = 1, m = price_trans.shape[1])
    for i in range(0, 4194):
        for j in range(0, 18):
            if quel[i] == j:
                quel_mat[i,j] = np.ones(1, 'bool')
                
    quel_mat = np.transpose(quel_mat)
    
    #Estimate housing
    if type_housing == 'formal':
        #hous = param["coeff_beta"] * (income1[quel_mat] - price_trans[quel_mat]) / R + param["coeff_alpha"] * param["basic_q"]
        hous = param["coeff_beta"] * (income1[quel_mat] - price_trans[:,:,0][quel_mat]) / np.matlib.repmat(R, m = 18, n = 1) + param["coeff_alpha"] * param["basic_q"]
    elif type_housing == 'backyard':
        hous = param["size_shack"] * np.ones((4194))
    elif type_housing == 'informal':
        hous = param["size_shack"] * np.ones((4194))

    #hous_formal_mat = np.ones(np.transpose(Ro).shape) * hous
    #depense_mat = np.ones(np.transpose(Ro).shape) * (hous * R)
    hous_formal_mat = hous
    depense_mat = (hous * R)
    Z = (income1) - (trans_tmp_cout_generalise[:,:,0]) - (depense_mat)
    Z[Z<=0] = 0
    utility = utilite_amenite(Z, hous_formal_mat, param, amenite, income1, 0)
    utility_max = np.transpose(Uo) #l'utilit? "max" est constante par centre
    utility_max_mat = np.matlib.repmat(utility_max, m = len(R), n = 1)
    utility = (np.abs(utility)) ** 0.01
    utility_max_mat = (np.abs(utility_max_mat)) ** 0.01 #1000 si 0.01%2000 si 0.005;11000 si 0.001;110000 si 0.0001
    param_lambda = param["lambda"] * np.ones(utility.shape) #lambda(1:20,:)=lambda(1:20,:)*0.96;%pour 0.01
    proba_log = -(utility_max_mat / np.transpose(utility) - 1) * np.transpose(param_lambda)
    
    lieu_zero = np.isnan(proba_log)
    proba_log[lieu_zero] = -100000

    medi = np.max(proba_log, 0)
    medi[np.isnan(medi)] = 0
    medi[np.isinf(medi)] = 0
    medi = np.matlib.repmat(medi, R_mat.shape[1], 1)
    proba_log = proba_log - medi

    #Number of jobs
    proba_log = proba_log + np.log(np.transpose(multi_proba))

    if sum(sum(~np.isreal(proba_log))) >= 1:
        print('nombres complexes dans proba!! pause !')

    #Exponential form
    proba = np.exp(proba_log)
    proba[np.transpose(Z) <= 0] = 0

    proba[lieu_zero] = 0
    proba[np.transpose(R_mat) <= 0] = 0

    #Normalization of the proba
    proba1 = np.sum(proba,0)
    proba = proba / np.matlib.repmat(proba1, (R_mat).shape[1], 1)
    proba[np.matlib.repmat(proba1, (R_mat).shape[1], 1) == 0] = 0

    #Housing construction
    if type_housing == 'formal':
        housing = housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in, rent_reference,interest_rate1)
    elif type_housing == 'backyard':
        housing = 1000000 * housing_backyard(R, grid, param, basic_q_formal, income1, price_trans_RDP)
    elif type_housing == 'informal':
        if option["double_storey_shacks"] == 0:
            housing = 1000000 * np.ones((4194))
            housing[R == 0] = 0
        elif option["double_storey_shacks"] == 1:
            housing = 1000000 * housing_informal(R, grid, param, poly, income1, price_trans, proba)

    limite = (income1 > price_trans[:,:,0]) & (np.transpose(proba) > 0) & (~np.isnan(price_trans[:,:,0])) & (R_mat > 0)
    proba = np.transpose(proba) * limite

    people_init = housing / hous * (np.sum(limite,0)>0)
    people_init[np.isnan(people_init)] = 0
    people_init_vrai = people_init * coeff_land_ici * 0.5 ** 2

    people_travaille = people_init_vrai * proba
    people_travaille[np.isnan(people_travaille)] = 0
    job_simul = np.sum(people_travaille, axis = 1)
    
    if type_housing == 'formal':
        R = np.maximum(R, transaction_cost_in)

    return job_simul,R,people_init,people_travaille,housing,hous,R_mat
