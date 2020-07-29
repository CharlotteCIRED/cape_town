# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:11:18 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
import numpy as np
import pandas as pd

#from solver.useful_functions_solver import *

def coeur_poly2(Uo, param, option, trans_tmp_cout_generalise, grille, transaction_cost_in, housing_limite_ici, loyer_de_ref, 
                construction_ici, interest_rate1, revenu1, multi_proba, prix_tc, prix_tc_RDP, coeff_land_ici, 
                coeff_landmax, poly, amenity, solus, uti, type_housing):
    
    """ Works both for formal or informal housing """
    
    Ro = solus((revenu1[:,0]), (np.transpose(Uo))) #D = 18 #n = 55 #XX et YY doivent être (55, 18)
    Ro[Ro < 0] = 0

    basic_q_formal = param["basic_q"]
    if (type_housing == 'backyard') | (type_housing == 'informal'):
        param["basic_q"] = 0   

    #Estimate bid rents using precalculate matrix
    if type_housing == 'formal':
        R_mat = solus(revenu1 - trans_tmp_cout_generalise[:,:,0], (np.transpose(np.matlib.repmat(Uo, n = 1, m = revenu1.shape[1]))/ (amenity)))
        R_mat[poly.formal == 0, :] = 0
    elif type_housing == 'backyard':
        amenity = amenity * param["amenity_backyard"]
        #R_mat = definit_R_informal(Uo, param, trans_tmp_cout_generalise, revenu1, amenity)
        R_mat = 1 / param["size_shack"] * (revenu1 - trans_tmp_cout_generalise[:,:,0] - (np.transpose(np.matlib.repmat(np.transpose(Uo), n = 1, m=revenu1.shape[1]))/(amenity * (param["size_shack"] - param["basic_q"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"])) 
        R_mat[poly.backyard == 0,:] = 0
    elif type_housing == 'informal':
        amenity = amenity * param["amenity_settlement"]
        R_mat = 1 / param["size_shack"] * (revenu1 - trans_tmp_cout_generalise[:,:,0] - (np.transpose(np.matlib.repmat(np.transpose(Uo), n = 1, m=revenu1.shape[1]))/(amenity * (param["size_shack"] - param["basic_q"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"]))        
        R_mat[poly.settlement == 0,:] = 0

    #R_mat = single(R_mat)
    R_mat[R_mat < 0] = 0

    #Estimate rents
    R = np.nanmax(R_mat, 0) #quel is the type of households with the highest bid-rent in each location
    quel = np.argmax(R_mat, 0) 
    
    #Estimate dwelling size
    temp = np.matlib.repmat(list(range(0, prix_tc.shape[0])), n = 1, m = prix_tc.shape[1])
    quel_mat = quel + temp 

    #Estimate housing
    if type_housing == 'formal':
        #hous = param["coeff_beta"] * (revenu1[quel_mat] - prix_tc[quel_mat]) / R + param["coeff_alpha"] * param["basic_q"]
        hous = param["coeff_beta"] * (revenu1 - prix_tc[:,:,0]) / np.matlib.repmat(R, m = 18, n = 1) + param["coeff_alpha"] * param["basic_q"]
    elif type_housing == 'backyard':
        hous = param["size_shack"] * np.ones(quel_mat.shape)
    elif type_housing == 'informal':
        hous = param["size_shack"] * np.ones(quel_mat.shape)

    hous_formal_mat = np.ones(np.transpose(Ro).shape) * hous
    depense_mat = np.ones(np.transpose(Ro).shape) * (hous * R)
    Z = (revenu1) - (trans_tmp_cout_generalise) - np.transpose(depense_mat)
    Z[Z<=0] = 0
    utility = utilite_amenite(Z, hous_formal_mat, param, amenity, revenu1, 0)
    utility_max = np.transpose(Uo) #l'utilit? "max" est constante par centre
    utility_max_mat = utility_max * np.ones(R.shape)
    utility = (np.abs(utility)) ** 0.01
    utility_max_mat = (np.abs(utility_max_mat)) ** 0.01 #1000 si 0.01%2000 si 0.005;11000 si 0.001;110000 si 0.0001
    param_lambda = param["lambda"] * np.ones(utility.shape) #lambda(1:20,:)=lambda(1:20,:)*0.96;%pour 0.01
    proba_log = -(np.matlib.repmat(utility_max_mat, n = 1, m= utility.shape[1]) / np.transpose(utility) - 1) * np.transpose(param_lambda)
    
    lieu_zero = np.isnan(proba_log)
    proba_log[lieu_zero] = -100000

    medi = np.max(proba_log, 0)
    medi[np.isnan(medi)] = 0
    medi[np.isinf(medi)] = 0
    medi = np.matlib.repmat(medi, len(R_mat), 1)
    proba_log = proba_log - medi

    #Number of jobs
    proba_log = proba_log + np.log(np.transpose(multi_proba))

    if sum(sum(~np.isreal(proba_log))) >= 1:
        disp('nombres complexes dans proba!! pause !')

    #Exponential form
    proba = np.exp(proba_log)
    proba[np.transpose(Z) <= 0] = 0

    proba[lieu_zero] = 0
    proba[R_mat <= 0] = 0

    #Normalization of the proba
    proba1 = np.sum(proba,0)
    proba = proba / np.matlib.repmat(proba1, len(R_mat), 1)
    proba[(np.ones((len(R_mat), 1)) * proba1) == 0] = 0

    #Housing construction
    if type_housing == 'formal':
        housing = housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in,loyer_de_ref,interest_rate1)
    elif type_housing == 'backyard':
        housing = 1000000 * housing_backyard(R, grille, param, basic_q_formal, revenu1, prix_tc_RDP)
    elif type_housing == 'informal':
        if option["double_storey_shacks"] == 0:
            housing = 1000000 * np.ones(quel_mat.shape)
            housing[R == 0] = 0
        elif option["double_storey_shacks"] == 1:
            housing = 1000000 * housing_informal(R, grille, param, poly, revenu1, prix_tc, proba)

    limite = (revenu1 > prix_tc) & (proba > 0) & (~np.isnan(prix_tc)) & (R_mat > 0)
    proba = proba * limite

    people_init = housing / hous * (sum(limite,1)>0)
    people_init[np.isnan(people_init)] = 0
    people_init_vrai = people_init * coeff_land_ici * 0.5 ** 2

    people_travaille = (np.ones(len(R_mat, 1), 1) * people_init_vrai) * proba
    people_travaille[np.isnan[people_travaille]] = 0
    job_simul = np.tranpose(np.sum(people_travaille, axis = 2))
    
    if type_housing == 'formal':
        R = np.max(R, transaction_cost_in)

    return job_simul,R,people_init,people_travaille,housing,hous,R_mat
