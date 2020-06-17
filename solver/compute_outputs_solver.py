# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:11:18 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
from numpy import np
from pandas import pd

def coeur_poly2(Uo,param,option,trans_tmp,grille,transaction_cost_in,housing_limite_ici,loyer_de_ref,construction_ici,interest_rate1,revenu1,multi_proba,prix_tc,prix_tc_RDP,coeff_land_ici, coeff_landmax,poly,amenity,solus,uti,type_housing):
    """ Works both for formal or informal housing """
    
    Ro = solus(revenu1(:,1)',Uo)
    Ro(Ro<0) = 0

    basic_q_formal = param["basic_q"]
    if (type_housing == 'backyard') | (type_housing == 'informal'):
        param["basic_q"] = 0   

    #Estimate bid rents using precalculate matrix
    if type_housing == 'formal':
        R_mat = definit_R_formal(Uo, param, trans_tmp, grille, revenu1, amenity, solus, uti)
        R_mat[poly.formal == 0, :] = 0
    elif type_housing == 'backyard':
        amenity = amenity * param["amenity_backyard"]
        R_mat = definit_R_informal(Uo, param, trans_tmp, revenu1, amenity)
        R_mat[poly.backyard == 0,:] = 0
    elif type_housing == 'informal':
        amenity = amenity * param["amenity_settlement"]
        R_mat = definit_R_informal(Uo, param, trans_tmp, revenu1, amenity)
        R_mat[poly.settlement == 0,:] = 0

    #R_mat = single(R_mat)
    R_mat[R_mat < 0] = 0

    #Estimate rents
    R, quel = max(R_mat[:,:],[],1) #quel is the type of households with the highest bid-rent in each location

    #Estimate dwelling size
    temp = [0:size(prix_tc,2) - 1] * size(prix_tc, 1)
    quel_mat = quel + temp 

    #Estimate housing
    if type_housing == 'formal':
        hous = param["coeff_beta"] * (revenu1(quel_mat) - prix_tc(quel_mat)) / R + param["coeff_alpha"] * param["basic_q"]
    elif type_housing == 'backyard':
        hous = param["size_shack"] * np.ones(size(quel_mat))
    elif type_housing == 'informal':
        hous = param["size_shack"] * np.ones(size(quel_mat))

    hous_formal_mat = np.ones(size(np.transpose(Ro))) * hous
    depense_mat = np.ones(size(np.transpose(Ro))) * (hous * R)
    Z = revenu1 - trans_tmp.cout_generalise - depense_mat
    Z[Z<=0] = 0
    utility = utilite_amenite(Z, hous_formal_mat, param, amenity, revenu1, 0)
    utility_max = np.tranpose(Uo) #l'utilit? "max" est constante par centre
    utility_max_mat = utility_max * np.ones(size(R))
    utility = (np.abs(utility)) ** 0.01
    utility_max_mat = (np.abs(utility_max_mat)) ** 0.01 #1000 si 0.01%2000 si 0.005;11000 si 0.001;110000 si 0.0001
    lambda = param["lambda"] * np.ones(size(utility)) #lambda(1:20,:)=lambda(1:20,:)*0.96;%pour 0.01
    proba_log = -(utility_max_mat / utility - 1) * lambda
    
    lieu_zero = np.isnan(proba_log)
    proba_log(lieu_zero) = -100000

    medi = np.max(proba_log,[],1)
    medi[np.isnan(medi)] = 0
    medi(np.isinf(medi)) = 0
    medi = np.ones(size(R_mat,1),1) * medi
    proba_log = proba_log - medi

    #Number of jobs
    proba_log = proba_log+log(double(multi_proba))

    if sum(sum(~isreal(proba_log)))>=1:
        disp('nombres complexes dans proba!! pause !')

    #Exponential form
    proba = np.exp(proba_log)
    proba(Z<=0) = 0

    proba[lieu_zero] = 0
    proba[R_mat<=0] = 0

    #Normalization of the proba
    proba1 = sum(proba,1)
    proba = proba / (np.ones(size(R_mat,1),1) * proba1)
    proba(((np.ones(size(R_mat,1),1) * proba1)) == 0) = 0
    proba = single(proba)

    #Housing construction
    if type_housing == 'formal':
        housing = housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in,loyer_de_ref,interest_rate1)
    elif type_housing == 'backyard':
        housing = 1000000 * housing_backyard(R, grille, param, basic_q_formal, revenu1, prix_tc_RDP)
    elif type_housing == 'informal':
        if option.double_storey_shacks == 0
            housing = 1000000 * np.ones(size(quel_mat))
            housing(R == 0) = 0
        elif option.double_storey_shacks == 1
            housing = 1000000 * housing_informal(R, grille, param, poly, revenu1, prix_tc, proba)

    limite = (revenu1 > prix_tc) & (proba > 0) & (~np.isnan(prix_tc)) & (R_mat > 0)
    proba = proba * limite

    people_init = housing / hous * (sum(limite,1)>0)
    people_init(np.isnan(people_init)) = 0
    people_init_vrai = people_init * coeff_land_ici * 0.5 ** 2

    people_travaille = (np.ones(size(R_mat,1),1) * people_init_vrai) * proba
    people_travaille(np.isnan(people_travaille)) = 0
    job_simul = np.tranpose(sum(people_travaille,2))
    
    if type_housing == 'formal':
        R = np.max(R, transaction_cost_in)

    return job_simul,R,people_init,people_travaille,housing,hous,R_mat
