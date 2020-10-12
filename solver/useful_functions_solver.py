# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:12:44 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
import numpy as np
import pandas as pd

def construction(param, macro_data, revenu):
    return (revenu / macro_data.revenu_ref) ** (- param["coeff_b"]) * param["coeff_A"]

def transaction_cost(param, macro_data, revenu):
        """ On suppose que le coût de transaction évolue proportionnellement au revenu. """
        return (revenu / macro_data.revenu_ref) * param["transaction_cost2011"]

def housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in,rent_reference,interest_rate1):
    """ Calculates the housing construction as a function of rents """
    if option["ajust_bati"] == 1:
        
        housing = construction_ici ** (1/param["coeff_a"])*(param["coeff_b"]/interest_rate1)**(param["coeff_b"]/param["coeff_a"])*(R)**(param["coeff_b"]/param["coeff_a"]) #Equation 6
        housing[(R < transaction_cost_in) & (~np.isnan(R))] = 0
        #housing(R < transaction_cost_in + param.tax_urban_edge_mat) = 0;
        housing[np.isnan(housing)] = 0
        housing = np.minimum(housing, (np.ones(housing.shape[0]) * np.min(housing_limite_ici)))
    
        #To add the construction on Mitchells_Plan
        housing = np.maximum(housing, param["housing_mini"])
    else:
        housing = param["housing_in"]
    
    return housing

def housing_backyard(R, grid, param, basic_q_formal, income1, price_trans_RDP):
    """ Calculates the backyard available for construction as a function of rents """

    housing = param["coeff_alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["coeff_beta"] * (income1[0,:] - price_trans_RDP) / ((param["backyard_size"]) * R)
    housing[income1[0,:] < price_trans_RDP] = param["coeff_alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["coeff_beta"] * (income1[0, income1[0,:] < price_trans_RDP]) / ((param["backyard_size"]) * R[income1[0,:] < price_trans_RDP])
    housing[R == 0] = 0
    housing = np.minimum(housing, 1)
    housing = np.maximum(housing, 0)

    return housing

def housing_informal(R, grille, param, poly, revenu1, prix_tc, proba):
    """ Calculates the backyard available for construction as a function of rents """

    net_income = sum(proba[poly.classes == 0, :] * (revenu1[poly.classes == 0, :] - prix_tc[poly.classes == 0, :])) / sum(proba[poly.classes == 0, :])
    housing = 1 + param["coeff_alpha"] / param["coeff_mu"] - net_income / R
    housing = np.max(housing, 1)
    housing = np.min(housing, 2)
    housing[R == 0] = 0

    return housing


#def definit_R_formal(Uo,param,trans_tmp_cout_generalise,grille,revenu1,amenite,solus,uti):
    
    """ Stone Geary utility function """
    #if amenite
    #factor_b = (np.matlib.repmat(Uo, n = 1, m = revenu1.shape[1])/ amenite)
    #factor_a = (revenu1) - trans_tmp_cout_generalise
    #R_mat = solus(revenu1 - trans_tmp_cout_generalise, (np.matlib.repmat(Uo, n = 1, m = revenu1.shape[1])/ amenite))
    #return R_mat

def definit_R_informal(Uo,param,trans_tmp_cout_generalise,income,amenity):

    R_mat = 1 / param["size_shack"] * (income - trans_tmp.cout_generalise - (repmat(np.tranpose(Uo),1,np.size(income,2))/(amenity * (param["size_shack"] - param["q0"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"]))
    return R_mat                                                                               

def utilite(Ro,revenu,basic_q,param):
    #Ro = np.transpose(np.matlib.repmat(Ro, n = 1, m = revenu.shape[1]))
    #print(Ro.shape)
    if (basic_q !=0):
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * np.sign(revenu-basic_q * Ro) * np.abs(revenu-basic_q * Ro) / (Ro ** param["coeff_beta"])
        utili[(1 - basic_q * Ro / revenu) < 0] = 0
    else:
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * revenu / (Ro ** param["coeff_beta"])

    utili[revenu==0] = 0
    return utili

def utilite_amenite(Z,hous, param, amenite, revenu,Ro):
    
    if Ro == 0:
        utili = Z ** (param["coeff_alpha"]) * ((hous) - param["q0"]) ** param["coeff_beta"]
    else:
        Ro = np.transpose(np.ones(len(revenu[1,:]), 1) * Ro)
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * np.sign(revenu - param["q0"] * Ro) * np.abs(revenu- param["q0"] * Ro) / (Ro ** param["coeff_beta"]) #Equation C2

    utili = utili * amenite
    utili[revenu==0] = 0
    return utili


def InterpolateIncomeNetOfCommutingCostsEvolution(trans,param,t):
    #computes transport generalized cost for a given year, by interpolation, using variable trans as input
    
    index1, index2, ponder1, ponder2 = CreatePonderation(t + param["baseline_year"], np.array([0, t]))
    return ponder1 * trans.incomeNetOfCommuting[:,:,index1] + ponder2 * trans.incomeNetOfCommuting[:,:,index2]

def CreatePonderation(value, vector):
    vectorCenter = vector - value
    valueMin = np.nanmin(np.abs(vectorCenter))
    index = np.argmin(np.abs(vectorCenter))

    if valueMin == 0:
        index1 = index
        index2 = index
        ponder1 = 1
        ponder2 = 0
    else:
        vecteurNeg = copy.deepcopy(vectorCenter)
        vecteurNeg[vecteurNeg>0] = np.nan
        close1 = np.nanmax(vecteurNeg)
        index1 = np.argmax(vecteurNeg)
    
        vecteurPos = copy.deepcopy(vectorCenter)
        vecteurPos[vecteurPos<0] = np.nan
        close2 = np.nanmin(vecteurPos)
        index2 = np.argmin(vecteurPos)
    
        ponder1 = np.abs(close1)/(close2 - close1)
        ponder2 = 1-ponder1
    
    return index1, index2, ponder1, ponder2

def InterpolateInterestRateEvolution(macro_data, T):
    numberYearsInterestRate = 3
    interestRateNYears = macro_data.interest_rate(np.arange(T - numberYearsInterestRate, T))
    interestRateNYears[interestRateNYears < 0] = np.nan
    interestRate = np.nanmean(interestRateNYears)/100
    return interestRate

def InterpolatePopulationEvolution(macro_data,t):
    return macro_data.population(t)

def InterpolateCoefficientConstruction(option, param, macro_data, income):

    coeff_A = param["coeff_A"]
    coeff_b = param["coeff_b"]

    return (income / macro_data.income_year_reference)**(-coeff_b) *coeff_A

def InterpolateLandCoefficientEvolution(land,option,param,T):
    landBackyard = land.spline_land_backyard(T)
    landRDP = land.spline_land_RDP(T)

    coeffLandPrivate = (land.spline_land_constraints(T) - landBackyard - land.informal - landRDP) * param["max_land_use"]
    coeffLandPrivate[coeffLandPrivate < 0] = 0
    coeffLandBackard = landBackyard * param["max_land_use_backyard"]
    coeffLandRDP = landRDP
    coeffLandSettlement = land.informal * param["max_land_use_settlement"]

    return np.array([coeffLandPrivate, coeffLandBackard, coeffLandSettlement, coeffLandRDP])

def InterpolateHousingLimitEvolution(land, option, param, T):
    return (T + param["baseline_year"] < 2018) * land.housing_limit + (T + param["baseline_year"] >= 2018) * land.housing_limit

def InterpolateAgriculturalRentEvolution(option, param, macro_data, t):
    output = macro_data.agricultural_rent(t)
    coeffKappaT = InterpolateCoefficientConstruction(option, param, macro_data, macro_data.income(t))
    return output ** (param["coeff_a"]) * (param["depreciation_rate"] + InterpolateInterestRateEvolution(macro_data, t)) / (coeffKappaT * param["coeff_b"] ** param["coeff_b"])
