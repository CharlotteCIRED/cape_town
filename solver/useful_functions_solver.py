# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:12:44 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
import numpy as np
import pandas as pd

def construction(param,macro,revenu):
    return (revenu / macro.revenu_ref) ** (- param["coeff_b"]) * param["coeff_grandA"]

def transaction_cost(param,macro,revenu):
        """ On suppose que le coût de transaction évolue proportionnellement au revenu. """
        return (revenu / macro.revenu_ref) * param["transaction_cost2011"]

def housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in,rent_reference,interest_rate1):
    """ Calculates the housing construction as a function of rents """

    housing = construction_ici ** (1/param["coeff_a"])*(param["coeff_b"]/interest_rate1)**(param["coeff_b"]/param["coeff_a"])*(R)**(param["coeff_b"]/param["coeff_a"]) #Equation 6
    housing[(R < transaction_cost_in) & (~np.isnan(R))] = 0
    #housing(R < transaction_cost_in + param.tax_urban_edge_mat) = 0;
    housing[np.isnan(housing)] = 0
    housing = np.minimum(housing, (np.ones(housing.shape[0]) * np.min(housing_limite_ici)))
    
    #To add the construction on Mitchells_Plan
    housing = np.maximum(housing, param["housing_mini"])
    
    return housing

def housing_backyard(R, grille, param, basic_q_formal, income1, price_trans_RDP):
    """ Calculates the backyard available for construction as a function of rents """

    housing = param["coeff_alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["coeff_beta"] * (income1[0,:] - price_trans_RDP) / ((param["backyard_size"]) * R)
    housing[income1[0,:] < price_trans_RDP[0,:]] = param["coeff_alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["coeff_beta"] * (income1[0, income1[0,:] < price_trans_RDP[0,:]]) / ((param["backyard_size"]) * R[income1[0,:] < price_trans_RDP[0,:]])
    housing[R == 0] = 0
    housing = np.min(housing, 1)
    housing = np.max(housing, 0)

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

    R_mat = 1 / param["size_shack"] * (income - trans_tmp.cout_generalise - (repmat(np.tranpose(Uo),1,np.size(income,2))/(amenity * (param["size_shack"] - param["basic_q"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"]))
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
        utili = Z ** (param["coeff_alpha"]) * ((hous) - param["basic_q"]) ** param["coeff_beta"]
    else:
        Ro = np.transpose(np.ones(len(revenu[1,:]), 1) * Ro)
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * np.sign(revenu - param["basic_q"] * Ro) * np.abs(revenu- param["basic_q"] * Ro) / (Ro ** param["coeff_beta"]) #Equation C2

    utili = utili * amenite
    utili[revenu==0] = 0
    return utili


"""
def courbes_ici(poly,job_simul_total,Jval,grille,rent,index_t,val_max, val_max_no_abs, val_moy,nombre,loyer_de_ref,precision):

    subplot(2,3,1)
    hold on
    plot(val_max(val_max~=0))
    plot(index_t,ylim, 'LineStyle','--', 'Color',[.7 .7 .7] )
    ylim([0 max(val_max)])
    title('max error')
    xlabel('number of iterations')
    ylabel('max error for the centers / abs')

    subplot(2,3,2)
    hold on
    plot(val_moy(val_moy~=0))
    plot(index_t,ylim, 'LineStyle','--', 'Color',[.7 .7 .7] )
    ylim([0 max(val_moy)])
    title('mean error')
    xlabel('number of iterations')
    ylabel('mean error in the employment centers')
    
    subplot(2,3,3)
    hold on
    plot(nombre(val_moy~=0))
    line([index_t index_t], ylim);
    plot(index_t,ylim, 'LineStyle','--', 'Color',[.7 .7 .7] )
    ylim([0 max(nombre)+1])
    title('numbers of errors')
    xlabel('number of iterations')
    ylabel('number of unsolved employment centers')

    subplot(2,3,4)
    hold on
    R_graph(:) = rent(index_t,1,:);
    plot(grille.dist(R_graph>loyer_de_ref),R_graph(R_graph>loyer_de_ref)/12,'.')
    plot(grille.dist(R_graph<=loyer_de_ref),R_graph(R_graph<=loyer_de_ref)/12,'.','Color',[1 0.6 0.6])
    title('monthly rent and Ro');
    xlim([0 max(grille.dist)+1])
    xlabel('distance from the CBD')
    ylabel('loyer (?/m?/mois)')

    subplot(2,3,5)
    hold on
    plot(val_max_no_abs(val_max_no_abs~=0))
    plot(index_t,ylim, 'LineStyle','--', 'Color',[.7 .7 .7] )
    ylim([-1 1])
    title('max error')
    xlabel('number of iteration')
    ylabel('max error / no abs')

    toc() """
