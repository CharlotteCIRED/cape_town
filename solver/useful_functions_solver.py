# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:12:44 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
from numpy import np
from pandas import pd

def construction(param,macro,revenu):
    return (revenu / macro.revenu_ref) ** (- param["coeff_b"]) * param["coeff_grandA"]

def transaction_cost(param,macro,revenu):
        """ On suppose que le coût de transaction évolue proportionnellement au revenu. """
        return (revenu / macro.revenu_ref) * param["transaction_cost2011"]
def cree_ponder(valeur,vecteur):
        vecteur_centre = vecteur - valeur
        valeur_mini, index = np.min(np.abs(vecteur_centre))

        if valeur_mini == 0:
            index1 = index
            index2 = index
            ponder1 = 1
            ponder2 = 0
        else:
            vecteur_neg = vecteur_centre
            vecteur_neg[vecteur_neg > 0] = np.nan
            rien1, index1 = np.max(vecteur_neg)  
            vecteur_pos = vecteur_centre
            vecteur_pos[vecteur_pos < 0] = np.nan
            rien2, index2 = np.min(vecteur_pos)       
            ponder1 = np.abs(rien1) / (rien2 - rien1)
            ponder2 = 1 - ponder1
            
        return index1, index2, ponder1, ponder2
    
    def prix2_polycentrique3(t_transport,cout_generalise,param,t):
        for index in range(0, len(t)):
            index1, index2, ponder1, ponder2 = cree_ponder(t[index] + param["year_begin"], t_transport)
            sortie[:,:,index] = ponder1 * cout_generalise[:,:,index1] + ponder2 * cout_generalise[:,:,index2]
            return sortie

def housing_construct(R,option,housing_limite_ici,construction_ici,param,transaction_cost_in,loyer_de_ref,interest_rate1,Profit_limite):
    """ Calculates the housing construction as a function of rents """

    housing = construction_ici ** (1/param["coeff_a"])*(param["coeff_b"]/interest_rate1)**(param["coeff_b"]/param["coeff_a"])*(R)**(param["coeff_b"]/param["coeff_a"])
    housing[R < transaction_cost_in] = 0
    #housing(R < transaction_cost_in + param.tax_urban_edge_mat) = 0;
    housing[np.isnan(housing)] = 0
    housing = np.min(housing, (np.ones(size(housing,1),1) * housing_limite_ici))
    
    #To add the construction on Mitchells_Plan
    housing = max(housing, param["housing_mini"])
    
    return housing

def housing_backyard(R, grille, param, basic_q_formal, revenu1, prix_tc_RDP):
    """ Calculates the backyard available for construction as a function of rents """

    housing = param["coeff_alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["coeff_beta"] * (revenu1(1,:) - prix_tc_RDP) / ((param["backyard_size"]) * R)
    housing(revenu1(1,:) < prix_tc_RDP) = param["coeff_alpha"] * (param["RDP_size"] + param["backyard_size"] - basic_q_formal) / (param["backyard_size"]) - param["coeff_beta"] * (revenu1(1, revenu1(1,:) < prix_tc_RDP)) / ((param["backyard_size"]) * R(revenu1(1,:) < prix_tc_RDP))
    housing[R == 0] = 0
    housing = np.min(housing, 1)
    housing = np.max(housing, 0)

    return housing

def housing_informal(R, grille, param, poly, revenu1, prix_tc, proba)
    """ Calculates the backyard available for construction as a function of rents """

    net_income = sum(proba(poly.class == 1, :) * (revenu1(poly.class == 1, :) - prix_tc(poly.class == 1, :))) / sum(proba(poly.class == 1, :))
    housing = 1 + param["coeff_alpha"] / param["coeff_mu"] - net_income / R
    housing = np.max(housing, 1)
    housing = np.min(housing, 2)
    housing[R == 0] = 0

    return housing


def definit_R_formal(Uo,param,trans_tmp,grille,revenu1,amenite,solus,uti):
    """ Stone Geary utility function """
    
    R_mat = solus(revenu1 - double(trans_tmp.cout_generalise), np.tranpose(Uo) * np.ones(1,size(revenu1,2)) / amenite)
    return R_mat

def definit_R_informal(Uo,param,trans_tmp,income,amenity):

    R_mat = 1 / param["size_shack"] * (income - trans_tmp.cout_generalise - (repmat(np.tranpose(Uo),1,np.size(income,2))/(amenity * (param["size_shack"] - param["basic_q"]) ** param["coeff_beta"])) ** (1 / param["coeff_alpha"]))
    return R_mat                                                                               

def utilite(Ro,revenu,basic_q,param):
    
    if (basic_q !=0):
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * np.sign(revenu-basic_q * Ro) * np.abs(revenu-basic_q * Ro) / (Ro ** param["coeff_beta"])
            utili((1 - basic_q * Ro / revenu)<0)=0
    else:
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * revenu / (Ro ** param["coeff_beta"])

    utili[revenu==0] = 0
    return utili

def utilite_amenite(Z,hous, param, amenite, revenu,Ro):
    
    if Ro == 0
        utili = Z ** (param["coeff_alpha"]) * (hous - param("basic_q"]) ** param["coeff_beta"]
    else
        Ro = np.transpose(np.ones(length(revenu(1,:)),1) * Ro)
        utili = param["coeff_alpha"] ** param["coeff_alpha"] * param["coeff_beta"] ** param["coeff_beta"] * np.sign(revenu - param["basic_q"] * Ro) * np.abs(revenu- param["basic_q"] * Ro) / (Ro ** param["coeff_beta"])

    utili = utili * amenite
    utili[revenu==0] = 0
    return utili

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

    toc()
