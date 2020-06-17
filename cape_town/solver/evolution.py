# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:17:47 2020

@author: Charlotte Liotta
"""

import copy
import numpy as np

from solver.compute_outputs_solver import *
from solver.useful_functions_solver import *

def nedum_lite_polycentrique_1_6(t, etat_initial, trans, grille, land, poly, param, macro, option):
    """ Modèle hybride entre le statique et le dynamique.
    
    Seul le bâti évolue dynamiquement. Les loyers et les densités sont à l'équilibre.
    Ne pas oublier de modifier les anticipations pour housing dans variable etat 
    (ne laisser que le cas housing_b i.e. celui sans anticipations).
    Iter_calc_lite : il faut beaucoup moins d'ítérations que dans la méthode d'euler : 
    10 itérations par an semble OK.
    """
    
    #Initialisation
    iter_calc_lite = param["iter_calc_lite"]
    etat_initial_init = copy.deepcopy(etat_initial)
    etat_initial = copy.deepcopy(etat_initial_init)
    longueur = len(etat_initial.people1)
    t_calc = np.arange(t[0], t[len(t)], (t[1]-t[0])/iter_calc_lite)
    option["ajust_bati"] = 0

    #Matrice de la solution
    simulation.hous = np.zeros([len(t), size(etat_initial.hous1)])
    simulation.rent = np.zeros([len(t), size(etat_initial.hous1)])
    simulation.people = np.zeros([len(t), size(etat_initial.people1)])
    simulation.housing = np.zeros([len(t), size(etat_initial.hous1)])

    for indice in range(0, len(t_calc)):
        t_temp = t_calc(indice)
        etat_tmp = etat_initial
    
        if indice > 0:
        
            if indice == len(t):
                print('stop')
        
            #On fait d'abord une simulation où on suppose que le bâti s'ajuste librement
            #Cela nous donne la cible housing_b des constructeurs
            option["ajust_bati"] = 1
            rev_temp = revenu2_polycentrique(macro,param,option,grille,poly,t_temp)
            rev_temp = np.transpose(rev_temp(:,1))
            Uobis = (etat_tmp.utility)./etat_tmp.revenu_in(:,1)'.*rev_temp;
            disp('Simulation without constraint');
            tmpi = NEDUM_basic_need_informal(t_temp,trans,option,land,grille,macro,param,poly,Uobis);
            option["ajust_bati"] = 0
            
            #Calcul de la dérivée de housing
            #Pour le formel, les constructeurs anticipent un mélange entre le loyer de t et t+1 si la construction est libre
            coeff_land = coeff_land_evol(land, option, param, t_calc(indice-1));
            deriv_housing_temp = evolution_simple_1_0(grille,land,trans,param,macro,option,poly,t_calc(indice-1),tmpi.rent1(1,:),tmpi.rent1(1,:),etat_tmp.housing1(1,:), etat_tmp.people_housing_type(1,:).*coeff_land(1,:) + etat_tmp.people_housing_type(4,:).*coeff_land(4,:));
            param.housing_in = etat_tmp.housing1(1,:) + deriv_housing_temp.*(t_calc(indice) - t_calc(indice-1));
        
            Uo_ici = (tmpi.utility + Uobis)/2;
            #simulation avec le nouveau housing et les nouveaux parametres
            disp('Simulation with constraint');
            etat_initial = NEDUM_basic_need_informal(t_temp,trans,option,land,grille,macro,param,poly,Uo_ici,param.housing_in);
       
            #Ro de la simulation libre
            etat_tmp.Uo_bis = tmpi.utility;
            etat_tmp.deriv_housing = deriv_housing_temp;

        else:
            
            etat_tmp.Uo_bis = etat_tmp.utility;
            etat_tmp.deriv_housing = zeros(size(etat_tmp.rent1(1,:)));
        
        if (indice-1)/iter_calc_lite-floor((indice-1)/iter_calc_lite) == 0:
        
            simulation.people_travaille((indice-1)/iter_calc_lite+1,:,:) = etat_initial.people_center;
            simulation.people_housing_type((indice-1)/iter_calc_lite+1,:,:) = etat_initial.people_housing_type;
            simulation.hous((indice-1)/iter_calc_lite+1,:,:) = etat_initial.hous1;
            simulation.rent((indice-1)/iter_calc_lite+1,:,:) = etat_initial.rent1;
            simulation.people((indice-1)/iter_calc_lite+1,:,:,:) = etat_initial.people1;
            simulation.erreur((indice-1)/iter_calc_lite+1,:,:) = etat_initial.erreur;
            simulation.housing((indice-1)/iter_calc_lite+1,:,:) = etat_initial.housing1;
            simulation.Uo_bis((indice-1)/iter_calc_lite+1,:,:) = etat_initial.utility;
            simulation.deriv_housing((indice-1)/iter_calc_lite+1,:,:) = etat_tmp.deriv_housing;
        
    if len(t) < len(t_calc):
        T = np.transpose(t)
    else
        T = np.tranpose(t_calc)

    option["ajust_bati"] = 1
    etat_initial = etat_initial_init
    simulation.T = T
    
    return simulation



def evolution_simple_1_0(grille,land,trans,param,macro,option,poly,t,rentA,rentB,housing1, people_formal1):
    persistent housing_aero
    persistent housing2008

    interest_rate1 = interest_rate(macro, t);

    T = np.tranpose(t)
    
    #Calcul des inputs dépendant du temps 
    #ATTENTION : equilibre.m est appelé 2 fois : d'abord dans évolution, avec
    #T=t qui est donc un scalaire, puis après la résolution de l'ODE dans
    #NEDUM.m avec T qui est un vecteur de tous les temps où on veut une valeur
    #ce code a donc été écrit pour accepter T comme scalaire ou vecteur

    tps = np.ones(size(T))
    matdist = np.ones(size(grille.dist))
    interest_rate1 = tps * interest_rate1
    interest_rate1 = interest_rate1 * matdist

    housing_limite_ici = housing_limite_evol(land,option,param,T)

    revenu_max = ppval(macro.spline_revenu,T)
    construction_new = construction(param, macro, revenu_max)
    transaction_cost_ici = transaction_cost(param, macro, revenu_max)

    construction_new = construction_new * matdist
    transaction_cost_ici = transaction_cost_ici * matdist
    rent_target_out = transaction_cost_ici #loyer en bord de ville pour un an

    rent = rentB
    rent_target = rent * (rent > rent_target_out)
    capland_target = ((construction_new) * param["coeff_b"] * rent_target / (interest_rate1 + param["depreciation_h"])) ** (1 / param["coeff_a"])
    housing_target_b = construction_new * capland_target ** param["coeff_b"]

    rent = rentA
    rent_target = rent * (rent > rent_target_out)
    capland_target = ((construction_new) * param["coeff_b"] * rent_target / (interest_rate1 + param.depreciation_h)) ** (1 / param["coeff_a"])
    housing_target_a = construction_new * capland_target ** param["coeff_b"]

    housing_target = housing_target_b * 0.5 + housing_target_a * 0.5 #Résultat principal: pondération entre le cas avec et sans anticipations

    
    housing_target = min(housing_target, housing_limite_ici) #on limite la hauteur de construction


    #Derivation with inertia
    sortie_housing = 1 / param.time_invest_h * (housing_target - housing1) * ((housing_target - housing1) > 0) + param["depreciation_h"] * (-housing1) * ((housing_target - housing1) <= 0)

    return sortie_housing #real(sortie_housing)
