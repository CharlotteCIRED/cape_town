# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:17:47 2020

@author: Charlotte Liotta
"""

import copy
import numpy as np
import math

from solver.compute_outputs_solver import *
from solver.useful_functions_solver import *
from data.functions_to_import_data import *
from solver.solver import *

def nedum_lite_polycentrique_1_6(t, etat_initial_erreur, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_utility, etat_initial_revenu_in, trans, grid, land, job, param, macro_data, option):
    """ Simulations in Cape Town.
    
    Housing evolves dynamically.
    Rents and density at equilibrium.
    !!! Do not forget to change the 
    
    Ne pas oublier de modifier les anticipations pour housing dans variable etat 
    (ne laisser que le cas housing_b i.e. celui sans anticipations).
    Iter_calc_lite : il faut beaucoup moins d'ítérations que dans la méthode d'euler : 
    10 itérations par an semble OK.
    """
    
    #Initialisation
    iter_calc_lite = param["iter_calc_lite"]
    etat_initial_erreur_init = copy.deepcopy(etat_initial_erreur)
    etat_initial_people_housing_type_init = copy.deepcopy(etat_initial_people_housing_type)
    etat_initial_people_center_init = copy.deepcopy(etat_initial_people_center)
    etat_initial_people1_init = copy.deepcopy(etat_initial_people1)
    etat_initial_hous1_init = copy.deepcopy(etat_initial_hous1)
    etat_initial_housing1_init = copy.deepcopy(etat_initial_housing1)
    etat_initial_rent1_init = copy.deepcopy(etat_initial_rent1)
    etat_initial_utility_init = copy.deepcopy(etat_initial_utility)
    etat_initial_revenu_in_init = copy.deepcopy(etat_initial_revenu_in)
    #etat_initial = copy.deepcopy(etat_initial_init)
    
    longueur = len(etat_initial_people1)
    t_calc = np.arange(t[0], t[len(t)-1], (t[1]-t[0])/iter_calc_lite)
    option["ajust_bati"] = 0

    #Matrice de la solution
    simulation_hous = np.zeros((len(t), etat_initial_hous1.shape[0], etat_initial_hous1.shape[1]))
    simulation_rent = np.zeros((len(t), etat_initial_hous1.shape[0], etat_initial_hous1.shape[1]))
    simulation_people = np.zeros((len(t), etat_initial_people1.shape[0], etat_initial_people1.shape[1], etat_initial_people1.shape[2]))
    simulation_housing = np.zeros((len(t), etat_initial_hous1.shape[0], etat_initial_hous1.shape[1]))
    simulation_people_travaille = np.zeros((len(t), etat_initial_people_center.shape[0], etat_initial_people_center.shape[1]))
    simulation_people_housing_type = np.zeros((len(t), etat_initial_people_housing_type.shape[0], etat_initial_people_housing_type.shape[1]))
    simulation_erreur = np.zeros((len(t), etat_initial_erreur.shape[0]))
    simulation_Uo_bis = np.zeros((len(t), etat_initial_utility.shape[0]))
    simulation_deriv_housing = np.zeros((len(t), etat_initial_people_housing_type.shape[0], etat_initial_people_housing_type.shape[1]))
        
    for index in range(0, len(t_calc)):
        t_temp = int(t_calc[index])
        etat_tmp_erreur = copy.deepcopy(etat_initial_erreur)
        etat_tmp_people_housing_type = copy.deepcopy(etat_initial_people_housing_type)
        etat_tmp_people_center = copy.deepcopy(etat_initial_people_center)
        etat_tmp_people1 = copy.deepcopy(etat_initial_people1)
        etat_tmp_hous1 = copy.deepcopy(etat_initial_hous1)
        etat_tmp_housing1 = copy.deepcopy(etat_initial_housing1)
        etat_tmp_rent1 = copy.deepcopy(etat_initial_rent1)
        etat_tmp_utility = copy.deepcopy(etat_initial_utility)
        etat_tmp_revenu_in = copy.deepcopy(etat_initial_revenu_in)
        
        if index > 0:
        
            if index == len(t):
                print('stop')
        
            #On fait d'abord une simulation où on suppose que le bâti s'ajuste librement
            #Cela nous donne la cible housing_b des constructeurs
            option["ajust_bati"] = 1
            rev_temp = revenu2_polycentrique(macro_data, param, option, grid, job, t_temp, 0)
            #rev_temp = np.transpose(rev_temp[:, 0])
            Uobis = etat_tmp_utility / np.transpose(etat_tmp_revenu_in[:, 0]) * rev_temp
            print('Simulation without constraint')
            simul_without_constraint_erreur, simul_without_constraint_job_simul, simul_without_constraint_people_housing_type, simul_without_constraint_people_center, simul_without_constraint_people1, simul_without_constraint_hous1, simul_without_constraint_housing1, simul_without_constraint_rent1, simul_without_constraint_R_mat, simul_without_constraint_capital_land1, simul_without_constraint_revenu_in, simul_without_constraint_limite1, simul_without_constraint_matrice_J, simul_without_constraint_mult, simul_without_constraint_utility, simul_without_constraint_impossible_population = NEDUM_basic_need_informal(t_temp, trans, option, land, grid, macro_data, param, job, Uobis)
            print('*** End of static resolution ***')
            
            option["ajust_bati"] = 0
            
            #Calcul de la dérivée de housing
            #Pour le formel, les constructeurs anticipent un mélange entre le loyer de t et t+1 si la construction est libre
            #Adaptation coeff land
            land_backyard = land.spline_land_backyard(t_calc[index - 1])
            land_RDP = land.spline_land_RDP(t_calc[index - 1])
            coeff_land_private = (land.spline_land_constraints(t_calc[index - 1]) - land_backyard - land.informal - land_RDP) * param["max_land_use"]
            coeff_land_private[coeff_land_private < 0] = 0
            coeff_land_backyard = land_backyard * param["max_land_use_backyard"]
            coeff_land_RDP = land_RDP
            coeff_land_settlement = land.informal * param["max_land_use_settlement"]
            coeff_land = np.array([coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP])

            deriv_housing_temp = evolution_simple_1_0(grid, land, trans, param, macro_data, option, job, t_calc[index - 1], simul_without_constraint_rent1[0, :], simul_without_constraint_rent1[0, :], etat_tmp_housing1[0, :], etat_tmp_people_housing_type[0, :] * coeff_land[0, :] + etat_tmp_people_housing_type[3, :] * coeff_land[3, :])
            param["housing_in"] = etat_tmp_housing1[0, :] + deriv_housing_temp * (t_calc[index] - t_calc[index - 1])
        
            Uo_ici = (simul_without_constraint_utility + Uobis)/2
            #simulation avec le nouveau housing et les nouveaux parametres
            print('Simulation with constraint')
            etat_initial_erreur, etat_initial_job_simul, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_R_mat, etat_initial_capital_land1, etat_initial_revenu_in, etat_initial_limite1, etat_initial_matrice_J, etat_initial_mult, etat_initial_utility, etat_initial_impossible_population  = NEDUM_basic_need_informal(t_temp, trans, option, land, grid, macro_data, param, job, Uo_ici)

            #Ro de la simulation libre
            etat_tmp_Uo_bis = simul_without_constraint_utility
            etat_tmp_deriv_housing = deriv_housing_temp

        else:
            
            etat_tmp_Uo_bis = etat_tmp_utility
            etat_tmp_deriv_housing = np.zeros(len(etat_tmp_rent1[0, :]))
        
        if ((index - 1)/iter_calc_lite) - math.floor((index - 1)/iter_calc_lite) == 0:
        
            simulation_people_travaille[int((index - 1) / iter_calc_lite + 1), :, :] = etat_initial_people_center
            simulation_people_housing_type[int((index - 1) / iter_calc_lite + 1), :, :] = etat_initial_people_housing_type
            simulation_hous[int((index - 1) / iter_calc_lite + 1), :, :] = etat_initial_hous1
            simulation_rent[int((index - 1) / iter_calc_lite + 1), :, :] = etat_initial_rent1
            simulation_people[int((index - 1) / iter_calc_lite + 1), :, :, :] = etat_initial_people1
            simulation_erreur[int((index - 1) / iter_calc_lite + 1), :] = etat_initial_erreur
            simulation_housing[int((index - 1) / iter_calc_lite + 1), :, :] = etat_initial_housing1
            simulation_Uo_bis[int((index - 1) / iter_calc_lite + 1), :] = etat_initial_utility
            simulation_deriv_housing[int((index - 1) / iter_calc_lite + 1), :, :] = etat_tmp_deriv_housing
        
    if len(t) < len(t_calc):
        T = np.transpose(t)
    else:
        T = np.transpose(t_calc)

    option["ajust_bati"] = 1
    etat_initial_erreur = copy.deepcopy(etat_initial_erreur_init)
    etat_initial_people_housing_type = copy.deepcopy(etat_initial_people_housing_type_init)
    etat_initial_people_center = copy.deepcopy(etat_initial_people_center_init)
    etat_initial_people1 = copy.deepcopy(etat_initial_people1_init)
    etat_initial_hous1 = copy.deepcopy(etat_initial_hous1_init)
    etat_initial_housing1 = copy.deepcopy(etat_initial_housing1_init)
    etat_initial_rent1 = copy.deepcopy(etat_initial_rent1_init)
    etat_initial_utility = copy.deepcopy(etat_initial_utility_init)
    etat_initial_revenu_in = copy.deepcopy(etat_initial_revenu_in_init)
    simulation_T = T
    
    return simulation_people_travaille, simulation_people_housing_type, simulation_hous, simulation_rent, simulation_people, simulation_erreur, simulation_housing, simulation_Uo_bis, simulation_deriv_housing, simulation_T



def evolution_simple_1_0(grid, land, trans, param, macro_data, option, job, t, rentA, rentB, housing1, people_formal1):
    #persistent housing_aero
    #persistent housing2008

    interest_rate1 = interest_rate(macro_data, int(t))

    T = np.transpose(t)
    
    #Calcul des inputs dépendant du temps 
    #ATTENTION : equilibre.m est appelé 2 fois : d'abord dans évolution, avec
    #T=t qui est donc un scalaire, puis après la résolution de l'ODE dans
    #NEDUM.m avec T qui est un vecteur de tous les temps où on veut une valeur
    #ce code a donc été écrit pour accepter T comme scalaire ou vecteur

    tps = np.ones(1) #♦LEN t
    matdist = np.ones(len(grid.dist))
    interest_rate1 = tps * interest_rate1
    interest_rate1 = interest_rate1 * matdist

    #housing_limite_ici = housing_limite_evol(land, option, param, T)
    housing_limite_ici = land.housing_limit

    revenu_max = macro_data.spline_revenu(T)
    construction_new = construction(param, macro_data, revenu_max)
    transaction_cost_ici = transaction_cost(param, macro_data, revenu_max)

    construction_new = construction_new * matdist
    transaction_cost_ici = transaction_cost_ici * matdist
    rent_target_out = transaction_cost_ici #loyer en bord de ville pour un an

    rent = copy.deepcopy(rentB)
    rent_target = rent * (rent > rent_target_out)
    capland_target = ((construction_new) * param["coeff_b"] * rent_target / (interest_rate1 + param["depreciation_rate"])) ** (1 / param["coeff_a"])
    housing_target_b = construction_new * capland_target ** param["coeff_b"]

    rent = copy.deepcopy(rentA)
    rent_target = rent * (rent > rent_target_out)
    capland_target = ((construction_new) * param["coeff_b"] * rent_target / (interest_rate1 + param["depreciation_rate"])) ** (1 / param["coeff_a"])
    housing_target_a = construction_new * capland_target ** param["coeff_b"]

    housing_target = housing_target_b * 0.5 + housing_target_a * 0.5 #Résultat principal: pondération entre le cas avec et sans anticipations

    
    housing_target = np.minimum(housing_target, housing_limite_ici) #on limite la hauteur de construction


    #Derivation with inertia
    sortie_housing = 1 / param["time_invest_h"] * (housing_target - housing1) * ((housing_target - housing1) > 0) + param["depreciation_rate"] * (-housing1) * ((housing_target - housing1) <= 0)

    return sortie_housing
