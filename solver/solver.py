# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:13:29 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
import numpy as np
import pandas as pd

from solver.compute_outputs_solver import *
from solver.useful_functions_solver import *
from data.functions_to_import_data import *

def NEDUM_basic_need_informal(t_ici, trans, option, land, grid, macro_data, param, poly, Uo_perso):
    """ Solver with a Stone-Geary utility function, n income classes and informal housing (settlement and backyard shacks) """

    t_ici = np.array(t_ici)
    revenu_tmp = interp1d((np.array(poly.annee) - param["year_begin"]), np.transpose(poly.avg_inc))
    revenu_tmp = revenu_tmp(t_ici)
    income1 = np.zeros([len(poly.Jx), len(grid.dist), 1])
    for index in range(0, 1):
        temp = np.matlib.repmat(revenu_tmp, 1, grid.dist.shape[0])
        income1[:,:,index] = np.reshape(temp, (len(poly.Jx), len(grid.dist)), order = 'F')  #income for the year
    income_avg = macro_data.spline_revenu(t_ici) #average income for the year of the simulation
    
    #price_trans = prix2_polycentrique2(trans, param, t_ici) #Final price for transport
    price_trans = prix2_polycentrique3(trans.t_transport, trans.cout_generalise, param, t_ici)
    price_trans[price_trans == 0] = np.nan
    price_time = prix2_polycentrique3(trans.t_transport, trans.prix_temps, param, t_ici) #Price associated with the time spent commuting
    price_time[price_time == 0] = np.nan
    
    #Interest rate
    interest_rate_3_years = macro_data.spline_notaires(np.array(list(range((t_ici - 5 + 1), t_ici))))
    interest_rate_3_years[interest_rate_3_years < 0] = np.nan
    interest_rate1 = np.nanmean(interest_rate_3_years) / 100

    #Population
    population = macro_data.spline_population(t_ici) #1.07M households in 2011
    RDP_total = macro_data.spline_RDP(t_ici) #336000 households live in RDP in 2016

    #Construction coefficient
    construction_ici = (income_avg / macro_data.revenu_ref) ** (- param["coeff_b"]) * param["coeff_grandA"] #Correspond en fait exactement au paramètres A (0.69)

    #Evolution of coeff_land
    land_backyard = land.spline_land_backyard(t_ici) #Les zones disponibles pour le backyarding et le logement subventionnées sont données
    land_RDP = land.spline_land_RDP(t_ici)

    coeff_land_private = (land.spline_land_constraints(t_ici) - land_backyard - land.informal - land_RDP) * land.coeff_landmax #Par défaut, toute zone qui n'est pas occupée par des logements informels ou subventionnés est disponible pour le logement formel
    coeff_land_private[coeff_land_private < 0] = 0
    coeff_land_backyard = land_backyard * land.coeff_landmax_backyard
    coeff_land_RDP = land_RDP
    coeff_land_settlement = land.informal * land.coeff_landmax_settlement
    land.coeff_land = np.array([coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP])
    land.RDP_houses_estimates = land.spline_estimate_RDP(t_ici)

    #Limit of housing construction
    #housing_limite_ici = housing_limite_evol(land, option, param, t_ici)
    housing_limite_ici = land.housing_limite
    #if nargin<10
        #housing_in = zeros(1,length(grid.dist));

    transaction_cost_in = (income_avg / macro_data.revenu_ref) * param["transaction_cost2011"] #Transaction cost is the rent at the city limit (per year)
    rent_reference = copy.deepcopy(transaction_cost_in)

    #Tax outside the urban edge
    param["tax_urban_edge_mat"] = np.zeros((1, len(grid.dist)))
    #if option.tax_out_urban_edge == 1
        #param.tax_urban_edge_mat(land.urban_edge == 0) = param.tax_urban_edge .* interest_rate1;

    #Estimation of the limit profit
    housing = construction_ici ** (1 / param["coeff_a"]) * (param["coeff_b"] / (interest_rate1 + param["depreciation_h"])) ** (param["coeff_b"] / param["coeff_a"]) * (rent_reference * 12 - transaction_cost_in) ** (param["coeff_b"] / param["coeff_a"]) * (rent_reference * 12 > transaction_cost_in)
    capital_land = (housing / construction_ici) ** (1 / param["coeff_b"])
    Profit_limite = housing * (rent_reference * 12 - transaction_cost_in * 12) - capital_land * (interest_rate1 + param["depreciation_h"])

    #Computation of the initial state
    etat_initial_erreur, etat_initial_job_simul, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_R_mat, etat_initial_capital_land1, etat_initial_revenu_in, etat_initial_limite1, etat_initial_matrice_J, etat_initial_mult, etat_initial_utility, etat_initial_impossible_population = compute_equilibrium_polycentrique_1_6(option, land, grid, macro_data, param, t_ici, rent_reference, housing_limite_ici, income1, income_avg, price_trans, interest_rate1, population, transaction_cost_in, construction_ici, poly, Profit_limite, Uo_perso, price_time, RDP_total)

    return etat_initial_erreur, etat_initial_job_simul, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_R_mat, etat_initial_capital_land1, etat_initial_revenu_in, etat_initial_limite1, etat_initial_matrice_J, etat_initial_mult, etat_initial_utility, etat_initial_impossible_population

def compute_equilibrium_polycentrique_1_6(option, land, grid, macro_data, param, t_ici, rent_reference, housing_limite_ici, income1, income_avg, price_trans, interest_rate1, population, transaction_cost_in, construction_ici, poly, Profit_limite, Uo_perso, price_time, RDP_total):
    
    max_iter_t = param["max_iter_t"]
    precision = param["precision"]

    #Preparation of the variables
    interest_rate1 = interest_rate1 + param["depreciation_h"]
    t_temp = copy.deepcopy(t_ici)
    param["revenu_ref"] = macro_data.revenu_ref #Je crois que c'est le revenu moyen sur plusieurs années
    Jval = interp1d(poly.annee, np.transpose(poly.Jval[range(0, len(poly.annee)),:]))
    Jval = Jval(t_temp + param["year_begin"]) #Répartition de la population par centre d'emploi
    avg_inc = interp1d(poly.annee, np.transpose(poly.avg_inc[range(0, len(poly.annee)), :]))
    avg_inc = avg_inc(t_temp + param["year_begin"]) #income in each employment center
    poly.income_mult = avg_inc / macro_data.spline_revenu(t_temp) #Inégalités de revenu

    #Class of each center and housing type
    formal_temp = poly.formal #Les 4 classes de ménages peuvent habiter dans le formel
    backyard_temp = poly.backyard #Seules les 2 classes de ménages les plus pauvres peuvent habiter dans des logements informels
    settlement_temp = poly.settlement
    poly.formal = np.zeros((len(avg_inc)))
    poly.backyard = np.zeros((len(avg_inc)))
    poly.settlement = np.zeros((len(avg_inc)))
    poly.classes = poly.classes[1, :]

    for i in range(0, param["multiple_class"]):
        if formal_temp[i] == 1:
            poly.formal[poly.classes == i] = 1   
        if backyard_temp[i] == 1:
            poly.backyard[poly.classes == i] = 1
        if settlement_temp[i] == 1:
            poly.settlement[poly.classes == i] = 1 #Dans quels types de logements peuvent habiter les gens qui travaillent dans les centres d'emploi ?

    #Ajust the population to remove the population in RDP
    
    ratio = population / np.sum(Jval)  
    Jval = Jval * ratio
    RDP_total = RDP_total * ratio
    Jval[poly.classes == 0] = Jval[poly.classes == 0] - RDP_total * Jval[poly.classes == 0] / sum(Jval[poly.classes == 0])
    Jx = poly.Jx
    Jy = poly.Jy
    J = np.array([Jval, Jx, Jy]) #Centres d'emploi et leurs coordonnées
    #J = single(J)

    #We define multi_proba (useful in the solver) here because it is faster
    multi_proba = np.matlib.repmat(np.transpose(J[0,:]), 1, len(grid.dist))
    multi_proba = np.reshape(multi_proba, (len(grid.dist), len(J[0,:])))

    #Commuting price for RDP households (useful for Backyarding)
    #Note: Households in RDP are allocated randomly to job centers
    a = np.empty((4, 24014, 1))
    for i in range(0,4): #Attention, je crois que le 4 correspond au fait qu'il y a 4 centres d'emploi accessibles à la première classe de ménages
        a[i, :] = Jval[poly.classes == 0][i] * price_trans[poly.classes == 0, :][0]
    price_trans_RDP = np.sum(a, 0) / sum(Jval[poly.classes == 0])

    #Amenities
    amenite = land.amenite
    amenite = np.ones((len(income1),1 )) * amenite #We transform amenities in a matrix with as many lines as employment centers

    #Useful functions
    uti = lambda Ro, revenu : utilite(Ro, revenu, param["basic_q"], param) #EQUATION C.2

    decomposition_rent = np.concatenate(([10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2)], np.arange(0.02, 0.081, 0.015), np.arange(0.1, 1.01, 0.02)))
    decomposition_income = np.concatenate(([10 ** (-5), 10 ** (-4), 10 ** (-3.5), 10 ** (-3), 10 ** (-2.5), 10 ** (-2), 0.03], np.arange(0.06, 1.01, 0.02)))

    choice_income = np.max(income1) * decomposition_income
    #income = np.transpose(np.matlib.repmat(choice_income, m = (len(choice_income)), n = 1))
    income = choice_income

    if param["basic_q"]== 0:
        choice_rent = 800 * 12 * decomposition_rent
        #rent = np.matlib.repmat(choice_rent, len(choice_income), 1)
        rent = choice_rent
    else:
        choice_rent = choice_income / param["basic_q"] #le loyer max correspond à celui pour lequel U=0
        rent = np.transpose(choice_rent) * decomposition_rent

    XX = income
    YY = uti(rent, income) #Utilité entre 0 et 15000
    ZZ = rent #Loyers entre 0 et 20 000 - Un chiffre réaliste semble entre 1500 et 2000 par an

    solus = lambda x, y : (griddata((XX,YY), ZZ ** param["coeff_beta"], (x, y))) ** (1 / param["coeff_beta"])

    #smaller grid for speed
    selected_pixels = (sum(land.coeff_land) > 0)
    land.coeff_land = land.coeff_land[:, selected_pixels]
    grid_temp = copy.deepcopy(grid)
    grid.dist = grid.dist[selected_pixels]
    housing_limite_ici = housing_limite_ici[selected_pixels]
    multi_proba = np.transpose(multi_proba[selected_pixels, :])
    price_trans = price_trans[:, selected_pixels]
    price_trans_RDP = price_trans_RDP[selected_pixels]
    price_time = price_time[:, selected_pixels, 0]
    param["housing_mini"] = param["housing_mini"][selected_pixels]
    param["housing_in"] = param["housing_in"][selected_pixels]
    #param.tax_urban_edge_mat = param.tax_urban_edge_mat[selected_pixels]
    income1 = income1[:, selected_pixels, 0]
    amenite = amenite[:, selected_pixels]

    #Estimation of the rent delta
    trans_tmp_cout_generalise = price_trans #Coût monétaire des transports vers chaque centre d'emploi pour la ville réduite
    trans_tmp_delta_loyer = (1 - trans_tmp_cout_generalise[:,:,0] / income1) ** (1 / param["coeff_beta"]) * (income1 > trans_tmp_cout_generalise[:,:,0]) #Revenu disponible puissance beta
    trans_tmp_min_transport = np.min(trans_tmp_cout_generalise, axis = 0) #Coût de transport vers le centre d'emploi le plus proche
    trans_tmp_price_time = price_time #Coût en terme de temps des transports vers chaque centre d'emploi

    #Solving the model

    #Useful variables
    deriv_U = np.zeros((max_iter_t, J.shape[1]))
    people = np.zeros((max_iter_t, 3, len(grid.dist))) #because we have 3 types of housing in the solver
    job_simul = np.zeros((max_iter_t, 3, J.shape[1]))
    job_simul_total = np.zeros((max_iter_t, J.shape[1]))
    rent = np.zeros((max_iter_t, 3, len(grid.dist)))
    val_max = np.zeros((max_iter_t))
    val_max_no_abs = np.zeros((max_iter_t))
    val_moy = np.zeros((max_iter_t))
    nombre = np.zeros((max_iter_t))
    erreur = np.zeros((max_iter_t, J.shape[1]))

    Uo = np.zeros((max_iter_t, J.shape[1])) #utility for each center = key variable that will be adjusted in the solver

    impossible_population = np.zeros((J.shape[1]), 'bool') # = 1 if we cannot reach the objective population
    number_impossible_mem = 0
    condition_possible = np.ones(1, 'bool') #exits the solver if we cannot reach the objective population

    #Definition of Uo
    Uo[0, :] = Uo_perso
    
    index_t = 0
    facteur_convergence_init = 0.025
    param["facteur_convergence"] = copy.deepcopy(facteur_convergence_init)

    #JUSQUE LA ON EST BONS !!!
    
    people_travaille = np.empty((3, 18, 4194))
    housing = np.empty(((3, 4194)))
    hous = np.empty(((3, 4194)))
    R_mat = np.empty((3, 18, 4194))
    
    #Formal housing
    job_simul[index_t, 0, :], rent[index_t, 0, :], people[index_t, 0, :], people_travaille[0, :, :], housing[0, :], hous[0, :], R_mat[0, :, :] = coeur_poly2(Uo[index_t, :], param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, land.coeff_land[0,:], 1, poly, amenite, solus, uti, 'formal')
    
    #Backyard housing
    job_simul[index_t, 1, :], rent[index_t, 1, :], people[index_t, 1, :], people_travaille[1,:,:], housing[1,:], hous[1,:], R_mat[1,:,:] = coeur_poly2(Uo[index_t,:], param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, land.coeff_land[1,:], land.coeff_landmax_backyard, poly, amenite, solus, uti, 'backyard')

    #Informal settlements
    job_simul[index_t, 2, :], rent[index_t, 2, :], people[index_t, 2, :], people_travaille[2, :, :], housing[2, :], hous[2, :], R_mat[2,:,:] = coeur_poly2(Uo[index_t, :], param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, land.coeff_land[2, :], land.coeff_landmax_settlement, poly, amenite, solus, uti, 'informal')

    #Total simulated population
    job_simul_total[index_t, :] = np.sum(job_simul[index_t, :, :], 0)

    #deriv_U will be used to adjust the utility levels
    deriv_U[index_t, :] = np.log((job_simul_total[index_t,:]+10)/(J[0,:]+10))
    deriv_U[index_t, :] = deriv_U[index_t,:] * param["facteur_convergence"]
    deriv_U[index_t, deriv_U[index_t,:] > 0] = deriv_U[index_t, deriv_U[index_t, :] > 0] * 1.1

    #Difference with reality
    erreur[index_t, :] = (job_simul_total[index_t, :] / Jval - 1) * 100
    val_max[index_t] = np.max(np.abs(job_simul_total[index_t, J[1, :]!=0] / Jval[J[1,:] != 0]- 1))
    val_max_no_abs[index_t] = -1
    val_moy[index_t] = np.mean(np.abs(job_simul_total[index_t, J[1,:]!=0]/ (Jval[J[1,:]!=0] + 0.001)-1))
    nombre[index_t] = np.sum(np.abs(job_simul_total[index_t, J[1,:]!=0] / Jval[J[1,:] !=0] - 1) > precision)

    #Memory
    index_memoire = copy.deepcopy(index_t)
    people_travaille_memoire = copy.deepcopy(people_travaille)
    housing_memoire = copy.deepcopy(housing)
    hous_memoire = copy.deepcopy(hous)
    val_moy_memoire = copy.deepcopy(nombre[index_memoire])


    while (index_t < (max_iter_t - 1)) & (val_max[index_t] > precision) & condition_possible:  #&&(val_max_abs>10)%(val_max(index_t)>0.0035)
    
        index_t = index_t + 1
    
        #Adjusting the level of utility
        Uo[index_t, :] = np.exp(np.log(Uo[index_t-1, :]) + deriv_U[index_t-1, :]) #.*income1/5.3910e+004;
    
        #Minimum and maximum levels of utility
        Uo[index_t, Uo[index_t,:] < 0] = 10
        Uo[index_t, impossible_population] = 10 #For the centers for which the objective cannot be attained (impossible_population = 1), utility level is set at an arbitrary low level
        Uo[index_t, J[1,:] == 0] = 10000000
    
        param["facteur_convergence"] = facteur_convergence_init / (1 + 1 * np.abs((job_simul_total[index_t, :] + 10) / (Jval + 10) - 1)) #.*(Jval./mean(Jval)).^0.3 %We adjust the parameter to how close we are from objective 
        param["facteur_convergence"] = param["facteur_convergence"] * (1 - 0.8 * index_t / max_iter_t)
        
        #Formal housing
        job_simul[index_t, 0, :], rent[index_t, 0, :], people[index_t, 0, :], people_travaille[0, :, :], housing[0, :], hous[0, :], R_mat[0, :, :] = coeur_poly2(Uo[index_t, :], param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, land.coeff_land[0, :], 1, poly, amenite, solus, uti, 'formal')
     
        #Backyard housing
        job_simul[index_t, 1, :], rent[index_t, 1, :], people[index_t, 1, :], people_travaille[1, :, :], housing[1, :], hous[1, :], R_mat[1, :, :] = coeur_poly2(Uo[index_t, :], param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, land.coeff_land[1, :], land.coeff_landmax_backyard, poly, amenite, solus, uti, 'backyard')

        #Informal settlements
        job_simul[index_t, 2, :], rent[index_t, 2, :], people[index_t, 2, :], people_travaille[2, :, :], housing[2, :], hous[2, :], R_mat[2, :, :] = coeur_poly2(Uo[index_t, :], param, option, trans_tmp_cout_generalise, grid, transaction_cost_in, housing_limite_ici, rent_reference, construction_ici, interest_rate1, income1, multi_proba, price_trans, price_trans_RDP, land.coeff_land[2, :], land.coeff_landmax_settlement, poly, amenite, solus, uti, 'informal')
     
    
        #Total simulated population
        job_simul_total[index_t, :] = np.sum(job_simul[index_t, :, :], 0)
    
        
        #deriv_U will be used to adjust the utility levels
        deriv_U[index_t, :] = np.log((job_simul_total[index_t, :] + 10) / (J[0, :] + 10))
        deriv_U[index_t, :] = deriv_U[index_t, :] * param["facteur_convergence"]
        deriv_U[index_t, deriv_U[index_t,:] > 0] = deriv_U[index_t, deriv_U[index_t,:] > 0] * 1.1

        #Variables to display
        erreur[index_t, :] = (job_simul_total[index_t, :] / Jval - 1) * 100
        val_max[index_t] = np.max(np.abs(job_simul_total[index_t, J[1,:] !=0] / Jval[J[1,:] != 0] - 1))
        m = np.argmax(np.abs(job_simul_total[index_t, J[1,:] !=0] / Jval[J[1,:] != 0] - 1))
        erreur_temp = (job_simul_total[index_t, J[1,:] !=0] / Jval[J[1,:] != 0] - 1)
        val_max_no_abs[index_t] = erreur_temp[m]
        val_moy[index_t] = np.mean(np.abs(job_simul_total[index_t, J[1, :] != 0] / (Jval[J[1,:] != 0] + 0.001) - 1))
        nombre[index_t] = sum(np.abs(job_simul_total[index_t, J[1, :] != 0] / Jval[J[1,:] != 0] - 1) > precision)
    
        #In case, for one type of households, it is impossible to attain the
        #objective population (basic need effect)
        if ((sum(Uo[index_t, :] < 1) > 0) & (max((job_simul_total[index_t, J[1,:] != 0]) / Jval[J[1,:] != 0] - 1) < precision)):
            impossible_population[Uo[index_t, :] < 1] = np.ones(1, 'bool')
        if (sum(impossible_population) + sum(np.abs(job_simul_total[index_t, J[1,:] != 0] / Jval[J[1,:] != 0] - 1) < precision)) >= len(poly.income_mult): #If we have to stop the solver
            if sum(impossible_population) == number_impossible_mem:
                condition_possible = np.zeros(1, 'bool') #We exit the solver
            else:
                number_impossible_mem = sum(impossible_population) #Gives the centers for which the model could not solve
    
        impossible_population[job_simul_total[index_t,:] > (1 + precision) * Jval] = 0 #In case there are problems with initialization
    
        #The best solution attained is stored in memory
        if nombre[index_t] <= val_moy_memoire:
            index_memoire = copy.deepcopy(index_t)
            people_travaille_memoire = copy.deepcopy(people_travaille)
            val_moy_memoire = copy.deepcopy(nombre[index_memoire])
            housing_memoire = copy.deepcopy(housing)
            hous_memoire = copy.deepcopy(hous)
        
        print(Uo[index_t, :])
        print(erreur[index_t, :])

    #RDP houses
    RDP_people = land.RDP_houses_estimates * RDP_total / sum(land.RDP_houses_estimates)
    RDP_construction = np.matlib.repmat(param["RDP_size"] / (param["RDP_size"] + param["backyard_size"]), 1, len(grid_temp.dist)) * 1000000
    RDP_dwelling_size = np.matlib.repmat(param["RDP_size"], 1, len(grid_temp.dist))
    people_travaille_with_RDP = np.zeros((4, len(poly.income_mult), len(grid_temp.dist)))
    people_travaille_with_RDP[0:3, :, selected_pixels] = people_travaille
    people_travaille_with_RDP[3, poly.classes == 0, :] = np.matlib.repmat(RDP_people, sum(poly.classes == 0), 1) * np.transpose(np.matlib.repmat((Jval[poly.classes == 0]), 24014, 1)) / sum(Jval[poly.classes == 0])

    #Outputs of the solver
    #Employment centers
    etat_initial_erreur = erreur[index_t, :]
    etat_initial_job_simul = job_simul[index_t, :, :]

    #Number of people
    etat_initial_people_housing_type = np.sum(people_travaille_with_RDP, axis = 0)
    etat_initial_people_center = np.sum(people_travaille_with_RDP, axis = 1)
    etat_initial_people1 = people_travaille_with_RDP
        
    #Housing and hous
    housing_export = np.zeros((3, len(grid_temp.dist)))
    hous_export = np.zeros((3, len(grid_temp.dist)))
    housing_export[:, selected_pixels] = housing
    hous_export[:, selected_pixels] = hous
    hous_export[hous_export <= 0] = np.nan
    etat_initial_hous1 = np.array([hous_export, RDP_dwelling_size])
    etat_initial_housing1 = np.array([housing_export, RDP_construction]) 

    #Rents (hh in RDP pay a rent of 0)
    rent_tmp = rent[index_t, :, :]
    rent_tmp_export = np.zeros((3, len(grid_temp.dist)))
    rent_tmp_export[:, selected_pixels] = rent_tmp
    rent_tmp_export[:, selected_pixels == 0] = np.nan
    etat_initial_rent1 = np.array([rent_tmp_export, np.zeros(len(grid_temp.dist))])
    R_mat_export = np.zeros((3, len(poly.Jx), len(grid_temp.dist)))
    R_mat_export[:,:,selected_pixels] = R_mat
    R_mat_export[:,:,selected_pixels == 0] = np.nan
    etat_initial_R_mat = R_mat_export

    #Other outputs
    etat_initial_capital_land1 = (housing / (param["coeff_grandA"])) ** (1/param["coeff_b"])
    etat_initial_revenu_in = income1
    etat_initial_limite1 = (etat_initial_people1 > 1)
    etat_initial_matrice_J = 0
    etat_initial_mult = 0
    etat_initial_utility = Uo[index_t,:]
    etat_initial_impossible_population = impossible_population

    return etat_initial_erreur, etat_initial_job_simul, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_R_mat, etat_initial_capital_land1, etat_initial_revenu_in, etat_initial_limite1, etat_initial_matrice_J, etat_initial_mult, etat_initial_utility, etat_initial_impossible_population

