# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 18:13:29 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import copy
from numpy import np
from pandas import pd

from solver.compute_outputs_solver import *
from solver.useful_functions_solver import *

def NEDUM_basic_need_informal(t_ici,trans,option,land,grille,macro,param,poly,Uo_perso,housing_in):
    """ Solver with a Stone-Geary utility function, n income classes and informal housing (settlement and backyard shacks) """

    revenu_tmp = interp1d(poly.annee - param["year_begin"], poly.avg_inc, t_ici)
    income1 = np.zeros([len(poly.Jx),len(grille.dist),len(t_ici)])
    for index in range(0, len(t_ici)):
        income1[:,:,index] = revenu_tmp[:,index] * np.ones(size(grille.dist))  #income for the year
    income_avg = ppval(macro.spline_revenu,t_ici) #average income for the year of the simulation
    
    price_trans = prix2_polycentrique2(trans, param, t_ici) #Final price for transport
    price_trans[price_trans == 0] = np.nan
    price_time = prix2_polycentrique3(trans.t_transport, trans.prix_temps, param, t_ici) #Price associated with the time spent commuting
    price_time[price_time == 0] = np.nan
    
    #Interest rate
    interest_rate_3_years = ppval(macro.spline_notaires, (t_ici - 5 + 1):t_ici)
    interest_rate_3_years[interest_rate_3_years < 0] = np.nan
    interest_rate1 = np.nanmean(interest_rate_3_years) / 100

    #Population
    population = ppval(macro.spline_population, t_ici)
    RDP_total = ppval(macro.spline_RDP, t_ici)

    #Construction coefficient
    construction_ici = (income_avg / macro.revenu_ref) ** (-param["coeff_b"]) * param["coeff_grandA"]

    #Evolution of coeff_land
    land_backyard = ppval(land.spline_land_backyard, t_ici)
    land_RDP = ppval(land.spline_land_RDP, t_ici)

    coeff_land_private = (ppval(land.spline_land_constraints, t_ici) - land_backyard - land.informal - land_RDP) * land.coeff_landmax
    coeff_land_private[coeff_land_private < 0] = 0
    coeff_land_backyard = land_backyard * land.coeff_landmax_backyard
    coeff_land_RDP = land_RDP
    coeff_land_settlement = land.informal * land.coeff_landmax_settlement
    land.coeff_land = [coeff_land_private, coeff_land_backyard, coeff_land_settlement, coeff_land_RDP]
    land.RDP_houses_estimates = ppval(land.spline_estimate_RDP,t_ici)

    #Limit of housing construction
    housing_limite_ici = housing_limite_evol(land, option, param, t_ici)
    #if nargin<10
        #housing_in = zeros(1,length(grille.dist));

    #Transaction cost is the rent at the city limit (per year)
    transaction_cost_in = (income_avg / macro.revenu_ref) * param["transaction_cost2011"]
    rent_reference = copy.deepcopy(transaction_cost_in)

    #Tax outside the urban edge
    param["tax_urban_edge_mat"] = np.zeros(1, length(grille.dist))
    #if option.tax_out_urban_edge == 1
        #param.tax_urban_edge_mat(land.urban_edge == 0) = param.tax_urban_edge .* interest_rate1;

    #Estimation of the limit profit
    housing = construction_ici ** (1 / param["coeff_a"]) * (param["coeff_b"] / (interest_rate1 + param["depreciation_h"])) ** (param["coeff_b"] / param["coeff_a"]) * (rent_reference * 12 - transaction_cost_in) ** (param["coeff_b"] / param["coeff_a"]) * (rent_reference * 12 > transaction_cost_in)
    capital_land = (housing / construction_ici) ** (1 / param["coeff_b"])
    Profit_limite = housing * (rent_reference * 12 - transaction_cost_in * 12) - capital_land * (interest_rate1 + param["depreciation_h"])

    #Computation of the initial state
    etat_initial = compute_equilibrium_polycentrique_1_6(option, land, grille, macro, param, t_ici, rent_reference, housing_limite_ici, income1, income_avg, price_trans, interest_rate1, population, transaction_cost_in, construction_ici, poly, Profit_limite, Uo_perso, price_time, housing_in, RDP_total)

    return etat_initial

def compute_equilibrium_polycentrique_1_6(option, land, grille, macro, param, t_ici, loyer_de_ref, housing_limite_ici, revenu1, revenu_max1, prix_tc, interest_rate1, population, transaction_cost_in, construction_ici, poly, Profit_limite, Uo_init, prix_temps, housing_in, RDP_total):
    
    param["housing_in"] = housing_in

    #Preparation of the variables
    interest_rate1 = interest_rate1 + param["depreciation_h"]
    t_temp = copy.deepcopy(t_ici)
    param["revenu_ref"] = macro.revenu_ref
    Jval = interp1d(poly.annee, poly.Jval(1:len(poly.annee),:), t_temp + param["year_begin"]) #employment centers
    avg_inc = interp1d(poly.annee, poly.avg_inc(1:len(poly.annee), :), t_temp + param["year_begin"]) #income of each class
    poly.income_mult = avg_inc / ppval(macro.spline_revenu, t_temp)

    #Class of each center and housing type
    formal_temp = poly.formal
    backyard_temp = poly.backyard
    settlement_temp = poly.settlement
    poly.formal = np.zeros(1, len(avg_inc))
    poly.backyard = np.zeros(1, len(avg_inc))
    poly.settlement = np.zeros(1, len(avg_inc))
    poly.class = poly.class[1,:]

    for i in range(1, param["multiple_class"]):
        if formal_temp(i) == 1
            poly.formal[poly.class == i] = 1   
        if backyard_temp(i) == 1
            poly.backyard[poly.class == i] = 1
        if settlement_temp(i) == 1
            poly.settlement[poly.class == i] = 1

    #Ajust the population to remove the population in RDP
    ratio = population ./ sum(Jval, 2)
    Jval = Jval * ratio
    RDP_total = RDP_total * ratio
    Jval[poly.class == 1] = Jval[poly.class == 1] - RDP_total * Jval[poly.class == 1] / sum(Jval[poly.class == 1])
    Jx = poly.Jx
    Jy = poly.Jy
    J = [Jval, Jx, Jy]
    #J = single(J)

    #We define multi_proba (useful in the solver) here because it is faster
    multi_proba = (np.transpose(J[1,:]) * np.ones(1,size(grille.dist, 2)))

    #Commuting price for RDP households (useful for Backyarding)
    #Note: Households in RDP are allocated randomly to job centers
    prix_tc_RDP = sum(np.transpose(Jval[poly.class == 1]) * prix_tc[poly.class == 1, :] , 1) / sum(Jval[poly.class == 1])

    #Amenities
    amenite = land.amenite
    amenite = np.ones(size(revenu1,1),1) * amenite #We transform amenities in a matrix with as many lines as employment centers

    #Useful functions
    uti = lambda Ro, revenu : utilite(Ro, revenu, param["basic_q"], param)

    decomposition_rent = np.concatenate(([10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2)], 
                                         np.arange(0.02, 0.081, 0.015), 
                                         np.arange(0.1, 1.01, 0.02)))
    
    decomposition_income = np.concatenate(([10 ** (-5), 10 ** (-4), 10 ** (-3.5), 10 ** (-3), 10 ** (-2.5), 10 ** (-2), 0.03], 
                                           np.arange(0.06, 1.01, 0.02)))

    choice_income = np.max(np.max(revenu1)) * decomposition_income
    income = np.transpose(np.matlib.repmat(choice_income, len(choice_income),1))

    if param.basic_q == 0:
        choice_rent = 800 * 12 * decomposition_rent
        rent = np.matlib.repmat(choice_rent, len(choice_income), 1)
    else:
        choice_rent = choice_income / param["basic_q"] #le loyer max correspond à celui pour lequel U=0
        rent = np.transpose(choice_rent) * decomposition_rent

    XX = income
    YY = uti(rent, income)
    ZZ = rent

    solus = lambda x, y : (griddata(XX,YY,ZZ ** param["coeff_beta"],x,y)) ** (1 / param["coeff_beta"])

    #smaller grid for speed
    selected_pixels = (sum(land.coeff_land) > 0)
    land.coeff_land = land.coeff_land[:,selected_pixels]
    grille_temp = grille
    grille.dist = grille.dist[selected_pixels]
    housing_limite_ici = housing_limite_ici[selected_pixels]
    multi_proba = multi_proba[:,selected_pixels]
    prix_tc = prix_tc[:,selected_pixels]
    prix_tc_RDP = prix_tc_RDP[:,selected_pixels]
    prix_temps = prix_temps[:,selected_pixels]
    param["housing_mini"] = param["housing_mini"][selected_pixels]
    param["housing_in"] = param["housing_in"][selected_pixels]
    #param.tax_urban_edge_mat = param.tax_urban_edge_mat[selected_pixels]
    revenu1 = revenu1[:,selected_pixels]
    amenite = amenite[:,selected_pixels]

    #Estimation of the rent delta
    trans_tmp.cout_generalise = prix_tc
    trans_tmp.delta_loyer = (1 - trans_tmp.cout_generalise / revenu1) ** (1 / param["coeff_beta"]) * (revenu1 > trans_tmp.cout_generalise)
    trans_tmp.min_transport = np.min(trans_tmp.cout_generalise, [], 1)
    trans_tmp.prix_temps = prix_temps

    #Solving the model

    #Useful variables
    deriv_U = np.zeros(max_iter_t, np.size(J,2))
    people = np.zeros(max_iter_t, 3, np.size(grille.dist, 2)) #because we have 3 types of housing in the solver
    job_simul = np.zeros(max_iter_t,3, np.size(J,2))
    job_simul_total = np.zeros(max_iter_t, np.size(J,2))
    rent = zeros(max_iter_t, 3, np.size(grille.dist,2))
    val_max = np.zeros(1, max_iter_t)
    val_max_no_abs = np.zeros(1, max_iter_t)
    val_moy = np.zeros(1, max_iter_t)
    nombre = np.zeros(1, max_iter_t)
    erreur = np.zeros(max_iter_t, size(J,2))

    Uo = np.zeros(max_iter_t,size(J,2)) #utility for each center = key variable that will be adjusted in the solver

    impossible_population = np.zeros((1, size(J,2)), dtype=bool) # = 1 if we cannot reach the objective population
    number_impossible_mem = 0
    condition_possible = np.ones(1, dtype = bool) #exits the solver if we cannot reach the objective population

    #Definition of Uo
    Uo(1,:) = Uo_init
    
    index_t = 1
    facteur_convergence_init = 0.025
    param["facteur_convergence"] = copy.deepcopy(facteur_convergence_init)

    #Formal housing
    job_simul[index_t, 1, :], rent[index_t, 1, :], people[index_t, 1, :], 
    people_travaille[1, :, :], housing[1, :], hous[1, :], R_mat[1, :, :] = 
    coeur_poly2(Uo[index_t, :], param, option, trans_tmp, grille, 
                transaction_cost_in, housing_limite_ici, loyer_de_ref, 
                construction_ici, interest_rate1, revenu1, multi_proba, 
                prix_tc, prix_tc_RDP, land.coeff_land[1,:], 1, poly, amenite, 
                solus, uti, 'formal')

    #Backyard housing
    job_simul[index_t, 2, :], rent[index_t, 2, :], people[index_t, 2, :], 
    people_travaille[2,:,:], housing[2,:], hous[2,:], R_mat[2,:,:] 
    = coeur_poly2(Uo[index_t,:], param, option, trans_tmp, grille, 
                  transaction_cost_in, housing_limite_ici,loyer_de_ref, 
                  construction_ici, interest_rate1, revenu1, multi_proba, 
                  prix_tc, prix_tc_RDP, land.coeff_land[2,:], 
                  land.coeff_landmax_backyard, poly, amenite, solus, uti, 'backyard')

    #Informal settlements
    job_simul[index_t, 3, :], rent[index_t, 3, :], people[index_t, 3, :], 
    people_travaille[3, :, :], housing[3, :], hous[3, :], R_mat[3,:,:]
    = coeur_poly2(Uo[index_t, :], param, option, trans_tmp, grille,
                  transaction_cost_in, housing_limite_ici, loyer_de_ref,
                  construction_ici, interest_rate1, revenu1, multi_proba,
                  prix_tc, prix_tc_RDP, land.coeff_land[3, :], 
                  land.coeff_landmax_settlement, poly, amenite, solus, uti, 'informal')

    #Total simulated population
    job_simul_total[index_t, :] = np.sum(job_simul[index_t, :, :], 2)

    #deriv_U will be used to adjust the utility levels
    deriv_Ui[ndex_t, :] = log((job_simul_total(index_t,:)+10)./(J(1,:)+10));
    deriv_U[index_t, :] = deriv_U(index_t,:).*param.facteur_convergence;
    deriv_U[index_t, deriv_U[index_t,:] > 0] = deriv_U[index_t, deriv_U[index_t, :] > 0] * 1.1

    #Difference with reality
    erreur[index_t, :] = (job_simul_total[index_t, :] / Jval - 1) * 100
    val_max[index_t] = np.max(np.abs(job_simul_total[index_t, J[1, :]!=0) / Jval[J[1,:] != 0 )- 1))
    val_max_no_abs[index_t] = -1
    val_moy[index_t] = np.mean(np.abs(job_simul_total(index_t,J(1,:)~=0)./(Jval(J(1,:)~=0)+0.001)-1))
    nombre[index_t] = np.sum(np.abs(job_simul_total[index_t,J[1,:]!=0] / Jval[J[1,:] !=0] - 1) > precision)

    #Memory
    index_memoire = copy.deepcopy(index_t)
    people_travaille_memoire = people_travaille
    housing_memoire = housing
    hous_memoire = hous
    val_moy_memoire = nombre[index_memoire]


while (index_t < max_iter_t) & (val_max[index_t] > precision) & condition_possible  #&&(val_max_abs>10)%(val_max(index_t)>0.0035)
    
    index_t = index_t + 1
    
    #Adjusting the level of utility
    Uo[index_t, :] = np.exp(np.log(Uo[index_t-1, :]) + deriv_U[index_t-1, :]) #.*revenu1/5.3910e+004;
    
    #Minimum and maximum levels of utility
    Uo[index_t, Uo[index_t,:] < 0] = 10
    Uo[index_t, impossible_population] = 10 #For the centers for which the objective cannot be attained (impossible_population = 1), utility level is set at an arbitrary low level
    Uo[index_t, J[1,:] == 0) = 10000000
    
    param["facteur_convergence"] = facteur_convergence_init / (1 + 1 * np.abs((job_simul_total(index_t, :) + 10) / (Jval + 10) - 1)) #.*(Jval./mean(Jval)).^0.3 %We adjust the parameter to how close we are from objective 
    param["facteur_convergence"] = param["facteur_convergence"] * (1 - 0.8 * index_t / max_iter_t)
        
    #Formal housing
    job_simul[index_t, 1, :], rent[index_t, 1, :], people[index_t, 1, :), people_travaille[1, :, :], housing[1, :], hous[1, :], R_mat[1, :, :] 
                                                          = coeur_poly2(Uo[index_t, :], param, option, trans_tmp, grille, transaction_cost_in, housing_limite_ici, loyer_de_ref, construction_ici, 
                                                                        interest_rate1, revenu1, multi_proba, prix_tc, prix_tc_RDP, land.coeff_land[1, :], 1, poly, amenite, solus, uti, 'formal')
     
    #Backyard housing                                                        
    job_simul[index_t, 2, :], rent[index_t, 2, :], people[index_t, 2, :), people_travaille[2, :, :], housing[2, :], hous[2, :], R_mat[2, :, :] 
                                                          = coeur_poly2(Uo[index_t, :], param, option, trans_tmp, grille, transaction_cost_in, housing_limite_ici, loyer_de_ref, construction_ici, 
                                                                        interest_rate1, revenu1, multi_proba, prix_tc, prix_tc_RDP, land.coeff_land[2, :], land.coeff_landmax_backyard, poly, amenite, solus, uti, 'backyard')

    #Informal settlements
    job_simul[index_t, 3, :], rent[index_t, 3, :], people[index_t, 3, :), people_travaille[3, :, :], housing[3, :], hous[3, :], R_mat[3, :, :] 
                                                          = coeur_poly2(Uo[index_t, :], param, option, trans_tmp, grille, transaction_cost_in, housing_limite_ici, loyer_de_ref, construction_ici, 
                                                                        interest_rate1, revenu1, multi_proba, prix_tc, prix_tc_RDP, land.coeff_land[3, :], land.coeff_landmax_settlement, poly, amenite, solus, uti, 'informal')
     
    
    #Total simulated population
    job_simul_total[index_t, :] = np.sum(job_simul[index_t, :, :], 2)
    
        
    #deriv_U will be used to adjust the utility levels
    deriv_U[index_t, :] = log((job_simul_total(index_t,:)+10)./(J(1,:)+10));
    deriv_U[index_t, :] = deriv_U(index_t,:).*param.facteur_convergence;
    deriv_U[index_t, deriv_U[index_t,:] > 0] = deriv_U[index_t, deriv_U[index_t,:] > 0] * 1.1

    #Variables to display
    erreur[index_t, :] = (job_simul_total[index_t, :] / Jval - 1) * 100
    val_max[index_t, :], m = max(np.abs(job_simul_total[index_t, J[1,:] !=0] / Jval[J[1,:] != 0] - 1))
    erreur_temp = (job_simul_total[index_t, J[1,:] !=0] / Jval[J[1,:] != 0] - 1)
    val_max_no_abs[index_t, :] = erreur_temp[m]
    val_moy[index_t, :] = np.mean(np.abs(job_simul_total[index_t, J[1, :] != 0] / (Jval[J[1,:]!=0]+0.001) - 1))
    nombre[index_t, :] = sum(np.abs(job_simul_total[index_t, J[1, :] != 0) / Jval[J[1,:]!=0] - 1) > precision)
    
    #In case, for one type of households, it is impossible to attain the
    #objective population (basic need effect)
    if ((sum(Uo[index_t, :] < 1) > 0) & (max((job_simul_total[index_t,J[1,:] != 0) / Jval(J[1,:]!=0) - 1)) < precision))/
        impossible_population(Uo[index_t, :] < 1) = np.ones(1, dtype = 'bool')
    if (sum(impossible_population) + sum(np.abs(job_simul_total(index_t,J[1,:]!=0) / Jval(J[1,:]!=0)-1)<precision))>= len(poly.income_mult): #If we have to stop the solver
        if sum(impossible_population) == number_impossible_mem:
            condition_possible = np.zeros(1, dtype = 'bool') #We exit the solver
        else:
           number_impossible_mem = sum(impossible_population) #Gives the centers for which the model could not solve
    
    impossible_population(job_simul_total[index_t,:] > (1 + precision) * Jval) = 0 #In case there are problems with initialization
    
    #The best solution attained is stored in memory
    if nombre[index_t] <= val_moy_memoire:
        index_memoire = index_t
        people_travaille_memoire = people_travaille
        val_moy_memoire = nombre[index_memoire] 
        housing_memoire = housing
        hous_memoire = hous

    #RDP houses
    RDP_people = land.RDP_houses_estimates * RDP_total / sum(land.RDP_houses_estimates)
    RDP_construction = np.matlib.repmat(param.RDP_size / (param["RDP_size"] + param["backyard_size"]), 1, len(grille_temp.dist)) * 1000000
    RDP_dwelling_size = np.matlib.repmat(param.RDP_size, 1, len(grille_temp.dist));
    people_travaille_with_RDP = np.zeros(4, len(poly.income_mult), len(grille_temp.dist))
    people_travaille_with_RDP[1:3, :, selected_pixels] = people_travaille
    people_travaille_with_RDP(4, poly.class == 1, :) = np.matlib.repmat(RDP_people, sum(poly.class == 1), 1) * np.transpose(Jval[poly.class == 1]) / sum(Jval[poly.class == 1])

    #Outputs of the solver

    #Employment centers
    etat_initial.erreur = erreur[index_t,:]
    etat_initial.job_simul[:,:] = job_simul[index_t,:,:]

    #Number of people
    etat_initial.people_housing_type[:,:] = sum(people_travaille_with_RDP, 2)
    etat_initial.people_center[:,:] = sum(people_travaille_with_RDP, 1)
    etat_initial.people1 = people_travaille_with_RDP

    #Housing and hous
    housing_export = np.zeros(3, len(grille_temp.dist))
    hous_export = np.zeros(3, len(grille_temp.dist))
    housing_export[:, selected_pixels] = housing
    hous_export[:, selected_pixels] = hous
    hous_export[hous <= 0] = np.nan
    etat_initial.hous1 = [hous_export, RDP_dwelling_size]
    etat_initial.housing1 = [housing_export, RDP_construction] 

    #Rents (hh in RDP pay a rent of 0)
    rent_tmp[:,:] = rent[index_t, :, :]
    rent_tmp_export = np.zeros(3, len(grille_temp.dist))
    rent_tmp_export[:, selected_pixels] = rent_tmp
    rent_tmp_export[:, selected_pixels == 0] = np.nan
    etat_initial.rent1[:,:] = [rent_tmp_export, np.zeros(1,len(grille_temp.dist))]
    R_mat_export = np.zeros(3, len(poly.Jx), len(grille_temp.dist))
    R_mat_export[:,:,selected_pixels] = R_mat
    R_mat_export[:,:,selected_pixels == 0] = np.nan
    etat_initial.R_mat = R_mat_export

    #Other outputs
    etat_initial.capital_land1 = (housing / (param["coeff_grandA"])) ** (1/param["coeff_b"]);
    etat_initial.revenu_in = revenu1;
    etat_initial.limite1 = (etat_initial.people1 > 1)
    etat_initial.matrice_J = 0
    etat_initial.mult = 0
    etat_initial.utility = Uo[index_t,:]
    etat_initial.impossible_population = impossible_population

    return etat_initial
