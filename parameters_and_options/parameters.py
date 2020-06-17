# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:29:16 2020

@author: Charlotte Liotta
"""

def choice_param():
    
    #Year of the calibration and simulation
    param = {"annee_equilibre" : 2011}
    param["annee_reference"] = 2011
    param["year_begin"] = 2011
    
    #Parameters of the consumption fonction
    param["coeff_beta"] = 0.25
    param["coeff_alpha"] = 1 - param["coeff_beta"]
    param["basic_q"] = 40
    param["mini_lotsize"] = 1
    param["coeff_mu"] = param["coeff_alpha"]
    
    #Parameters of the production fonction
    param["coeff_grandA"] = 0.69
    param["coeff_b"] = 0.55
    param["coeff_a"] = 1-param["coeff_b"]
    param["delay"] = 0
    param["depreciation_h"] = 0.03
    param["interest_rate1"] = 0.0250

    #Transportation costs
    param["facteur_logit_min"] = 6
    param["facteur_logit"] = 3
    param["limite_temps"] = 20.2000;
    param["moyenne_pente_smooth"] = 3;
    param["prix_temps"] = 1
    param["prix_temps2"] = param["prix_temps"] * 0.6
    param["prix_tsport"] = 60

    #Housing limit
    param["rayon_historique"] = 20
    param["taille_limite1"] = 0.5
    param["taille_limite2"] = 0.5

    #Param for dynamic evolution
    param["time_invest_h"] = 10
    param["time_infra_km"] = 1

    #Limit of the city (density)
    param["borne"] = 30000

    param["coeff_sigma"] = 1
    param["ratio"] = 0
    param["iter_calc_lite"] = 1

    #Paramètres de la résolution statique
    param["max_iter_t"] = 400
    param["precision"] = 0.025

    #seuil du nombre d'emplois au-delà duquel on garde le centre d'emploi
    param["seuil_emplois"] = 20000
    param["pas"] = 2

    param["transaction_cost2011"] = 700

    param["lambda"] = 1500

    #Land-use constraints
    param["coeff_landmax"] = 0.7;

    #Multiple income classes
    param["multiple_class"] = 4
    param["income_distribution"] = [0, 1, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4]

    #Households size for transport cost
    param["taille_menage_transport"] = [1.14, 1.94, 1.94, 1.94]
    
    #Informal settlements
    param["size_shack"] = 20 #the size of a backyard shack
    param["RDP_size"] = 40 #in m2 ; the land area occupied by a RDP house
    param["backyard_size"] = 70 #in m2 ; size of the backyard of a RDP house
    param["backyard_size_future"] = param["backyard_size"]
    param["coeff_landmax_backyard"] = 0.45
    param["coeff_landmax_settlement"] = 0.2
    param["amenity_backyard"] = 0.38
    param["amenity_settlement"] = 0.37
    
    #Transportation
    param["metro_waiting_time"] = 10
    param["speed_walking"] = 5 #Average walking speed = 5 km/h

    print('*** Parameters imported succesfully ***')
    
    return param