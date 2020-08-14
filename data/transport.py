# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:15:09 2020

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd
from data.functions_to_import_data import *
import numpy.matlib
from scipy.interpolate import griddata
import scipy.io
import copy

class TransportData:
        
    def __init__(self):
        
        self
        
    def import_transport_data(self, option, grid, macro_data, param, job, t_trafic):
        """ Compute travel times and costs """

        referencement = job.referencement
        complement_trajet_voiture = 0
        complement_trajet_pieds = 0
        complement_trajet_TC = 0
        trans_nbre_modes = 5

        #Import distances and durations by subway
        distance_metro_2, duration_metro_2 = import_metro_data(job, grid, param)

        #Load distance and duration with each transportation mode
        transport_time_grid = scipy.io.loadmat('./2. Data/Basile data/Transport_times_GRID.mat')
        transport_time_sp = scipy.io.loadmat('./2. Data/Basile data/Transport_times_SP.mat')
        
        distance_car = transport_time_grid["distanceCar"] #devrait faire len(job.corresp). Pour le moment, matrice OD 185 * 20014 (car on garde 185 centres d'emploi)
        duration_car = transport_time_grid["durationCar"]
        distance_metro = transport_time_grid["distanceCar"]
        duration_metro = transport_time_grid["durationTrain"] #Est-ce que train = métro ?
        duration_minibus = transport_time_grid["durationMinibus"]
        duration_bus = transport_time_grid["durationBus"]
        
        if option["polycentric"] == 0:
            distance_car = distance_car[135]
            duration_car = duration_car[135]
            distance_metro = distance_metro[135]
            duration_metro = duration_metro[135]
            duration_minibus = duration_minibus[135]
            duration_bus = duration_bus[135]
            nb_center = 1
        elif option["polycentric"] == 1:
            distance_car = distance_car[(10, 34, 40, 108, 135, 155),:]
            duration_car = duration_car[(10, 34, 40, 108, 135, 155),:]
            distance_metro = distance_metro[(10, 34, 40, 108, 135, 155),:]
            duration_metro = duration_metro[(10, 34, 40, 108, 135, 155),:]
            duration_minibus = duration_minibus[(10, 34, 40, 108, 135, 155),:]
            duration_bus = duration_bus[(10, 34, 40, 108, 135, 155),:]
            nb_center = 6
            
        #Monocentric : 135
        
        #Polycentric
        #5101 (CBD): 135
        #2002 (Bellville): 40
        #1201 (Epping): 10
        #1553 (Claremont): 34
        #3509 (Sommerset West): 108
        #5523 (Table View + Century City): 165
        
        #Points where all data are available                                     
        trans_reliable = np.ones(len(grid.dist))
        if option["polycentric"] == 0:
            trans_reliable[np.isnan(duration_car)] = 0
            trans_reliable[np.isnan(duration_metro)] = 0
            trans_reliable[np.isnan(duration_minibus)] = 0
            trans_reliable[np.isnan(duration_bus)] = 0
        elif option["polycentric"] == 1:
            trans_reliable = np.ones((nb_center, len(grid.dist)))
            for i in range(0, nb_center):
                trans_reliable[np.isnan(duration_car[:,i])] = 0
                trans_reliable[np.isnan(duration_metro[:,i])] = 0
                trans_reliable[np.isnan(duration_minibus[:,i])] = 0
                trans_reliable[np.isnan(duration_bus[:,i])] = 0
        
        #Define time and distance variables
        if option["polycentric"] == 1:
            LongueurTotale_VP = np.array([distance_car[0], distance_car[0], distance_car[0], distance_car[0], distance_car[1], distance_car[1], distance_car[1], distance_car[2], distance_car[2], distance_car[2], distance_car[2], distance_car[4], distance_car[4], distance_car[4], distance_car[4], distance_car[5], distance_car[5], distance_car[5]])
            LongueurTotale_VP = LongueurTotale_VP * 1.2
            LongueurEnVehicule_TC = distance_metro_2 #pour le coût en métro, les coûts dépendent de la distance à la gare centrale du Cap (LongueurEnVehicule_TC n'est donc pas la distance parcourue en TC)
        else:
            if param["nb_of_income_classes"] == 4: 
                LongueurTotale_VP = np.array([distance_car, distance_car, distance_car, distance_car])
                LongueurTotale_VP = LongueurTotale_VP * 1.2
                LongueurEnVehicule_TC = distance_metro_2
            elif param["nb_of_income_classes"] == 12: 
                LongueurTotale_VP = np.array([distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car])
                LongueurTotale_VP = LongueurTotale_VP * 1.2
                LongueurEnVehicule_TC = distance_metro_2
            
        increment = range(0, len(job.quel))
        
        #Public transportation price
        prix_metro_2012_km = 1.5 / 40 * macro_data.spline_fuel(2012 - param["baseline_year"]) / macro_data.spline_inflation(2015 - param["baseline_year"])  #0.164
        prix_metro_2012_fixe_mois = 121.98 * macro_data.spline_inflation(2012 - param["baseline_year"]) / macro_data.spline_inflation(2015 - param["baseline_year"]) #4.48*40
        prix_taxi_2012_km = 0.785
        prix_taxi_2012_fixe_mois = 4.32 * 40
        prix_bus_2012_km = 0.522
        prix_bus_2012_fixe_mois = 6.24 * 40

        #Correct for inflation
        inflation = macro_data.spline_inflation(t_trafic)
        infla_2012 = macro_data.spline_inflation(2012 - param["baseline_year"])
        prix_metro_km = prix_metro_2012_km * inflation / infla_2012
        prix_metro_fixe_mois = prix_metro_2012_fixe_mois * inflation / infla_2012
        prix_taxi_km = prix_taxi_2012_km * inflation / infla_2012
        prix_taxi_fixe_mois = prix_taxi_2012_fixe_mois * inflation / infla_2012
        prix_bus_km = prix_bus_2012_km * inflation / infla_2012
        prix_bus_fixe_mois = prix_bus_2012_fixe_mois * inflation /infla_2012

        #Fixed price for private cars
        prix_fixe_vehicule_mois_2012 = 350
        prix_fixe_vehicule_mois = prix_fixe_vehicule_mois_2012 * inflation / infla_2012
        prix_essence = macro_data.spline_fuel(t_trafic)
        prix_essence_mois = np.zeros(prix_essence.shape)
        prix_essence_mois = prix_essence * 2 * 20
        
        if option["polycentric"] == 1:
            nb_employment_center = 18
        else:
            if param["nb_of_income_classes"] == 4:
                nb_employment_center = 4
            elif param["nb_of_income_classes"] == 12:
                nb_employment_center = 12
    
        #Transport times
        if option["polycentric"] == 1:
            TEMPSHPM = np.array([duration_car[0], duration_car[0], duration_car[0], duration_car[0], duration_car[1], duration_car[1], duration_car[1], duration_car[2], duration_car[2], duration_car[2], duration_car[2], duration_car[4], duration_car[4], duration_car[4], duration_car[4], duration_car[5], duration_car[5], duration_car[5]])
            TEMPS_MINIBUS = np.array([duration_minibus[0], duration_minibus[0], duration_minibus[0], duration_minibus[0], duration_minibus[1], duration_minibus[1], duration_minibus[1], duration_minibus[2], duration_minibus[2], duration_minibus[2], duration_minibus[2], duration_minibus[4], duration_minibus[4], duration_minibus[4], duration_minibus[4], duration_minibus[5], duration_minibus[5], duration_minibus[5]])
            TEMPS_BUS = np.array([duration_bus[0], duration_bus[0], duration_bus[0], duration_bus[0], duration_bus[1], duration_bus[1], duration_bus[1], duration_bus[2], duration_bus[2], duration_bus[2], duration_bus[2], duration_bus[4], duration_bus[4], duration_bus[4], duration_bus[4], duration_bus[5], duration_bus[5], duration_bus[5]])
            TEMPSTC = duration_metro_2 #duration_metro
        else:
            if param["nb_of_income_classes"] == 4:
                TEMPSHPM = np.array([duration_car, duration_car, duration_car, duration_car])
                TEMPS_MINIBUS = np.array([duration_minibus, duration_minibus, duration_minibus, duration_minibus])
                TEMPS_BUS = np.array([duration_bus, duration_bus, duration_bus, duration_bus])
                TEMPSTC = duration_metro_2 #duration_metro
            elif param["nb_of_income_classes"] == 12:
                TEMPSHPM = np.array([duration_car, duration_car, duration_car, duration_car, duration_car, duration_car, duration_car, duration_car, duration_car, duration_car, duration_car, duration_car])
                TEMPS_MINIBUS = np.array([duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus, duration_minibus])
                TEMPS_BUS = np.array([duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus, duration_bus])
                TEMPSTC = duration_metro_2 #duration_metro
        
        temps_pieds_temp = (LongueurTotale_VP) / param["walking_speed"] * 60 + complement_trajet_pieds
        temps_pieds_temp[np.isnan(TEMPSHPM)] = np.nan #si on ne fait pas ça, on a des 0 au lieu d'avoir des nan
        temps_pieds_temp = pd.DataFrame(temps_pieds_temp)
        temps_sortie = np.empty((nb_employment_center, 24014, 5))
        temps_sortie[:,:,0] = temps_pieds_temp #temps pour rejoindre à pieds
        temps_sortie[:,:,1] = pd.DataFrame(TEMPSTC + complement_trajet_TC) #temps en TC
        temps_sortie[:,:,2] = pd.DataFrame(TEMPSHPM + complement_trajet_voiture) #temps en voiture
        temps_sortie[:,:,3] = pd.DataFrame(TEMPS_MINIBUS + complement_trajet_TC) #temps en minibus-taxis
        temps_sortie[:,:,4] = pd.DataFrame(TEMPS_BUS + complement_trajet_TC) #temps en bus
        #temps_sortie=single(temps_sortie);

        #Interpolate with public transport time - function of the number of km
        mult_prix_sortie = np.empty((nb_employment_center, 24014, 5))
        mult_prix_sortie[:,:,0] = np.zeros((mult_prix_sortie[:,:,0]).shape)
        mult_prix_sortie[:,:,1] = pd.DataFrame(LongueurEnVehicule_TC)
        mult_prix_sortie[:,:,2] = pd.DataFrame(LongueurTotale_VP)
        mult_prix_sortie[:,:,3] = pd.DataFrame(LongueurTotale_VP)
        mult_prix_sortie[:,:,4] = pd.DataFrame(LongueurTotale_VP)

        if option["polycentric"] == 1:
            prix_sortie_unitaire = np.empty((prix_essence_mois.shape[0], 5))
            prix_sortie_unitaire[:,0] = np.ones(prix_essence_mois.shape)
        else:
            if param["nb_of_income_classes"] == 4:
                prix_sortie_unitaire = np.empty((prix_essence_mois.shape[0], 5))
                prix_sortie_unitaire[:,0] = np.ones(prix_essence_mois.shape)
            else: 
                prix_sortie_unitaire = np.empty((1, 5))
                prix_sortie_unitaire[:,0] = np.ones(1)
        prix_sortie_unitaire[:,1] = prix_metro_km * 20 * 2 * 12
        prix_sortie_unitaire[:,2] = prix_essence_mois * 12
        prix_sortie_unitaire[:,3] = prix_taxi_km * 2 * 20 * 12
        prix_sortie_unitaire[:,4] = prix_bus_km * 2 * 20 * 12

        #Distance
        distance_sortie = np.empty((nb_employment_center, 24014, 5))
        distance_sortie[:,:,0] = pd.DataFrame(LongueurTotale_VP)
        distance_sortie[:,:,1] = pd.DataFrame(LongueurEnVehicule_TC)
        distance_sortie[:,:,2] = pd.DataFrame(LongueurTotale_VP)
        distance_sortie[:,:,3] = pd.DataFrame(LongueurTotale_VP)
        distance_sortie[:,:,4] = pd.DataFrame(LongueurTotale_VP)

        trans_distance_sortie = distance_sortie #trans.distance_sortie = single(distance_sortie)
        prix_monetaire = np.zeros((nb_employment_center, 24014, trans_nbre_modes))
        if param["nb_of_income_classes"] == 12:
            cout_generalise = (np.zeros((nb_employment_center, 24014, 1)))
            quel = (np.zeros((nb_employment_center, 24014, 1)))
        else:
            cout_generalise = (np.zeros((len(job.code_emploi_poly[referencement]), 24014, len(t_trafic))))
            quel = (np.zeros((len(job.code_emploi_poly[referencement]), 24014, len(t_trafic))))

        mult = cout_generalise
        cout_generalise_ancien = cout_generalise
        tbis = t_trafic

        taille_menage_mat = np.matlib.repmat(param["household_size"], 1, int(len(job.code_emploi_init) / param["nb_of_income_classes"]))
        taille_menage_mat = np.matlib.repmat(np.transpose(taille_menage_mat.squeeze()[job.quel]), 1, len(grid.dist)) #pour prendre en compte des tailles de ménages différentes
        taille_menage_mat = np.reshape(taille_menage_mat, (len(grid.dist), nb_employment_center)) #24014 * 4
        
        if isinstance(tbis, int):
            a = 1
        else:
            a = len(tbis)
        #compute transport price for each year
        for index in range(0, a):
            for index2 in range(0, trans_nbre_modes):
                prix_monetaire[:,:,index2] = prix_sortie_unitaire[index, index2] * mult_prix_sortie[:, :, index2] #On multiplie le prix par km par la distance pour avoir le prix sur le total du trajet
                prix_monetaire[:,:,index2] = prix_monetaire[:, :, index2] * np.transpose(taille_menage_mat) #On multiplie le tout par le nombre de personne par ménage, qui varie selon la classe de ménage (d'où le format 24014 * 4 * 5)
            #trans_cout_generalise = copy.deepcopy(prix_monetaire) #sert juste pour mettre dans la fonction revenu2_polycentrique          
            
            #add fixed costs
            if a > 1:
                prix_monetaire[:,:,1] = prix_monetaire[:,:,1] + prix_metro_fixe_mois[index] * 12 * np.transpose(taille_menage_mat) #train, avec abonnement mensuel
                prix_monetaire[:,:,2] = prix_monetaire[:,:,2] + prix_fixe_vehicule_mois[index] * 12 * np.transpose(taille_menage_mat) #voiture
                prix_monetaire[:,:,3] = prix_monetaire[:,:,3] + prix_taxi_fixe_mois[index] * 12 * np.transpose(taille_menage_mat) #minibus-taxi
                prix_monetaire[:,:,4] = prix_monetaire[:,:,4] + prix_bus_fixe_mois[index] * 12 * np.transpose(taille_menage_mat) #bus
            else:
                prix_monetaire[:,:,1] = prix_monetaire[:,:,1] + prix_metro_fixe_mois * 12 * np.transpose(taille_menage_mat) #train, avec abonnement mensuel
                prix_monetaire[:,:,2] = prix_monetaire[:,:,2] + prix_fixe_vehicule_mois * 12 * np.transpose(taille_menage_mat) #voiture
                prix_monetaire[:,:,3] = prix_monetaire[:,:,3] + prix_taxi_fixe_mois * 12 * np.transpose(taille_menage_mat) #minibus-taxi
                prix_monetaire[:,:,4] = prix_monetaire[:,:,4] + prix_bus_fixe_mois * 12 * np.transpose(taille_menage_mat) #bus
                
            number_hour_week = 40
            number_weeks = 52
            
            revenu_ici = revenu2_polycentrique(macro_data, param, option, grid, job, t_trafic, index) #4, 2014, 7 ---> ne devrait pas dépendant du temps (2014 * 4)
            #revenu_ici = np.matlib.repmat(revenu_ici, 2014)
            income_per_hour = revenu_ici / number_weeks / number_hour_week
            prix_temps = np.empty((24014, nb_employment_center , 5))
            for i in range(0,nb_employment_center):
                prix_temps[:,i,:] = temps_sortie[i,:,:] * param2["cost_of_time"] * income_per_hour[i]  / 60 * 2 * 20 * 12
            #prix_temps = temps_sortie * param["prix_temps"] * income_per_hour / 60 * 2 * 20 * 12 #21014 * 4 * 2011* 4 * 5
        
            #if NON_LIN == 1:
                #prix_temps[temps_sortie > param["limite_temps"]] = (param["limite_temps"] * param["prix_temps"] + (temps_sortie[temps_sortie > param["limite_temps"]] - param["limite_temps"]) * param["prix_temps2"]) * income_per_hour[temps_sortie > param["limite_temps"]] / 60 * 2 * 20 * 12
            
            prix_temps = np.swapaxes(prix_temps, 0, 1)
            prix_final = prix_monetaire + prix_temps #20014 * 4 * 5
            
            #if index == 1:
            #    trans_prix_monetaire_init = prix_monetaire
            #    trans_prix_temps_init = prix_temps
       
            #if option["LOGIT"] == 1:
             #   mini_prix = np.empty((24014, 4, 5))
             #   mini_prix[:,:,0] = np.min(prix_final, axis = 2)
             #   mini_prix[:,:,1] = mini_prix[:,:,0]
             #   mini_prix[:,:,2] = mini_prix[:,:,0]
             #   mini_prix[:,:,3] = mini_prix[:,:,0]
             #   mini_prix[:,:,4] = mini_prix[:,:,0]

             #   coeff_logit = param["facteur_logit"] / mini_prix
             #   mode_logit = logit(coeff_logit, prix_final, trans.nbre_modes)
        
             #   mult[:,:,index] = (pour_moyenne_logit(coeff_logit, prix_final) / param["facteur_logit"])
             #   cout_generalise[:,:,index] = single(pour_moyenne_logit(coeff_logit,prix_final) / coeff_logit[:,:,0])
        
             #   cout_generalise_ancien[:,:,index] = (sum(prix_final * mode_logit,3))
        
             #   quel = mode_logit #ATTENTION, trans.quel ne depend plus du temps ici
                
             #   trans_prix_temps[:,:,index] = sum(quel * prix_temps, 3)
            #else:
            cout_generalise[:,:,index] = np.min(prix_final, axis = 2)
            quel[:,:,index] = np.argmin(prix_final, axis = 2)        
           
        self.reliable = trans_reliable #number of points where data are available. 24014
        self.distance_sortie = trans_distance_sortie #distance to employment center. nb of employment center * 24014 * 5 transportation modes.
        self.cout_generalise = cout_generalise #transportation costs (choosing the transportation mode that minimizes transportation costs). nb of employment centers * 24014 * 7 years
        self.prix_temps = prix_temps #Opportunity cost of transport. nb of employment centers * 24014 * 5 transportation modes
        self.t_transport = t_trafic + param["baseline_year"] #Years.
        self.quel = quel #Transportation mode. nb of employment centers * 24014 * 7 years.
        self.mult = mult #pareil que coût generalisé
        self.temps_sortie = temps_sortie #transportation time. nb of employment centers * 24014 * 5 transportation modes
        