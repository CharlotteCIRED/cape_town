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
        
    def import_transport_data(self, option, grid, macro_data, param, job, households_data, yearTraffic, extrapolate):
        """ Compute travel times and costs """

        #referencement = job.referencement
        complement_trajet_voiture = 0
        complement_trajet_pieds = 0
        complement_trajet_TC = 0
        trans_nbre_modes = 5
        
        listCenter = np.unique(job.corresp)
        coordinatesCenter = np.array([np.unique(job.xCenter), np.unique(job.yCenter)])

        transport_matrices = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/transportMatrices.mat')
        X = transport_matrices["X"].squeeze()
        Y = transport_matrices["Y"].squeeze()
        durationBus = transport_matrices["bus"]
        durationCar = transport_matrices["cars"]
        distanceCar = transport_matrices["distance_vol_oiseau"]
        durationMinibus = transport_matrices["taxi"]
        durationTrain = transport_matrices["train"]       
        durationTrain[:, X > -8000] = np.nan
        
        distanceCar_bak = distanceCar[job.selectedCenters, :]
        durationCar_bak = durationCar[job.selectedCenters, :]
        durationTrain_bak = durationTrain[job.selectedCenters, :]
        durationMinibus_bak = durationMinibus[job.selectedCenters, :]
        durationBus_bak = durationBus[job.selectedCenters, :]
        
        xInterp = grid.horiz_coord
        yInterp = grid.vert_coord
        
        distanceCar = np.zeros((len(job.corresp), len(xInterp)))
        durationCar = np.zeros((len(job.corresp), len(xInterp)))
        durationTrain = np.zeros((len(job.corresp), len(xInterp)))
        durationMinibus = np.zeros((len(job.corresp), len(xInterp)))
        durationBus = np.zeros((len(job.corresp), len(xInterp)))

        option["loadTransportTime"] = 1
        if option["loadTransportTime"] == 0:  
            for i in range(0, len(listCenter)):
                distanceCar[i,:] = griddata_extra(X/1000, Y/1000, distanceCar_bak[i, :], xInterp, yInterp, extrapolate, coordinatesCenter[:, i])
                durationCar[i,:] = griddata_extra(X/1000, Y/1000, durationCar_bak[i,:], xInterp, yInterp, extrapolate, coordinatesCenter[:, i])
                durationTrain[i,:] = griddata_extra(X/1000, Y/1000, durationTrain_bak[i,:], xInterp, yInterp, extrapolate, coordinatesCenter[:, i])
                durationMinibus[i,:] = griddata_extra(X/1000, Y/1000, durationMinibus_bak[i,:], xInterp, yInterp, extrapolate, coordinatesCenter[:, i])
                durationBus[i,:] = griddata_extra(X/1000, Y/1000, durationBus_bak[i,:], xInterp, yInterp, extrapolate, coordinatesCenter[:, i])
        else: 
            transport_times = scipy.io.loadmat('./2. Data/Basile data/Transport_times_GRID.mat')
            distanceCar = transport_times["distanceCar"]
            durationCar = transport_times["durationCar"]
            durationTrain = transport_times["durationTrain"]
            durationMinibus = transport_times["durationMinibus"]
            durationBus = transport_times["durationBus"]

        distanceTrain = distanceCar
        LengthPrivateCar = distanceCar
        LengthInVehiculePubTransit = distanceTrain
         
        #Price per km and fixed costs

        #Inputs from own analysis (from Roux 2013)
        priceTrainPerKMMonth = 0.164 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceTrainFixedMonth = 4.48 * 40 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceTaxiPerKMMonth = 0.785 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceTaxiFixedMonth = 4.32 * 40 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceBusPerKMMonth = 0.522 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceBusFixedMonth = 6.24 * 40 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])

        #Correct for inflation
        inflation = macro_data.inflation(yearTraffic)
        infla_2012 = macro_data.inflation(2012 - param["baseline_year"])
        priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
        priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
        priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
        priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
        priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
        priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012

        priceFixedVehiculeMonth = 400 
        priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012
        priceFuel = macro_data.fuel_cost(yearTraffic)

        #Transport times
        timePV = durationCar
        timeTrain = durationTrain #duration_metro_2
        timeTaxi = durationMinibus
        timeBus = durationBus

        #For each year, we esimate the price per km for cars
        priceFuelPerKMMonth = np.zeros(len(priceFuel))
        for index in range(0, len(yearTraffic)):        
            taxAccrossTime = 0
            priceFuelPerKMMonth[index] = priceFuel[index] + taxAccrossTime
            
        ### Transport times and costs as matrices

        #Time by each mode, aller-retour, en minute
        timeWalkingTemp = LengthPrivateCar / param["walking_speed"] * 60 * 1.2 * 2 + complement_trajet_pieds
        timeWalkingTemp[np.isnan(timePV)] = np.nan
        timeOutput = np.empty((timeWalkingTemp.shape[0], timeWalkingTemp.shape[1], 5))
        timeOutput[:,:,0] = timeWalkingTemp
        timeOutput[:,:,1] = timeTrain + complement_trajet_TC
        timeOutput[:,:,2] = timePV + complement_trajet_voiture
        timeOutput[:,:,3] = timeTaxi + complement_trajet_TC
        timeOutput[:,:,4] = timeBus + complement_trajet_TC

        #Length (km) using each mode
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:,:,0] = np.zeros((timeOutput[:,:,0].shape))
        multiplierPrice[:,:,1] = LengthInVehiculePubTransit
        multiplierPrice[:,:,2] = LengthPrivateCar
        multiplierPrice[:,:,3] = LengthPrivateCar
        multiplierPrice[:,:,4] = LengthPrivateCar
        
        #Number of worked days per year
        numberDaysPerYear = 235

        #Multiplying by 235 (days per year)
        pricePerKM = np.empty((len(priceFuelPerKMMonth), 5))
        pricePerKM[:, 0] = np.zeros(len(priceFuelPerKMMonth))
        pricePerKM[:, 1] = priceTrainPerKMMonth*numberDaysPerYear
        pricePerKM[:, 2] = priceFuelPerKMMonth*numberDaysPerYear          
        pricePerKM[:, 3] = priceTaxiPerKMMonth*numberDaysPerYear
        pricePerKM[:, 4] = priceBusPerKMMonth*numberDaysPerYear

        #Distances (not useful to calculate price but useful output)
        distanceOutput = np.empty((timeOutput.shape))
        distanceOutput[:,:,0] = LengthPrivateCar
        distanceOutput[:,:,1] = LengthInVehiculePubTransit
        distanceOutput[:,:,2] = LengthPrivateCar
        distanceOutput[:,:,3] = LengthPrivateCar
        distanceOutput[:,:,4] = LengthPrivateCar

        #Monetary cost

        #Monetary price per year
        monetaryCost = np.zeros((len(job.codeCentersPolycentric), timeOutput.shape[1], 5))
        trans_monetaryCost = np.zeros((len(job.codeCentersPolycentric), timeOutput.shape[1], 5, len(yearTraffic)))
        for index in range(0, len(yearTraffic)):    
            for index2 in range(0, 5):
                monetaryCost[:,:,index2] = pricePerKM[index,index2] * multiplierPrice[:,:,index2]
                monetaryCost[:,:,index2] = monetaryCost[:,:,index2] #.*householdSizeMat;
        
            #Adding fixed costs
            monetaryCost[:,:,1] = monetaryCost[:,:,1] + priceTrainFixedMonth[index] * 12 #train (monthly fare)
            monetaryCost[:,:,2] = monetaryCost[:,:,2] + priceFixedVehiculeMonth[index] * 12 #private car
            monetaryCost[:,:,3] = monetaryCost[:,:,3] + priceTaxiFixedMonth[index] * 12 #minibus-taxi
            monetaryCost[:,:,4] = monetaryCost[:,:,4] + priceBusFixedMonth[index] * 12 #bus

            trans_monetaryCost[:,:,:,index] = copy.deepcopy(monetaryCost)

        #Cost associated with time

        numberHourWorkedPerDay= 8
        costTime = (timeOutput * param["timeCost"]) / (60 * numberHourWorkedPerDay) #en h de transport par h de travail
   
        param_lambda = param["lambda"].squeeze()
        incomeNetOfCommuting = np.zeros((param["nb_of_income_classes"], len(grid.dist), len(yearTraffic)))
        averageIncome = np.zeros((param["nb_of_income_classes"], len(grid.dist), len(yearTraffic)))
        modalShares = np.zeros((len(job.incomeCentersInit), len(grid.dist), 5, param["nb_of_income_classes"], len(yearTraffic)))
        ODflows = np.zeros((len(job.incomeCentersInit), len(grid.dist), param["nb_of_income_classes"], len(yearTraffic)))
    
        for index in range(0, len(yearTraffic)):
            incomeGroup = InterpolateIncomeEvolution(macro_data, param, option, grid, job, yearTraffic[index])
            incomeGroupRef = InterpolateIncomeEvolution(macro_data, param, option, grid, job, 0)
            incomeCenters = job.incomeCentersInit * incomeGroup[0, :] / incomeGroupRef[0, :]
            incomeNetOfCommuting[:,:,index], modalShares[:,:,:,:,index], ODflows[:,:,:,index], averageIncome[:,:,index], monetaryCost_v2, timeCost_v2 = ComputeIncomeNetOfCommuting(param, costTime, trans_monetaryCost, grid, job, households_data, param_lambda, incomeCenters, index)

        self.distanceOutput = distanceOutput
        self.monetaryCost = trans_monetaryCost
        self.timeCost = costTime
        self.incomeNetOfCommuting = incomeNetOfCommuting
        self.modalShares = modalShares
        self.ODflows = ODflows
        self.averageIncome = averageIncome
        self.yearTransport = yearTraffic + param["baseline_year"]
        self.timeOutput = timeOutput
        
    

"""
        
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
        
        if option["nb_employment_center"] == 6:
            nb_employment_center = 18
        elif option["nb_employment_center"] == 185:
            nb_employment_center = 740
        else:
            if option["nb_employment_center"] == 1 & param["nb_of_income_classes"] == 4:
                nb_employment_center = 4
            elif option["nb_employment_center"] == 1 &  param["nb_of_income_classes"] == 12:
                nb_employment_center = 12
    
        #Transport times
        if option["nb_employment_center"] == 6:
            TEMPSHPM = np.array([duration_car[0], duration_car[0], duration_car[0], duration_car[0], duration_car[1], duration_car[1], duration_car[1], duration_car[2], duration_car[2], duration_car[2], duration_car[2], duration_car[4], duration_car[4], duration_car[4], duration_car[4], duration_car[5], duration_car[5], duration_car[5]])
            TEMPS_MINIBUS = np.array([duration_minibus[0], duration_minibus[0], duration_minibus[0], duration_minibus[0], duration_minibus[1], duration_minibus[1], duration_minibus[1], duration_minibus[2], duration_minibus[2], duration_minibus[2], duration_minibus[2], duration_minibus[4], duration_minibus[4], duration_minibus[4], duration_minibus[4], duration_minibus[5], duration_minibus[5], duration_minibus[5]])
            TEMPS_BUS = np.array([duration_bus[0], duration_bus[0], duration_bus[0], duration_bus[0], duration_bus[1], duration_bus[1], duration_bus[1], duration_bus[2], duration_bus[2], duration_bus[2], duration_bus[2], duration_bus[4], duration_bus[4], duration_bus[4], duration_bus[4], duration_bus[5], duration_bus[5], duration_bus[5]])
            TEMPSTC = duration_metro_2 #duration_metro
        elif option["nb_employment_center"] == 185:
            TEMPSHPM = np.empty((740, 24014))
            TEMPS_MINIBUS = np.empty((740, 24014))
            TEMPS_BUS = np.empty((740, 24014))
            for i in range(0, 185):
                TEMPSHPM[4 * i, :] = duration_car[i]
                TEMPSHPM[4 * i + 1, :] = duration_car[i]
                TEMPSHPM[4 * i + 2, :] = duration_car[i]
                TEMPSHPM[4 * i + 3, :] = duration_car[i]         
                TEMPS_MINIBUS[4 * i, :] = duration_minibus[i]
                TEMPS_MINIBUS[4 * i + 1, :] = duration_minibus[i]
                TEMPS_MINIBUS[4 * i + 2, :] = duration_minibus[i]
                TEMPS_MINIBUS[4 * i + 3, :] = duration_minibus[i]    
                TEMPS_BUS[4 * i, :] = duration_bus[i]
                TEMPS_BUS[4 * i + 1, :] = duration_bus[i]
                TEMPS_BUS[4 * i + 2, :] = duration_bus[i]
                TEMPS_BUS[4 * i + 3, :] = duration_bus[i]    
            TEMPSTC = duration_metro_2 #duration_metro
        else:
            if option["nb_employment_center"] == 1 & param["nb_of_income_classes"] == 4:
                TEMPSHPM = np.array([duration_car, duration_car, duration_car, duration_car])
                TEMPS_MINIBUS = np.array([duration_minibus, duration_minibus, duration_minibus, duration_minibus])
                TEMPS_BUS = np.array([duration_bus, duration_bus, duration_bus, duration_bus])
                TEMPSTC = duration_metro_2 #duration_metro
            elif option["nb_employment_center"] == 1 & param["nb_of_income_classes"] == 12:
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

        if option["nb_employment_center"] == 6:
            prix_sortie_unitaire = np.empty((prix_essence_mois.shape[0], 5))
            prix_sortie_unitaire[:,0] = np.ones(prix_essence_mois.shape)
        elif option["nb_employment_center"] == 185:
            prix_sortie_unitaire = np.empty((prix_essence_mois.shape[0], 5))
            prix_sortie_unitaire[:,0] = np.ones(prix_essence_mois.shape)
        elif option["nb_employment_center"] == 1 & param["nb_of_income_classes"] == 4:
            prix_sortie_unitaire = np.empty((prix_essence_mois.shape[0], 5))
            prix_sortie_unitaire[:,0] = np.ones(prix_essence_mois.shape)
        elif option["nb_employment_center"] == 1 & param["nb_of_income_classes"] == 12: 
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
        
        prix_temps_bis = np.empty((nb_employment_center, 24014, len(t_trafic)))
 
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
            prix_temps = np.empty((24014, nb_employment_center, 5))
            for i in range(0,nb_employment_center):
                prix_temps[:,i,:] = temps_sortie[i,:,:] * param["cost_of_time"] * income_per_hour[i]  / 60 * 2 * 20 * 12
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
            for i in range(0, quel.shape[0]):
                for j in range(0, quel.shape[1]):
                    prix_temps_bis[:, :, index]  = prix_temps[:, :, int(quel[i, j, index])]
           
        self.reliable = trans_reliable #number of points where data are available. 24014
        self.distance_sortie = trans_distance_sortie #distance to employment center. nb of employment center * 24014 * 5 transportation modes.
        self.cout_generalise = cout_generalise #transportation costs (choosing the transportation mode that minimizes transportation costs). nb of employment centers * 24014 * 7 years
        self.prix_temps = prix_temps_bis #Opportunity cost of transport. nb of employment centers * 24014 * 5 transportation modes
        self.t_transport = t_trafic + param["baseline_year"] #Years.
        self.quel = quel #Transportation mode. nb of employment centers * 24014 * 7 years.
        self.mult = mult #pareil que coût generalisé
        self.temps_sortie = temps_sortie #transportation time. nb of employment centers * 24014 * 5 transportation modes
        
        
        
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
        
        if option["nb_employment_center"] == 1:
            distance_car = distance_car[135]
            duration_car = duration_car[135]
            distance_metro = distance_metro[135]
            duration_metro = duration_metro[135]
            duration_minibus = duration_minibus[135]
            duration_bus = duration_bus[135]
            nb_center = 1
        elif option["nb_employment_center"] == 6:
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
        if option["nb_employment_center"] == 1:
            trans_reliable[np.isnan(duration_car)] = 0
            trans_reliable[np.isnan(duration_metro)] = 0
            trans_reliable[np.isnan(duration_minibus)] = 0
            trans_reliable[np.isnan(duration_bus)] = 0
        elif option["nb_employment_center"] == 6 | option["nb_employment_center"] == 185:
            trans_reliable = np.ones((nb_center, len(grid.dist)))
            for i in range(0, nb_center):
                trans_reliable[np.isnan(duration_car[:,i])] = 0
                trans_reliable[np.isnan(duration_metro[:,i])] = 0
                trans_reliable[np.isnan(duration_minibus[:,i])] = 0
                trans_reliable[np.isnan(duration_bus[:,i])] = 0
        
        #Define time and distance variables
        if option["nb_employment_center"] == 6:
            LongueurTotale_VP = np.array([distance_car[0], distance_car[0], distance_car[0], distance_car[0], distance_car[1], distance_car[1], distance_car[1], distance_car[2], distance_car[2], distance_car[2], distance_car[2], distance_car[4], distance_car[4], distance_car[4], distance_car[4], distance_car[5], distance_car[5], distance_car[5]])
            LongueurTotale_VP = LongueurTotale_VP * 1.2
            LongueurEnVehicule_TC = distance_metro_2 #pour le coût en métro, les coûts dépendent de la distance à la gare centrale du Cap (LongueurEnVehicule_TC n'est donc pas la distance parcourue en TC)
        elif option["nb_employment_center"] == 185:
            LongueurTotale_VP = np.empty((740, 24014))
            for i in range(0, 185):
                LongueurTotale_VP[4 * i, :] = distance_car[i]
                LongueurTotale_VP[4 * i + 1, :] = distance_car[i]
                LongueurTotale_VP[4 * i + 2, :] = distance_car[i]
                LongueurTotale_VP[4 * i + 3, :] = distance_car[i]              
            LongueurTotale_VP = LongueurTotale_VP * 1.2
            LongueurEnVehicule_TC = distance_metro_2
        elif (param["nb_employment_center"] == 1) & (param["nb_of_income_classes"] == 4): 
            LongueurTotale_VP = np.array([distance_car, distance_car, distance_car, distance_car])
            LongueurTotale_VP = LongueurTotale_VP * 1.2
            LongueurEnVehicule_TC = distance_metro_2
        elif (param["nb_employment_center"] == 1) & (param["nb_of_income_classes"] == 12): 
            LongueurTotale_VP = np.array([distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car, distance_car])
            LongueurTotale_VP = LongueurTotale_VP * 1.2
            LongueurEnVehicule_TC = distance_metro_2
            
        increment = range(0, len(job.quel))"""