# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:15:09 2020

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd
from data.functions_to_import_data import *

def charges_temps_polycentrique_CAPE_TOWN_3(option,grille,macro,param,poly,t_trafic):
    """ Compute travel times and costs """

    referencement = poly.referencement

    #Compléments aux trajets
    complement_trajet_voiture = 0
    complement_trajet_pieds = 0
    complement_trajet_TC = 0

    trans.nbre_modes = 5

    #Modèle métro
    distance_metro_2, duration_metro_2 = import_donnees_metro_poly(poly, grille, param)


    load transport_time

    distance_car = distance_vol_oiseau
    duration_car = cars
    duration_metro = train
    distance_metro = distance_vol_oiseau
    duration_minibus = taxi
    duration_bus = bus
    
    distance_car_bak = distance_car
    duration_car_bak = duration_car
    duration_metro_bak = duration_metro
    distance_metro_bak = distance_metro
    duration_minibus_bak = duration_minibus
    duration_bus_bak = duration_bus
    distance_car = np.zeros(len(poly.corresp),size(grille.dist,2))
    duration_car = np.zeros(len(poly.corresp),size(grille.dist,2))
    distance_metro = np.zeros(len(poly.corresp),size(grille.dist,2))
    duration_metro = np.zeros(len(poly.corresp),size(grille.dist,2))
    duration_minibus = np.zeros(len(poly.corresp),size(grille.dist,2))
    duration_bus = np.zeros(len(poly.corresp),size(grille.dist,2))

    for i in range(0, len(poly.corresp)):
        distance_car(i,:) = griddata_hier(X/1000,Y/1000,distance_car_bak(poly.corresp(i),:)',grille.coord_horiz,grille.coord_vert)'
        duration_car(i,:) = griddata_hier(X/1000,Y/1000,duration_car_bak(poly.corresp(i),:)',grille.coord_horiz,grille.coord_vert)'
        distance_metro(i,:) = griddata_hier(X/1000,Y/1000,distance_metro_bak(poly.corresp(i),:)',grille.coord_horiz,grille.coord_vert)'
        duration_metro(i,:) = griddata_hier(X/1000,Y/1000,duration_metro_bak(poly.corresp(i),:)',grille.coord_horiz,grille.coord_vert)'
        duration_minibus(i,:) = griddata_hier(X/1000,Y/1000,duration_minibus_bak(poly.corresp(i),:)',grille.coord_horiz,grille.coord_vert)'
        duration_bus(i,:) = griddata_hier(X/1000,Y/1000,duration_bus_bak(poly.corresp(i),:)',grille.coord_horiz,grille.coord_vert)'
    
    trans.reliable = np.ones(0,len(grille.dist)) #sert à savoir où sont les points où on a toutes les données
    for i in range(0, len(poly.Jx)):
        trans.reliable(np.isnan(duration_car(i,:))) = 0
        trans.reliable(np.isnan(duration_metro(i,:))) = 0
        trans.reliable(np.isnan(duration_minibus(i,:))) = 0
        trans.reliable(np.isnan(duration_bus(i,:))) = 0

    #Définition des variables de temps et distance
    LongueurTotale_VP = distance_car *1.2

    #pour le coût en métro, les coûts dépendent de la distance à la gare
    #centrale du Cap (LongueurEnVehicule_TC n'est donc pas la distance parcourue en TC)
    LongueurEnVehicule_TC = distance_metro_2

    increment = range(0, len(poly.quel))

    #Price for public transportation
    prix_metro_2012_km = 1.5 / 40 * ppval(macro.spline_inflation, 2012 - param["year_begin"]) / ppval(macro.spline_inflation, 2015 - param["year_begin"])  #0.164
    prix_metro_2012_fixe_mois = 121.98 * ppval(macro.spline_inflation, 2012 - param["year_begin"]) / ppval(macro.spline_inflation, 2015-param["year_begin"]) #4.48*40
    prix_taxi_2012_km = 0.785
    prix_taxi_2012_fixe_mois = (4.32*40)
    prix_bus_2012_km = 0.522
    prix_bus_2012_fixe_mois = 6.24 * 40

    #Correct for inflation
    inflation = ppval(macro.spline_inflation, t_trafic)
    infla_2012 = ppval(macro.spline_inflation, 2012 - param["year_begin"])
    prix_metro_km = prix_metro_2012_km * inflation / infla_2012
    prix_metro_fixe_mois = prix_metro_2012_fixe_mois * inflation / infla_2012
    prix_taxi_km = prix_taxi_2012_km * inflation / infla_2012
    prix_taxi_fixe_mois = prix_taxi_2012_fixe_mois * inflation / infla_2012
    prix_bus_km = prix_bus_2012_km * inflation / infla_2012
    prix_bus_fixe_mois = prix_bus_2012_fixe_mois * inflation /infla_2012

    #Fixed price for private cars
    prix_fixe_vehicule_mois_2012 = 350
    prix_fixe_vehicule_mois = prix_fixe_vehicule_mois_2012 * inflation / infla_2012
    prix_essence = ppval(macro.spline_carburant,t_trafic)
    prix_essence_mois = np.zeros(size(prix_essence))
    prix_essence_mois = prix_essence * 2 * 20
    
    #Transport times
    TEMPSHPM = duration_car
    TEMPSTC = duration_metro #duration_metro_2
    TEMPS_MINIBUS = duration_minibus
    TEMPS_BUS = duration_bus

    temps_pieds_temp = LongueurTotale_VP/param.speed_walking*60 + complement_trajet_pieds;
    temps_pieds_temp(isnan(TEMPSHPM)) = NaN;%si on ne fait pas ça, on a des 0 au lieu d'avoir des nan
    temps_sortie(:,:,1) = temps_pieds_temp #temps pour rejoindre à pieds
    temps_sortie(:,:,2) = TEMPSTC + complement_trajet_TC #temps en TC
    temps_sortie(:,:,3) = TEMPSHPM + complement_trajet_voiture #temps en voiture
    temps_sortie(:,:,4) = TEMPS_MINIBUS + complement_trajet_TC #temps en minibus-taxis
    temps_sortie(:,:,5) = TEMPS_BUS + complement_trajet_TC #temps en bus
    #temps_sortie=single(temps_sortie);

    #interpolation avec prix transport en commun en fonction nombre km
    mult_prix_sortie(:,:,1) = np.zeros(size(temps_sortie(:,:,1)))
    mult_prix_sortie(:,:,2) = LongueurEnVehicule_TC
    mult_prix_sortie(:,:,3) = LongueurTotale_VP
    mult_prix_sortie(:,:,4) = LongueurTotale_VP
    mult_prix_sortie(:,:,5) = LongueurTotale_VP

    prix_sortie_unitaire(:,1) = np.ones(size(prix_essence_mois))
    prix_sortie_unitaire(:,2) = prix_metro_km*20*2*12
    prix_sortie_unitaire(:,3) = prix_essence_mois*12
    prix_sortie_unitaire(:,4) = prix_taxi_km*2*20*12
    prix_sortie_unitaire(:,5) = prix_bus_km*2*20*12

    #distances parcourues (ne sert pas dans le calcul mais est une donnée utile)
    distance_sortie(:,:,1) = LongueurTotale_VP
    distance_sortie(:,:,2) = LongueurEnVehicule_TC
    distance_sortie(:,:,3) = LongueurTotale_VP
    distance_sortie(:,:,4) = LongueurTotale_VP
    distance_sortie(:,:,5) = LongueurTotale_VP

    trans.distance_sortie = distance_sortie #trans.distance_sortie = single(distance_sortie)

    prix_monetaire = np.zeros(len(poly.code_emploi_poly(referencement)),size(temps_sortie,2),trans.nbre_modes)
    cout_generalise = single(np.zeros(len(poly.code_emploi_poly(referencement)),size(temps_sortie,2),len(t_trafic)))
    quel = uint8(np.zeros(size(poly.code_emploi_poly(referencement),1),size(temps_sortie,2)))

    mult = cout_generalise
    cout_generalise_ancien = cout_generalise

    tbis=t_trafic

    taille_menage_mat = np.matlibb.repmat(param["taille_menage_transport"], 1, len(poly.code_emploi_init) / param["multiple_class"])
    taille_menage_mat = np.transpose(taille_menage_mat(poly.quel)) * np.ones(1, len(grille.dist)) #pour prendre en compte des tailles de ménages différentes

    #boucle sur le temps
    for index in range(0, len(tbis)):
        for index2 in range(0, trans.nbre_modes):
            prix_monetaire(:,:,index2) = prix_sortie_unitaire(index,index2) * mult_prix_sortie(:,:,index2)
            prix_monetaire(:,:,index2) = prix_monetaire(:,:,index2) * taille_menage_mat
        transi.cout_generalise = prix_monetaire #sert juste pour mettre dans la fonction revenu2_polycentrique
    
        revenu_ici = revenu2_polycentrique(macro, param,option,grille,poly,t_trafic(index))
        revenu_ici = np.matlib.repmat(revenu_ici,[1 1 size(prix_monetaire,3)])
    
        #ajout des couts fixes
        prix_monetaire(:,:,2) = prix_monetaire(:,:,2) + prix_metro_fixe_mois(index) * 12 * taille_menage_mat #train, avec abonnement mensuel
        prix_monetaire(:,:,3) = prix_monetaire(:,:,3) + prix_fixe_vehicule_mois(index) * 12 *taille_menage_mat #voiture
        prix_monetaire(:,:,4) = prix_monetaire(:,:,4) + prix_taxi_fixe_mois(index) * 12 * taille_menage_mat #minibus-taxi
        prix_monetaire(:,:,5) = prix_monetaire(:,:,5) + prix_bus_fixe_mois(index) * 12 *taille_menage_mat #bus

        number_hour_week = 40
        number_weeks = 52
        income_per_hour = revenu_ici / number_weeks / number_hour_week
        prix_temps = temps_sortie * param["prix_temps"] * income_per_hour / 60 * 2 * 20 * 12
        
        if NON_LIN==1:
            prix_temps[temps_sortie > param.limite_temps] = (param["limite_temps"] * param["prix_temps"] + (temps_sortie[temps_sortie > param["limite_temps"]] - param["limite_temps"]) * param["prix_temps2"]) * income_per_hour(temps_sortie > param["limite_temps"]) / 60 * 2 * 20 * 12
                                                                
        prix_final = prix_monetaire + prix_temps
    
        if index == 1:
            trans.prix_monetaire_init = prix_monetaire
            trans.prix_temps_init = prix_temps
       
        if option.LOGIT == 1:
            mini_prix(:,:,1) = min(prix_final,[],3);
            mini_prix(:,:,2) = mini_prix(:,:,1);
            mini_prix(:,:,3) = mini_prix(:,:,1);
            mini_prix(:,:,4) = mini_prix(:,:,1);
            mini_prix(:,:,5) = mini_prix(:,:,1);

            coeff_logit = param["facteur_logit"] / mini_prix
            mode_logit = logit(coeff_logit, prix_final, trans.nbre_modes)
        
            mult(:,:,index) = single(pour_moyenne_logit(coeff_logit,prix_final) / param["facteur_logit"]);
            cout_generalise(:,:,index) = single(pour_moyenne_logit(coeff_logit,prix_final) / coeff_logit(:,:,1))
        
            cout_generalise_ancien(:,:,index) = single(sum(prix_final * mode_logit,3))
        
            quel = single(mode_logit) #ATTENTION, trans.quel ne depend plus du temps ici
        
            trans.prix_temps(:,:,index) = sum(quel * prix_temps,3)
        else
           [cout_generalise(:,:,index),quel(:,:,index)] = min(single(prix_final),[],trans.nbre_modes)

    trans.t_transport = t_trafic + param["year_begin"]
    trans.cout_generalise = cout_generalise

    trans.quel = quel
    trans.mult = mult
    trans.temps_sortie = temps_sortie

def import_donnees_metro_poly(poly, grille,param):
    """ import and estimate transport time by the metro """
    
    metro_station = pd.read_csv('metro_station_poly.csv')
    
    station_line_time = ["Bellvill1_B", "Bellvill2_M", "Bellvill3_S", "Bonteheuwel1_C", "Bonteheuwel2_B", "Bonteheuwel3_K", "Capeflats", "Malmesbury", "Simonstown", "Worcester"]

    duration = np.zeros(len(ID_station))
    
    for i in range (0, len(ID_station)): #matrice des temps O-D entre les stations de m?tro
        for j in range(0, i):
            if (i == j):
                duration(i,j)=0
            elif sum(station_line_time[i,:] * station_line_time[j,:]) > 0: #pas besoin de faire de changement
                temps = np.abs(station_line_time(j, station_line_time[i,:] * station_line_time[j,:] > 0) - station_line_time(i,station_line_time(i,:) * station_line_time(j,:)>0))
                duration(i,j) = min(temps) + param.metro_waiting_time
                duration(j,i) = duration(i,j)
            else: #il faut faire un changement
                line_i = station_line_time(i,:) > 0
                line_j = station_line_time(j,:) > 0
                noeud = false(length(ID_station),1)
                for k = 1:length(ID_station):
                    if (sum(station_line_time(k,:) * station_line_time(i,:)) > 0) & (sum(station_line_time(k,:) * station_line_time(j,:)) > 0):
                        noeud(k) = true
                temps1 = (np.abs(np.matlib.repmat(station_line_time(j, line_j > 0), sum(noeud), 1) - station_line_time(noeud, line_j > 0)))
                temps2 = (np.abs(np.matlib.repmat(station_line_time(i, line_i > 0), sum(noeud), 1) - station_line_time(noeud, line_i > 0)))
                duration(i,j) = min(min(temps1,[],2) + min(temps2,[],2))
                duration(i,j) = duration(i,j) + 2 * param["metro_waiting_time"]
                duration(j,i) = duration(i,j)

    #pour chaque point de grille la station la plus proche, et distance
    ID_station_grille = griddata(X_cape / 1000, Y_cape / 1000, ID_station, grille.coord_horiz, grille.coord_vert, 'nearest')
    distance_grille = np.zeros(len(grille.coord_horiz), 1)

    #Pour chaque centre d'emploi la station la plus proche, et distance
    ID_station_center = griddata(X_cape / 1000, Y_cape / 1000, ID_station, poly.Jx, poly.Jy,'nearest')
    distance_center = np.zeros(len(poly.Jx), 1)
    for i in range(0, len(poly.Jx)):
        distance_center(i) = np.sqrt((poly.Jx(i) - X_cape(ID_station_center(i)) / 1000) ** 2 + (poly.Jy(i) - Y_cape(ID_station_center(i)) / 1000) ** 2)

    #calcul de la matrice des durées
    duration_metro = np.zeros(len(grille.dist), len(poly.Jx))
    distance_metro = np.zeros(len(grille.dist), len(poly.Jx))
    for i in range(0, len(grille.coord_horiz)):
        distance_grille(i) = np.sqrt((grille.coord_horiz(i) - X_cape(ID_station_grille(i)) / 1000) ** 2 + (grille.coord_vert(i) - Y_cape(ID_station_grille(i)) / 1000) ** 2)
        for j in range(0, len(poly.Jx)):
            duration_metro(i,j) = (distance_grille(i) + distance_center(j)) * 1.2 / (param["speed_walking"] / 60) + duration(ID_station_grille(i), ID_station_center(j))
            distance_metro(i,j) = max(np.sqrt((grille.xcentre - X_cape(ID_station_grille(i)) / 1000) ** 2 + (grille.ycentre - Y_cape(ID_station_grille(i)) / 1000) ** 2), np.sqrt((grille.xcentre - X_cape(ID_station_center(j)) / 1000) ** 2 + (grille.ycentre - Y_cape(ID_station_center(j)) / 1000) ** 2))

    duration_metro = np.transpose(duration_metro)
    distance_metro = np.transpose(distance_metro)

    return distance_metro, duration_metro