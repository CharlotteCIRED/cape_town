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

class TransportData:
        
    def __init__(self):
        
        self
        
    def charges_temps_polycentrique_CAPE_TOWN_3(self, option, grille, macro, param, poly, t_trafic):
        """ Compute travel times and costs """

        referencement = poly.referencement
        complement_trajet_voiture = 0
        complement_trajet_pieds = 0
        complement_trajet_TC = 0
        trans_nbre_modes = 5

        #Modèle métro
        distance_metro_2, duration_metro_2 = import_donnees_metro_poly(poly, grille, param)

        #Load distance and duration with each transportation mode
        transport_time_grid = scipy.io.loadmat('./2. Data/Transport_times_GRID.mat')
        transport_time_sp = scipy.io.loadmat('./2. Data/Transport_times_SP.mat')
        #load transport times
        #distance_car = distance_vol_oiseau
        #duration_car = cars
        #duration_metro = train
        #distance_metro = distance_vol_oiseau
        #duration_minibus = taxi
        #duration_bus = bus   
        #distance_car_bak = distance_car
        #duration_car_bak = duration_car
        #duration_metro_bak = duration_metro
        #distance_metro_bak = distance_metro
        #duration_minibus_bak = duration_minibus
        #duration_bus_bak = duration_bus
        #distance_car = np.zeros(len(poly.corresp), len(grille.dist))
        #duration_car = np.zeros(len(poly.corresp), len(grille.dist))
        #distance_metro = np.zeros(len(poly.corresp), len(grille.dist))
        #duration_metro = np.zeros(len(poly.corresp), len(grille.dist))
        #duration_minibus = np.zeros(len(poly.corresp), len(grille.dist))
        #duration_bus = np.zeros(len(poly.corresp), len(grille.dist))

        #Extrapolation des données sur toute la grille
        #for i in range(0, len(poly.corresp)):
            #distance_car[i,:] = np.transpose(griddata_hier(X/1000, Y/1000, np.transpose(distance_car_bak[poly.corresp[i], :]), grille.coord_horiz, grille.coord_vert))
            #duration_car[i,:] = np.transpose(griddata_hier(X/1000, Y/1000, np.transpose(duration_car_bak[poly.corresp[i], :]), grille.coord_horiz, grille.coord_vert))
            #distance_metro[i,:] = np.transpose(griddata_hier(X/1000, Y/1000, np.transpose(distance_metro_bak[poly.corresp[i], :]), grille.coord_horiz, grille.coord_vert))
            #duration_metro[i,:] = np.transpose(griddata_hier(X/1000, Y/1000, np.transpose(duration_metro_bak[poly.corresp[i], :]), grille.coord_horiz, grille.coord_vert))
            #duration_minibus[i,:] = np.transpose(griddata_hier(X/1000, Y/1000, np.transpose(duration_minibus_bak[poly.corresp[i], :]), grille.coord_horiz, grille.coord_vert))
            #duration_bus[i,:] = np.transpose(griddata_hier(X/1000, Y/1000, np.transpose(duration_bus_bak[poly.corresp[i], :]), grille.coord_horiz, grille.coord_vert))

        
        distance_car = transport_time_grid["distanceCar"] #devrait faire len(poly.corresp). Pour le moment, matrice OD 185 * 20014 (car on garde 185 centres d'emploi)
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
        elif option["polycentric"] == 1:
            distance_car = distance_car[(135, 40, 10, 34, 108, 165),:]
            duration_car = duration_car[(135, 40, 10, 34, 108, 165),:]
            distance_metro = distance_metro[(135, 40, 10, 34, 108, 165),:]
            duration_metro = duration_metro[(135, 40, 10, 34, 108, 165),:]
            duration_minibus = duration_minibus[(135, 40, 10, 34, 108, 165),:]
            duration_bus = duration_bus[(135, 40, 10, 34, 108, 165),:]
            
        #Pour le centre : restreindre à 135
        #5101 (CBD): 135
        #2002 (Bellville): 40
        #1201 (Epping): 10
        #1553 (Claremont): 34
        #3509 (Sommerset West): 108
        #5523 (Table View + Century City): 165
        
        #Points pour lesquels on a toutes les données                                     
        trans_reliable = np.ones(len(grille.dist))
        if option["polycentric"] == 0:
            trans_reliable[np.isnan(duration_car)] = 0
            trans_reliable[np.isnan(duration_metro)] = 0
            trans_reliable[np.isnan(duration_minibus)] = 0
            trans_reliable[np.isnan(duration_bus)] = 0

        #Définition des variables de temps et distance
        LongueurTotale_VP = distance_car * 1.2
        LongueurEnVehicule_TC = distance_metro_2  #pour le coût en métro, les coûts dépendent de la distance à la gare centrale du Cap (LongueurEnVehicule_TC n'est donc pas la distance parcourue en TC)

        increment = range(0, len(poly.quel))
        
        #Price for public transportation
        prix_metro_2012_km = 1.5 / 40 * macro_data.spline_inflation(2012 - param["year_begin"]) / macro_data.spline_inflation(2015 - param["year_begin"])  #0.164
        prix_metro_2012_fixe_mois = 121.98 * macro_data.spline_inflation(2012 - param["year_begin"]) / macro_data.spline_inflation(2015 - param["year_begin"]) #4.48*40
        prix_taxi_2012_km = 0.785
        prix_taxi_2012_fixe_mois = 4.32 * 40
        prix_bus_2012_km = 0.522
        prix_bus_2012_fixe_mois = 6.24 * 40

        #Correct for inflation
        inflation = macro_data.spline_inflation(t_trafic)
        infla_2012 = macro_data.spline_inflation(2012 - param["year_begin"])
        prix_metro_km = prix_metro_2012_km * inflation / infla_2012
        prix_metro_fixe_mois = prix_metro_2012_fixe_mois * inflation / infla_2012
        prix_taxi_km = prix_taxi_2012_km * inflation / infla_2012
        prix_taxi_fixe_mois = prix_taxi_2012_fixe_mois * inflation / infla_2012
        prix_bus_km = prix_bus_2012_km * inflation / infla_2012
        prix_bus_fixe_mois = prix_bus_2012_fixe_mois * inflation /infla_2012

        #Fixed price for private cars
        prix_fixe_vehicule_mois_2012 = 350
        prix_fixe_vehicule_mois = prix_fixe_vehicule_mois_2012 * inflation / infla_2012
        prix_essence = macro_data.spline_carburant(t_trafic)
        prix_essence_mois = np.zeros(prix_essence.shape)
        prix_essence_mois = prix_essence * 2 * 20
    
        #Transport times
        TEMPSHPM = duration_car
        TEMPSTC = duration_metro #duration_metro_2
        TEMPS_MINIBUS = duration_minibus
        TEMPS_BUS = duration_bus
        temps_pieds_temp = LongueurTotale_VP / param["speed_walking"] * 60 + complement_trajet_pieds
        temps_pieds_temp[np.isnan(TEMPSHPM)] = np.nan #si on ne fait pas ça, on a des 0 au lieu d'avoir des nan
        temps_pieds_temp = pd.DataFrame(temps_pieds_temp)
        temps_sortie = np.empty((24014,1, 5))
        temps_sortie[:,:,0] = temps_pieds_temp #temps pour rejoindre à pieds
        temps_sortie[:,:,1] = pd.DataFrame(TEMPSTC + complement_trajet_TC) #temps en TC
        temps_sortie[:,:,2] = pd.DataFrame(TEMPSHPM + complement_trajet_voiture) #temps en voiture
        temps_sortie[:,:,3] = pd.DataFrame(TEMPS_MINIBUS + complement_trajet_TC) #temps en minibus-taxis
        temps_sortie[:,:,4] = pd.DataFrame(TEMPS_BUS + complement_trajet_TC) #temps en bus
        #temps_sortie=single(temps_sortie);

        #interpolation avec prix transport en commun en fonction nombre km
        mult_prix_sortie = np.empty((24014, 1, 5))
        mult_prix_sortie[:,:,0] = np.zeros((temps_sortie[:,:,1]).shape)
        mult_prix_sortie[:,:,1] = pd.DataFrame(LongueurEnVehicule_TC[0])
        mult_prix_sortie[:,:,2] = pd.DataFrame(LongueurTotale_VP)
        mult_prix_sortie[:,:,3] = pd.DataFrame(LongueurTotale_VP)
        mult_prix_sortie[:,:,4] = pd.DataFrame(LongueurTotale_VP)

        prix_sortie_unitaire = np.empty((prix_essence_mois.shape[0], 5))
        prix_sortie_unitaire[:,0] = np.ones(prix_essence_mois.shape)
        prix_sortie_unitaire[:,1] = prix_metro_km * 20 * 2 * 12
        prix_sortie_unitaire[:,2] = prix_essence_mois * 12
        prix_sortie_unitaire[:,3] = prix_taxi_km * 2 * 20 * 12
        prix_sortie_unitaire[:,4] = prix_bus_km * 2 * 20 * 12

        #distances parcourues (ne sert pas dans le calcul mais est une donnée utile)
        distance_sortie = np.empty((24014, 1, 5))
        distance_sortie[:,:,0] = pd.DataFrame(LongueurTotale_VP)
        distance_sortie[:,:,1] = pd.DataFrame(LongueurEnVehicule_TC[0])
        distance_sortie[:,:,2] = pd.DataFrame(LongueurTotale_VP)
        distance_sortie[:,:,3] = pd.DataFrame(LongueurTotale_VP)
        distance_sortie[:,:,4] = pd.DataFrame(LongueurTotale_VP)

        trans_distance_sortie = distance_sortie #trans.distance_sortie = single(distance_sortie)
        #prix_monetaire = np.zeros((len(poly.code_emploi_poly[referencement]), temps_sortie.shape[1], trans_nbre_modes))
        prix_monetaire = np.zeros((24014, 4, trans_nbre_modes))
        cout_generalise = (np.zeros((len(poly.code_emploi_poly[referencement]), temps_sortie.shape[1], len(t_trafic))))
        quel = (np.zeros((poly.code_emploi_poly[referencement].shape[0], temps_sortie.shape[1])))

        mult = cout_generalise
        cout_generalise_ancien = cout_generalise
        tbis = t_trafic

        taille_menage_mat = np.matlib.repmat(param["taille_menage_transport"], 1, int(len(poly.code_emploi_init) / param["multiple_class"]))
        taille_menage_mat = np.matlib.repmat(np.transpose(taille_menage_mat.squeeze()[poly.quel]), 1, len(grille.dist)) #pour prendre en compte des tailles de ménages différentes
        taille_menage_mat = np.reshape(taille_menage_mat,(len(grille.dist), 4))
        
        #boucle sur le temps
        for index in range(0, len(tbis)):
            for index2 in range(0, trans_nbre_modes):
                prix_monetaire[:,:,index2] = prix_sortie_unitaire[index, index2] * mult_prix_sortie[:, :, index2]
                prix_monetaire[:,:,index2] = prix_monetaire[:, :, index2] * taille_menage_mat
            trans_cout_generalise = prix_monetaire #sert juste pour mettre dans la fonction revenu2_polycentrique          
            
            revenu_ici = revenu2_polycentrique(macro_data, param, option, grille, poly, t_trafic, index)
            revenu_ici = np.matlib.repmat(revenu_ici, np.array([1, 1, size(prix_monetaire, 3)]))
    
            #ajout des couts fixes
            prix_monetaire[:,:,1] = prix_monetaire[:,:,1] + prix_metro_fixe_mois[index] * 12 * taille_menage_mat #train, avec abonnement mensuel
            prix_monetaire[:,:,2] = prix_monetaire[:,:,2] + prix_fixe_vehicule_mois[index] * 12 * taille_menage_mat #voiture
            prix_monetaire[:,:,3] = prix_monetaire[:,:,3] + prix_taxi_fixe_mois[index] * 12 * taille_menage_mat #minibus-taxi
            prix_monetaire[:,:,4] = prix_monetaire[:,:,4] + prix_bus_fixe_mois[index] * 12 * taille_menage_mat #bus

            number_hour_week = 40
            number_weeks = 52
            income_per_hour = revenu_ici / number_weeks / number_hour_week
            prix_temps = temps_sortie * param["prix_temps"] * income_per_hour / 60 * 2 * 20 * 12
        
            if NON_LIN == 1:
                prix_temps[temps_sortie > param["limite_temps"]] = (param["limite_temps"] * param["prix_temps"] + (temps_sortie[temps_sortie > param["limite_temps"]] - param["limite_temps"]) * param["prix_temps2"]) * income_per_hour[temps_sortie > param["limite_temps"]] / 60 * 2 * 20 * 12
                                                                
            prix_final = prix_monetaire + prix_temps
    
            if index == 1:
                trans_prix_monetaire_init = prix_monetaire
                trans_prix_temps_init = prix_temps
       
            if option["LOGIT"] == 1:
                mini_prix[:,:,0] = min(prix_final,[],3)
                mini_prix[:,:,1] = mini_prix[:,:,0]
                mini_prix[:,:,2] = mini_prix[:,:,0]
                mini_prix[:,:,3] = mini_prix[:,:,0]
                mini_prix[:,:,4] = mini_prix[:,:,0]

                coeff_logit = param["facteur_logit"] / mini_prix
                mode_logit = logit(coeff_logit, prix_final, trans.nbre_modes)
        
                mult[:,:,index] = (pour_moyenne_logit(coeff_logit, prix_final) / param["facteur_logit"])
                cout_generalise[:,:,index] = single(pour_moyenne_logit(coeff_logit,prix_final) / coeff_logit[:,:,0])
        
                cout_generalise_ancien[:,:,index] = (sum(prix_final * mode_logit,3))
        
                quel = mode_logit #ATTENTION, trans.quel ne depend plus du temps ici
                
                trans_prix_temps[:,:,index] = sum(quel * prix_temps, 3)
            else:
                cout_generalise[:,:,index] = min((prix_final),[],trans_nbre_modes)
                quel[:,:,index] = np.argmin((prix_final),[],trans_nbre_modes)
           
        trans_t_transport = t_trafic + param["year_begin"]
        trans_cout_generalise = cout_generalise

        trans_quel = quel
        trans_mult = mult
        trans_temps_sortie = temps_sortie
           
        self.reliable = trans_reliable
        self.distance_sortie = trans_distance_sortie
        self.cout_generalise = trans_cout_generalise
        self.prix_monetaire_init = trans_prix_monetaire_init
        self.prix_temps_init = trans_prix_temps_init
        self.prix_temps = trans_prix_temps
        self.t_transport = trans_t_transport
        self.quel = trans_quel
        self.mult = trans_mult
        self.temps_sortie = trans_temps_sortie
        




'''
liste={'bustsh','carstt','taxitsh','totaltsh','traintt','bustt','cartsh','taxitt','traintsh','walktsh'};

for i=liste
    i=char(i);
    eval([i,'=importfile_mat(''./data/transport/',i,'.csv'');'])
end

%importation et sauvegarde de la liste des codes des zones
liste1=importfile_mat('./data/transport/carstt.csv',1, 1);
liste2=liste1;
%save('liste.mat','liste1','liste2');

%% importation des coordonn?es des zones

%fichier de donn?es sur els zones: emplois et coordonn?es
[X,Y,TZ2013,Area,diss,Zones,BY5Origins,BY5Destina,PTODOrigin,PTODDestin,emploi_TAZ,emploi_T_1,emploi_T_2,emploi_T_3,emploi_T_4,emploi_T_5,emploi_T_6,emploi_T_7,emploi_T_8,job_total,job_dens,Ink1,Ink2,Ink3,Ink4,job_tot] ...
    = importfile_TAZ('./data/TAZ_amp_2013_proj_centro2.csv');

%on r?-ordonne les temps de transport comme il faut, pour qu'ils soient
%dans le m?me ordre que les zones
cars=zeros(size(TZ2013));
bus=zeros(size(TZ2013));
taxi=zeros(size(TZ2013));
train=zeros(size(TZ2013));
for index1=1:length(TZ2013),
   for index2=1:length(TZ2013),
       choix1=(liste1==TZ2013(index1));
       choix2=(liste2==TZ2013(index2));
       if (sum(choix1)>0)&&(sum(choix2)>0),
           cars(index1,index2)=carstt(choix1,choix2);
           bus(index1,index2)=bustt(choix1,choix2);
           taxi(index1,index2)=taxitt(choix1,choix2);
           train(index1,index2)=traintt(choix1,choix2);
       end
   end
   waitbar(index1/length(TZ2013));
end

%calcul de la distance ? vol d'oiseau d'une zone ? l'autre
distance_vol_oiseau=zeros(size(TZ2013));
for index=1:length(TZ2013),
    distance_vol_oiseau(:,index)=sqrt((X-X(index)).^2+(Y-Y(index)).^2)/1000;
end

%sauvegarde de tout ?a
save('transport_time','cars','bus','train','taxi', 'X','Y','distance_vol_oiseau','Area');

%% OPTIONNEL: calcul des densit;es d'emploi liss?es apr surface

emploi=zeros(size(TZ2013));
for index1=1:length(TZ2013),
    garde=(cars(index1,:)<=5);
    garde(cars(index1,:)==0)=0;
    garde(index1)=1;
    if sum(garde)>0,
       %pour avoir la densit? d'emploi
       emploi(index1)=sum(job_total(garde).*Area(garde))./sum(Area(garde));
       %pour avoir la somme des smplois ? moins d'un certain temps en
       %voiture
       %emploi(index1)=sum(job_total(garde));
    end
end

%% (pour tracer une carte)
trace_points_carte( cdata, X/1000,Y/1000,emploi);
'''
