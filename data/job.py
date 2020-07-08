# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:47:04 2020

@author: Charlotte Liotta
"""

import copy
import numpy.matlib
import pandas as pd
import numpy as np

class ImportEmploymentData:
        
    def __init__(self):
        
        self

    def import_employment_data(self, grille, param, option, macro_data, t):
        
        #Import data
        TAZ = pd.read_csv('./2. Data/TAZ_amp_2013_proj_centro2.csv') #Number of jobs per Transport Zone (TZ)
        TAZ2013_centre = pd.read_csv('./2. Data/TAZ_final.csv') #The 44 main employment centers
    
        #Coordinates of employment centers
        listou = copy.deepcopy(TAZ2013_centre.TAZ2013_centre)
        Xou = np.zeros(len(listou))
        You = np.zeros(len(listou))
        corresp = np.zeros(len(listou))
        increment = np.array([i for i in range(len(TAZ.TZ2013))])
        for index1 in range(0, len(listou)):
            Xou[index1] = TAZ.X[TAZ.TZ2013 == listou[index1]]
            You[index1] = TAZ.Y[TAZ.TZ2013 == listou[index1]]
            corresp[index1] = increment[TAZ.TZ2013 == listou[index1]]
     
        #Data on employment centers of each income group            
        J_data = np.transpose([TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink2/3, TAZ.Ink2/3, TAZ.Ink2/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3])
    
        poly_code_emploi_init = copy.deepcopy(TAZ.TZ2013)
        poly_code_emploi_init_simple = copy.deepcopy(TAZ.TZ2013)
        ID_centre = copy.deepcopy(TAZ.TZ2013)    
        XCoord = TAZ.X
        YCoord = TAZ.Y

        #Total number of households and average income per class (12 classes)
        #year_income_distribution = param["year_begin"] + t
        year_income_distribution = [x + param["year_begin"] for x in t]
        if len(t) == 1:
            total_bracket = np.transpose(np.concatenate([np.ones((1, len(t))), (macro_data.spline_pop_inc_distribution(t))]))
            avg_inc_bracket = np.transpose(np.concatenate([np.zeros((1, len(t))), (macro_data.spline_inc_distribution(t))]))
        else:
            total_bracket = np.transpose(np.concatenate([np.ones((1, len(t))), (macro_data.spline_pop_inc_distribution(t))])) #Total number of households per class
            avg_inc_bracket = np.transpose(np.concatenate([np.zeros((1, len(t))), (macro_data.spline_inc_distribution(t))])) #Average income for each class

        avg_inc_class = np.zeros((len(year_income_distribution), param["multiple_class"]))
        total_class = np.zeros((len(year_income_distribution), param["multiple_class"]))
 
        #Total number of households and average income per class (4 classes)
        for j in range(0, param["multiple_class"] + 1):
            total_class[:, j-1] = np.sum(total_bracket[:, (param["income_distribution"] == j)], axis = 1)
            avg_inc_class[:, j-1] = np.sum(avg_inc_bracket[:, (param["income_distribution"] == j)] * total_bracket[:, param["income_distribution"] == j], axis = 1) / total_class[:, j-1]

        #Selection of employment centers
        poly_code_emploi_init = np.zeros((param["multiple_class"] * len(ID_centre)))
        Jx = np.zeros((param["multiple_class"] * len(ID_centre))) #Coordonnées X des centres d'emploi
        Jy = np.zeros((param["multiple_class"] * len(ID_centre))) #Coordonnées Y des centres d'emploi
        Jval1 = np.zeros((len(year_income_distribution), param["multiple_class"] * len(ID_centre)))
        avg_inc = np.zeros((len(year_income_distribution), param["multiple_class"] * len(ID_centre))) #Revenu moyen de chaque classe pour chaque centre d'emploi
        classes = np.zeros((len(year_income_distribution), param["multiple_class"] * len(ID_centre))) #Nombre de personnes de chaque classe pour chaque centre d'emploi
        
        #Duplication of employment centers for the several income groups
        for i in range(0, len(ID_centre)):
            for j in range(0, param["multiple_class"]):
                poly_code_emploi_init[param["multiple_class"] * (i) + j] = ID_centre[i]
                Jx[param["multiple_class"] * (i) + j] = XCoord[i] / 1000 #Coordonnées X des centres d'emploi
                Jy[param["multiple_class"] * (i) + j] = YCoord[i] / 1000 #Coordonnées Y des centres d'emploi
                Jval1[:, param["multiple_class"] * (i) + j] = np.transpose(numpy.matlib.repmat(sum(J_data[i, param["income_distribution"] == j]), len(year_income_distribution), 1)) #Nombre de personnes de chacune des 4 classes qui va travailler dans chaque centre d'emploi
                avg_inc[:, param["multiple_class"] * (i) + j] = avg_inc_class[:, j] #Revenu moyen de chaque classe de ménage pour chaque centre d'emploi et chaque année
                classes[:, param["multiple_class"] * (i) + j] = j  #Nombre de personnes de chaque classe de ménage pour chaque centre d'emploi et chaque année
                
        ID_centre_poly = np.array(list(range(0, len(poly_code_emploi_init))))

        #Selection of employment centers to keep
        poly_quel = np.zeros(len(poly_code_emploi_init), 'bool')
        if option["polycentric"] == 1:   #On choisit manuellement 6 centres d'emploi à garder, mais on pourrait en prendre plus
            poly_quel[poly_code_emploi_init == 5101] = np.ones(1, 'bool') #CBD
            poly_quel[poly_code_emploi_init == 2002] = np.ones(1, 'bool') #Bellville
            poly_quel[poly_code_emploi_init == 1201] = np.ones(1, 'bool') #Epping
            poly_quel[poly_code_emploi_init == 1553] = np.ones(1, 'bool') #Claremont
            poly_quel[poly_code_emploi_init == 3509] = np.ones(1, 'bool') #Sommerset West
            poly_quel[poly_code_emploi_init == 5523] = np.ones(1, 'bool') #Table View + Century City
            poly_quel[Jval1[0,:] <= 0] = np.zeros(1, 'bool')
            nb_centres = 6
        elif option["polycentric"] == 0: 
            poly_quel[poly_code_emploi_init == 5101] = np.ones(1, 'bool') #CBD
            nb_centres = 1
            
        #Rescale to include for each center all the jobs located in a defined buffer zone
        distance_buffer = 4
        Jval_temp = Jval1[:, poly_quel] #Nombre de personnes de chaque classe qui vont travailler dans chaque centre d'emploi qu'on a gardé pour chaque date - temporaire
        sum_class_j = np.zeros((len(year_income_distribution), param["multiple_class"])) #Nombre de personnes de chaque classe qui vont travailler dans chaque centre d'emploi qu'on a gardé pour chaque date - final
        Jdistance = np.sqrt(((numpy.matlib.repmat(Jx, len(Jx[poly_quel]), 1) - np.transpose(numpy.matlib.repmat((Jx[poly_quel]), len(Jx), 1)))**2) + ((numpy.matlib.repmat(Jy, len(Jy[poly_quel]), 1) - np.transpose(numpy.matlib.repmat((Jy[poly_quel]), len(Jy), 1)))**2)) #Distance de chaque centre d'emploi aux centres d'emploi qu'on a gardés
        
        for i in range(0, len(year_income_distribution)):
            for j in range(0, param["multiple_class"]):
                Jval_i = Jval1[i,:] #Nombre de personnes de chaque clase qui vont travailler dans chaque centre d'emploi pour l'année i
                class_i = classes[i,:] #Aide à interpréter Jval_i : à quelle classe est-ce que chaque colonne correspond ?
                poly_class_i = classes[i, poly_quel] #à quelle classe est-ce que chaque colonne correspond ? Seulement pour les centres d'emploi qu'on a retenus
                Jval_temp[i, poly_class_i == j] = np.dot(Jval_i[class_i == j], (np.transpose(Jdistance[(poly_class_i == j),][:,(class_i == j)]) < distance_buffer)) #Nombre de personnes d'une classe donnée qui vont travailler dans chaque centre d'emploi (ou à moins de 4 km) pour l'année i
                sum_class_j[i, j] = sum(Jval_temp[i, poly_class_i == j])

        #Remove the employment centers that are not significant enough
        ID_centre_poly_quel = ID_centre_poly[poly_quel]
        quel_temp = np.ones(sum(poly_quel), 'bool')
        for j in range(0, param["multiple_class"]):
            quel_temp = np.where((Jval_temp[0,] / np.concatenate(np.matlib.repmat(sum_class_j[0,], nb_centres, 1)) < 0.1), np.zeros(1, 'bool'), quel_temp) #Si moins de 10% des gens vont travailler dans ce centre d'emploi, on ne le compte pas
        Jval_temp = Jval_temp[:, quel_temp]
        ID_centre_poly_remove = ID_centre_poly_quel[quel_temp == np.zeros(1, 'bool')]
        for j in range(0, len(poly_quel)):
            poly_quel[j] = np.where((ID_centre_poly[j] in ID_centre_poly_remove), np.zeros(1, 'bool'), poly_quel[j])
        poly_Jx = Jx[poly_quel]
        poly_Jy = Jy[poly_quel]
        poly_classes = classes[:, poly_quel]
        poly_avg_inc = avg_inc[:, poly_quel]

        #Rescale to keep the correct global income distribution
        sum_class_quel = np.zeros((len(year_income_distribution), param["multiple_class"]))
        Jval2 = np.zeros(Jval_temp.shape)  
        for j in range(0, param["multiple_class"]):  
            sum_class_quel[:, j] = np.sum(Jval_temp[:, poly_classes[0,:] == j], axis = 1)
            nb_centre_quel = sum(poly_classes[0, :] == j)   
            Jval2[:, poly_classes[0,:] == j] = Jval_temp[:, poly_classes[0,:] == j] * np.transpose(numpy.matlib.repmat(total_class[:, j] /sum_class_quel[:, j], nb_centre_quel, 1))
        
        #Export 
        annee_Jval = year_income_distribution
        poly_total_hh_class = total_class
        poly_annee = annee_Jval
        increment = range(0, len(poly_avg_inc[0, :]))
        poly_referencement = increment
        poly_garde = poly_quel
        poly_code_emploi_poly = poly_code_emploi_init[poly_quel]
        poly_avg_inc = avg_inc[:, poly_quel]
        poly_classes = classes[:, poly_quel]
        increment = np.array(list(range(0, len(TAZ.TZ2013))))
        corresp = np.zeros(len(poly_Jx))
        for index1 in range(0, len(poly_Jx)):
            corresp[index1] = increment[TAZ.TZ2013 == poly_code_emploi_poly[index1]]
        poly_corresp = np.transpose(corresp)
        poly_Jval_pour_garde = Jval1
        poly_Jval = Jval2

        #Select which income class can live in informal / formal settlements
        poly_formal = np.array([1, 1, 1, 1])
        poly_backyard = np.array([1, 1, 0, 0])
        poly_settlement = np.array([1, 1, 0, 0])
        
        self.code_emploi_init_simple = poly_code_emploi_init_simple
        self.code_emploi_init = poly_code_emploi_init
        self.quel = poly_quel
        self.class_i = poly_class_i
        self.Jx = poly_Jx
        self.Jy = poly_Jy
        self.avg_inc = poly_avg_inc
        self.total_hh_class = poly_total_hh_class
        self.annee = poly_annee
        self.referencement = poly_referencement
        self.garde = poly_garde
        self.code_emploi_poly = poly_code_emploi_poly
        self.avg_inc = poly_avg_inc
        self.classes = poly_classes
        self.corresp = poly_corresp
        self.Jval_pour_garde = poly_Jval_pour_garde
        self.Jval = poly_Jval
        self.formal = poly_formal
        self.backyard = poly_backyard
        self.settlement = poly_settlement
