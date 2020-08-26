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
        
        # %% Import data
        TAZ = pd.read_csv('./2. Data/Basile data/TAZ_amp_2013_proj_centro2.csv') #Number of jobs per Transport Zone (TZ)
        TAZ2013_centre = pd.read_csv('./2. Data/Basile data/TAZ_final.csv') #The 44 main employment centers
    
        #Alt
        #data_job_2015 = pd.read_csv('./2. Data/Basile data/BASE_2015.csv') #Employment centers 2015

        # %% Coordinates of employment centers
        listou = copy.deepcopy(TAZ2013_centre.TAZ2013_centre)
        Xou = np.zeros(len(listou))
        You = np.zeros(len(listou))
        corresp = np.zeros(len(listou))
        increment = np.array([i for i in range(len(TAZ.TZ2013))])
        for index1 in range(0, len(listou)):
            Xou[index1] = TAZ.X[TAZ.TZ2013 == listou[index1]]
            You[index1] = TAZ.Y[TAZ.TZ2013 == listou[index1]]
            corresp[index1] = increment[TAZ.TZ2013 == listou[index1]]
        poly_code_emploi_init = copy.deepcopy(TAZ.TZ2013)
        poly_code_emploi_init_simple = copy.deepcopy(TAZ.TZ2013)
        ID_centre = copy.deepcopy(TAZ.TZ2013)    
        XCoord = TAZ.X
        YCoord = TAZ.Y
     
        # %% Data on employment centers of each income group            
        J_data = np.transpose([TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink2/3, TAZ.Ink2/3, TAZ.Ink2/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3])
        
        #Alt
        #J_data = np.transpose([data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12, data_job_2015.JOBS/12]) #Number of employees in each TZ for the 12 income classes
        #ID_centre = data_job_2015.TZ2015
        # %% Total number of households and average income per class (12 classes)
        year_income_distribution = [x + param["baseline_year"] for x in t]
        if len(t) == 1:
            total_bracket = np.transpose(np.concatenate([np.ones((1, len(t))), (macro_data.spline_pop_inc_distribution(t))]))
            avg_inc_bracket = np.transpose(np.concatenate([np.zeros((1, len(t))), (macro_data.spline_inc_distribution(t))]))
        else:
            total_bracket = np.transpose(np.concatenate([np.ones((1, len(t))), (macro_data.spline_pop_inc_distribution(t))])) #Total number of households per class
            avg_inc_bracket = np.transpose(np.concatenate([np.zeros((1, len(t))), (macro_data.spline_inc_distribution(t))])) #Average income for each class

        avg_inc_class = np.zeros((len(year_income_distribution), param["nb_of_income_classes"]))
        total_class = np.zeros((len(year_income_distribution), param["nb_of_income_classes"]))
 
        # %% Total number of households and average income per class (4 classes)
        for j in range(0, param["nb_of_income_classes"] + 1):
            total_class[:, j-1] = np.sum(total_bracket[:, (param["income_distribution"] == j)], axis = 1)
            avg_inc_class[:, j-1] = np.sum(avg_inc_bracket[:, (param["income_distribution"] == j)] * total_bracket[:, param["income_distribution"] == j], axis = 1) / total_class[:, j-1]

        # %% Selection of employment centers
        poly_code_emploi_init = np.zeros((param["nb_of_income_classes"] * len(ID_centre)))
        Jx = np.zeros((param["nb_of_income_classes"] * len(ID_centre))) #Coordonnées X des centres d'emploi
        Jy = np.zeros((param["nb_of_income_classes"] * len(ID_centre))) #Coordonnées Y des centres d'emploi
        Jval1 = np.zeros((len(year_income_distribution), param["nb_of_income_classes"] * len(ID_centre)))
        avg_inc = np.zeros((len(year_income_distribution), param["nb_of_income_classes"] * len(ID_centre))) #Revenu moyen de chaque classe pour chaque centre d'emploi
        classes = np.zeros((len(year_income_distribution), param["nb_of_income_classes"] * len(ID_centre))) #Nombre de personnes de chaque classe pour chaque centre d'emploi
        
        # %% Duplication of employment centers for the several income groups
        for i in range(0, len(ID_centre)):
            for j in range(0, param["nb_of_income_classes"]):
                poly_code_emploi_init[param["nb_of_income_classes"] * (i) + j] = ID_centre[i]
                Jx[param["nb_of_income_classes"] * (i) + j] = XCoord[i] / 1000 #Coordonnées X des centres d'emploi
                Jy[param["nb_of_income_classes"] * (i) + j] = YCoord[i] / 1000 #Coordonnées Y des centres d'emploi
                Jval1[:, param["nb_of_income_classes"] * (i) + j] = np.transpose(numpy.matlib.repmat(sum(J_data[i, param["income_distribution"] == j]), len(year_income_distribution), 1)) #Nombre de personnes de chacune des 4 classes qui va travailler dans chaque centre d'emploi
                avg_inc[:, param["nb_of_income_classes"] * (i) + j] = avg_inc_class[:, j] #Revenu moyen de chaque classe de ménage pour chaque centre d'emploi et chaque année
                classes[:, param["nb_of_income_classes"] * (i) + j] = j  #Nombre de personnes de chaque classe de ménage pour chaque centre d'emploi et chaque année
                
        ID_centre_poly = np.array(list(range(0, len(poly_code_emploi_init))))

        # %% Selection of employment centers to keep
        poly_quel = np.zeros(len(poly_code_emploi_init), 'bool')
        if option["nb_employment_center"] == 185:   #On choisit manuellement 6 centres d'emploi à garder, mais on pourrait en prendre plus
            for i in range(1, int(len(poly_code_emploi_init) / param["nb_of_income_classes"]) + 1):
                if ((Jval1[0, param["nb_of_income_classes"] * i - 1] + Jval1[0, param["nb_of_income_classes"] * i - 2] + Jval1[0, param["nb_of_income_classes"] * i - 3] + Jval1[0, param["nb_of_income_classes"] * i - 4]) > 2478):
                    poly_quel[param["nb_of_income_classes"] * i - 1] = np.ones(1, 'bool')
                    poly_quel[param["nb_of_income_classes"] * i - 2] = np.ones(1, 'bool')
                    poly_quel[param["nb_of_income_classes"] * i - 3] = np.ones(1, 'bool')
                    poly_quel[param["nb_of_income_classes"] * i - 4] = np.ones(1, 'bool')
                    nb_centres = 185
        elif option["nb_employment_center"] == 6:
            poly_quel[poly_code_emploi_init == 5101] = np.ones(1, 'bool') #CBD
            poly_quel[poly_code_emploi_init == 2002] = np.ones(1, 'bool') #Bellville
            poly_quel[poly_code_emploi_init == 1201] = np.ones(1, 'bool') #Epping
            poly_quel[poly_code_emploi_init == 1553] = np.ones(1, 'bool') #Claremont
            poly_quel[poly_code_emploi_init == 3509] = np.ones(1, 'bool') #Sommerset West
            poly_quel[poly_code_emploi_init == 5523] = np.ones(1, 'bool') #Table View + Century City
            poly_quel[Jval1[0,:] <= 0] = np.zeros(1, 'bool')
            nb_centres = 6
            
            #Alt
            #Jx_zone = np.empty(6)
            #Jy_zone = np.empty(6)
            #Jobs_zone = np.empty(6)
            #CBD
            #zone = ((data_job_2015.MZ2015 > 0) & (data_job_2015.MZ2015 < 5)) | (data_job_2015.MZ2015 == 8)
            #Jx_zone[0] = sum(data_job_2015.JOBS[zone] * data_job_2015.Xcoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jy_zone[0] = sum(data_job_2015.JOBS[zone] * data_job_2015.Ycoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jobs_zone[0] = sum(data_job_2015.JOBS[zone])
            
            #Bellville
            #zone = (data_job_2015.MZ2015 == 11) | (data_job_2015.MZ2015 == 19) | (data_job_2015.MZ2015 == 20)
            #Jx_zone[1] = sum(data_job_2015.JOBS[zone] * data_job_2015.Xcoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jy_zone[1] = sum(data_job_2015.JOBS[zone] * data_job_2015.Ycoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jobs_zone[1] = sum(data_job_2015.JOBS[zone])
    
            #Epping - Airport
            #zone = (data_job_2015.MZ2015 == 12) | (data_job_2015.MZ2015 == 21)  | (data_job_2015.MZ2015 == 29)
            #Jx_zone[2] = sum(data_job_2015.JOBS[zone] * data_job_2015.Xcoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jy_zone[2] = sum(data_job_2015.JOBS[zone] * data_job_2015.Ycoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jobs_zone[2] = sum(data_job_2015.JOBS[zone])
            
            #Claremont-Rondebosch
            #zone = (data_job_2015.MZ2015 == 7) | (data_job_2015.MZ2015 == 15)
            #Jx_zone[3] = sum(data_job_2015.JOBS[zone] * data_job_2015.Xcoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jy_zone[3] = sum(data_job_2015.JOBS[zone] * data_job_2015.Ycoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jobs_zone[3] = sum(data_job_2015.JOBS[zone])
            
            #Strand
            #zone = (data_job_2015.MZ2015 == 35)
            #Jx_zone[4] = sum(data_job_2015.JOBS[zone] * data_job_2015.Xcoord[zone] / 1000)  / sum(data_job_2015.JOBS[zone])
            #Jy_zone[4] = sum(data_job_2015.JOBS[zone] * data_job_2015.Ycoord[zone] / 1000)  / sum(data_job_2015.JOBS[zone])
            #Jobs_zone[4] = sum(data_job_2015.JOBS[zone])
    
            #Century City
            #zone = (data_job_2015.MZ2015 == 5) | (data_job_2015.MZ2015 == 6)
            #Jx_zone[5] = sum(data_job_2015.JOBS[zone] * data_job_2015.Xcoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jy_zone[5] = sum(data_job_2015.JOBS[zone] * data_job_2015.Ycoord[zone] / 1000) / sum(data_job_2015.JOBS[zone])
            #Jobs_zone[5] = sum(data_job_2015.JOBS[zone])
    
        elif option["nb_employment_center"] == 1: 
            poly_quel[poly_code_emploi_init == 5101] = np.ones(1, 'bool') #CBD
            nb_centres = 1
            
            #Alt
            #Jx_zone = grid.x_center
            #Jy_zone = grid.y_center
            #Jobs_zone = sum(data_job_2015.JOBS)
            
        # %% Rescale to include for each center all the jobs located in a defined buffer zone
        distance_buffer = 4
        Jval_temp = Jval1[:, poly_quel] #Nombre de personnes de chaque classe qui vont travailler dans chaque centre d'emploi qu'on a gardé pour chaque date - temporaire
        sum_class_j = np.zeros((len(year_income_distribution), param["nb_of_income_classes"])) #Nombre de personnes de chaque classe qui vont travailler dans chaque centre d'emploi qu'on a gardé pour chaque date - final
        Jdistance = np.sqrt(((numpy.matlib.repmat(Jx, len(Jx[poly_quel]), 1) - np.transpose(numpy.matlib.repmat((Jx[poly_quel]), len(Jx), 1)))**2) + ((numpy.matlib.repmat(Jy, len(Jy[poly_quel]), 1) - np.transpose(numpy.matlib.repmat((Jy[poly_quel]), len(Jy), 1)))**2)) #Distance de chaque centre d'emploi aux centres d'emploi qu'on a gardés
        
        for i in range(0, len(year_income_distribution)):
            for j in range(0, param["nb_of_income_classes"]):
                Jval_i = Jval1[i,:] #Nombre de personnes de chaque classe qui vont travailler dans chaque centre d'emploi pour l'année i
                class_i = classes[i,:] #Aide à interpréter Jval_i : à quelle classe est-ce que chaque colonne correspond ?
                poly_class_i = classes[i, poly_quel] #à quelle classe est-ce que chaque colonne correspond ? Seulement pour les centres d'emploi qu'on a retenus
                Jval_temp[i, poly_class_i == j] = np.dot(Jval_i[class_i == j], (np.transpose(Jdistance[(poly_class_i == j),][:,(class_i == j)]) < distance_buffer)) #Nombre de personnes d'une classe donnée qui vont travailler dans chaque centre d'emploi (ou à moins de 4 km) pour l'année i
                sum_class_j[i, j] = sum(Jval_temp[i, poly_class_i == j])

        # %% Remove the employment centers that are not significant enough
        if option["nb_employment_center"] == 6:    
            ID_centre_poly_quel = ID_centre_poly[poly_quel]
            quel_temp = np.ones(sum(poly_quel), 'bool')
            for j in range(0, param["nb_of_income_classes"]):
                quel_temp = np.where((Jval_temp[0,] / np.concatenate(np.matlib.repmat(sum_class_j[0,], nb_centres, 1)) < 0.1), np.zeros(1, 'bool'), quel_temp) #Si moins de 10% des gens vont travailler dans ce centre d'emploi, on ne le compte pas
            Jval_temp = Jval_temp[:, quel_temp]
            ID_centre_poly_remove = ID_centre_poly_quel[quel_temp == np.zeros(1, 'bool')]
            for j in range(0, len(poly_quel)):
                poly_quel[j] = np.where((ID_centre_poly[j] in ID_centre_poly_remove), np.zeros(1, 'bool'), poly_quel[j])
        poly_Jx = Jx[poly_quel]
        poly_Jy = Jy[poly_quel]
        poly_classes = classes[:, poly_quel]
        poly_avg_inc = avg_inc[:, poly_quel]

        # %% Rescale to keep the correct global income distribution
        sum_class_quel = np.zeros((len(year_income_distribution), param["nb_of_income_classes"]))
        Jval2 = np.zeros(Jval_temp.shape)  
        for j in range(0, param["nb_of_income_classes"]):  
            sum_class_quel[:, j] = np.sum(Jval_temp[:, poly_classes[0,:] == j], axis = 1)
            nb_centre_quel = sum(poly_classes[0, :] == j)   
            Jval2[:, poly_classes[0,:] == j] = Jval_temp[:, poly_classes[0,:] == j] * np.transpose(numpy.matlib.repmat(total_class[:, j] /sum_class_quel[:, j], nb_centre_quel, 1))
        
        # %% Export 
        increment = np.array(list(range(0, len(TAZ.TZ2013))))
        corresp = np.zeros(len(poly_Jx))
        for index1 in range(0, len(poly_Jx)):
            corresp[index1] = increment[TAZ.TZ2013 == poly_code_emploi_init[poly_quel][index1]]



        self.code_emploi_init_simple = poly_code_emploi_init_simple #List of the employment centers
        self.code_emploi_init = poly_code_emploi_init #List of the employment centers * the number of income groups
        self.quel = poly_quel #Keeps only the employment centers that we want to keep and the income groups that are working in those employment centers
        self.class_i = poly_class_i #List of the income groups that we keep
        self.Jx = poly_Jx #X coordinates of the employment centers that we keep
        self.Jy = poly_Jy #Y coordinates of the employment centers that we keep
        self.avg_inc = avg_inc[:, poly_quel] #Average income for each year and each employment center
        self.total_hh_class = total_class #Number of each household working in each class for each year
        self.annee = year_income_distribution #Years of the analysis
        self.referencement = range(0, len(poly_avg_inc[0, :])) #Number of income groups
        self.garde = poly_quel #Employment centers and income groups that we keep
        self.code_emploi_poly = poly_code_emploi_init[poly_quel] #Code of the employment centers that we keep
        self.classes = classes[0, poly_quel] #Income groups that we keep, for each year
        self.corresp = np.transpose(corresp) #Code TAZ auquel ça correspond
        self.Jval_pour_garde = Jval1
        self.Jval = Jval2 #Number of households of each income group per employment center and per year
        self.formal = np.array([1, 1, 1, 1]) #Select which income class can live in formal settlements
        self.backyard = np.array([1, 1, 0, 0]) #Select which income class can live in backyard settlements
        self.settlement = np.array([1, 1, 0, 0]) #Select which income class can live in informal settlements