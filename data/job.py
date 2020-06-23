# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:47:04 2020

@author: Charlotte Liotta
"""

import copy
import numpy.matlib

def import_emploi_CAPE_TOWN2(grille, param, option, macro_data, data_courbe, t):

    #[X,Y,TZ2013,Area,diss,Zones,BY5Origins,BY5Destina,PTODOrigin,PTODDestin,emploi_TAZ,emploi_T_1,emploi_T_2,emploi_T_3,emploi_T_4,emploi_T_5,emploi_T_6,emploi_T_7,emploi_T_8,job_total,job_dens,Ink1,Ink2,Ink3,Ink4,job_tot] ...
    #= importfile_TAZ(strcat(path_nedum, 'TAZ_amp_2013_proj_centro2.csv'))
    
    TAZ = pd.read_csv('./2. Data/TAZ_amp_2013_proj_centro2.csv')
    #load emploi.mat
    #job_total = emploi';
    #emploi_T_5 = emploi_T_5bis';
    #emploi_T_6 = emploi_T_6bis';
    #emploi_T_7 = emploi_T_7bis';
    #emploi_T_8 = emploi_T_8bis';

    TAZ2013_centre = pd.read_csv('./2. Data/TAZ_final.csv')
    listou = copy.deepcopy(TAZ2013_centre.TAZ2013_centre)
    Xou = np.zeros(len(listou))
    You = np.zeros(len(listou))
    corresp = np.zeros(len(listou))
    increment = np.array([i for i in range(len(TAZ.TZ2013))])
    for index1 in range(0, len(listou)):
        Xou[index1] = TAZ.X[TAZ.TZ2013 == listou[index1]]
        You[index1] = TAZ.Y[TAZ.TZ2013 == listou[index1]]
        corresp[index1] = increment[TAZ.TZ2013 == listou[index1]]
               
    J_data = np.transpose([TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink2/3, TAZ.Ink2/3, TAZ.Ink2/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3])

    poly_code_emploi_init = copy.deepcopy(TAZ.TZ2013)
    poly_code_emploi_init_simple = copy.deepcopy(TAZ.TZ2013)
    ID_centre = copy.deepcopy(TAZ.TZ2013)    
    XCoord = TAZ.X
    YCoord = TAZ.Y

    #Total number of households per class

    year_income_distribution = param["year_begin"] + t
    if len(t) == 1:
        total_bracket = [np.ones((len(t),1)), np.transpose(macro_data.spline_pop_inc_distribution(t))]
        avg_inc_bracket = [np.zeros((len(t),1)), np.transpose(macro_data.spline_inc_distribution(t))]
    else:
        total_bracket = [np.ones((len(t), 1)), np.transpose(macro_data.spline_pop_inc_distribution(t))]
        avg_inc_bracket = [np.zeros((len(t),1)), macro_data.spline_inc_distribution(t)]

    avg_inc_class = np.zeros((len(year_income_distribution), param["multiple_class"]))
    total_class = np.zeros((len(year_income_distribution), param["multiple_class"]))
 
    #total income distribution in the city
    for j in range(0, param["multiple_class"]):
        total_class[:, j] = sum(total_bracket[:, param["income_distribution"] == j], axis = 1)
        avg_inc_class[:, j] = sum(avg_inc_bracket[:, param["income_distribution"] == j] * total_bracket[:, param["income_distribution"] == j], axis = 1) / total_class[:, j]

    #Selection of employment centers
    poly_code_emploi_init = np.zeros((1, param["multiple_class"] * len(ID_centre)))
    Jx = np.zeros((1, param["multiple_class"] * len(ID_centre)))
    Jy = np.zeros((1, param["multiple_class"] * len(ID_centre)))
    Jval1 = np.zeros((len(year_income_distribution), param["multiple_class"] * len(ID_centre)))
    avg_inc = np.zeros((len(year_income_distribution), param["multiple_class"] * len(ID_centre)))
    classes = np.zeros((len(year_income_distribution), param["multiple_class"] * len(ID_centre)))


    #Duplication of employment centers for the several income groups
    for i in range(0, len(ID_centre)):
        for j in range(0, param["multiple_class"]):
            print(i)
            print(j)
            poly_code_emploi_init[0, param["multiple_class"] * (i - 1) + j] = ID_centre[i]
            Jx[param["multiple_class"] * (i-1) + j] = XCoord[i] / 1000
            Jy[param["multiple_class"] * (i-1) + j] = YCoord[i] / 1000
            Jval1[:, param["multiple_class"] * (i-1) + j] = np.transpose(numpy.matlib.repmat(sum(J_data[i, param["income_distribution"] == j]), len(year_income_distribution), 1))
            avg_inc[:, param["multiple_class"]*(i-1)+j] = avg_inc_class[:,j]
            classes[:, param["multiple_class"]*(i-1)+j] = j

    ID_centre_poly = range(0, len(poly.code_emploi_init))

    #Selection of employment centers to keep

    if option["polycentric"] == 1:
    
        poly_quel = np.zeros(len(poly_code_emploi_init), 'bool')

        #Manual sorting 
        poly_quel[poly_code_emploi_init == 5101] = np.ones(1, 'bool') #CBD
        poly_quel[poly_code_emploi_init == 2002] = np.ones(1, 'bool') #Bellville
        poly_quel[poly_code_emploi_init == 1201] = np.ones(1, 'bool') #Epping
        poly_quel[poly_code_emploi_init == 1553] = np.ones(1, 'bool') #Claremont
        poly_quel[poly_code_emploi_init == 3509] = np.ones(1, 'bool') #Sommerset West
        poly_quel[poly_code_emploi_init == 5523] = np.ones(1, 'bool') #Table View + Century City

        poly_quel[Jval1[0,:] <= 0] = np.zeros(1, 'bool')

    elif option["polycentric"] == 0:
    
        poly_quel = np.zeros(len(poly_code_emploi_init), 'bool')
        poly_quel[poly_code_emploi_init == 5101] = np.ones(1, 'bool') #CBD

    #Rescale of number of jobs after selection
    #Rescale to include for each center all the jobs located in a defined buffer zone

    distance_buffer = 4
    Jval_temp = Jval1[:, poly_quel]
    sum_class_j = np.zeros(len(year_income_distribution), param["multiple_class"])
    Jdistance = np.sqrt((Jx - np.transpose(Jx[poly_quel]))**2) + ((Jy - np.transpose(Jy[poly_quel])) ** 2)

    for i in range(0, len(year_income_distribution)):
        for j in range(0, param["multiple_class"]):
            Jval_i = Jval1[i,:]
            class_i = classes[i,:]
            poly_class_i = classes[i, poly_quel]
            Jval_temp[i, poly_class_i == j] = Jval_i[class_i == j] * (np.transpose(Jdistance(poly_class_i == j, class_i == j)) < distance_buffer)
            sum_class_j[i, j] = sum(Jval_temp[i, poly_class_i == j])

    #Remove the employment centers that are not significant enough
    ID_centre_poly_quel = ID_centre_poly[poly_quel]
    quel_temp = np.ones(sum(poly_quel))
    for j in range(0, param["multiple_class"]):
        quel_temp[Jval_temp[1, poly_class_i == j] / sum_class_j[1, j] < 0.1] = 0

    quel_temp = np.where(quel_temp == 0, 'False', 'True')
    Jval_temp = Jval_temp[:,quel_temp]
    ID_centre_poly_remove = ID_centre_poly_quel[quel_temp == 0]
    poly_quel(ismember(ID_centre_poly, ID_centre_poly_remove)) = 0
    poly_Jx = Jx[poly_quel]
    poly_Jy = Jy[poly_quel]
    poly_classes = classes[:,poly_quel]
    poly_avg_inc = avg_inc[:,poly_quel]

    #Rescale to keep the correct global income distribution

    sum_class_quel = np.zeros(len(year_income_distribution), param["multiple_class"])
    Jval2 = np.zeros(size(Jval_temp))
    
    for j in range(0, param["multiple_class"]):
    
        sum_class_quel[:,j] = sum(Jval_temp[:, poly_classes[0,:] == j], axis = 1)
        nb_centre_quel = sum(poly.classes[1,:] == j)
    
        Jval2[:, poly_classes[1,:] == j] = Jval_temp[:,poly_classes[1,:] == j] * numpy.matlib.repmat(total_class[:, j] / sum_class_quel[:, j], 1, nb_centre_quel)
        
    #Export 
    annee_Jval = year_income_distribution
    poly_total_hh_class = total_class

    poly_annee = annee_Jval

    increment = range(0, len(poly_avg_inc[0, :]))
    poly_referencement = increment
    poly_garde = poly.quel

    poly_code_emploi_poly = poly_code_emploi_init[poly_quel]
    poly_avg_inc = avg_inc[:, poly_quel]
    poly_classes = classes[:, poly_quel]

    increment = range(0, len(TAZ.TZ2013))
    corresp = np.zeros(len(poly.Jx))
    for index1 in range(0, len(poly_Jx)):
        corresp[index1] = increment(TAZ.TZ2013 == poly_code_emploi_poly[index1])

    poly_corresp = np.transpose(corresp)

    poly_Jval_pour_garde = Jval1
    poly_Jval = Jval2

    #Who can live in informal / formal settlements
    poly_formal = np.array([1, 1, 1, 1])
    poly_backyard = np.array([1, 1, 0, 0])
    poly_settlement = np.array([1, 1, 0, 0])

"""
def importfile_TAZ(filename):
    startRow = 2
    endRow = inf

    formatSpec = '%f%f%f%f%f%f%f%f%f%f%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

    fileID = fopen(filename,'r')

    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
    for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    fclose(fileID);

    X = dataArray{:, 1};
    Y = dataArray{:, 2};
    TZ2013 = dataArray{:, 3};
    Area = dataArray{:, 4};
    diss = dataArray{:, 5};
    Zones = dataArray{:, 6};
    BY5Origins = dataArray{:, 7};
    BY5Destina = dataArray{:, 8};
    PTODOrigin = dataArray{:, 9};
    PTODDestin = dataArray{:, 10};
    emploi_TAZ = dataArray{:, 11};
    emploi_T_1 = dataArray{:, 12};
    emploi_T_2 = dataArray{:, 13};
    emploi_T_3 = dataArray{:, 14};
    emploi_T_4 = dataArray{:, 15};
    emploi_T_5 = dataArray{:, 16};
    emploi_T_6 = dataArray{:, 17};
    emploi_T_7 = dataArray{:, 18};
    emploi_T_8 = dataArray{:, 19};
    job_total = dataArray{:, 20};
    job_dens = dataArray{:, 21};
    Ink1 = dataArray{:, 22};
    Ink2 = dataArray{:, 23};
    Ink3 = dataArray{:, 24};
    Ink4 = dataArray{:, 25};
    job_tot = dataArray{:, 26};

    return X,Y,TZ2013,Area,diss,Zones,BY5Origins,BY5Destina,PTODOrigin,PTODDestin,emploi_TAZ,emploi_T_1,emploi_T_2,emploi_T_3,emploi_T_4,emploi_T_5,emploi_T_6,emploi_T_7,emploi_T_8,job_total,job_dens,Ink1,Ink2,Ink3,Ink4,job_tot

def importfile_centers(filename, startRow, endRow)

    delimiter = '';
    if nargin<=2
        startRow = 2;
        endRow = inf;

    formatSpec = '%f%[^\n\r]';

    fileID = fopen(filename,'r');

    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
    for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
        dataArray{1} = [dataArray{1};dataArrayBlock{1}];
    fclose(fileID);

    return dataArray{:, 1};
"""