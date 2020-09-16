# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:08:34 2020

@author: Charlotte Liotta
"""

import pandas as pd
import numpy as np
import scipy
from scipy.interpolate import interp1d
import numpy.matlib
from scipy.interpolate import griddata

def SP_to_grid_2011_1(data_SP, SP_Code, grid):  
    grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP_intersect.csv', sep = ';')   
    data_grid = np.zeros(len(grid.dist))   
    for index in range(0, len(grid.dist)):  
        intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])
        area_exclu = 0       
        for i in range(0, len(intersect)):     
            if len(data_SP[SP_Code == intersect[i]]) == 0:                      
                area_exclu = area_exclu + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])])
            else:
                data_grid[index] = data_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * data_SP[SP_Code == intersect[i]]       
        if area_exclu > 0.9 * sum(grid_intersect.Area[grid_intersect.ID_grille == grid.ID[index]]):
            data_grid[index] = np.nan         
        else:
            if (sum(grid_intersect.Area[grid_intersect.ID_grille == grid.ID[index]]) - area_exclu) > 0:
                data_grid[index] = data_grid[index] / (sum(grid_intersect.Area[grid_intersect.ID_grille == grid.ID[index]]) - area_exclu)
            else:
               data_grid[index] = np.nan 
                
    return data_grid


def SP_to_grid_2011_2(data_SP, SP_Code, grid):
    
    grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP_intersect.csv', sep = ';')  
    data_grid = np.zeros(len(grid.dist))   
    for index in range(0, len(grid.dist)): 
        intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])
        for i in range(0, len(intersect)): 
            if len(data_SP[SP_Code == intersect[i]]) != 0:  
                data_grid[index] = data_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * data_SP[SP_Code == intersect[i]] / sum(grid_intersect.Area[grid_intersect.SP_CODE == intersect[i]])

    return data_grid

def SP_to_grid_2001(data_SP, Code_SP_2001, grid):
    #pour des variables extensives
    grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP2001_intersect.csv', sep = ';') 
    data_grid = np.zeros(len(grid.dist)) 
    for index in range(0, len(grid.dist)):
        intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])     
        for i in range(0, len(intersect)): 
            if len(data_SP[Code_SP_2001 == intersect[i]]) != 0:
                data_grid[index] = data_grid[index] + sum(grid_intersect.area_intersection[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * data_SP[Code_SP_2001 == intersect[i]] / sum(grid_intersect.area_intersection[grid_intersect.SP_CODE == intersect[i]])
    return data_grid

def SAL_to_grid(data_SAL, SAL_Code_conversion, grid):
    #to transform data at the Census 2011 SAL level to data at the grid level
    grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SAL_intersect.csv', sep = ';') 
    data_grid = np.zeros(len(grid.dist))

    for index in range (0, len(grid.dist)):
        intersect = np.unique(grid_intersect.OBJECTID_1[grid_intersect.ID_grille == grid.ID[index]])
        if len(intersect) == 0:
            data_grid[index] = np.nan
        else:
            for i in range(0, len(intersect)):
                if len(data_SAL[SAL_Code_conversion == intersect[i]]) > 0:
                    data_grid[index] = data_grid[index] + sum(grid_intersect.Area_inter[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.OBJECTID_1 == intersect[i])]) * data_SAL[SAL_Code_conversion == intersect[i]]
            if sum(grid_intersect.Area_inter[grid_intersect.ID_grille == grid.ID[index]]) < 150000:
                data_grid[index] = np.nan
            else:
                data_grid[index] = data_grid[index] / (sum(grid_intersect.Area_inter[grid_intersect.ID_grille == grid.ID[index]]))
    return data_grid

def import_SAL_land_use(grid):
    sal_ea_inters = pd.read_csv('./2. Data/Basile data/SAL_EA_inters_data_landuse.csv', sep = ';') 
    urb = sal_ea_inters.Collective_living_quarters + sal_ea_inters.Formal_residential + sal_ea_inters.Informal_residential
    non_urb = sal_ea_inters.Commercial + sal_ea_inters.Farms + sal_ea_inters.Industrial + sal_ea_inters.Informal_residential + sal_ea_inters.Parks_and_recreation + sal_ea_inters.Small_Holdings + sal_ea_inters.Vacant
    return urb /(urb+non_urb)

def prix2_polycentrique3(t_transport, cout_generalise, param, t):
    index1, index2, ponder1, ponder2 = cree_ponder(t + param["baseline_year"], t_transport)
    sortie = np.zeros((cout_generalise.shape[0], cout_generalise.shape[1]))
    sortie[:, :] = (ponder1 * cout_generalise[:, :, index1]) + (ponder2 * cout_generalise[:, :, index2])    
    return sortie

def cree_ponder(valeur,vecteur):
    vecteur_centre = vecteur - valeur
    valeur_mini = np.min(np.abs(vecteur_centre))
    index = np.argmin(np.abs(vecteur_centre))

    if valeur_mini == 0:
        index1 = index
        index2 = index
        ponder1 = 1
        ponder2 = 0
    else:
        vecteur_neg = vecteur_centre
        vecteur_neg[vecteur_neg > 0] = np.nan
        rien1 = np.max(vecteur_neg)
        index1 = np.argmax(vecteur_neg)
    
        vecteur_pos = vecteur_centre
        vecteur_pos[vecteur_pos < 0] = np.nan
        rien2 = np.min(vecteur_pos)
        index2 = np.argmin(vecteur_pos)
    
        ponder1 = np.abs(rien1) / (rien2 - rien1)
        ponder2 = 1 - ponder1
    return index1, index2, ponder1, ponder2

def griddata_hier(a,b,c,x,y):
    test = ~np.isnan(c)

    #scatteredInterpolant does not extrapolate if 'none' is the second method parameter
    #'linear' is a linear extrapolation based on the gradient at the border (can cause problem locally - decreasing transport times)
    surface_linear = griddata(a[test], b[test], c[test], 'linear')
    surface_nearest = griddata(a[test], b[test], c[test], 'nearest')

    #If the extrapolated data is lower than the nearest neighboor, we take the nearest neighboor
    return max(surface_linear(np.transpose(x), np.transpose(y)), surface_nearest(np.transpose(x), np.transpose(y)))

def import_metro_data(job, grid, param):
    """ Import and estimate transport time by metro """
    
    metro_station = pd.read_csv('./2. Data/Basile data/metro_station_poly.csv', sep = ';')   
    station_line_time = metro_station[["Bellvill1_B", "Bellvill2_M", "Bellvill3_S", "Bonteheuwel1_C", "Bonteheuwel2_B", "Bonteheuwel3_K", "Capeflats", "Malmesbury", "Simonstown", "Worcester"]]
    duration = np.zeros((len(metro_station.ID_station), len(metro_station.ID_station)))
    
    for i in range (0, len(metro_station.ID_station)): #matrice des temps O-D entre les stations de métro
        for j in range(0, i):
            if (i == j):
                duration[i, j] = 0
            elif np.dot(station_line_time.iloc[i], station_line_time.iloc[j]) > 0: #pas besoin de faire de changement
                temps = np.abs(station_line_time.iloc[j] - station_line_time.iloc[i])
                temps = np.where((station_line_time.iloc[j] == 0) | (station_line_time.iloc[i] == 0), np.nan, temps)
                duration[i,j] = np.nanmin(temps) + param["waiting_time_metro"]
                duration[j,i] = duration[i,j]
            else: #il faut faire un changement
                line_i = station_line_time.iloc[i] > 0
                line_j = station_line_time.iloc[j] > 0
                noeud = np.zeros((len(metro_station.ID_station),1), 'bool')
                for k in range(0, len(metro_station.ID_station)):
                    if (sum(station_line_time.iloc[k] * station_line_time.iloc[i]) > 0) & (sum(station_line_time.iloc[k] * station_line_time.iloc[j]) > 0):
                        noeud[k] = np.ones(1, 'bool')
                temps1 = (np.abs(numpy.matlib.repmat(station_line_time.iloc[j][line_j].squeeze(), int(sum(noeud)), 1) - station_line_time.loc[noeud, (np.array(line_j))]))
                temps2 = (np.abs(numpy.matlib.repmat(station_line_time.iloc[i][line_i].squeeze(), int(sum(noeud)), 1) - station_line_time.loc[noeud, (np.array(line_i))]))
                duration[i,j] = np.amin(np.amin(temps1, axis = 1) + np.amin(temps2, axis = 1))
                duration[i,j] = duration[i,j] + 2 * param["waiting_time_metro"]
                duration[j,i] = duration[i,j]

    #pour chaque point de grille la station la plus proche, et distance
    ID_station_grille = scipy.interpolate.griddata((metro_station.X_cape / 1000, metro_station.Y_cape / 1000), (metro_station.ID_station - 1), (grid.horiz_coord, grid.vert_coord), method = 'nearest')
    distance_grille = np.zeros((len(grid.horiz_coord), 1))

    #Pour chaque centre d'emploi la station la plus proche, et distance
    ID_station_center = scipy.interpolate.griddata((metro_station.X_cape / 1000, metro_station.Y_cape / 1000), (metro_station.ID_station - 1), (job.Jx, job.Jy), method = 'nearest')
    distance_center = np.zeros((len(job.Jx), 1))
    for i in range(0, len(job.Jx)):
        distance_center[i] = np.sqrt((job.Jx[i] - metro_station.X_cape[ID_station_center[i]] / 1000) ** 2 + (job.Jy[i] - metro_station.Y_cape[ID_station_center[i]] / 1000) ** 2)

    #calcul de la matrice des durées
    duration_metro = np.zeros((len(grid.dist), len(job.Jx)))
    distance_metro = np.zeros((len(grid.dist), len(job.Jx)))
    for i in range(0, len(grid.horiz_coord)):
        distance_grille[i] = np.sqrt(((grid.horiz_coord[i] - metro_station.X_cape[ID_station_grille[i]] / 1000) ** 2) + ((grid.vert_coord[i] - metro_station.Y_cape[ID_station_grille[i]] / 1000) ** 2))
        for j in range(0, len(job.Jx)):
            duration_metro[i,j] = (distance_grille[i] + distance_center[j]) * 1.2 / (param["walking_speed"] / 60) + duration[ID_station_grille[i], ID_station_center[j]]
            distance_metro[i,j] = max(np.sqrt(((grid.x_center - metro_station.X_cape[ID_station_grille[i]] / 1000) ** 2) + (grid.y_center - metro_station.Y_cape[ID_station_grille[i]] / 1000) ** 2), np.sqrt((grid.x_center - metro_station.X_cape[ID_station_center[j]] / 1000) ** 2 + (grid.y_center - metro_station.Y_cape[ID_station_center[j]] / 1000) ** 2))

    duration_metro = np.transpose(duration_metro)
    distance_metro = np.transpose(distance_metro)

    return distance_metro, duration_metro

def revenu2_polycentrique(macro, param, option, grid, job, t_trafic, index):
    #evolution du revenu...
    #revenu_tmp = interp1d(np.array(poly.annee) - param["year_begin"], poly.avg_inc, T)
    revenu_tmp = interp1d(np.array(job.annee) - param["baseline_year"], np.transpose(job.avg_inc))
    #revenu = np.zeros(((len(poly.Jx), len(grille.dist)))) #Pour chaque classe de ménage et chaque centre d'emploi, habitant en chaque point de la ville, chaque année
    #temp = np.matlib.repmat(pd.DataFrame(revenu_tmp(t_trafic[index])), 1, (grille.dist).shape)
    #revenu[:,:, index] = np.reshape(temp, 4, (grille.dist).shape)
    if isinstance(t_trafic, int):
        revenu = revenu_tmp(t_trafic)
    else:
        revenu = revenu_tmp(t_trafic[index])
    return revenu

def interest_rate(macro_data, T):
    number_years_interest_rate = 5
    interest_rate_3_years = macro_data.spline_notaires(range(int(T) - number_years_interest_rate + 1, int(T)))
    interest_rate_3_years[interest_rate_3_years < 0] = np.nan
    return np.nanmean(interest_rate_3_years)/100


def InterpolateIncomeEvolution(macro_data, param, option, grid, job, T):
    income_tmp = interp1d(job.year - param["baseline_year"], np.transpose(job.averageIncomeGroup))
    income_tmp = income_tmp(int(T))
    income = np.matlib.repmat(income_tmp, len(grid.dist), 1)

    return income

def ComputeIncomeNetOfCommuting(param, costTime, trans_monetaryCost, grid, poly, data, param_lambda, incomeCenters, year):

    annualToHourly = 1 / (8*20*12)
    costTime[np.isnan(costTime)] = 10 ** 2
    monetaryCost = trans_monetaryCost[:, :, :, year] * annualToHourly #en coût par heure
    monetaryCost[np.isnan(monetaryCost)] = 10**3 * annualToHourly
    incomeCenters = incomeCenters * annualToHourly
    xInterp = grid.horiz_coord
    yInterp = grid.vert_coord
    modalShares = np.zeros((len(incomeCenters), costTime.shape[1], 5, param["nb_of_income_classes"]))
    ODflows = np.zeros((len(incomeCenters), costTime.shape[1], param["nb_of_income_classes"]))
    incomeNetOfCommuting = np.zeros((param["nb_of_income_classes"], costTime.shape[1]))
    averageIncome = np.zeros((param["nb_of_income_classes"], costTime.shape[1]))

    for j in range(0, param["nb_of_income_classes"]):
        
        #Household size varies with transport costs
        householdSize = param["household_size"][j]
        whichCenters = incomeCenters[:,j] > -100000
        incomeCentersGroup = incomeCenters[whichCenters, j]
           
        #Transport costs and employment allocation (cout par heure)
        transportCostModes = householdSize * monetaryCost[whichCenters,:,:] + (costTime[whichCenters,:,:] * np.repeat(np.transpose(np.matlib.repmat(incomeCentersGroup, costTime.shape[1], 1))[:, :, np.newaxis], 5, axis=2))
        
        #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
        valueMax = (np.min(param_lambda * transportCostModes, axis = 2) - 500)
        
        #Modal shares
        modalShares[whichCenters,:,:,j] = np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2))  / np.repeat(np.nansum(np.exp(-param_lambda *transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2)[:, :, np.newaxis], 5, axis=2)
        
        #Transport costs
        transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2) - valueMax))
        
        #minIncome is also to prevent diverging exponentials
        minIncome = np.nanmax(param_lambda  * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - 700
        
        #OD flows
        #ODflows[whichCenters,:,j] = np.exp(param_lambda * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1)) - transportCost) - minIncome) / np.transpose(np.matlib.repmat(np.nansum(np.exp(param_lambda * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1)) - transportCost) - minIncome), 1), 24014, 1))
        ODflows[whichCenters,:,j] = np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome), 0)

        #Income net of commuting (correct formula)
        incomeNetOfCommuting[j,:] = 1 /param_lambda * (np.log(np.nansum(np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome), 0)) + minIncome)
        
        #Average income earned by a worker
        averageIncome[j,:] = np.nansum(ODflows[whichCenters,:,j] * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))))


    incomeNetOfCommuting = incomeNetOfCommuting / annualToHourly
    averageIncome = averageIncome / annualToHourly
    
    return incomeNetOfCommuting, modalShares, ODflows, averageIncome


def griddata_extra(a,b,c,x,y, extrapolate, coordinate_center):
#Like a griddata but with extrapolation outside the area of data availability

    test = ~np.isnan(c)
    if sum(test) < 10:   
        output = np.empty(len(x))
        print('problematic employment: %g', coordinate_center[0], coordinate_center[1])
    else:

        #scatteredInterpolant does not extrapolate if 'none' is the second method parameter

        if extrapolate == 1:

            x_center = coordinate_center[0]
            y_center = coordinate_center[1]

            #Extrapolation by linear trend on angle slices
            interpolated_values = griddata((a[test], b[test]), c[test], (x, y), method='linear')
            extrapolated_values = np.empty(len(interpolated_values))
            interv_angle = np.pi/10
            angles = np.arange(interv_angle/2, (2*np.pi - interv_angle/2), interv_angle)
            grid_angles = np.arctan2(y - y_center, x - x_center) + np.pi
            grid_distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
    
            for theta in range(0, len(angles)):
                which_angles = (grid_angles >= angles[theta] - interv_angle / 2) & (grid_angles <= angles[theta] + interv_angle / 2)
                if sum(which_angles * ~np.isnan(interpolated_values)) != 0:
                    coefficient = np.nanmean(interpolated_values[which_angles] / grid_distance[which_angles])
                    max_value = np.max(interpolated_values[which_angles & ~np.isnan(interpolated_values)])
                    which_max = np.argmax(interpolated_values[which_angles & ~np.isnan(interpolated_values)])
                    max_distance = grid_distance[which_angles & ~np.isnan(interpolated_values)]
                    max_distance = max_distance.iloc[which_max]
                    extrapolated_values[which_angles] = max_value + coefficient * (grid_distance[which_angles] - max_distance)
            output = interpolated_values
            output[np.isnan(output)] = extrapolated_values[np.isnan(output)]
    
        else:
  
            output = griddata((a[test], b[test]), c[test], (x, y), method='linear')
            
            
        return output

