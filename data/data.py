# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:02:26 2020

@author: Charlotte Liotta
"""

from data.functions_to_import_data import *
import copy
import pandas as pd
import numpy as np
import math

class ImportHouseholdsData:
        
    def __init__(self):
        
        self
        
    def import_data(self, grid, param, option):
        
        if option["load_households_data"] == 0:
            data = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/data.mat')['data']
            
            self.Code_SP_2011 = data['spCode'][0][0].squeeze() #Administrative divisions, 1046
            self.X_SP_2011 = data['spX'][0][0].squeeze() #X coordinates, 1046
            self.Y_SP_2011 = data['spY'][0][0].squeeze() #Y coordinates, 1046
            self.distance_SP_2011 = data['sp2011Distance'][0][0].squeeze() #distance to the city center, 1046
            self.area_SP_2011 = data['sp2011Area'][0][0].squeeze() #SP area, 1046 
            self.CT_SP_2011 = data['sp2011CapeTown'][0][0].squeeze() #CT or not ? True or False, 921/1046
            self.Mitchells_Plain_SP_2011 = data['sp2011MitchellsPlain'][0][0].squeeze() #Mitchells Plain or not ? True or False, 19/1046   
            self.Mitchells_Plain_grid_2011 = data['MitchellsPlain'][0][0].squeeze() #Mitchells Plain or not ? True or False, 220/24014   
        
            self.income_grid_2011 = data['gridAverageIncome'][0][0].squeeze() #Average income, 24014
            self.income_SP_2011 = data['sp2011AverageIncome'][0][0].squeeze() #Average income, 1046
        
            self.income_12_class_SP_2011 = data['sp2011IncomeDistribution12Class'][0][0].squeeze() #Number of households per income class, 1046*12
            self.income_n_class_SP_2011 = data['sp2011IncomeDistributionNClass'][0][0].squeeze()   #Number of households per income class, 1046*4        
            self.nb_poor_grid_2011 = data['gridNumberPoor'][0][0].squeeze() #nb of poor households per grid cell, 24014
            self.nb_rich_grid_2011 = data['gridNumberRich'][0][0].squeeze() #nb of rich households per grid cell, 24014
        
            self.median_income_2011 = data['medianIncome'][0][0].squeeze() #Median income of each of the income class, 12
            self.income_groups_limits = data['thresholdIncomeDistribution'][0][0].squeeze() #Boundaries of the 4 income classes
        
            self.total_dwellings_SP_2011 = data['spTotalDwellings'][0][0].squeeze() #Number of dwellings per administrative division, 1046
            self.formal_grid_2011 = data['gridFormal'][0][0].squeeze() #Number of formal settlements per grid cell, 24014. 158316 in total
            self.backyard_SP_2011 = data['spInformalBackyard'][0][0].squeeze() #Number of informal settlements in backyard per SP division, 1046. 85007 in total.
            self.informal_SP_2011 = data['spInformalSettlement'][0][0].squeeze() #Number of informal settlements per SP division, 1046. 158316 in total
            self.backyard_grid_2011 = data['gridInformalBackyard'][0][0].squeeze() #Number of informal settlements in backyard per grid cell, 24014
            self.informal_grid_2011 = data['gridInformalSettlement'][0][0].squeeze() #Number of informal settlements per grid cell, 24014
        
            self.Code_SP_2001 = data['sp2001Code'][0][0].squeeze()  #Administrative divisions, 1013
            self.X_SP_2001 = data['sp2001X'][0][0].squeeze()  #X coordinates, 1013
            self.Y_SP_2001 = data['sp2001Y'][0][0].squeeze()  #Y coordinates, 1013
            self.distance_SP_2001 = data['sp2001DistanceCenter'][0][0].squeeze() #distance to the city center, 1013
            self.area_SP_2001 = data['sp2001Area'][0][0].squeeze()    #SP area, 1013   
        
            self.income_12_class_SP_2001 = data['sp2001Distribution12Class'][0][0].squeeze() #Number of persons per income class, 1013*12
            self.income_n_class_SP_2001 = data['sp2001DistributionNClass'][0][0].squeeze() #Number of persons per income class, 1013*4 
            self.nb_poor_SP_2001 = data['sp2001NumberPoor'][0][0].squeeze() #nb of poor households per administrative division, 1013
            self.nb_rich_SP_2001 = data['sp2001NumberRich'][0][0].squeeze()  #nb of rich households per administrative division, 1013
            self.CT_SP_2001 = data['sp2001CapeTown'][0][0].squeeze() #CT or not ? True or False, 683/1046
        
            self.formal_SP_2001 = data['sp2001Formal'][0][0].squeeze() #Number of formal dwellings per administrative division, 1013. 806909
            self.backyard_SP_2001 = data['sp2001Backyard'][0][0].squeeze() #Number of informal settlements in backyard per SP division, 1013. 42075 in total.
            self.informal_SP_2001 = data['sp2001Settlement'][0][0].squeeze()   #Number of informal settlements per SP division, 1013. 149258 in total.
            self.formal_grid_2001 = data['grid2001Formal'][0][0].squeeze() #Number of formal dwellings per grid cell.
            self.backyard_grid_2001 = data['grid2001Backyard'][0][0].squeeze() #Number of backyard settlements per grid cell.
            self.informal_grid_2001 = data['grid2001Settlement'][0][0].squeeze() #Number of backyard settlements per grid cell.
        
            self.total_number_per_income_class = data['totalNumberPerIncomeGroup'][0][0].squeeze() #Number of persons per income class, in CT
            self.total_number_per_income_bracket = data['totalNumberPerIncomeBracket'][0][0].squeeze() #Number of persons per income class, in CT
        
            self.sale_price_SP = data['spPrice'][0][0].squeeze() #Sale price, 1046
            self.sale_price_year = data['yearPrice'][0][0].squeeze() #2001 and 2011
        
            self.spDwellingSize = data['spDwellingSize'][0][0].squeeze()
            self.spFormalDensityHFA = data['spFormalDensityHFA'][0][0].squeeze()
            self.SP_is_CT = data['SP_is_CT'][0][0].squeeze()
            self.spUnconstrainedArea = data['spUnconstrainedArea'][0][0].squeeze()
            self.gridFormalDensityHFA = data['gridFormalDensityHFA'][0][0].squeeze()
            self.limitCapeTown = data['limitCapeTown'][0][0].squeeze()
    
            self.GV_count_RDP = data['gridCountRDPfromGV'][0][0].squeeze() #Number of subsidized housing, 24014
            self.GV_area_RDP = data['gridAreaRDPfromGV'][0][0].squeeze() #Surface of subsidized housing, 24014
            self.household_size_group = data['householdSizeIncomeGroup'][0][0].squeeze() #Household size as a function of income groups

        else:
            # %% Census 2011: Income and dwellings data 2011
        
            dwellings_data_2011 = pd.read_csv('./2. Data/Basile data/sub-places-dwelling-statistics.csv', sep = ';')
        
            #Data at the SP level (administrative division)
            X_SP_2011 = dwellings_data_2011.CoordX/1000
            Y_SP_2011 = dwellings_data_2011.CoordY/1000
            distance_SP_2011 = np.sqrt(((X_SP_2011 - grid.x_center) ** 2) + ((Y_SP_2011 - grid.y_center) ** 2))
            area_SP_2011 = dwellings_data_2011.ALBERS_ARE
            CT_SP_2011 = (dwellings_data_2011.MN_CODE == 199)
            Code_SP_2011 = dwellings_data_2011.SP_CODE
        
            #Average income
            income_grid_2011 = SP_to_grid_2011_1(dwellings_data_2011.Data_census_dwelling_INC_average, Code_SP_2011, grid)
            income_SP_2011 = dwellings_data_2011.Data_census_dwelling_INC_average
        
            #Income classes
            income_12_class_SP_2011 = pd.DataFrame({'class1': dwellings_data_2011.Data_census_dwelling_INC_0, 'class2':dwellings_data_2011.Data_census_dwelling_INC_1_4800, 'class3': dwellings_data_2011.Data_census_dwelling_INC_4801_9600, 'class4': dwellings_data_2011.Data_census_dwelling_INC_9601_19600, 'class5': dwellings_data_2011.Data_census_dwelling_INC_19601_38200, 'class6': dwellings_data_2011.Data_census_dwelling_INC_38201_76400, 'class7': dwellings_data_2011.Data_census_dwelling_INC_76401_153800, 'class8':dwellings_data_2011.Data_census_dwelling_INC_153801_307600, 'class9':dwellings_data_2011.Data_census_dwelling_INC_307601_614400, 'class10': dwellings_data_2011.Data_census_dwelling_INC_614001_1228800, 'class11':dwellings_data_2011.Data_census_dwelling_INC_1228801_2457600, 'class12': dwellings_data_2011.Data_census_dwelling_INC_2457601_more})
            income_n_class_SP_2011 = np.zeros((len(Code_SP_2011), param["nb_of_income_classes"]))    
            for i in range(0, param["nb_of_income_classes"]):
                income_n_class_SP_2011[:,i] = np.sum(income_12_class_SP_2011.iloc[:, (param["income_distribution"]) == i + 1], axis = 1)         
        
            #Poor and rich
            middle_class = math.floor(param["nb_of_income_classes"] / 2)
            poor = (param["income_distribution"] <= middle_class)
            rich = (param["income_distribution"] > middle_class)
            nb_poor_grid_2011 = SP_to_grid_2011_2(np.sum(income_12_class_SP_2011.iloc[:, poor], axis = 1), Code_SP_2011, grid)
            nb_rich_grid_2011 = SP_to_grid_2011_2(np.sum(income_12_class_SP_2011.iloc[:, rich], axis = 1), Code_SP_2011, grid)
            
            #Mitchells Plain
            Mitchells_Plain_SP_2011 = (dwellings_data_2011.MP_CODE == 199039)
            Mitchells_Plain_grid_2011 = (SP_to_grid_2011_2(Mitchells_Plain_SP_2011, Code_SP_2011, grid) > 0)
         
            #Type de logement
            dwelling_type = pd.DataFrame({'col1': dwellings_data_2011.Data_census_dwelling_House_concrete_block_structure, 'col2': dwellings_data_2011.Data_census_dwelling_Traditional_dwelling, 'col3': dwellings_data_2011.Data_census_dwelling_Flat_apartment, 'col4': dwellings_data_2011.Data_census_dwelling_Cluster_house, 'col5': dwellings_data_2011.Data_census_dwelling_Townhouse, 'col6': dwellings_data_2011.Data_census_dwelling_Semi_detached_house, 'col7': dwellings_data_2011.Data_census_dwelling_House_flat_room_in_backyard, 'col8': dwellings_data_2011.Data_census_dwelling_Informal_dwelling_in_backyard, 'col9': dwellings_data_2011.Data_census_dwelling_Informal_dwelling_settlement, 'col10': dwellings_data_2011.Data_census_dwelling_Room_flatlet_on_property_or_larger_dw, 'col11': dwellings_data_2011.Data_census_dwelling_Caravan_tent, 'col12': dwellings_data_2011.Data_census_dwelling_Other, 'col13': dwellings_data_2011.Data_census_dwelling_Unspecified, 'col14': dwellings_data_2011.Data_census_dwelling_Not_applicable}) #Number of house of each type per SP
            total_dwellings_SP_2011 = np.sum(dwelling_type, axis = 1)
            backyard_SP_2011 = dwelling_type.iloc[:, 7]
            informal_SP_2011 = dwelling_type.iloc[:, 8]
            backyard_SP_2011[np.isnan(backyard_SP_2011)] = 0
            informal_SP_2011[np.isnan(informal_SP_2011)] = 0
            backyard_grid_2011 = SP_to_grid_2011_2(backyard_SP_2011, Code_SP_2011, grid)
            informal_grid_2011 = SP_to_grid_2011_2(informal_SP_2011, Code_SP_2011, grid)
            formal_grid_2011 = SP_to_grid_2011_2(total_dwellings_SP_2011 - informal_SP_2011 - backyard_SP_2011, Code_SP_2011, grid)
        
            # %% Income distribution 2011
            
            income_2011 = pd.read_csv('./2. Data/Basile data/Income_distribution_2011.csv', sep=",")
            median_income_2011 = income_2011.INC_med #Median income for each income group
            income_groups_limits = np.zeros(param["nb_of_income_classes"]) #Min. income for each income class   
            for j in range(0, param["nb_of_income_classes"]):
                income_groups_limits[j] = np.max(income_2011.INC_max.iloc[param["income_distribution"] == j + 1])
 
            # %% Census 2001: Income and dwellings data 2001
            income_2001 = pd.read_csv('./2. Data/Basile data/Census_2001_income.csv', sep = ";")
        
            #Define a grid with SP data
            X_SP_2001 = income_2001.X_2001 / 1000
            Y_SP_2001 = income_2001.Y_2001 / 1000
            Code_SP_2001 = income_2001.SP_CODE
            distance_SP_2001 = np.sqrt(((X_SP_2001 - grid.x_center) ** 2) + ((Y_SP_2001 - grid.y_center) ** 2))
            area_SP_2001 = income_2001.Area_sqm / 1000000 #in km2

            #Income Classes
            income_12_class_SP_2001 = pd.DataFrame({'col1': income_2001.Census_2001_inc_No_income, 'col2': income_2001.Census_2001_inc_R1_4800, 'col3': income_2001.Census_2001_inc_R4801_9600, 'col4': income_2001.Census_2001_inc_R9601_19200, 'col5': income_2001.Census_2001_inc_R19201_38400, 'col6': income_2001.Census_2001_inc_R38401_76800, 'col7': income_2001.Census_2001_inc_R76801_153600, 'col8': income_2001.Census_2001_inc_R153601_307200, 'col9': income_2001.Census_2001_inc_R307201_614400, 'col10': income_2001.Census_2001_inc_R614401_1228800, 'col11': income_2001.Census_2001_inc_R1228801_2457600, 'col12': income_2001.Census_2001_inc_R2457601_more})
            income_n_class_SP_2001 = np.zeros((len(income_2001.SP_CODE), param["nb_of_income_classes"]))
            for i in range(0, param["nb_of_income_classes"]):
                income_n_class_SP_2001[:,i] = np.sum(income_12_class_SP_2001.iloc[:, (param["income_distribution"]) - 1 == i], axis = 1)

            #Poor and rich
            nb_poor_SP_2001 = np.sum(income_12_class_SP_2001.iloc[:, poor], axis = 1)
            nb_rich_SP_2001 = np.sum(income_12_class_SP_2001.iloc[:, rich], axis = 1) #We keep the same categories than for 2011 (is it relevant?)
        
            CT_SP_2001 = (Code_SP_2001 > 17000000) & (Code_SP_2001 < 18000000)       
            total_dwellings_grid_2001 = SP_to_grid_2001(nb_poor_SP_2001 + nb_rich_SP_2001, Code_SP_2001, grid)

            #Total number of people per income class for 2001 and 2011
            total_number_per_income_class = np.array([sum(income_n_class_SP_2001[CT_SP_2001, :]), sum(income_n_class_SP_2011[CT_SP_2011, :])])
            total_number_per_income_bracket = ([sum(np.array(income_12_class_SP_2001)[CT_SP_2001, :]), sum(np.array(income_12_class_SP_2011)[CT_SP_2011, :])])
        
            # %% Dwelling types 2001
            
            dwellings_data_2001 = pd.read_csv('./2. Data/Basile data/Census_2001_dwelling_type.csv', sep = ';')
        
            formal = dwellings_data_2001.House_brick_structure_separate_stand + dwellings_data_2001.Flat_in_block + dwellings_data_2001.semi_detached_house + dwellings_data_2001.House_flat_in_backyard + dwellings_data_2001.Room_flatlet_shared_property + dwellings_data_2001.Caravan_tent + dwellings_data_2001.Ship_boat
            backyard = dwellings_data_2001.Informal_dwelling_in_backyard
            informal = dwellings_data_2001.Traditional_dwelling_traditional_materials + dwellings_data_2001.Informal_dwelling_NOT_backyard

            formal_SP_2001 = np.zeros(len(Code_SP_2001))
            backyard_SP_2001 = np.zeros(len(Code_SP_2001))
            informal_SP_2001 = np.zeros(len(Code_SP_2001))
    
            for i in range(0, len(Code_SP_2001)):
                match = (dwellings_data_2001.SP_Code == Code_SP_2001[i])
                formal_SP_2001[i] = formal[match]
                backyard_SP_2001[i] = backyard[match]
                informal_SP_2001[i] = informal[match]

            formal_grid_2001 = SP_to_grid_2001(formal_SP_2001, Code_SP_2001, grid)
            backyard_grid_2001 = SP_to_grid_2001(backyard_SP_2001, Code_SP_2001, grid)
            informal_grid_2001 = SP_to_grid_2001(informal_SP_2001, Code_SP_2001, grid)

            # %% Real estate data 2012
        
            #Sales data were previously treaty and aggregated at the SP level on R
            sale_price = pd.read_csv('./2. Data/Basile data/SalePriceStat_SP.csv', sep = ',')
        
            #Median sale price for each SP for 2011 and 2001
            sale_price_SP = np.zeros((3, len(Code_SP_2011)))
            for i in range(0, len(Code_SP_2011)):
                if (sum(sale_price.SP_CODE == Code_SP_2011[i]) == 1):
                    sale_price_SP[2][i] = sale_price.Median_2011[sale_price.SP_CODE == Code_SP_2011[i]]
                    sale_price_SP[1][i] = sale_price.Median_2006[sale_price.SP_CODE == Code_SP_2011[i]]
                    sale_price_SP[0][i] = sale_price.Median_2001[sale_price.SP_CODE == Code_SP_2011[i]]
            sale_price_year = np.array([2001, 2006, 2011])
            sale_price_SP[sale_price_SP == 0] = np.nan
        
            # %% Density/Dwellings: Cape Town's data
        
            sal_coord = pd.read_csv('./2. Data/Basile data/Res_SAL_coord.csv', sep = ';')
        
            #Define a grid for SAL data
            X_SAL = sal_coord.X_sal_CAPE / 1000
            Y_SAL = sal_coord.Y_sal_CAPE / 1000
            distance_SAL = np.sqrt(((X_SAL - grid.x_center) ** 2) + ((Y_SAL - grid.y_center) ** 2))
            area_SAL = sal_coord.Area_sqm
        
            #Nombre et taille des logements, densités de logements et de population
            DENS_DU_formal = sal_coord.FMD / (sal_coord.Area_sqm / 1000000) #nombre de logements formels / km2
            DU_Size = sal_coord.AvgDUSz #taille moyenne des logements formels
            DENS_HFA_formal = (sal_coord.SR_Ext + sal_coord.STS_Ext) / (sal_coord.Area_sqm/1000000) #nombre de m2 construits formel / km2
            DENS_HFA_informal = (sal_coord.BF_ExtAdj /(sal_coord.Area_sqm / 1000000)) - DENS_HFA_formal #nombre de m2 construits informel / km2
            DENS_HFA_informal[DENS_HFA_informal < 0] = 0
            DENS_DU = sal_coord.DUs / (sal_coord.Area_sqm / 1000000) #nombre de personnes par km2        
            Code_conversion_SAL = sal_coord.OBJECTID_1
            Code_SAL = sal_coord.SAL_CODE
            Code_SP_SAL = sal_coord.SP_CODE
            SAL_total_formal_HFA = sal_coord.SR_Ext + sal_coord.STS_Ext

            #Nombre et taille des logements, densités de logements et de population - format grid
            DENS_HFA_formal_grid = SAL_to_grid(DENS_HFA_formal, Code_conversion_SAL, grid)
            DENS_HFA_informal_grid = SAL_to_grid(DENS_HFA_informal, Code_conversion_SAL, grid)
            DENS_HFA_informal_grid[(DENS_HFA_informal_grid > 2000000)] = np.nan
            DENS_DU_grid = SAL_to_grid(DENS_DU, Code_conversion_SAL, grid)
            DU_Size_grid = SAL_to_grid(DU_Size, Code_conversion_SAL, grid)
            DU_Size_grid[DU_Size_grid > 600] = np.nan
            
            coeff_land_SAL = import_SAL_land_use(grid)
            coeff_land_SAL_grid = SAL_to_grid(coeff_land_SAL, Code_conversion_SAL, grid)
            limit_Cape_Town = ~np.isnan(coeff_land_SAL_grid)

            # %% RDP houses (= subsidized housing) from GV2012
            
            rdp_houses = pd.read_csv('./2. Data/Basile data/GV2012_grid_RDP_count2.csv', sep = ';')
        
            # %% Data dwelling sizes and available area per Subplace (aggregated with R)
            
            dwelling_size_area_urb_constr = pd.read_csv('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/data_Cape_Town/SP_dwelling_size_area_urb_constrained.csv')
            spDwellingSize = 10000 * np.ones(len(Code_SP_2011))
            spFormalDensityHFA = np.zeros(len(Code_SP_2011))
            SP_is_CT = np.zeros(len(Code_SP_2011))
            spUnconstrainedArea = np.zeros(len(Code_SP_2011))
            for i in range(0, len(Code_SP_2011)):
                if sum(dwelling_size_area_urb_constr.SP_CODE == Code_SP_2011[i]) == 1:
                    which = (dwelling_size_area_urb_constr.SP_CODE == Code_SP_2011[i])
                    spDwellingSize[i] = dwelling_size_area_urb_constr.MEAN_RES_D[which]
                    spFormalDensityHFA[i] = dwelling_size_area_urb_constr.dens_HFA[which]
                    SP_is_CT[i] = dwelling_size_area_urb_constr.is_Cape_Town[which]
                    spUnconstrainedArea[i] = dwelling_size_area_urb_constr.unconstrained[which]
        
            spDwellingSize[spDwellingSize == 10000] = np.nan

            gridFormalDensityHFA = SP_to_grid_2011_1(spFormalDensityHFA, Code_SP_2011, grid)
            limitCapeTown = SP_to_grid_2011_1(SP_is_CT, Code_SP_2011, grid) > 0

            # %% Put it into a DataCourbe structure
            self.Code_SP_2011 = Code_SP_2011 #Administrative divisions, 1046
            self.X_SP_2011 = X_SP_2011 #X coordinates, 1046
            self.Y_SP_2011 = Y_SP_2011 #Y coordinates, 1046
            self.distance_SP_2011 = distance_SP_2011 #distance to the city center, 1046
            self.area_SP_2011 = area_SP_2011 #SP area, 1046 
            self.CT_SP_2011 = CT_SP_2011 #CT or not ? True or False, 921/1046
            self.Mitchells_Plain_SP_2011 = Mitchells_Plain_SP_2011 #Mitchells Plain or not ? True or False, 19/1046   
            self.Mitchells_Plain_grid_2011 = Mitchells_Plain_grid_2011 #Mitchells Plain or not ? True or False, 220/24014   
        
            self.income_grid_2011 = income_grid_2011 #Average income, 24014
            self.income_SP_2011 = income_SP_2011 #Average income, 1046
        
            self.income_12_class_SP_2011 = income_12_class_SP_2011 #Number of households per income class, 1046*12
            self.income_n_class_SP_2011 = income_n_class_SP_2011   #Number of households per income class, 1046*4        
            self.nb_poor_grid_2011 = nb_poor_grid_2011 #nb of poor households per grid cell, 24014
            self.nb_rich_grid_2011 = nb_rich_grid_2011 #nb of rich households per grid cell, 24014
        
            self.median_income_2011 = median_income_2011 #Median income of each of the income class, 12
            self.income_groups_limits = income_groups_limits #Boundaries of the 4 income classes
        
            self.total_dwellings_SP_2011 = total_dwellings_SP_2011 #Number of dwellings per administrative division, 1046
            self.formal_grid_2011 = formal_grid_2011 #Number of formal settlements per grid cell, 24014. 158316 in total
            self.backyard_SP_2011 = backyard_SP_2011 #Number of informal settlements in backyard per SP division, 1046. 85007 in total.
            self.informal_SP_2011 = informal_SP_2011 #Number of informal settlements per SP division, 1046. 158316 in total
            self.backyard_grid_2011 = backyard_grid_2011 #Number of informal settlements in backyard per grid cell, 24014
            self.informal_grid_2011 = informal_grid_2011 #Number of informal settlements per grid cell, 24014
        
            self.Code_SP_2001 = Code_SP_2001  #Administrative divisions, 1013
            self.X_SP_2001 = X_SP_2001  #X coordinates, 1013
            self.Y_SP_2001 = Y_SP_2001  #Y coordinates, 1013
            self.distance_SP_2001 = distance_SP_2001 #distance to the city center, 1013
            self.area_SP_2001 = area_SP_2001    #SP area, 1013   
        
            self.income_12_class_SP_2001 = income_12_class_SP_2001 #Number of persons per income class, 1013*12
            self.income_n_class_SP_2001 = income_n_class_SP_2001 #Number of persons per income class, 1013*4 
            self.nb_poor_SP_2001 = nb_poor_SP_2001 #nb of poor households per administrative division, 1013
            self.nb_rich_SP_2001 = nb_rich_SP_2001  #nb of rich households per administrative division, 1013
            self.CT_SP_2001 = CT_SP_2001 #CT or not ? True or False, 683/1046
        
            self.formal_SP_2001 = formal_SP_2001 #Number of formal dwellings per administrative division, 1013. 806909
            self.backyard_SP_2001 = backyard_SP_2001 #Number of informal settlements in backyard per SP division, 1013. 42075 in total.
            self.informal_SP_2001 = informal_SP_2001   #Number of informal settlements per SP division, 1013. 149258 in total.
            self.formal_grid_2001 = formal_grid_2001 #Number of formal dwellings per grid cell.
            self.backyard_grid_2001 = backyard_grid_2001 #Number of backyard settlements per grid cell.
            self.informal_grid_2001 = informal_grid_2001 #Number of backyard settlements per grid cell.
        
            self.total_number_per_income_class = total_number_per_income_class #Number of persons per income class, in CT
            self.total_number_per_income_bracket = total_number_per_income_bracket #Number of persons per income class, in CT
        
            self.sale_price_SP = sale_price_SP #Sale price, 1046
            self.sale_price_year = sale_price_year #2001 and 2011
        
            self.spDwellingSize = spDwellingSize
            self.spFormalDensityHFA = spFormalDensityHFA
            self.SP_is_CT = SP_is_CT
            self.spUnconstrainedArea = spUnconstrainedArea
            self.gridFormalDensityHFA = gridFormalDensityHFA
            self.limitCapeTown = limitCapeTown
    
            self.GV_count_RDP = np.transpose(rdp_houses.count_RDP) #Number of subsidized housing, 24014
            self.GV_area_RDP = np.transpose(rdp_houses.area_RDP) #Surface of subsidized housing, 24014
            self.household_size_group = np.array([6.556623149, 1.702518978, 0.810146856, 1.932265222]) #Household size as a function of income groups


    """
        # %% Enumerator Area definition from the 2011 census - informal settlements areas
        
        ea_data = pd.read_csv('./2. Data/Basile data/EA_definition_CPT_CAPE.csv', sep = ';')
        area_urb_from_EA_SP = np.zeros(len(Code_SP_2011))
        formal_dens_HFA_SP = np.zeros(len(Code_SP_2011))
        for i in range(0, len(Code_SP_2011)):
            area_urb_from_EA_SP[i] = sum(ea_data.ALBERS_ARE[(ea_data.SP_CODE == Code_SP_2011[i]) & ((ea_data.EA_TYPE_C == 1) | (ea_data.EA_TYPE_C == 6) )]) #Surface en km2 des EA type 1 et 6
            formal_dens_HFA_SP[i] = sum(SAL_total_formal_HFA[Code_SP_SAL == Code_SP_2011[i]] / 1000000) / area_urb_from_EA_SP[i] #Densités de logements dans les EA de type 1 et 6
        formal_dens_HFA_SP[(area_urb_from_EA_SP / np.transpose(area_SP_2011)) < 0.2] = np.nan
        formal_dens_HFA_SP[formal_dens_HFA_SP > 3] = np.nan

        # %% Residential construction data at the SP level
        
        sp_res_data = pd.read_csv('./2. Data/Basile data/SP_res_data.csv', sep = ';')
        SP_CODE_SAL = copy.deepcopy(sp_res_data.SP_CODE)
        floor_factor_SP = 1000000 * np.ones(len(Code_SP_2011))
        share_urbanised_SP = 1000000 * np.ones(len(Code_SP_2011))
        dwelling_size_SP = 1000000 * np.ones(len(Code_SP_2011))
        for i in range(0, len(Code_SP_2011)):
            if sum(SP_CODE_SAL == Code_SP_2011[i]) > 0:
                floor_factor_SP[i] = sp_res_data.FF[SP_CODE_SAL == Code_SP_2011[i]]
                share_urbanised_SP[i] = sp_res_data.Res_share[SP_CODE_SAL == Code_SP_2011[i]]
                dwelling_size_SP[i] = sp_res_data.avg_DUsz[SP_CODE_SAL == Code_SP_2011[i]]
        floor_factor_SP[floor_factor_SP == 1000000] = np.nan
        floor_factor_SP[np.isinf(floor_factor_SP)] = np.nan
        share_urbanised_SP[share_urbanised_SP == 1000000] = np.nan
        dwelling_size_SP[dwelling_size_SP == 1000000] = np.nan
        """