# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:02:26 2020

@author: Charlotte Liotta
"""

from data.functions_to_import_data import *
import copy


class ImportDataCourbe:
    """ Class definig a grid with:
        - ID
        - coord_horiz
        - coord_vert
        - xcentre, ycentre
        - dist """
        
    def __init__(self):
        
        self
        
    def import_data(self, grille, param):
        
        precision = 1

        #if option.LOAD_DATA == 0
            #load(strcat('.', slash, 'precalculations', slash, 'data_courbe'))
            #disp('Data loaded directly')
            #else    
            
            #Données recensement 2011

        #Subplace data from the Census 2011
        dwellings_data = pd.read_csv('./2. Data/sub-places-dwelling-statistics.csv', sep = ';')
        data_courbe_SP_Code = dwellings_data.SP_CODE
        data_courbe_income_grid = data_SP_vers_grille(dwellings_data.Data_census_dwelling_INC_average, data_courbe_SP_Code, grille)
        data_courbe_income_SP = dwellings_data.Data_census_dwelling_INC_average
        data_courbe_SP_income_12_class = pd.DataFrame({'class1': dwellings_data.Data_census_dwelling_INC_0, 'class2':dwellings_data.Data_census_dwelling_INC_1_4800, 'class3': dwellings_data.Data_census_dwelling_INC_4801_9600, 'class4': dwellings_data.Data_census_dwelling_INC_9601_19600, 'class5': dwellings_data.Data_census_dwelling_INC_19601_38200, 'class6': dwellings_data.Data_census_dwelling_INC_38201_76400, 'class7': dwellings_data.Data_census_dwelling_INC_76401_153800, 'class8':dwellings_data.Data_census_dwelling_INC_153801_307600, 'class9':dwellings_data.Data_census_dwelling_INC_307601_614400, 'class10': dwellings_data.Data_census_dwelling_INC_614001_1228800, 'class11':dwellings_data.Data_census_dwelling_INC_1228801_2457600, 'class12': dwellings_data.Data_census_dwelling_INC_2457601_more})
        data_courbe_SP_income_n_class = np.zeros((len(dwellings_data.SP_CODE), param["multiple_class"]))
    
        for i in range(0, param["multiple_class"]):
            data_courbe_SP_income_n_class[:,i] = np.sum(data_courbe_SP_income_12_class.iloc[:,(param["income_distribution"]) - 1 == i], axis = 1)         
        data_courbe_SP_X = dwellings_data.CoordX/1000
        data_courbe_SP_Y = dwellings_data.CoordY/1000
        data_courbe_SP_2011_distance = np.sqrt(((data_courbe_SP_X - grille.xcentre) ** 2) + ((data_courbe_SP_Y - grille.ycentre) ** 2))
        data_courbe_SP_2011_area = dwellings_data.ALBERS_ARE #in km2
        data_courbe_SP_2011_CT = (dwellings_data.MN_CODE == 199)
        data_courbe_SP_2011_Mitchells_Plain = (dwellings_data.MP_CODE == 199039)

        middle_class = math.floor(param["multiple_class"] / 2)
        poor = (param["income_distribution"] <= middle_class)
        rich = (param["income_distribution"] > middle_class)

        data_courbe_nb_poor_grid = data_SP_vers_grille_alternative(np.sum(data_courbe_SP_income_12_class.iloc[:, poor], axis = 1), data_courbe_SP_Code, grille)
        data_courbe_nb_rich_grid = data_SP_vers_grille_alternative(np.sum(data_courbe_SP_income_12_class.iloc[:, rich], axis = 1), data_courbe_SP_Code, grille)
        data_courbe_Mitchells_Plain = (data_SP_vers_grille_alternative(data_courbe_SP_2011_Mitchells_Plain, data_courbe_SP_Code, grille) > 0)

        #Data on the income distribution
        income_distrib = pd.read_csv('./2. Data/Income_distribution_2011.csv', sep=",")
        data_courbe_INC_med = income_distrib.INC_med
        data_courbe_limit = np.zeros(param["multiple_class"])
        for j in range(0, param["multiple_class"]):
            data_courbe_limit[j] = np.max(income_distrib.INC_max.iloc[param["income_distribution"] == j])

        #Type de logement
        data_type = pd.DataFrame({'col1': dwellings_data.Data_census_dwelling_House_concrete_block_structure, 'col2': dwellings_data.Data_census_dwelling_Traditional_dwelling, 'col3': dwellings_data.Data_census_dwelling_Flat_apartment, 'col4': dwellings_data.Data_census_dwelling_Cluster_house, 'col5': dwellings_data.Data_census_dwelling_Townhouse, 'col6': dwellings_data.Data_census_dwelling_Semi_detached_house, 'col7': dwellings_data.Data_census_dwelling_House_flat_room_in_backyard, 'col8': dwellings_data.Data_census_dwelling_Informal_dwelling_in_backyard, 'col9': dwellings_data.Data_census_dwelling_Informal_dwelling_settlement, 'col10': dwellings_data.Data_census_dwelling_Room_flatlet_on_property_or_larger_dw, 'col11': dwellings_data.Data_census_dwelling_Caravan_tent, 'col12': dwellings_data.Data_census_dwelling_Other, 'col13': dwellings_data.Data_census_dwelling_Unspecified, 'col14': dwellings_data.Data_census_dwelling_Not_applicable})
        data_courbe_SP_total_dwellings = np.sum(data_type, axis = 1)
        data_courbe_SP_informal_backyard = data_type.iloc[:, 8]
        data_courbe_SP_informal_settlement = data_type.iloc[:, 9]
        data_courbe_SP_informal_backyard[np.isnan(data_courbe_SP_informal_backyard)] = 0
        data_courbe_SP_informal_settlement[np.isnan(data_courbe_SP_informal_settlement)] = 0
        data_courbe_informal_backyard_grid = data_SP_vers_grille_alternative(data_courbe_SP_informal_backyard, data_courbe_SP_Code, grille)
        data_courbe_informal_settlement_grid = data_SP_vers_grille_alternative(data_courbe_SP_informal_settlement, data_courbe_SP_Code, grille)
        data_courbe_formal_grid = data_SP_vers_grille_alternative(data_courbe_SP_total_dwellings - data_courbe_SP_informal_settlement - data_courbe_SP_informal_backyard, data_courbe_SP_Code, grille)

        #Données densité pop 2001
        income_2001 = pd.read_csv('./2. Data/Census_2001_income.csv', sep = ";")
        data_courbe_SP_2001_X = income_2001.X_2001 / 1000
        data_courbe_SP_2001_Y = income_2001.Y_2001 / 1000
        data_courbe_SP_2001_Code = income_2001.SP_CODE
        data_courbe_SP_2001_dist = np.sqrt(((data_courbe_SP_2001_X - grille.xcentre) ** 2) + ((data_courbe_SP_2001_Y - grille.ycentre) ** 2))
        data_courbe_SP_2001_area = income_2001.Area_sqm / 1000000 #in km2

        #Income Classes
        data_courbe_SP_2001_12_class = pd.DataFrame({'col1': income_2001.Census_2001_inc_No_income, 'col2': income_2001.Census_2001_inc_R1_4800, 'col3': income_2001.Census_2001_inc_R4801_9600, 'col4': income_2001.Census_2001_inc_R9601_19200, 'col5': income_2001.Census_2001_inc_R19201_38400, 'col6': income_2001.Census_2001_inc_R38401_76800, 'col7': income_2001.Census_2001_inc_R76801_153600, 'col8': income_2001.Census_2001_inc_R153601_307200, 'col9': income_2001.Census_2001_inc_R307201_614400, 'col10': income_2001.Census_2001_inc_R614401_1228800, 'col11': income_2001.Census_2001_inc_R1228801_2457600, 'col12': income_2001.Census_2001_inc_R2457601_more})
        data_courbe_SP_2001_income_n_class = np.zeros((len(income_2001.SP_CODE), param["multiple_class"]))
        for i in range(0, param["multiple_class"]):
            data_courbe_SP_2001_income_n_class[:,i] = np.sum(data_courbe_SP_2001_12_class.iloc[:,(param["income_distribution"]) - 1 == i], axis = 1)

        #Density of people
        data_courbe_SP_2001_nb_poor = np.sum(data_courbe_SP_2001_12_class.iloc[:, poor], axis = 1)
        data_courbe_SP_2001_nb_rich = np.sum(data_courbe_SP_2001_12_class.iloc[:, rich], axis = 1) #We keep the same categories than for 2011 (is it relevant?)
        
        data_courbe_SP_2001_CT = (data_courbe_SP_2001_Code > 17000000) & (data_courbe_SP_2001_Code < 18000000)
        
        data_courbe_people_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe_SP_2001_nb_poor + data_courbe_SP_2001_nb_rich, data_courbe_SP_2001_Code, grille)

        #Dwelling types
        dwelling_type_2001 = pd.read_csv('./2. Data/Census_2001_dwelling_type.csv', sep = ';')
        formal = dwelling_type_2001.House_brick_structure_separate_stand + dwelling_type_2001.Flat_in_block + dwelling_type_2001.semi_detached_house + dwelling_type_2001.House_flat_in_backyard + dwelling_type_2001.Room_flatlet_shared_property + dwelling_type_2001.Caravan_tent + dwelling_type_2001.Ship_boat
        backyard = dwelling_type_2001.Informal_dwelling_in_backyard
        informal = dwelling_type_2001.Traditional_dwelling_traditional_materials + dwelling_type_2001.Informal_dwelling_NOT_backyard

        data_courbe_SP_2001_formal = np.zeros(len(data_courbe_SP_2001_Code))
        data_courbe_SP_2001_backyard = np.zeros(len(data_courbe_SP_2001_Code))
        data_courbe_SP_2001_informal = np.zeros(len(data_courbe_SP_2001_Code))
    
        for i in range(0, len(data_courbe_SP_2001_Code)):
            match = (dwelling_type_2001.SP_Code == data_courbe_SP_2001_Code[i])
            data_courbe_SP_2001_formal[i] = formal[match]
            data_courbe_SP_2001_backyard[i] = backyard[match]
            data_courbe_SP_2001_informal[i] = informal[match]

        data_courbe_formal_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe_SP_2001_formal, data_courbe_SP_2001_Code, grille)
        data_courbe_backyard_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe_SP_2001_backyard, data_courbe_SP_2001_Code, grille)
        data_courbe_informal_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe_SP_2001_informal, data_courbe_SP_2001_Code, grille)

        #Total number of people per income class

        #For 2001 and 2011
        data_courbe_total_number_per_income_class = np.array([sum(data_courbe_SP_2001_income_n_class[data_courbe_SP_2001_CT, :]), sum(data_courbe_SP_income_n_class[data_courbe_SP_2011_CT, :])])
        data_courbe_total_number_per_income_bracket = np.array([sum(np.array(data_courbe_SP_2001_12_class)[data_courbe_SP_2001_CT, :]), sum(np.array(data_courbe_SP_income_12_class)[data_courbe_SP_2011_CT, :])])

        #Real estate data 2012
        #Sales data were previously treaty and aggregated at the SP level on R
        sale_price = pd.read_csv('./2. Data/SalePriceStat_SP.csv', sep = ',')
        data_courbe_SP_price = np.zeros((2, len(data_courbe_SP_Code)))
        for i in range(0, len(data_courbe_SP_Code)):
            if (sum(sale_price.SP_CODE == data_courbe_SP_Code[i]) == 1):
                data_courbe_SP_price[1][i] = sale_price.Median_2011[sale_price.SP_CODE == data_courbe_SP_Code[i]]
                data_courbe_SP_price[0][i] = sale_price.Median_2001[sale_price.SP_CODE == data_courbe_SP_Code[i]]
        data_courbe_year_price = np.array([2001, 2011])
        data_courbe_SP_price[data_courbe_SP_price == 0] = np.nan
        data_courbe_X_price = np.transpose(data_courbe_SP_X)
        data_courbe_Y_price = np.transpose(data_courbe_SP_Y)
        data_courbe_distance_price = np.sqrt(((data_courbe_SP_X - grille.xcentre) ** 2) + ((data_courbe_SP_Y - grille.ycentre) ** 2))

        #Données SAL 2011 de la ville du Cap
        sal_coord = pd.read_csv('./2. Data/Res_SAL_coord.csv', sep = ';')
        data_courbe_X_sal = sal_coord.X_sal_CAPE / 1000
        data_courbe_Y_sal = sal_coord.Y_sal_CAPE / 1000
        data_courbe_dist_sal = np.sqrt(((data_courbe_X_sal - grille.xcentre) ** 2) + ((data_courbe_Y_sal - grille.ycentre) ** 2))
        data_courbe_DENS_DU_formal = sal_coord.FMD / (sal_coord.Area_sqm / 1000000) #nombre de logements formels / km2
        data_courbe_DU_Size = sal_coord.AvgDUSz
        data_courbe_DENS_HFA_formal = (sal_coord.SR_Ext + sal_coord.STS_Ext) / (sal_coord.Area_sqm/1000000) #nombre de m2 construits formel / km2
        data_courbe_DENS_HFA_informal = (sal_coord.BF_ExtAdj /(sal_coord.Area_sqm / 1000000)) - data_courbe_DENS_HFA_formal #nombre de m2 construits formel / km2
        data_courbe_DENS_HFA_informal[data_courbe_DENS_HFA_informal < 0] = 0
        data_courbe_DENS_DU = sal_coord.DUs / (sal_coord.Area_sqm / 1000000) #nombre de personnes par km2
        data_courbe_SAL_Code_conversion = sal_coord.OBJECTID_1
        data_courbe_SAL_Code = sal_coord.SAL_CODE
        data_courbe_SP_Code_SAL = sal_coord.SP_CODE
        data_courbe_SAL_area = sal_coord.Area_sqm
        data_courbe_SAL_total_formal_HFA = sal_coord.SR_Ext + sal_coord.STS_Ext

        data_courbe_DENS_HFA_formal_grid = data_CensusSAL_vers_grille(data_courbe_DENS_HFA_formal, data_courbe_SAL_Code_conversion, grille)
        data_courbe_DENS_HFA_informal_grid = data_CensusSAL_vers_grille(data_courbe_DENS_HFA_informal, data_courbe_SAL_Code_conversion, grille)
        data_courbe_DENS_HFA_informal_grid[(data_courbe_DENS_HFA_informal_grid > 2000000)] = np.nan

        data_courbe_DENS_DU_grid = data_CensusSAL_vers_grille(data_courbe_DENS_DU, data_courbe_SAL_Code_conversion, grille)
        
        data_courbe_DU_Size_grid = data_CensusSAL_vers_grille(data_courbe_DU_Size, data_courbe_SAL_Code_conversion, grille)
        data_courbe_DU_Size_grid[data_courbe_DU_Size_grid > 600] = np.nan

        data_courbe_coeff_land_sal = import_data_SAL_landuse(grille)
        data_courbe_coeff_land_sal_grid = data_CensusSAL_vers_grille(data_courbe_coeff_land_sal, data_courbe_SAL_Code_conversion, grille)
        data_courbe_limit_Cape_Town = ~np.isnan(data_courbe_coeff_land_sal_grid)
    
        #Density of construction at the SP level

        #We need the surface for formal residential use, from EA data (Census 2011)
        ea_data = pd.read_csv('./2. Data/EA_definition_CPT_CAPE.csv', sep = ';')

        data_courbe_SP_area_urb_from_EA = np.zeros(len(data_courbe_SP_Code))
        data_courbe_SP_formal_dens_HFA = np.zeros(len(data_courbe_SP_Code))

        for i in range(0, len(data_courbe_SP_Code)):
            data_courbe_SP_area_urb_from_EA[i] = sum(ea_data.ALBERS_ARE[(ea_data.SP_CODE == data_courbe_SP_Code[i]) & ((ea_data.EA_TYPE_C == 1) | (ea_data.EA_TYPE_C == 6) )]) #in km2
            data_courbe_SP_formal_dens_HFA[i] = sum(data_courbe_SAL_total_formal_HFA[data_courbe_SP_Code_SAL == data_courbe_SP_Code[i]] / 1000000) / data_courbe_SP_area_urb_from_EA[i]

        data_courbe_SP_formal_dens_HFA[(data_courbe_SP_area_urb_from_EA / np.transpose(data_courbe_SP_2011_area)) < 0.2] = np.nan
        data_courbe_SP_formal_dens_HFA[data_courbe_SP_formal_dens_HFA > 3] = np.nan

        #Residential construction data at the SP level
        sp_res_data = pd.read_csv('./2. Data/SP_res_data.csv', sep = ';')
        SP_CODE_SAL = copy.deepcopy(sp_res_data.SP_CODE)
        data_courbe_SP_floor_factor = 1000000 * np.ones(len(data_courbe_SP_Code))
        data_courbe_SP_share_urbanised = 1000000 * np.ones(len(data_courbe_SP_Code))
        data_courbe_SP_dwelling_size = 1000000 * np.ones(len(data_courbe_SP_Code))
        for i in range(0, len(data_courbe_SP_Code)):
            if sum(SP_CODE_SAL == data_courbe_SP_Code[i]) > 0:
                data_courbe_SP_floor_factor[i] = sp_res_data.FF[SP_CODE_SAL == data_courbe_SP_Code[i]]
                data_courbe_SP_share_urbanised[i] = sp_res_data.Res_share[SP_CODE_SAL == data_courbe_SP_Code[i]]
                data_courbe_SP_dwelling_size[i] = sp_res_data.avg_DUsz[SP_CODE_SAL == data_courbe_SP_Code[i]]
        data_courbe_SP_floor_factor[data_courbe_SP_floor_factor == 1000000] = np.nan
        data_courbe_SP_share_urbanised[data_courbe_SP_share_urbanised == 1000000] = np.nan
        data_courbe_SP_dwelling_size[data_courbe_SP_dwelling_size == 1000000] = np.nan


        #RDP houses from GV2012
        rdp_houses = pd.read_csv('./2. Data/GV2012_grid_RDP_count2.csv', sep = ';')
        data_courbe_GV_count_RDP = np.transpose(rdp_houses.count_RDP)
        data_courbe_GV_area_RDP = np.transpose(rdp_houses.area_RDP)

        #Household size as a function of income groups 
        #Data from 2011 Census (Claus' estimation)
        data_courbe_household_size_group = np.array([6.556623149, 1.702518978, 0.810146856, 1.932265222])
    
        self.SP_Code = data_courbe_SP_Code
        self.income_grid = data_courbe_income_grid
        self.income_SP = data_courbe_income_SP
        self.SP_income_12_class = data_courbe_SP_income_12_class
        self.SP_income_n_class = data_courbe_SP_income_n_class      
        self.SP_X = data_courbe_SP_X
        self.SP_Y = data_courbe_SP_Y
        self.SP_2011_distance = data_courbe_SP_2011_distance
        self.SP_2011_area = data_courbe_SP_2011_area
        self.SP_2011_CT = data_courbe_SP_2011_CT
        self.SP_2011_Mitchells_Plain = data_courbe_SP_2011_Mitchells_Plain
        self.nb_poor_grid = data_courbe_nb_poor_grid
        self.nb_rich_grid = data_courbe_nb_rich_grid
        self.Mitchells_Plain = data_courbe_Mitchells_Plain
        self.INC_med = data_courbe_INC_med
        self.limit = data_courbe_limit        
        self.SP_total_dwellings = data_courbe_SP_total_dwellings
        self.SP_informal_backyard = data_courbe_SP_informal_backyard
        self.SP_informal_settlement = data_courbe_SP_informal_settlement
        self.informal_backyard_grid = data_courbe_informal_backyard_grid
        self.informal_settlement_grid = data_courbe_informal_settlement_grid
        self.formal_grid = data_courbe_formal_grid
        self.SP_2001_X = data_courbe_SP_2001_X
        self.SP_2001_Y = data_courbe_SP_2001_Y
        self.SP_2001_Code = data_courbe_SP_2001_Code
        self.SP_2001_dist = data_courbe_SP_2001_dist
        self.SP_2001_area = data_courbe_SP_2001_area
        self.SP_2001_12_class = data_courbe_SP_2001_12_class
        self.SP_2001_income_n_class = data_courbe_SP_2001_income_n_class
        self.SP_2001_nb_poor = data_courbe_SP_2001_nb_poor
        self.SP_2001_nb_rich = data_courbe_SP_2001_nb_rich
        self.SP_2001_CT = data_courbe_SP_2001_CT
        self.people_2001_grid = data_courbe_people_2001_grid
        self.SP_2001_formal = data_courbe_SP_2001_formal
        self.SP_2001_backyard = data_courbe_SP_2001_backyard
        self.SP_2001_informal = data_courbe_SP_2001_informal   
        self.formal_2001_grid = data_courbe_formal_2001_grid
        self.backyard_2001_grid = data_courbe_backyard_2001_grid
        self.informal_2001_grid = data_courbe_informal_2001_grid
        self.total_number_per_income_class = data_courbe_total_number_per_income_class
        self.total_number_per_income_bracket = data_courbe_total_number_per_income_bracket
        self.SP_price = data_courbe_SP_price
        self.year_price = data_courbe_year_price
        self.X_price = data_courbe_X_price
        self.Y_price = data_courbe_Y_price
        self.distance_price = data_courbe_distance_price
        self.X_sal = data_courbe_X_sal
        self.Y_sal = data_courbe_Y_sal
        self.dist_sal = data_courbe_dist_sal
        self.DENS_DU_formal = data_courbe_DENS_DU_formal
        self.DU_Size = data_courbe_DU_Size
        self.DENS_HFA_formal = data_courbe_DENS_HFA_formal
        self.DENS_HFA_informal = data_courbe_DENS_HFA_informal
        self.DENS_DU = data_courbe_DENS_DU
        self.SAL_Code_conversion = data_courbe_SAL_Code_conversion
        self.SAL_Code = data_courbe_SAL_Code
        self.SP_Code_SAL = data_courbe_SP_Code_SAL
        self.SAL_area = data_courbe_SAL_area
        self.total_formal_HFA = data_courbe_SAL_total_formal_HFA
        self.DENS_HFA_formal_grid = data_courbe_DENS_HFA_formal_grid
        self.DENS_HFA_informal_grid = data_courbe_DENS_HFA_informal_grid
        self.DENS_DU_grid = data_courbe_DENS_DU_grid
        self.DU_Size_grid = data_courbe_DU_Size_grid
        self.coeff_land_sal = data_courbe_coeff_land_sal
        self.coeff_land_sal_grid = data_courbe_coeff_land_sal_grid
        self.limit_Cape_Town = data_courbe_limit_Cape_Town 
        self.SP_area_urb_grom_EA = data_courbe_SP_area_urb_from_EA
        self.formal_dens_HFA = data_courbe_SP_formal_dens_HFA
        self.SP_floor_factor = data_courbe_SP_floor_factor
        self.SP_share_urbanised = data_courbe_SP_share_urbanised
        self.SP_dwelling_size = data_courbe_SP_dwelling_size       
        self.GV_count_RDP = data_courbe_GV_count_RDP
        self.GV_area_RDP = data_courbe_GV_area_RDP
        self.household_size_group = data_courbe_household_size_group

