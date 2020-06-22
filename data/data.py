# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:02:26 2020

@author: Charlotte Liotta
"""

from data.functions_to_import_data import *

def data_polycentrisme_CAPE_TOWN(grille, param):

    precision = 1

    #if option.LOAD_DATA == 0
        #load(strcat('.', slash, 'precalculations', slash, 'data_courbe'))
        #disp('Data loaded directly')
    #else    

    #Données recensement 2011

    #Subplace data from the Census 2011
    dwellings_data = pd.read_csv('./2. Data/sub-places-dwelling-statistics.csv', sep = ';')
    data_courbe = pd.DataFrame()
    data_courbe_SP_Code = dwellings_data.SP_CODE
    data_courbe_income_grid = data_SP_vers_grille(dwellings_data.Data_census_dwelling_INC_average, data_courbe, grille)
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

    data_courbe_nb_poor_grid = data_SP_vers_grille_alternative(np.sum(data_courbe_SP_income_12_class.iloc[:, poor], axis = 1), data_courbe, grille)
    data_courbe_nb_rich_grid = data_SP_vers_grille_alternative(np.sum(data_courbe_SP_income_12_class.iloc[:, rich], axis = 1), data_courbe, grille)
    data_courbe_Mitchells_Plain = (data_SP_vers_grille_alternative(data_courbe_SP_2011_Mitchells_Plain, data_courbe, grille) > 0)

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
    data_courbe_informal_backyard_grid = data_SP_vers_grille_alternative(data_courbe_SP_informal_backyard, data_courbe, grille)
    data_courbe_informal_settlement_grid = data_SP_vers_grille_alternative(data_courbe_SP_informal_settlement, data_courbe, grille)
    data_courbe_formal_grid = data_SP_vers_grille_alternative(data_courbe_SP_total_dwellings - data_courbe_SP_informal_settlement - data_courbe_SP_informal_backyard, data_courbe, grille)

    #Données densité pop 2001
    income_2001 = pd.read_csv('./2. Data/Census_2001_income.csv', sep = ";")
    data_courbe_SP_2001_X = income_2001.X_2001 / 1000
    data_courbe_SP_2001_Y = income_2001.Y_2001 / 1000
    data_courbe_SP_2001_Code = income_2001.SP_CODE
    data_courbe_SP_2001_dist = np.sqrt(((data_courbe_SP_2001_X - grille.xcentre) ** 2) + ((data_courbe_SP_2001_Y - grille.ycentre) ** 2))
    data_courbe_SP_2001_area = income_2001.Area_sqm / 1000000 #in km2

    #Income Classes
    data_courbe_SP_2001_12_class = pd.DataFrame({'col1': income_2001.Census_2001_inc_No_income, 'col2': income_2001.Census_2001_inc_R1_4800, 'col3': income_2001.Census_2001_inc_R4801_9600, 'col4': income_2001.Census_2001_inc_R9601_19200, 'col5': income_2001.Census_2001_inc_R19201_38400, 'col6': income_2001.Census_2001_inc_R38401_76800, 'col7': income_2001.Census_2001_inc_R76801_153600, 'col8': income_2001.Census_2001_inc_R153601_307200, 'col9': income_2001.Census_2001_inc_R307201_614400, 'col10': income_2001.Census_2001_inc_R614401_1228800, 'col11': income_2001.Census_2001_inc_R1228801_2457600, 'col12': income_2001.Census_2001_inc_R2457601_more})
    for i in range(0, param["multiple_class"]):
        data_courbe_SP_2001_income_n_class[:,i] = np.sum(data_courbe.SP_2001_12_class.iloc[:,(param["income_distribution"]) - 1 == i], axis = 1)

    #Density of people
    data_courbe_SP_2001_nb_poor = np.sum(data_courbe_SP_2001_12_class.iloc[:, poor], axis = 1)
    data_courbe_SP_2001_nb_rich = np.sum(data_courbe_SP_2001_12_class[:, rich], axis = 1) #We keep the same categories than for 2011 (is it relevant?)
    
    data_courbe_SP_2001_CT = (data_courbe_SP_2001_Code > 17000000) & (data_courbe_SP_2001_Code < 18000000)

    data_courbe_people_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe_SP_2001_nb_poor + data_courbe_SP_2001_nb_rich, data_courbe, grille)

    #Dwelling types
    importfile([path_nedum,'Census_2001_dwelling_type.csv'])
    formal = House_brick_structure_separate_stand + Flat_in_block + semi_detached_house + House_flat_in_backyard + Room_flatlet_shared_property + Caravan_tent + Ship_boat;
    backyard = 	Informal_dwelling_in_backyard;
    informal = Traditional_dwelling_traditional_materials + Informal_dwelling_NOT_backyard;

    data_courbe.SP_2001_formal = zeros(1,length(data_courbe.SP_2001_Code));
    data_courbe.SP_2001_backyard = zeros(1,length(data_courbe.SP_2001_Code));
    data_courbe.SP_2001_informal = zeros(1,length(data_courbe.SP_2001_Code));
    
    for i = 1:length(data_courbe.SP_2001_Code):
        match = SP_Code == data_courbe.SP_2001_Code(i);
        data_courbe.SP_2001_formal(i) = formal(match);
        data_courbe.SP_2001_backyard(i) = backyard(match);
        data_courbe.SP_2001_informal(i) = informal(match);

    data_courbe.formal_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe.SP_2001_formal, data_courbe, grille);
    data_courbe.backyard_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe.SP_2001_backyard, data_courbe, grille);
    data_courbe.informal_2001_grid = data_SP_2001_vers_grille_alternative(data_courbe.SP_2001_informal, data_courbe, grille);

    #Total number of people per income class

    #For 2001 and 2011
    data_courbe.total_number_per_income_class = [sum(data_courbe.SP_2001_income_n_class(data_courbe.SP_2001_CT,:)); sum(data_courbe.SP_income_n_class(data_courbe.SP_2011_CT, :))];
    data_courbe.total_number_per_income_bracket = [sum(data_courbe.SP_2001_12_class(data_courbe.SP_2001_CT,:)); sum(data_courbe.SP_income_12_class(data_courbe.SP_2011_CT, :))];

    #Real estate data 2012
    #Sales data were previously treaty and aggregated at the SP level on R
    importfile([path_nedum,'SalePriceStat_SP.csv'])
    data_courbe.SP_price=zeros(2,length(data_courbe.SP_Code));
    for i=1:length(data_courbe.SP_Code):
        if sum(x0x22SP_CODE0x22==data_courbe.SP_Code(i)) == 1:
            data_courbe.SP_price(2,i) = x0x22Median_20110x22(x0x22SP_CODE0x22==data_courbe.SP_Code(i));
            data_courbe.SP_price(1,i) = x0x22Median_20010x22(x0x22SP_CODE0x22==data_courbe.SP_Code(i));
    data_courbe.year_price = [2001 2011];
    data_courbe.SP_price(data_courbe.SP_price==0) = NaN;
    data_courbe.X_price = data_courbe.SP_X';
    data_courbe.Y_price = data_courbe.SP_Y';
    data_courbe.distance_price = sqrt((data_courbe.SP_X-grille.xcentre).^2+(data_courbe.SP_Y-grille.ycentre).^2);

    #Données SAL 2011 de la ville du Cap
    importfile([path_nedum,'Res_SAL_coord.csv']);
    data_courbe.X_sal = X_sal_CAPE./1000;
    data_courbe.Y_sal = Y_sal_CAPE./1000;
    data_courbe.dist_sal = sqrt((data_courbe.X_sal-grille.xcentre).^2+(data_courbe.Y_sal-grille.ycentre).^2);
    data_courbe.DENS_DU_formal = FMD./(Area_sqm./1000000); %nombre de logements formels / km2
    data_courbe.DU_Size = AvgDUSz; 
    data_courbe.DENS_HFA_formal = (SR_Ext+STS_Ext)./(Area_sqm./1000000);%nombre de m2 construits formel / km2
    data_courbe.DENS_HFA_informal = (BF_ExtAdj./(Area_sqm./1000000))-data_courbe.DENS_HFA_formal;%nombre de m2 construits formel / km2
    data_courbe.DENS_HFA_informal(data_courbe.DENS_HFA_informal<0) = 0;
    data_courbe.DENS_DU = DUs./(Area_sqm./1000000); %nombre de personnes par km2
    data_courbe.SAL_Code_conversion = OBJECTID_1;
    data_courbe.SAL_Code = SAL_CODE;
    data_courbe.SP_Code_SAL = SP_CODE;
    data_courbe.SAL_area = Area_sqm;
    data_courbe.SAL_total_formal_HFA = SR_Ext + STS_Ext;

    data_courbe.DENS_HFA_formal_grid = data_CensusSAL_vers_grille(data_courbe.DENS_HFA_formal, data_courbe, grille);
    data_courbe.DENS_HFA_informal_grid = data_CensusSAL_vers_grille(data_courbe.DENS_HFA_informal, data_courbe, grille);
    data_courbe.DENS_HFA_informal_grid(data_courbe.DENS_HFA_informal_grid > 2000000) = NaN;

    data_courbe.DENS_DU_grid = data_CensusSAL_vers_grille(data_courbe.DENS_DU, data_courbe, grille);
    
    data_courbe.DU_Size_grid = data_CensusSAL_vers_grille(data_courbe.DU_Size, data_courbe, grille);
    data_courbe.DU_Size_grid(data_courbe.DU_Size_grid > 600) = NaN;

    data_courbe.coeff_land_sal = import_data_SAL_landuse(grille);
    data_courbe.coeff_land_sal_grid = data_CensusSAL_vers_grille(data_courbe.coeff_land_sal, data_courbe,grille);
    data_courbe.limit_Cape_Town = ~isnan(data_courbe.coeff_land_sal_grid);
    
    #Density of construction at the SP level

    #We need the surface for formal residential use, from EA data (Census 2011)
    importfile([path_nedum, 'EA_definition_CPT_CAPE.csv']);

    data_courbe.SP_area_urb_from_EA = zeros(1, length(data_courbe.SP_Code));
    data_courbe.SP_formal_dens_HFA = zeros(1, length(data_courbe.SP_Code));

    for i = 1:length(data_courbe.SP_Code)
        data_courbe.SP_area_urb_from_EA(i) = sum(ALBERS_ARE((SP_CODE == data_courbe.SP_Code(i)) & (EA_TYPE_C == 1 | EA_TYPE_C == 6 ))); % in km2
        data_courbe.SP_formal_dens_HFA(i) = sum(data_courbe.SAL_total_formal_HFA(data_courbe.SP_Code_SAL == data_courbe.SP_Code(i))./1000000) ./ data_courbe.SP_area_urb_from_EA(i);

    data_courbe.SP_formal_dens_HFA(data_courbe.SP_area_urb_from_EA ./ data_courbe.SP_2011_area' < 0.2) = NaN;
    data_courbe.SP_formal_dens_HFA(data_courbe.SP_formal_dens_HFA > 3) = NaN;

    #Residential construction data at the SP level
    importfile([path_nedum, 'SP_res_data.csv']);
    SP_CODE_SAL = SP_CODE;
    data_courbe.SP_floor_factor = 1000000.*ones(1,length(data_courbe.SP_Code));
    data_courbe.SP_share_urbanised = 1000000.*ones(1,length(data_courbe.SP_Code));
    data_courbe.SP_dwelling_size = 1000000.*ones(1,length(data_courbe.SP_Code));
    for i = 1:length(data_courbe.SP_Code)
        if sum(SP_CODE_SAL == data_courbe.SP_Code(i)) > 0 
            data_courbe.SP_floor_factor(i) = FF(SP_CODE_SAL == data_courbe.SP_Code(i));
            data_courbe.SP_share_urbanised(i) = Res_share(SP_CODE_SAL == data_courbe.SP_Code(i));
            data_courbe.SP_dwelling_size(i) = avg_DUsz(SP_CODE_SAL == data_courbe.SP_Code(i));
    data_courbe.SP_floor_factor(data_courbe.SP_floor_factor == 1000000) = NaN;
    data_courbe.SP_share_urbanised(data_courbe.SP_share_urbanised == 1000000) = NaN;
    data_courbe.SP_dwelling_size(data_courbe.SP_dwelling_size == 1000000) = NaN;


    #RDP houses from GV2012
    importfile([path_nedum, 'GV2012_grid_RDP_count2.csv']);
    data_courbe.GV_count_RDP = count_RDP';
    data_courbe.GV_area_RDP = area_RDP';

    #Household size as a function of income groups 
    #Data from 2011 Census (Claus' estimation)
    data_courbe.household_size_group = [6.556623149, 1.702518978, 0.810146856, 1.932265222];
    
    #Construction 
    param["housing_in"] = data_courbe.DENS_HFA_formal_grid./land.coeff_land(1,:).*1.1;
    param["housing_in(~isfinite(param.housing_in))"] = 0
    param["housing_in(param.housing_in>2*10^6)"] = 2*10^6
    param["housing_in(param.housing_in<0)"] = 0

    param["housing_mini"] = zeros(1,length(grille.dist))
    param["housing_mini(data_courbe.Mitchells_Plain)"] = data_courbe.DENS_HFA_formal_grid(data_courbe.Mitchells_Plain)./land.coeff_land(1,data_courbe.Mitchells_Plain);
    param["housing_mini(land.coeff_land(1,:) < 0.1 | isnan(param.housing_mini))"] = 0
    
    return data_courbe, param



    


