# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:27:47 2020

@author: Charlotte Liotta
"""
from scipy.interpolate import griddata
import copy
from data.job import *
from data.transport import *
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.io

class Amenity:
        
    def __init__(self):
        
        self
        
    def calibration(self, grid, param, macro, job, option, trans, households_data, land, t_amenity):

        # %% Preparation of data (at the SP_level)
        
        t_trafic_amenity = t_amenity - param["baseline_year"]

        #Income (SP level)
        income = households_data.income_SP_2011 
        revenu_ref = macro_data.spline_revenu(t_trafic_amenity)
        revenu_max = np.max(income)

        #Are transport data reliable ? (SP level)
        reliable_transport = griddata(np.c_[grid.horiz_coord, grid.vert_coord], trans.reliable, np.c_[households_data.X_SP_2011, households_data.Y_SP_2011])
    
        #Transport data, SP level
        cout_generalise = prix2_polycentrique3(trans.t_transport, trans.cout_generalise, param, t_trafic_amenity)
        trans_SP = np.empty((param["nb_of_income_classes"], len(households_data.Y_SP_2011)))
        for i in range(0, param["nb_of_income_classes"]):
            trans_SP[i,:] = griddata(np.c_[grid.horiz_coord, grid.vert_coord], cout_generalise[i,:], np.c_[households_data.X_SP_2011, households_data.Y_SP_2011])

        #Transport data with 12 classes, SP level
        param2 = copy.deepcopy(param)
        param2["nb_of_income_classes"] = 12 #to assign several household classes to each job center
        param2["household_size"] = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]) #household size for transport costs, à calibrer sur des données réelles
        param2["income_distribution"] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        job2 = ImportEmploymentData()
        job2.import_employment_data(grid, param2, option2, macro_data, [t_trafic_amenity, t_trafic_amenity + 10])    
        trans2 = TransportData()
        trans2.import_transport_data(option2, grid, macro_data, param2, job2, t_trafic_amenity)
        cout_generalise2 = prix2_polycentrique3(trans2.t_transport, trans2.cout_generalise, param, t_trafic_amenity)
        trans_SP2 = np.empty((param2["nb_of_income_classes"], len(households_data.X_SP_2011)))
        for i in range (0, param2["nb_of_income_classes"]):
            trans_SP2[i,:] = griddata(np.c_[grid.horiz_coord, grid.vert_coord], cout_generalise2[i,:], np.c_[households_data.X_SP_2011, households_data.Y_SP_2011])

        #Transport costs at the SP level (averaging between the income classes)
        trans_avg = np.zeros((len(income)))
        for i in range(0, len(households_data.income_SP_2011)):
            inc = interp1d(job2.annee, job2.avg_inc,  axis = 0)
            inc = inc(t_amenity)
            spline_trans_avg = interp1d(np.transpose(inc), trans_SP2[:,i])
            trans_avg[i] = spline_trans_avg(households_data.income_SP_2011[i])
            
        #Average income and transport cost for each class, array with the SP locations of each class
        classes = np.ones((len(income)))
        income_class = np.empty(len(classes))
        amenity_trans = np.empty(len(classes))
        for i in range(0, len(classes)):
            for j in range(0, param["nb_of_income_classes"]):
                if income[i] > households_data.income_groups_limits[j]:
                    classes[i] = j
            spline = interp1d(job.annee, job.avg_inc[:, int(classes[i])])
            income_class[i] = spline(t_amenity)
            amenity_trans[i] = np.transpose(trans_SP[int(classes[i]), i])

        #Income net of transportation costs at the SP level
        net_income = np.transpose(households_data.income_SP_2011) - trans_avg

        #Price
        spline_price = interp1d(households_data.sale_price_year, households_data.sale_price_SP, axis = 0)
        price = spline_price(t_amenity)
        price = households_data.sale_price_SP[1]
        distance = np.sqrt(((households_data.X_SAL - grid.x_center) ** 2) + ((households_data.Y_SAL - grid.y_center) ** 2)) #Inutile, juste pour des courbes de contrôle

        #Dwelling size and density HFA
        dwelling_size = copy.deepcopy(households_data.dwelling_size_SP)
        dwelling_size[dwelling_size > 1000] = np.nan
        construction = households_data.formal_dens_HFA_SP / param["max_land_use"]
        construction[construction > 2] = np.nan
        amenite_rent_HFA = price * (param["depreciation_rate"] + interest_rate(macro_data, t_amenity - param["baseline_year"])) / construction

        # %% Calibration of the parameters of the utility function
        
        #Data preparation
        quel_general = (amenite_rent_HFA < np.nanquantile(amenite_rent_HFA, 0.9)) & (reliable_transport > 0.95) & (price > 0) & (classes >= 1) & (~np.isnan(amenite_rent_HFA)) #What SP do we keep? 
        quel_general = (reliable_transport > -1)
        net_income_reg = net_income[quel_general]
        amenite_rent_HFA_reg = amenite_rent_HFA[quel_general]
        dwelling_size_reg = np.transpose(dwelling_size[quel_general])

        #STEP1: ESTIMATION OF BASIC_Q AS THE MINIMUM
        amenity_basic_q = np.nanmin(dwelling_size[households_data.informal_SP_2011 < 1])
        #amenity_basic_q = 31.6
        
        #STEP2: CALIBRATION OF BETA
        Y = dwelling_size_reg - param["q0"]
        X = net_income_reg / amenite_rent_HFA_reg - param["q0"]
        model = sm.OLS(Y, X, missing='drop').fit() #Formule 5.28
        model.summary()
        amenity_coeff_beta = model.params.squeeze()
        amenity_coeff_beta = 0.25
    
        #STEP3: CALIBRATION OF THE WEIGHTS ON AMENITIES  
    
        #Regression on amenities
        residu = np.abs(amenity_coeff_beta ** amenity_coeff_beta * (1 - amenity_coeff_beta) ** (1 - amenity_coeff_beta) * (income_class - amenity_trans - param["q0"] * amenite_rent_HFA)) / ((amenite_rent_HFA) ** (amenity_coeff_beta)) #Formule 5.27 de Basile: U/A
        ratio_utility = np.zeros((param["nb_of_income_classes"] - 1))
        utility_normalized = np.ones((len(classes)))
        for i in range(0, param["nb_of_income_classes"] - 1):
            print(i)
            residual_poor = residu.squeeze()[(classes == i) & ((income_class - amenity_trans - amenity_basic_q * amenite_rent_HFA) > 0)]
            residual_rich = residu.squeeze()[(classes == i + 1) & ((income_class - amenity_trans - amenity_basic_q * amenite_rent_HFA) > 0)]
            m = np.argmin(np.abs(residual_poor - np.nanquantile(residual_poor, 0.1)))
            M = np.argmin(np.abs(residual_rich - np.nanquantile(residual_rich, 0.9)))
            ratio_utility[i] = residual_rich[M] / residual_poor[m]
            ratio_utility_u1 = np.prod(ratio_utility, where = (~np.isnan(ratio_utility) & ~np.isinf(ratio_utility) & (ratio_utility > 0)))
            utility_normalized[classes == i] = ratio_utility_u1  #Estimation of the ratio of utilities for each income class and the one above
            
        residual_reg = amenity_coeff_beta * np.log(np.abs(amenite_rent_HFA)) - np.log(np.abs(amenity_coeff_beta ** amenity_coeff_beta * (1 - amenity_coeff_beta) ** (1 - amenity_coeff_beta) * (income_class - amenity_trans - param["q0"] * amenite_rent_HFA))) + np.log(utility_normalized) #Estimation de log(A/u)
        residual_reg = residual_reg.squeeze()
        residual_reg[amenite_rent_HFA <= 0] = np.nan
        residual_reg[(income_class - amenity_trans - param["q0"] * amenite_rent_HFA) <= 0] = np.nan

        quel = (~np.isnan(amenite_rent_HFA) & (~np.isnan(residual_reg))) & (reliable_transport==1) & (~np.isnan(amenite_rent_HFA))

        amenities_sp = pd.read_csv('./2. Data/Basile data/SP_amenities.csv')

        airport_cone2 = copy.deepcopy(amenities_sp.airport_cone)
        airport_cone2[amenities_sp.airport_cone == 55] = 1
        airport_cone2[amenities_sp.airport_cone == 60] = 1
        airport_cone2[amenities_sp.airport_cone == 65] = 1
        airport_cone2[amenities_sp.airport_cone == 70] = 1
        airport_cone2[amenities_sp.airport_cone == 75] = 1
    
        #dist_RDP = 2
        #if (dist_RDP != 2):
            #matrix_distance = ((repmat(grid.horiz_coord, len(households_data.SP_X), 1) - repmat(households_data.SP_X, 1, len(grid.horiz_coord))) ** 2 + (repmat(grid.coord_vert, len(households_data.SP_Y), 1) - repmat(households_data.SP_Y, 1, len(grid.coord_vert))) ** 2) < dist_RDP ** 2
            #SP_distance_RDP = np.transpose((land.RDP_houses_estimates > 5) * np.transpose(matrix_distance)) > 1
        #else:
            #load(strcat('.', slash, 'precalculations', slash, 'SP_distance_RDP'))
        sp_distance_rdp = scipy.io.loadmat('./2. Data/Basile data/SPdistanceRDP.mat')

        table_regression = pd.DataFrame(data= np.transpose([np.transpose(residual_reg), amenities_sp.distance_distr_parks < 2, amenities_sp.distance_ocean < 2, ((amenities_sp.distance_ocean < 4) & (amenities_sp.distance_ocean > 2)), amenities_sp.distance_world_herit < 2, amenities_sp.distance_urban_herit < 2, amenities_sp.distance_UCT < 2, airport_cone2, np.log(1 + amenities_sp.slope), amenities_sp.distance_train < 2, amenities_sp.distance_protected_envir < 2, np.log(1 + sp_distance_rdp["SP_distance_RDP"]).squeeze(), amenities_sp.distance_power_station < 2]))
        table_reg = table_regression[quel]
        table_reg.columns = ['residu', 'distance_distr_parks', 'distance_ocean', 'distance_ocean2', 'distance_world_herit', 'distance_urban_herit', 'distance_UCT', 'airport_cone2', 'slope', 'distance_train', 'distance_protected_envir', 'RDP_proximity', 'distance_power_station']
        Y = table_reg.residu
        X = table_reg[["distance_distr_parks", "distance_ocean", "distance_ocean2", "distance_urban_herit", "airport_cone2", "slope", "distance_protected_envir", "distance_train"]]
        X = sm.add_constant(X)
        model_amenity = sm.OLS(Y, X).fit()
        model_amenity.summary()
        
        #Extrapolation at the grid level
        grid_amenity = pd.read_csv('./2. Data/Basile data/grid_amenities.csv', sep = ',')
        #grid_amenity = pd.read_csv('./2. Data/grid_amenity.csv', sep = ';')

        airport_cone2 = copy.deepcopy(grid_amenity.airport_cone)
        airport_cone2[grid_amenity.airport_cone == 55] = 1
        airport_cone2[grid_amenity.airport_cone == 60] = 1
        airport_cone2[grid_amenity.airport_cone == 65] = 1
        airport_cone2[grid_amenity.airport_cone == 70] = 1
        airport_cone2[grid_amenity.airport_cone == 75] = 1

        #if dist_RDP != 2:
            #matrix_distance = ((numpy.matlib.repmat(grid.coord_horiz, len(grid.coord_horiz), 1) - numpy.matlib.repmat(np.transpose(grid.coord_horiz), 1, len(grid.coord_horiz))) ** 2 + (repmat(grid.coord_vert, len(grid.coord_horiz), 1) - repmat(np.transpose(grid.coord_vert), 1, len(grid.coord_horiz))) ** 2) < dist_RDP ** 2
            #grid_distance_RDP = np.transpose(land.RDP_houses_estimates > 5 * np.transpose(matrix_distance)) > 1
        #else:
            #load(strcat('.', slash, 'precalculations', slash, 'grid_distance_RDP'))
        grid_distance_rdp = scipy.io.loadmat('./2. Data/Basile data/gridDistanceRDP.mat')

        table_predictors = np.transpose(pd.DataFrame(data= ([grid_amenity.distance_distr_parks < 2, grid_amenity.distance_ocean < 2, ((amenities_sp.distance_ocean < 4) & (amenities_sp.distance_ocean > 2)), grid_amenity.distance_urban_herit < 2, airport_cone2, np.log(1 + grid_amenity.slope), grid_amenity.distance_protected_envir < 2, amenities_sp.distance_train < 2])))

        estimated_amenities = (amenity_coeff_beta * (np.array([model_amenity.params[0], model_amenity.params[1], model_amenity.params[2], model_amenity.params[3], model_amenity.params[4], model_amenity.params[5], model_amenity.params[6], model_amenity.params[7]]) * table_predictors)) # A vérifier  
        estimated_amenities = np.exp(estimated_amenities.astype(float))
        
        amenity_utility = np.matlib.repmat(ratio_utility, 8, 1) * np.transpose(np.matlib.repmat(np.exp(model_amenity.params[0]), 3, 1)) #Compute utilities using amenities and their weights

        # %% Calibration of the construction function (RH = A.b.H^b)

        construction = households_data.formal_dens_HFA_SP / param["max_land_use"]

        quel_construct = (construction > 0.05) & (construction < 3) & (np.transpose(households_data.distance_SP_2011) < 20) & (classes > 0) & (amenite_rent_HFA > np.nanquantile(amenite_rent_HFA, 0.1)) & (amenite_rent_HFA < np.nanquantile(amenite_rent_HFA, 0.9))
        model_construction = sm.OLS(np.log(price[quel_construct]), np.log(construction[quel_construct] * 1000000), missing = 'drop').fit()
        model_construction.summary()
        
        amenity_coeff_b = model_construction.coef_
        #amenity_coeff_b = 0.25
        amenity_coeff_a = 1 - amenity_coeff_b
        amenity_coeff_A = (1 / amenity_coeff_b ** amenity_coeff_b) * np.exp(model_construction.intercept_ * amenity_coeff_b)
        #amenity_coeff_A = 0.04
        
        # %% Calibration of the construction function for 2001
    
        total_2001_RDP = macro_data.spline_RDP(2001 - param["baseline_year"])
        grid_2001_private_housing = np.maximum(np.zeros(24014), households_data.formal_grid_2001 - households_data.GV_count_RDP / sum(households_data.GV_count_RDP) * total_2001_RDP)
        formal_2001_SP_2011 = griddata(np.c_[grid.horiz_coord, grid.vert_coord], grid_2001_private_housing, np.c_[households_data.X_SP_2011, households_data.Y_SP_2011])

        quel = (np.transpose(formal_2001_SP_2011) > 0) & (~np.isnan(households_data.sale_price_SP[0 , ])) & (households_data.dwelling_size_SP > 0) & (households_data.dwelling_size_SP < 500) & (np.transpose(households_data.distance_SP_2011) < 20)
        
        lm = LinearRegression()
        model_constr_2001 = lm.fit(pd.DataFrame(np.log(households_data.sale_price_SP[0 , quel])), np.transpose(np.log(formal_2001_SP_2011[quel] * households_data.dwelling_size_SP[quel])))

        amenity_coeff_b_2001 = model_constr_2001.coef_
        amenity_coeff_a_2001 = 1 - amenity_coeff_b_2001
        amenity_coeff_A_2001 = np.exp(model_constr_2001.intercept_) / (amenity_coeff_b_2001 ** amenity_coeff_b_2001)

        self.income = income
        self.reliable_transport = reliable_transport
        self.trans_avg = trans_avg
        self.classes = classes
        self.income_class = income_class
        self.trans = amenity_trans
        self.dwelling_size = dwelling_size
        self.construction = construction
        self.rent_HFA = amenite_rent_HFA
        self.basic_q = amenity_basic_q
        self.rent_HFA_reg = amenite_rent_HFA_reg
        self.dwelling_size_reg = dwelling_size_reg
        self.coeff_beta = amenity_coeff_beta
        self.estimated_amenities = estimated_amenities
        self.model_amenity = model_amenity
        self.utility = amenity_utility
        self.coeff_b = amenity_coeff_b
        self.coeff_a = amenity_coeff_a
        self.coeff_A = amenity_coeff_A
        self.model_construction = model_construction
        self.coeff_b_2001 = amenity_coeff_b_2001
        self.coeff_a_2001 = amenity_coeff_a_2001
        self.coeff_A_2001 = amenity_coeff_A_2001
    