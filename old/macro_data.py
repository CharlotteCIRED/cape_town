# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:46:14 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
from scipy.interpolate import PPoly
import pandas as pd
import numpy.matlib
import numpy as np

class MacroData:
    
    def __init__(self):
        
        self
        
    def import_macro_data(self, param, option, t):

        method = 'linear' #Méthode pour faire les interpolations (linear ou spline)

        # %% Evolution of inflation, incomes, interest rates, fuel prices, and population
        scenario = pd.read_csv('./2. Data/Basile data/Scenario_CAPE_TOWN_Claus_inputs.csv', sep = ";")
        spline_inflation = interp1d(scenario.year_infla[~np.isnan(scenario.inflation_base_2010)] - param["baseline_year"], scenario.inflation_base_2010[~np.isnan(scenario.inflation_base_2010)], method)
        spline_revenu = interp1d(scenario.Year[~np.isnan(scenario.Inc_avg)] - param["baseline_year"], scenario.Inc_avg[~np.isnan(scenario.Inc_avg)], method)
        revenu_ref = spline_revenu(param["baseline_year"] - param["baseline_year"])
        spline_interest_rate = interp1d(scenario.year_interest_rate[~np.isnan(scenario.year_interest_rate)] - param["baseline_year"], scenario.real_interest_rate[~np.isnan(scenario.year_interest_rate)], method) #Interest rate
        spline_fuel = interp1d(scenario.carbu_annee[~np.isnan(scenario.carbu_annee)] - param["baseline_year"], scenario.carbu_carbu[~np.isnan(scenario.carbu_annee)]/100, method)
        spline_population = interp1d(scenario.Year[~np.isnan(scenario.Pop_HH_total)] - param["baseline_year"], scenario.Pop_HH_total[~np.isnan(scenario.Pop_HH_total)], method)
        spline_pop_inc_distribution = interp1d(scenario.Year[~np.isnan(scenario.Inc_avg)] - param["baseline_year"], [scenario.Pop_11_class1[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class2[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class3[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class4[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class5[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class6[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class7[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class8[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class9[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class10[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class11[~np.isnan(scenario.Inc_avg)],], method)
        revenu = spline_revenu(np.transpose(t))
        
        # %% Distribution of incomes
        income_2011 = pd.read_csv('./2. Data/Basile data/Income_distribution_2011.csv')
        income_distribution = numpy.matlib.repmat(income_2011.INC_med, len(scenario.Year), 1) #Revenu médian des 12 classes de ménages pour toutes les années étudiées
        income_distribution[scenario.Year > 2011, :] = income_distribution[scenario.Year > 2011, :] * numpy.matlib.repmat(np.transpose([spline_inflation(scenario.Year[scenario.Year > 2011] - param["baseline_year"]) / spline_inflation(2011 - param["baseline_year"])]), 1, np.size(income_distribution, axis = 1))
        spline_inc_distribution = interp1d(scenario.Year[~np.isnan(scenario.Inc_avg)] - param["baseline_year"], np.transpose(income_distribution[~np.isnan(scenario.Inc_avg), 1:12]), method)

        # %% Number of RDP houses
        RDP_2011 = 320969
        if option["future_construction_RDP"] == 1:
            spline_RDP = interp1d([2001 - param["baseline_year"], 2011 - param["baseline_year"], 2041 - param["baseline_year"]], [RDP_2011 - 10*5000, RDP_2011, RDP_2011 + 30*5000], method)
        else: 
            spline_RDP = interp1d([2001 - param["baseline_year"], 2011 - param["baseline_year"], 2018 - param["baseline_year"], 2041 - param["baseline_year"]], [RDP_2011 - 10*5000, RDP_2011, RDP_2011 + 7*5000, RDP_2011 + 7*5000], method)

        self.spline_inflation = spline_inflation
        self.spline_revenu = spline_revenu
        self.revenu_ref = revenu_ref
        self.spline_inc_distribution = spline_inc_distribution
        self.spline_notaires = spline_interest_rate
        self.spline_fuel = spline_fuel
        self.spline_RDP = spline_RDP
        self.spline_population = spline_population
        self.spline_pop_inc_distribution = spline_pop_inc_distribution
        self.revenu = revenu