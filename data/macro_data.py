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

        # %% Evolution de l'inflation, des revenus, des taux d'intéret et du prix du carburant, de la population et des revenus
        scenario = pd.read_csv('./2. Data/Basile data/Scenario_CAPE_TOWN_Claus_inputs.csv', sep = ";")
        macro_spline_inflation = interp1d(scenario.year_infla[~np.isnan(scenario.inflation_base_2010)] - param["year_begin"], scenario.inflation_base_2010[~np.isnan(scenario.inflation_base_2010)], method)
        macro_spline_revenu = interp1d(scenario.Year[~np.isnan(scenario.Inc_avg)] - param["year_begin"], scenario.Inc_avg[~np.isnan(scenario.Inc_avg)], method)
        macro_revenu_ref = macro_spline_revenu(param["annee_reference"] - param["year_begin"])
        macro_spline_notaires = interp1d(scenario.year_interest_rate[~np.isnan(scenario.year_interest_rate)] - param["year_begin"], scenario.real_interest_rate[~np.isnan(scenario.year_interest_rate)], method) #Interest rate
        macro_spline_carburant = interp1d(scenario.carbu_annee[~np.isnan(scenario.carbu_annee)] - param["year_begin"], scenario.carbu_carbu[~np.isnan(scenario.carbu_annee)]/100, method)
        macro_spline_population = interp1d(scenario.Year[~np.isnan(scenario.Pop_HH_total)] - param["year_begin"], scenario.Pop_HH_total[~np.isnan(scenario.Pop_HH_total)], method)
        macro_spline_pop_inc_distribution = interp1d(scenario.Year[~np.isnan(scenario.Inc_avg)] - param["year_begin"], [scenario.Pop_11_class1[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class2[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class3[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class4[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class5[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class6[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class7[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class8[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class9[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class10[~np.isnan(scenario.Inc_avg)], scenario.Pop_11_class11[~np.isnan(scenario.Inc_avg)],], method)
        macro_revenu = macro_spline_revenu(np.transpose(t))
        
        # %% Distribution of incomes
        income_2011 = pd.read_csv('./2. Data/Basile data/Income_distribution_2011.csv')
        income_distribution = numpy.matlib.repmat(income_2011.INC_med, len(scenario.Year), 1) #Revenu médian des 12 classes de ménages pour toutes les années étudiées
        income_distribution[scenario.Year > 2011, :] = income_distribution[scenario.Year > 2011, :] * numpy.matlib.repmat(np.transpose([macro_spline_inflation(scenario.Year[scenario.Year > 2011] - param["year_begin"]) / macro_spline_inflation(2011 - param["year_begin"])]), 1, np.size(income_distribution, axis = 1))
        macro_spline_inc_distribution = interp1d(scenario.Year[~np.isnan(scenario.Inc_avg)] - param["year_begin"], np.transpose(income_distribution[~np.isnan(scenario.Inc_avg), 1:12]), method)

        # %% Number of RDP houses
        RDP_2011 = 320969
        if option["future_construction_RDP"] == 1:
            macro_spline_RDP = interp1d([2001 - param["year_begin"], 2011 - param["year_begin"], 2041 - param["year_begin"]], [RDP_2011 - 10*5000, RDP_2011, RDP_2011 + 30*5000], method)
        else: 
            macro_spline_RDP = interp1d([2001 - param["year_begin"], 2011 - param["year_begin"], 2018 - param["year_begin"], 2041 - param["year_begin"]], [RDP_2011 - 10*5000, RDP_2011, RDP_2011 + 7*5000, RDP_2011 + 7*5000], method)

        self.spline_inflation = macro_spline_inflation
        self.spline_revenu = macro_spline_revenu
        self.revenu_ref = macro_revenu_ref
        self.spline_inc_distribution = macro_spline_inc_distribution
        self.spline_notaires = macro_spline_notaires
        self.spline_carburant = macro_spline_carburant
        self.spline_RDP = macro_spline_RDP
        self.spline_population = macro_spline_population
        self.spline_pop_inc_distribution = macro_spline_pop_inc_distribution
        self.revenu = macro_revenu