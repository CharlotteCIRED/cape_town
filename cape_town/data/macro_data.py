# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:46:14 2020

@author: Charlotte Liotta
"""

from scipy.interpolate import interp1d
import pandas as pd

def import_macro_data(param,options,t):

    method = 'linear' #Méthode pour faire les interpolations (linear ou spline)
    
    #Inflation (pour les coûts de transport) et revenus
    scenario  = pd.read_csv('Scenario_CAPE_TOWN_Claus_inputs.csv')
    macro.spline_inflation = interp1d(year_infla[~np.isnan(inflation_base_2010)] - param["year_begin"], inflation_base_2010[~np.isnan(inflation_base_2010)], method, 'pp')
    macro.spline_revenu = interp1d(Year[~np.isnan(Inc_avg)] - param["year_begin"], Inc_avg(~np.isnan(Inc_avg)), method, 'pp')
    macro.revenu_ref = ppval(macro.spline_revenu, param["annee_reference"] - param["year_begin"])

    #Distribution of incomes
    open('Income_distribution_2011.txt')
    income_distribution = np.matlib.repmat(INC_med, length(Year), 1)
    income_distribution[Year > 2011, :] = income_distribution[Year > 2011, :] * np.matlib.repmat(ppval(macro.spline_inflation, Year[Year > 2011] - param["year_begin"] / ppval(macro.spline_inflation, 2011 - param["year_begin"]), 1, size(income_distribution,2))
    macro.spline_inc_distribution = interp1d(Year[~np.isnan(Inc_avg)] - param["year_begin"], income_distribution[~np.isnan(Inc_avg), 2:12], method, 'pp')

    #Interest rate, price of fuel
    macro.spline_notaires = interp1d(year_interest_rate[~np.isnan(year_interest_rate)] - param["year_begin"], real_interest_rate[~np.isnan(year_interest_rate)], method, 'pp') #Interest rate
    macro.spline_carburant = interp1d(carbu_annee[~np.isnan(carbu_annee)] - param["year_begin"], carbu_carbu[~np.isnan(carbu_annee)]/100, method, 'pp')

    #Number of RDP houses in Cape Town
    RDP_2011 = 320969
    if option.future_construction_RDP == 1:
        macro.spline_RDP = interp1d([2001 - param["year_begin"], 2011 - param["year_begin"], 2041 - param["year_begin"]], [RDP_2011 - 10*5000, RDP_2011, RDP_2011 + 30*5000], method, 'pp')
    else: 
        macro.spline_RDP = interp1d([2001 - param["year_begin"], 2011 - param["year_begin"], 2018 - v, 2041 - param["year_begin"]], [RDP_2011 - 10*5000, RDP_2011, RDP_2011 + 7*5000, RDP_2011 + 7*5000], method, 'pp')

    #Data for total population
    macro.spline_population = interp1d(Year[~np.isnan(Pop_HH_total)] - param["year_begin"], Pop_HH_total[~np.isnan(Pop_HH_total)], method, 'pp')
    macro.spline_pop_inc_distribution = interp1d(Year[~np.isnan(Inc_avg)] - param["year_begin"], [Pop_11_class1[~np.isnan(Inc_avg)], Pop_11_class2[~np.isnan(Inc_avg)], Pop_11_class3[~np.isnan(Inc_avg)], Pop_11_class4[~np.isnan(Inc_avg)], Pop_11_class5[~np.isnan(Inc_avg)], Pop_11_class6[~np.isnan(Inc_avg)], Pop_11_class7[~np.isnan(Inc_avg)], Pop_11_class8[~np.isnan(Inc_avg)], Pop_11_class9[~np.isnan(Inc_avg)], Pop_11_class10[~np.isnan(Inc_avg)], Pop_11_class11[~np.isnan(Inc_avg)],], method, 'pp')
    
    #Definition du revenu
    revenu = ppval(macro.spline_revenu, np.tranpose(t))

    return macro