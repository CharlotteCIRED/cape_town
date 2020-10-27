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
        
    def import_macro_data(self, param, option):
        
        method = 'linear' #Méthode pour faire les interpolations (linear ou spline)

        pathScenarios = "C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/data_Cape_Town/Scenarios"

        # 1. Import distribution of incomes

        scenario_income_distribution = pd.read_csv(pathScenarios + '/Scenario_inc_distrib_' + option["scenarioIncomeDistribution"] + '.csv', sep = ';')
        income_2011 = pd.read_csv('./2. Data/Basile data/Income_distribution_2011.csv')

        splinePopulationIncomeDistribution = interp1d(np.array([2001, 2011, 2040]) - param["baseline_year"], np.transpose([income_2011.Households_nb_2001, income_2011.Households_nb, scenario_income_distribution.Households_nb_2040]), method)
        averageIncome2001 = np.sum(income_2011.Households_nb_2001 * income_2011.INC_med) / sum(income_2011.Households_nb_2001)
        averageIncome2011 = np.sum(income_2011.Households_nb * income_2011.INC_med) / sum(income_2011.Households_nb)
        averageIncome2040 = np.sum(scenario_income_distribution.Households_nb_2040 * scenario_income_distribution.INC_med_2040) / sum(scenario_income_distribution.Households_nb_2040)

        # 2. Import other data from the scenario

        ScenarioPop = pd.read_csv(pathScenarios + '/Scenario_pop_' + option["scenarioPop"] + '.csv', sep = ';')
        ScenarioInterestRate = pd.read_csv(pathScenarios + '/Scenario_interest_rate_' + option["scenarioInterestRate"] + '.csv', sep = ';')
        ScenarioPriceFuel = pd.read_csv(pathScenarios + '/Scenario_price_fuel_' + option["scenarioPriceFuel"] + '.csv', sep = ';')
        ScenarioInflation = pd.read_csv(pathScenarios + '/Scenario_inflation_' + option["scenarioInflation"] + '.csv', sep = ';')

        spline_inflation = interp1d(ScenarioInflation.Year_infla[~np.isnan(ScenarioInflation.inflation_base_2010)] - param["baseline_year"], ScenarioInflation.inflation_base_2010[~np.isnan(ScenarioInflation.inflation_base_2010)], method)

        #Income - after 2011, income evolves as inflation (and it is the same for income for each income group)
        yearInc = ScenarioInflation.Year_infla[~np.isnan(ScenarioInflation.inflation_base_2010)]
        yearInc = yearInc[(yearInc > 2000) & (yearInc < 2041)]
        Inc_year_infla = interp1d([2001, 2011, 2040], [averageIncome2001, averageIncome2011, averageIncome2040], method)(yearInc)
       
        inflaRef = spline_inflation(yearInc[(yearInc > param["baseline_year"])] - param["baseline_year"]) / spline_inflation(0)
        Inc_year_infla[yearInc > param["baseline_year"]] = Inc_year_infla[yearInc == param["baseline_year"]] * inflaRef
        splineIncome = interp1d(yearInc - param["baseline_year"], Inc_year_infla,  method)
        incomeYearReference = splineIncome(0)

        #Rescale income by inflation (after 2011)
        incomeDistribution = np.array([income_2011.INC_med, income_2011.INC_med, income_2011.INC_med, scenario_income_distribution.INC_med_2040])
        incomeDistribution[ScenarioPop.Year_pop > 2011, :] = incomeDistribution[ScenarioPop.Year_pop > 2011, :] * np.matlib.repmat(spline_inflation(ScenarioPop.Year_pop[ScenarioPop.Year_pop > 2011] - param["baseline_year"]) / spline_inflation(2011 - param["baseline_year"]), 1, incomeDistribution.shape[1])
        splineIncomeDistribution = interp1d(ScenarioPop.Year_pop[~np.isnan(ScenarioPop.Year_pop)] - param["baseline_year"], np.transpose(incomeDistribution[~np.isnan(ScenarioPop.Year_pop), :]), method)

        spline_interest_rate = interp1d(ScenarioInterestRate.Year_interest_rate[~np.isnan(ScenarioInterestRate.real_interest_rate)] - param["baseline_year"], ScenarioInterestRate.real_interest_rate[~np.isnan(ScenarioInterestRate.real_interest_rate)], method) #Interest rate
        spline_fuel = interp1d(ScenarioPriceFuel.Year_fuel[~np.isnan(ScenarioPriceFuel.price_fuel)] - param["baseline_year"], ScenarioPriceFuel.price_fuel[~np.isnan(ScenarioPriceFuel.price_fuel)]/100, method)
        spline_population = interp1d(ScenarioPop.Year_pop[~np.isnan(ScenarioPop.HH_total)] - param["baseline_year"], ScenarioPop.HH_total[~np.isnan(ScenarioPop.HH_total)], method)
        

        # 3. Import the scenario for RDP/BNG houses

        RDP_2011 = min(2.2666e+05, sum(income_2011.formal[param["income_distribution"] == 1])) #(estimated as sum(data.gridFormal(data.countRDPfromGV > 0)))    % RDP_2011 = 320969; %227409; % Where from?
        RDP_2001 = min(1.1718e+05, sum(income_2011.Households_nb_2001[param["income_distribution"] == 1])) #(estimated as sum(data.gridFormal(data.countRDPfromGV > 0)))  % 262452; % Estimated by nb inc_1 - BY - settlement in 2001
        splineRDP = interp1d([2001 - param["baseline_year"], 2011 - param["baseline_year"], 2018 -  param["baseline_year"], 2041 - param["baseline_year"]], [RDP_2001, RDP_2011, RDP_2011 + 7*5000, RDP_2011 + 7*5000 + 23 * param["futureRatePublicHousing"]], method)
        
        # 4. Import evolution of agricultural land

        agriculturalRent2040 = param["agriculturalRent2011"] * spline_inflation(2040 - param["baseline_year"]) / spline_inflation(2011 - param["baseline_year"])
        splineAgriculturalRent = interp1d([2001 - param["baseline_year"], 2011 - param["baseline_year"], 2040 - param["baseline_year"]], [param["agriculturalRent2001"], param["agriculturalRent2011"], agriculturalRent2040], method)

        self.pop_inc_distribution = splinePopulationIncomeDistribution
        self.inflation = spline_inflation
        self.income = splineIncome
        self.income_year_reference = incomeYearReference
        self.income_distribution = splineIncomeDistribution
        self.interest_rate = spline_interest_rate
        self.population = spline_population
        self.rdp = splineRDP
        self.agricultural_rent = splineAgriculturalRent
        self.fuel_cost = spline_fuel

