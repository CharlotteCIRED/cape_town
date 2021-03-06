# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:49:51 2020

@author: Charlotte Liotta
"""

def choice_options():
    
    #Urban edge
    option = {"taxOutUrbanEdge" : 0}
    option["urban_edge"] = 0 #1 mean we keep the urban edge
    
    #Solver
    option["adjustHousingSupply"] = 1 #For the solver
    
    #Scenario numbers
    option["scenarioPop"] = '2'
    option["scenarioIncomeDistribution"] = '2'
    option["scenarioInflation"] = '1'
    option["scenarioInterestRate"] = '1'
    option["scenarioPriceFuel"] = '1'

    return option
