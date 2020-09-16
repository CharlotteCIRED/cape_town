# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:49:51 2020

@author: Charlotte Liotta
"""

def choice_options():
    
    #Scenarios RDP
    option = {"future_construction_RDP" : 1} #Assumptions on future subsidized housing construction
    option["taxOutUrbanEdge"] = 0
    option["urban_edge"] = 0 #1 mean we keep the urban edge
    
    #Solver
    option["adjustHousingSupply"] = 1 #For the solver
    option["ownInitializationSolver"] = 0
    
    #Autres
    option["logit"] = 1 #Logit for the modal allocation and cross-commuting
    option["sortEmploymentCenters"] = 1    
    
    #Scenario numbers
    option["scenarioPop"] = '2'
    option["scenarioIncomeDistribution"] = '2'
    option["scenarioInflation"] = '1'
    option["scenarioInterestRate"] = '1'
    option["scenarioPriceFuel"] = '1'

    return option
