# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:34:23 2020

@author: Charlotte Liotta
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import optimize
import math
import copy
import scipy.io

from parameters_and_options.parameters import *
from parameters_and_options.options import *
from data.amenity import *
from data.data import *
from data.grille import *
from data.job import *
from data.land import *
from data.macro_data import *
from data.transport import *
from solver.solver import *
from solver.evolution import *

print("\n*** NEDUM-Cape-Town - Polycentric Version - Formal and Informal housing ***\n")

# %% Choose parameters and options

print("\n*** Load parameters and options ***\n")

param = choice_param()
option = choice_options()

t = np.array([0, 1, 2, 3, 4, 5, 6]) #Years of the scenario

# %% Import land-use, households and macro data

print("\n*** Load data ***\n")

#Import grid
grid = SimulGrid()
grid.create_grid()

#Import data on households (Census, Housing, Income,...)
households_data = ImportHouseholdsData()
households_data.import_data(grid, param)

#Import land-use data
land = Land()
land.import_land_use(grid, option, param, households_data, macro_data)
param = add_construction_parameters(param, households_data, land, grid)

#Import macro data (population, inflation,...)
macro_data = MacroData()
macro_data.import_macro_data(param, option, t)

# %% Calibration of the model

print("\n*** Calibration ***\n")

#Monocentric calibration
option2 = copy.deepcopy(option)
option2["polycentric"] = 0
job = ImportEmploymentData()
job.import_employment_data(grid, param, option2, macro_data, t)
trans = TransportData()
trans.import_transport_data(option2, grid, macro_data, param, job, t)

#Calibration using amenities
amenity = Amenity()
amenity.calibration(grid, param, macro_data, job, option2, trans, households_data, land, 2011)
land.amenite = amenity.estimated_amenities / np.mean(amenity.estimated_amenities)
#land.amenite = np.ones(24014)
param["coeff_beta"] = amenity.coeff_beta
param["coeff_alpha"] = 1 - param["coeff_beta"]
param["basic_q"] = amenity.basic_q
param["coeff_b"] = amenity.coeff_b
param["coeff_a"] = 1 - amenity.coeff_b
param["coeff_A"] = amenity.coeff_A * 1.3

# %% Import employment and transportation data

print("\n*** Load employment and transportation data ***\n")

job = ImportEmploymentData()
job.import_employment_data(grid, param, option, macro_data, t)
trans = TransportData()
trans.import_transport_data(option, grid, macro_data, param, job, t) 

# %% Initial state

print('*** Initial state ***')
print("Year", t[0] + param["baseline_year"])
Uo_perso = 8000
etat_initial_erreur, etat_initial_job_simul, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_R_mat, etat_initial_capital_land1, etat_initial_revenu_in, etat_initial_limite1, etat_initial_matrice_J, etat_initial_mult, etat_initial_utility, etat_initial_impossible_population = solver(t[0], trans, option, land, grid, macro_data, param, job, Uo_perso)
print('*** End of static resolution ***')
print("Mean abs. error on the nb. of workers per employment center:", int(np.mean(np.abs(etat_initial_erreur))), " %")
print("Total population:", int(sum(etat_initial_people1)))
#stat_initiales = compute_stat_initiales_formal_informal(trans, land, grid, macro_data, param,option,etat_initial,poly,t[0])

#Plot outputs
#trace_courbes_formal(grid, etat_initial, stat_initiales, macro_data, data_courbe, land, param, macro, option, t[0])
#trace_courbes_informal_abs(grid, etat_initial, stat_initiales, data_courbe, land, param, macro_data, poly, option, t[0])
#export_for_maps_initial_benchmark(grid, land, etat_initial, stat_initiales, data_courbe)
#print('*** Outputs exported ***')

# %% Scenarios

print('*** Beginning of the scenarios ***')
simulation_people_travaille, simulation_people_housing_type, simulation_hous, simulation_rent, simulation_people, simulation_erreur, simulation_housing, simulation_Uo_bis, simulation_deriv_housing, simulation_T = scenarios(t, etat_initial_erreur, etat_initial_people_housing_type, etat_initial_people_center, etat_initial_people1, etat_initial_hous1, etat_initial_housing1, etat_initial_rent1, etat_initial_utility, etat_initial_revenu_in, trans, grid, land, job, param, macro_data, option)
#stat_dynamics = compute_stat_finales_Cape_Town(macro_data, option, trans, land, grid, param, poly, simulation)
print('*** End of the scenario ***')
for i in range(0, len(t)):
    print("Year", t[i] + param["baseline_year"], "-", "Mean abs. error:", int(np.mean(np.abs(simulation_erreur[i]))), " % -", "Total population :", int(sum(simulation_people[i])))
