# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:34:23 2020

@author: Charlotte Liotta
"""

import pandas as pd 
import numpy as np
import timeit #pour mesurer les temps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #pour les grapohes en 3d
import scipy as sc
from scipy import optimize
import math
import copy

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
from plot_and_export_outputs.courbes_formal import *
from plot_and_export_outputs.courbes_informal import *
from plot_and_export_outputs.export_for_maps import *
from plot_and_export_outputs.stats_initiales import *
from plot_and_export_outputs.stats_dynamics import *

print("\n*** NEDUM-Cape-Town - Polycentric Version - Formal and Informal housing ***\n")

# %% Parameters and data

print("\n*** Load parameters and options ***\n")
param = choice_param()
option = choice_options()

print("\n*** Load data ***\n")

#Grid, land-use constraints and macro data
t = [0, 1, 2, 3, 4, 5, 6] #to go up to 2017
grid = SimulGrid()
grid.create_grid()
data_courbe = data_polycentrisme_CAPE_TOWN(grid, param)
land, param = import_coeff_land_CAPE_TOWN2(grid, option2, param, data_courbe)
macro_data = import_macro_data(param, options, t)

#Income groups in monocentric
option2 = copy.deepcopy(option)
option2["polycentric"] = 0
poly = import_emploi_CAPE_TOWN2(grid, param, option2, macro_data, data_courbe, t)
trans = charges_temps_polycentrique_CAPE_TOWN_3(option2,grid,macro_data,param,poly,t)

#Amenities
amenity = amenity_calibration_parameters_v3(grid,param, macro_data, poly, option2, trans, data_courbe, land, 2011)
land.amenite = amenity.estimated_amenities / np.mean(amenity.estimated_amenities)
param["coeff_beta"] = amenity.coeff_beta
param["coeff_alpha"] = 1 - param.coeff_beta
param["basic_q"] = amenity.basic_q
param["coeff_b"] = amenity.coeff_b
param["coeff_a"] = 1 - amenity.coeff_b
param["coeff_grandA"] = amenity.coeff_grandA * 1.3

#Job centers and transportation data
poly = import_emploi_CAPE_TOWN2(grid, param, option, macro_data, data_courbe, t)
trans = charges_temps_polycentrique_CAPE_TOWN_3(option, grid, macro_data, param, poly, t) 

# %% Initial state

#Solver
print('*** Initial state ***')
Uo_init = 0
etat_initial = NEDUM_basic_need_informal(t[0],trans,option,land,grid,macro_data,param,poly,Uo_init,param["housing_in"])
print('*** End of static resolution ***')

#Initial statistics
stat_initiales = compute_stat_initiales_formal_informal(trans,land,grid,macro_data,param,option,etat_initial,poly,t[0])

#Plot outputs
trace_courbes_formal(grid, etat_initial, stat_initiales, macro_data, data_courbe, land, param, macro, option, t[0])
trace_courbes_informal_abs(grid, etat_initial, stat_initiales, data_courbe, land, param, macro_data, poly, option, t[0])
export_for_maps_initial_benchmark(grid, land, etat_initial, stat_initiales, data_courbe)
print('*** Outputs exported ***')

# %% Evolution

print('*** Beginning of the evolution ***')
simulation = nedum_lite_polycentrique_1_6(t, etat_initial, trans, grid, land, poly, param, macro_data, option)
stat_dynamics = compute_stat_finales_Cape_Town(macro_data, option, trans, land, grid, param, poly, simulation)
print('*** End of the evolution ***')