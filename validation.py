# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:53:33 2020

@author: Charlotte Liotta
"""

from data.functions_to_import_data import *
from data.grid import *
from data.flood import *
from scipy.interpolate import griddata
from plot_and_export_outputs.export_outputs_flood_damages import *

import numpy as np
import matplotlib.pyplot as plt

print("\n*** Validation ***\n")

# %% Floods

# 1.Graph comparing estimated and real damages for each housing type

#Data
floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"
flood = FloodData()
flood.import_floods_data()
grid = SimulGrid()
grid.create_grid()
count_formal = households_data.formal_grid_2011 - households_data.GV_count_RDP
count_formal[count_formal < 0] = 0
mat1 = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations scenarios - 201908.mat')
simul1 = mat1["simulation_noUE"]
simul1_rent = simul1["rent"][0][0]
interest_rate = param["interest_rate"] + param["depreciation_rate"]
dwelling_size = SP_to_grid_2011_1(households_data.spDwellingSize, households_data.Code_SP_2011, grid)

#Compute damages
formal_structure_cost = dwelling_size * simul1_rent[0, 0, :] / interest_rate
damages_data = compute_damages(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011, formal_structure_cost, initialState_contentCost, flood)
formal_structure_cost = (initialState_rent[0,:] * initialState_dwellingSize[0, :]) / interest_rate
damages_simul = compute_damages(floods, path_data, initialState_householdsHousingType[0, :], initialState_householdsHousingType[3, :], initialState_householdsHousingType[2, :], initialState_householdsHousingType[1, :], formal_structure_cost, initialState_contentCost, flood)

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.labelleft'] = True

data = [[annualize_damages(damages_simul.formal_structure_damages), annualize_damages(damages_simul.formal_content_damages)],
[annualize_damages(damages_data.formal_structure_damages), annualize_damages(damages_data.formal_content_damages)]]
X = np.arange(2)
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = "Simulation")
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = "Data")
#plt.legend()
plt.ylim(0, 180000000)
plt.title("Formal")
text(0.125, 170000000, "Structures")
text(1.125, 170000000, "Contents")
plt.show()

data = [[annualize_damages(damages_simul.subsidized_structure_damages), annualize_damages(damages_simul.subsidized_content_damages)],
[annualize_damages(damages_data.subsidized_structure_damages), annualize_damages(damages_data.subsidized_content_damages), ]]
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = "Simulation")
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = "Data")
plt.ylim(0, 85000)
plt.title("Subsidized")
text(0.125, 82000, "Structures")
text(1.125, 82000, "Contents")
plt.show()

data = [[annualize_damages(damages_simul.informal_structure_damages), annualize_damages(damages_simul.informal_content_damages)],
[annualize_damages(damages_data.informal_structure_damages), annualize_damages(damages_data.informal_content_damages), ]]
plt.bar(X + 0.00, data[0], color = 'b', width = 0.25, label = "Simulation")
plt.bar(X + 0.25, data[1], color = 'g', width = 0.25, label = "Data")
plt.ylim(0, 180000)
plt.title("Informal")
text(0.125, 160000, "Structures")
text(1.125, 160000, "Contents")
plt.show()

# 2. Graph comparing esimated and real flood depth for each housing type
formal_shirt = [stats_per_housing_type_data.flood_depth_formal[5], stats_per_housing_type_data.flood_depth_subsidized[5], stats_per_housing_type_data.flood_depth_informal[5], stats_per_housing_type_data.flood_depth_backyard[5]]
tshirt2 = [stats_per_housing_type_simul.flood_depth_formal[2], stats_per_housing_type_simul.flood_depth_subsidized[2], stats_per_housing_type_simul.flood_depth_informal[2], stats_per_housing_type_simul.flood_depth_backyard[2]]
formal_shirt2 = [stats_per_housing_type_simul.flood_depth_formal[5], stats_per_housing_type_simul.flood_depth_subsidized[5], stats_per_housing_type_simul.flood_depth_informal[5], stats_per_housing_type_simul.flood_depth_backyard[5]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.arange(len(quarter))
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, np.array(tshirt), color=colors[1], edgecolor='white', width=barWidth, label='20 years')
plt.bar(r, np.array(formal_shirt) - (np.array(tshirt)), bottom=(np.array(tshirt)), color=colors[2], edgecolor='white', width=barWidth, label='100 years')
plt.bar(r + 0.25, np.array(tshirt2), color=colors[1], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(formal_shirt2) - np.array(tshirt2), bottom=np.array(tshirt2), color=colors[2], edgecolor='white', width=barWidth)
plt.legend()
plt.xticks(r, quarter, fontweight='bold')
plt.ylabel("Flood depth (cm)")
plt.show()

# 3. Graph comparing estimated and real proportion of flood-prone dwellings for each housing type

stats_per_housing_type_data = compute_stats_per_housing_type(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011)
stats_per_housing_type_simul = compute_stats_per_housing_type(floods, path_data, initialState_householdsHousingType[0, :], initialState_householdsHousingType[3, :], initialState_householdsHousingType[2, :], initialState_householdsHousingType[1, :])

quarter = ["FP", "FS", "IS", "IB"]
jeans = [stats_per_housing_type_data.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_data.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_data.fraction_informal_in_flood_prone_area[2], stats_per_housing_type_data.fraction_backyard_in_flood_prone_area[2]]
tshirt = [stats_per_housing_type_data.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_data.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_data.fraction_informal_in_flood_prone_area[5], stats_per_housing_type_data.fraction_backyard_in_flood_prone_area[5]]
formal_shirt = [stats_per_housing_type_data.fraction_formal_in_flood_prone_area[9], stats_per_housing_type_data.fraction_subsidized_in_flood_prone_area[9], stats_per_housing_type_data.fraction_informal_in_flood_prone_area[9], stats_per_housing_type_data.fraction_backyard_in_flood_prone_area[9]]
jeans2 = [stats_per_housing_type_simul.fraction_formal_in_flood_prone_area[2], stats_per_housing_type_simul.fraction_subsidized_in_flood_prone_area[2], stats_per_housing_type_simul.fraction_informal_in_flood_prone_area[2], stats_per_housing_type_simul.fraction_backyard_in_flood_prone_area[2]]
tshirt2 = [stats_per_housing_type_simul.fraction_formal_in_flood_prone_area[5], stats_per_housing_type_simul.fraction_subsidized_in_flood_prone_area[5], stats_per_housing_type_simul.fraction_informal_in_flood_prone_area[5], stats_per_housing_type_simul.fraction_backyard_in_flood_prone_area[5]]
formal_shirt2 = [stats_per_housing_type_simul.fraction_formal_in_flood_prone_area[9], stats_per_housing_type_simul.fraction_subsidized_in_flood_prone_area[9], stats_per_housing_type_simul.fraction_informal_in_flood_prone_area[9], stats_per_housing_type_simul.fraction_backyard_in_flood_prone_area[9]]
colors = ['#FF9999', '#00BFFF','#C1FFC1','#CAE1FF','#FFDEAD']
r = np.arange(len(quarter))
barWidth = 0.25
plt.figure(figsize=(10,7))
plt.bar(r, jeans, color=colors[0], edgecolor='white', width=barWidth, label="20 years")
plt.bar(r, np.array(tshirt) - np.array(jeans), bottom=np.array(jeans), color=colors[1], edgecolor='white', width=barWidth, label='100 years')
plt.bar(r, np.array(formal_shirt) - (np.array(tshirt)), bottom=(np.array(tshirt)), color=colors[2], edgecolor='white', width=barWidth, label='1000 years')
plt.bar(r + 0.25, np.array(jeans2), color=colors[0], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(tshirt2) - np.array(jeans2), bottom=np.array(jeans2), color=colors[1], edgecolor='white', width=barWidth)
plt.bar(r + 0.25, np.array(formal_shirt2) - np.array(tshirt2), bottom=np.array(tshirt2), color=colors[2], edgecolor='white', width=barWidth)
plt.legend()
plt.xticks(r, quarter, fontweight='bold')
plt.ylabel("Dwellings in flood-prone areas (%)")
plt.show()

# %% Others

#Graph comparing estimated and real distance to the employments center
dataDistanceDistribution = np.array([45.6174222, 18.9010734, 14.9972971, 9.6725616, 5.9425438, 2.5368754, 0.9267125, 0.3591011, 1.0464129])
timeOutput, distanceOutput, monetaryCost, costTime = import_transport_costs(option, grid, macro_data, param, job, households_data, [0, 1], 1)

modalShares, incomeCenters, distanceDistribution = EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job, households_data)


def EstimateIncome(param, timeOutput, distanceOutput, monetaryCost, costTime, job, households_data):
    #Solve for income per employment centers for different values of lambda

    print('Estimation of local incomes, and lambda parameter')

    annualToHourly = 1 / (8*20*12)
    bracketsTime = np.array([0, 15, 30, 60, 90, np.nanmax(np.nanmax(np.nanmax(timeOutput)))])
    bracketsDistance = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 200])

    timeCost = copy.deepcopy(costTime) #en h de transport par h de travail
    timeCost[np.isnan(timeCost)] = 10 ** 2
    monetary_cost = monetaryCost * annualToHourly
    monetary_cost[np.isnan(monetary_cost)] = 10 ** 3 * annualToHourly
    transportTimes = timeOutput / 2
    transportDistances = distanceOutput[:, :, 0]

    modalSharesTot = np.zeros(5)
    incomeCentersSave = np.zeros((len(job.jobsCenters[:,0,0]), 4))
    distanceDistribution = np.zeros((len(bracketsDistance) - 1))

    param_lambda = 4.27
        
    print('Estimating for lambda = ', param_lambda)
        
    incomeCentersAll = -math.inf * np.ones((len(job.jobsCenters[:,0,0]), 4))
    modalSharesGroup = np.zeros((5, 4))
    timeDistributionGroup = np.zeros((len(bracketsTime) - 1, 4))
    distanceDistributionGroup = np.zeros((len(bracketsDistance) - 1, 4))

    for j in range(0, 4):
        #Household size varies with transport costs
        householdSize = param["household_size"][j]
            
        averageIncomeGroup = job.averageIncomeGroup[0, j] * annualToHourly
    
        print('incomes for group ', j)
        
        whichJobsCenters = job.jobsCenters[:, j, 0] > 600
        popCenters = job.jobsCenters[whichJobsCenters, j, 0]
        data_class = SP_to_grid_2011_2(households_data.income_n_class_SP_2011[:, j], households_data.Code_SP_2011, grid)
        popResidence = copy.deepcopy(data_class) * sum(job.jobsCenters[whichJobsCenters, j, 0]) / np.nansum(data_class)
        
        funSolve = lambda incomeCentersTemp: fun0(incomeCentersTemp, averageIncomeGroup, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)

        maxIter = 150
        tolerance = 0.001
        if j == 0:
            factorConvergenge = 0.008
        elif j == 1:
            factorConvergenge = 0.005
        else:
            factorConvergenge = 0.0005
        
        index = 0
        error = np.zeros((len(popCenters), maxIter))
        scoreIter = np.zeros(maxIter)
        errorMax = 1
        
        #Initializing the solver
        incomeCenters = np.zeros((sum(whichJobsCenters), maxIter))
        #incomeCenters[:, 0] =  averageIncomeGroup * (popCenters / np.nanmean(popCenters)) ** (0.1)
        incomeCenters[:, 0] =  1
        error[:, 0] = funSolve(incomeCenters[:, 0])
      
        while ((index <= maxIter - 2) & (errorMax > tolerance)):
            
            index = index + 1
            incomeCenters[:,index] = incomeCenters[:, max(index-1, 0)] + factorConvergenge * averageIncomeGroup * error[:, max(index - 1,0)] / popCenters
            error[:,index] = funSolve(incomeCenters[:,index])
            errorMax = np.nanmax(np.abs(error[:, index] / popCenters))
            scoreIter[index] = np.nanmean(np.abs(error[:, index] / popCenters))
            print(index)
            print(errorMax)
            
        if (index > maxIter - 1):
            scoreBest = np.amin(scoreIter)
            bestSolution = np.argmin(scoreIter)
            incomeCenters[:, index] = incomeCenters[:, bestSolution]
            print(' - max iteration reached - mean error', scoreBest)
        else:
            print(' - computed - max error', errorMax)
        
        
        incomeCentersRescaled = incomeCenters[:, iter] * averageIncomeGroup / ((np.nansum(incomeCenters[:, iter] * popCenters) / np.nansum(popCenters)))
        modalSharesGroup[:,j] = modalShares(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, param_lambda)
        incomeCentersAll[whichJobsCenters,j] = incomeCentersRescaled
        
        distanceDistributionGroup[:,j] = computeDistributionCommutingDistances(incomeCentersRescaled, popCenters, popResidence, monetary_cost[whichJobsCenters,:,:] * householdSize, timeCost[whichJobsCenters,:,:] * householdSize, transportDistances[whichJobsCenters,:], bracketsDistance, param_lambda)

        modalSharesTot = np.nansum(modalSharesGroup, 1) / np.nansum(np.nansum(modalSharesGroup))
        incomeCentersSave = incomeCentersAll / annualToHourly
        distanceDistribution = np.nansum(distanceDistributionGroup, 1) / np.nansum(np.nansum(distanceDistributionGroup))

    return modalSharesTot, incomeCentersSave, distanceDistribution

def fun0(incomeCenters_here, meanIncome, popCenters, popResidence, monCost, tCost, param_lambda):
    """ Computes error in employment allocation """

    incomeCentersFull = incomeCenters_here * meanIncome / ((np.nansum(incomeCenters_here * popCenters) / np.nansum(popCenters)))
    transportCostModes = monCost + tCost * incomeCentersFull[:, None, None]
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500
    transportCost = - 1 / param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCentersFull[:, None] - transportCost))) - 500
    score = popCenters - np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCentersFull[:, None] - transportCost) - minIncome), 0)[None, :] * popResidence[None, :], 1)

    return score


def modalShares(incomeCenters, popCenters, popResidence, monetaryCost, timeCost, param_lambda):
    """ Computes total modal shares """

    #Transport cost by modes
    transportCostModes = monetaryCost + timeCost * incomeCenters[:, None, None]
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda  * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]
    transportCost = - 1 / param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters[:, None] - transportCost))) - 500
    modalSharesTot = np.nansum(np.nansum(modalSharesTemp * (np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * (incomeCenters[:, None] - transportCost) - minIncome)))[:, :, None] * popResidence, 1), 0)
    return modalSharesTot


def computeDistributionCommutingDistances(incomeCenters_here, popCenters, popResidence, monCost, tCost, transportDistance, bracketsDistance, param_lambda):

    #Transport cost by modes
    transportCostModes = monCost + tCost * incomeCenters_here[:, None, None]

    #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
    valueMax = np.nanmin(param_lambda * transportCostModes, 2) - 500

    #Compute modal shares
    modalSharesTemp = np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]) / np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)[:, :, None]

    #Multiply by OD flows
    transportCost = - 1/param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + valueMax[:, :, None]), 2)) - valueMax)

    #minIncome is also to prevent diverging exponentials
    minIncome = np.nanmax(np.nanmax(param_lambda * (incomeCenters_here[:, None] - transportCost))) - 500

    #Total distribution of times
    nbCommuters = np.zeros(len(bracketsDistance) - 1)
    for k in range(0, len(bracketsDistance)-1):
        which = (transportDistance > bracketsDistance[k]) & (transportDistance <= bracketsDistance[k + 1]) & (~np.isnan(transportDistance))
        nbCommuters[k] = np.nansum(np.nansum(np.nansum(which[:, :, None] * modalSharesTemp * np.exp(param_lambda * (incomeCenters_here[:, None] - transportCost) - minIncome)[:, :, None] / np.nansum(np.exp(param_lambda * (incomeCenters_here[:, None] - transportCost) - minIncome), 0)[None, :, None] * popResidence[None, :, None], 1)))

    return nbCommuters



#Results of the regression on amenities and amenity score

#Population density
xData = grid.dist
yData = (households_data.nb_poor_grid_2011 + households_data.nb_rich_grid_2011) / 0.25
xSimul = grid.dist
ySimul = np.nansum(initialState_householdsHousingType, 0) / 0.25

df = pd.DataFrame(data = np.transpose(np.array([xData, yData, ySimul])), columns = ["x", "yData", "ySimul"])
df["round"] = round(df.x)
new_df = df.groupby(['round']).mean()
q1_df = df.groupby(['round']).quantile(0.25)
q3_df = df.groupby(['round']).quantile(0.75)

plt.plot(np.arange(max(df["round"] + 1)), new_df.yData, color = "black", label = "Data")
plt.plot(np.arange(max(df["round"] + 1)), new_df.ySimul, color = "green", label = "Simul")
axes = plt.axes()
axes.set_ylim([0, 2000])
axes.set_xlim([0, 50])
axes.fill_between(np.arange(max(df["round"] + 1)), q1_df.ySimul, q3_df.ySimul, color = "lightgreen")
plt.legend()

#Housing types
xData = grid.dist
formal_data = (households_data.formal_grid_2011) / 0.25
backyard_data = (households_data.backyard_grid_2011) / 0.25
informal_data = (households_data.informal_grid_2011) / 0.25
formal_simul = (initialState_householdsHousingType[0, :] + initialState_householdsHousingType[3, :]) / 0.25
informal_simul = (initialState_householdsHousingType[2, :]) / 0.25
backyard_simul = (initialState_householdsHousingType[1, :]) / 0.25

df = pd.DataFrame(data = np.transpose(np.array([xData, formal_data, backyard_data, informal_data, formal_simul, backyard_simul, informal_simul])), columns = ["xData", "formal_data", "backyard_data", "informal_data", "formal_simul", "backyard_simul", "informal_simul"])
df["round"] = round(df.xData)
new_df = df.groupby(['round']).mean()

plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_data, color = "black", label = "Data")
plt.plot(np.arange(max(df["round"] + 1)), new_df.formal_simul, color = "green", label = "Simul")
axes = plt.axes()
axes.set_ylim([0, 1600])
axes.set_xlim([0, 40])
plt.title("Formal")
plt.legend()

plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_data, color = "black", label = "Data")
plt.plot(np.arange(max(df["round"] + 1)), new_df.informal_simul, color = "green", label = "Simul")
axes = plt.axes()
axes.set_ylim([0, 600])
axes.set_xlim([0, 40])
plt.title("Informal")
plt.legend()

plt.plot(np.arange(max(df["round"] + 1)), new_df.backyard_data, color = "black", label = "Data")
plt.plot(np.arange(max(df["round"] + 1)), new_df.backyard_simul, color = "green", label = "Simul")
axes = plt.axes()
axes.set_ylim([0, 600])
axes.set_xlim([0, 40])
plt.title("Backyard")
plt.legend()

#Housing prices

priceSimul = (initialState_rent[0, :] * param["coeff_A"] * param["coeff_b"] ** param["coeff_b"] / (interest_rate)) ** (1/param["coeff_a"])
priceSimulPricePoints = griddata(np.transpose(np.array([grid.horiz_coord, grid.vert_coord])), priceSimul, np.transpose(np.array([households_data.X_SP_2011, households_data.Y_SP_2011])))

xData = np.sqrt((households_data.X_SP_2011 - grid.x_center) ** 2 + (households_data.Y_SP_2011-grid.y_center) ** 2)
yData = households_data.sale_price_SP[2,:]
xSimulation = xData
ySimulation = priceSimulPricePoints

df = pd.DataFrame(data = np.transpose(np.array([xData, yData, ySimulation])), columns = ["xData", "yData", "ySimulation"])
df["round"] = round(df.xData)
new_df = df.groupby(['round']).mean()

which = ~np.isnan(new_df.yData) & ~np.isnan(new_df.ySimulation)

plt.plot(new_df.xData[which], new_df.yData[which], color = "black", label = "Data")
plt.plot(new_df.xData[which], new_df.ySimulation[which], color = "green", label = "Simul")
axes = plt.axes()

plt.legend()


def annualize_damages(array):
    interval0 = 1 - (1/5)    
    interval1 = (1/5) - (1/10)
    interval2 = (1/10) - (1/20)
    interval3 = (1/20) - (1/50)
    interval4 = (1/50) - (1/75)
    interval5 = (1/75) - (1/100)
    interval6 = (1/100) - (1/200)
    interval7 = (1/200) - (1/250)
    interval8 = (1/250) - (1/500)
    interval9 = (1/500) - (1/1000)
    interval10 = (1/1000)
    return 0.5 * ((interval0 * 0) + (interval1 * array[0]) + (interval2 * array[1]) + (interval3 * array[2]) + (interval4 * array[3]) + (interval5 * array[4]) + (interval6 * array[5]) + (interval7 * array[6]) + (interval8 * array[7]) + (interval9 * array[8]) + (interval10 * array[9]))
