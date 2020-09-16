# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:47:04 2020

@author: Charlotte Liotta
"""

import copy
import numpy.matlib
import pandas as pd
import numpy as np
import scipy.io

class ImportEmploymentData:
        
    def __init__(self):
        
        self

    def import_employment_data(self, grille, param, option, macro_data, t):
        
        # %% Import data
        TAZ = pd.read_csv('./2. Data/Basile data/TAZ_amp_2013_proj_centro2.csv') #Number of jobs per Transport Zone (TZ)

        #Number of employees in each TZ for the 12 income classes
        jobsCenters12Class = np.array([np.zeros(len(TAZ.Ink1)), TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink1/3, TAZ.Ink2/2, TAZ.Ink2/2, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink3/3, TAZ.Ink4/3, TAZ.Ink4/3, TAZ.Ink4/3])
        
        codeCentersInitial = TAZ.TZ2013
        xCoord = TAZ.X / 1000
        yCoord = TAZ.Y / 1000
        
        #Total number of households per class 
        yearIncomeDistribution = param["baseline_year"] + t
        totalBracket = macro_data.pop_inc_distribution(t)
        avgIncomeBracket = macro_data.income_distribution(t)

        #Total income distribution in the city
        avgIncomeGroup = np.zeros((len(yearIncomeDistribution), param["nb_of_income_classes"]))
        totalGroup = np.zeros((len(yearIncomeDistribution), param["nb_of_income_classes"]))
        for j in range(0, param["nb_of_income_classes"]):
            totalGroup[:, j] = np.sum(totalBracket[(param["income_distribution"] == j + 1), :], axis = 0)
            avgIncomeGroup[:, j] = np.sum(avgIncomeBracket[(param["income_distribution"] == j + 1), :] * totalBracket[param["income_distribution"] == j + 1, :], axis = 0) / totalGroup[:, j]


        selectedCenters = sum(jobsCenters12Class, 0) > 2500

        #Where we don't have reliable transport data
        selectedCenters[xCoord > -10] = np.zeros(1, 'bool')
        selectedCenters[yCoord > -3719] = np.zeros(1, 'bool')
        selectedCenters[(xCoord > -20) & (yCoord > -3765)] = np.zeros(1, 'bool')
        selectedCenters[codeCentersInitial == 1010] = np.zeros(1, 'bool')
        selectedCenters[codeCentersInitial == 1012] = np.zeros(1, 'bool')
        selectedCenters[codeCentersInitial == 1394] = np.zeros(1, 'bool')
        selectedCenters[codeCentersInitial == 1499] = np.zeros(1, 'bool')
        selectedCenters[codeCentersInitial == 4703] = np.zeros(1, 'bool')

        xCenter = xCoord[selectedCenters]
        yCenter = yCoord[selectedCenters]

        #Number of workers per group for the selected 
        jobsCentersNgroup = np.zeros((len(xCoord), param["nb_of_income_classes"]))
        for j in range(0, param["nb_of_income_classes"]):
             jobsCentersNgroup[:, j] = np.sum(jobsCenters12Class[param["income_distribution"] == j + 1, :], 0)

        jobsCentersNgroup = jobsCentersNgroup[selectedCenters, :]

        #Rescale of number of jobs after selection

        #Rescale to keep the correct global income distribution
        jobsCentersNGroupRescaled = np.zeros((jobsCentersNgroup.shape[0], jobsCentersNgroup.shape[1], len(yearIncomeDistribution)))
        for i in range(0, len(yearIncomeDistribution)):
            jobsCentersNGroupRescaled[:, :, i] = jobsCentersNgroup * totalGroup[i, :] / np.sum(jobsCentersNgroup, 0)

        #Export 
        yearCenters = yearIncomeDistribution
        totalHouseholdsGroup = totalGroup
        year = yearCenters

        codeCentersPolycentric = codeCentersInitial[selectedCenters]
        averageIncomeGroup = avgIncomeGroup

        increment = np.arange(0,len(TAZ.TZ2013))

        corresp = np.zeros(len(xCenter))
        for i in range(0, len(xCenter)):
            corresp[i] = increment[(TAZ.TZ2013 == codeCentersPolycentric.iloc[i])]
            
        jobsCentersMemory = jobsCenters12Class
        jobsCenters = jobsCentersNGroupRescaled

        whichYearInit = np.argmin(np.abs(param["baseline_year"] - year))
        jobsCenterInit = jobsCenters[:, :, whichYearInit]

        self.codeCentersInitial = codeCentersInitial
        self.selectedCenters = selectedCenters
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.totalHouseholdsGroup = totalHouseholdsGroup
        self.year = year
        self.codeCentersPolycentric = codeCentersPolycentric
        self.averageIncomeGroup = averageIncomeGroup
        increment = np.arange(0,len(TAZ.TZ2013))
        self.increment = increment[np.transpose(selectedCenters)]
        self.corresp = corresp
        self.jobsCentersMemory = jobsCentersMemory
        self.jobsCenters = jobsCenters
        self.jobsCenterInit = jobsCenterInit
        self.incomeCentersInit = scipy.io.loadmat('C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/0. Precalculated inputs/incomeCentersKeep.mat')['incomeCentersKeep']
        self.formal = np.array([1, 1, 1, 1]) #Select which income class can live in formal settlements
        self.backyard = np.array([1, 1, 0, 0]) #Select which income class can live in backyard settlements
        self.settlement = np.array([1, 1, 0, 0]) #Select which income class can live in informal settlements
        self.incomeGroup = np.matlib.repmat([1,2,3,4], len(year), 1)

