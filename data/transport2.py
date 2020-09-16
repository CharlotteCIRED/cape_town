# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:53:22 2020

@author: Charlotte Liotta
"""

class TransportData:
        
    def __init__(self):
        
        self
        
    def import_transport_data(self, option, grid, macro_data, param, job, households_data, yearTraffic, extrapolate):
        """ Compute travel times and costs """

        #### STEP 1: IMPORT TRAVEL TIMES AND COSTS

        # Import travel times and distances
        transport_times = scipy.io.loadmat('./2. Data/Basile data/Transport_times_GRID.mat')
        
        #Price per km
        priceTrainPerKMMonth = 0.164 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceTrainFixedMonth = 4.48 * 40 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceTaxiPerKMMonth = 0.785 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceTaxiFixedMonth = 4.32 * 40 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceBusPerKMMonth = 0.522 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        priceBusFixedMonth = 6.24 * 40 * macro_data.inflation(2011 - param["baseline_year"]) / macro_data.inflation(2013 - param["baseline_year"])
        inflation = macro_data.inflation(yearTraffic)
        infla_2012 = macro_data.inflation(2012 - param["baseline_year"])
        priceTrainPerKMMonth = priceTrainPerKMMonth * inflation / infla_2012
        priceTrainFixedMonth = priceTrainFixedMonth * inflation / infla_2012
        priceTaxiPerKMMonth = priceTaxiPerKMMonth * inflation / infla_2012
        priceTaxiFixedMonth = priceTaxiFixedMonth * inflation / infla_2012
        priceBusPerKMMonth = priceBusPerKMMonth * inflation / infla_2012
        priceBusFixedMonth = priceBusFixedMonth * inflation / infla_2012
        priceFuelPerKMMonth = macro_data.fuel_cost(yearTraffic)
        
        #Fixed costs
        priceFixedVehiculeMonth = 400 
        priceFixedVehiculeMonth = priceFixedVehiculeMonth * inflation / infla_2012
        
        #### STEP 2: TRAVEL TIMES AND COSTS AS MATRIX
        
        #parameters
        numberDaysPerYear = 235
        numberHourWorkedPerDay= 8
        annualToHourly = 1 / (8*20*12)
        

        #Time by each mode, aller-retour, en minute
        timeOutput = np.empty((transport_times["durationTrain"].shape[0], transport_times["durationTrain"].shape[1], 5))
        timeOutput[:,:,0] = transport_times["distanceCar"] / param["walking_speed"] * 60 * 1.2 * 2
        timeOutput[:,:,0][np.isnan(transport_times["durationCar"])] = np.nan
        timeOutput[:,:,1] = copy.deepcopy(transport_times["durationTrain"])
        timeOutput[:,:,2] = copy.deepcopy(transport_times["durationCar"])
        timeOutput[:,:,3] = copy.deepcopy(transport_times["durationMinibus"])
        timeOutput[:,:,4] = copy.deepcopy(transport_times["durationBus"])

        #Length (km) using each mode
        multiplierPrice = np.empty((timeOutput.shape))
        multiplierPrice[:,:,0] = np.zeros((timeOutput[:,:,0].shape))
        multiplierPrice[:,:,1] = transport_times["distanceCar"]
        multiplierPrice[:,:,2] = transport_times["distanceCar"]
        multiplierPrice[:,:,3] = transport_times["distanceCar"]
        multiplierPrice[:,:,4] = transport_times["distanceCar"]

        #Multiplying by 235 (days per year)
        pricePerKM = np.empty((len(priceFuelPerKMMonth), 5))
        pricePerKM[:, 0] = np.zeros(len(priceFuelPerKMMonth))
        pricePerKM[:, 1] = priceTrainPerKMMonth*numberDaysPerYear
        pricePerKM[:, 2] = priceFuelPerKMMonth*numberDaysPerYear          
        pricePerKM[:, 3] = priceTaxiPerKMMonth*numberDaysPerYear
        pricePerKM[:, 4] = priceBusPerKMMonth*numberDaysPerYear
        
        #Distances (not useful to calculate price but useful output)
        distanceOutput = np.empty((timeOutput.shape))
        distanceOutput[:,:,0] = transport_times["distanceCar"]
        distanceOutput[:,:,1] = transport_times["distanceCar"]
        distanceOutput[:,:,2] = transport_times["distanceCar"]
        distanceOutput[:,:,3] = transport_times["distanceCar"]
        distanceOutput[:,:,4] = transport_times["distanceCar"]

        #Monetary price per year
        monetaryCost = np.zeros((len(job.codeCentersPolycentric), timeOutput.shape[1], 5))
        trans_monetaryCost = np.zeros((len(job.codeCentersPolycentric), timeOutput.shape[1], 5, len(yearTraffic)))
        for index in range(0, len(yearTraffic)):    
            for index2 in range(0, 5):
                monetaryCost[:,:,index2] = pricePerKM[index,index2] * multiplierPrice[:,:,index2]
                monetaryCost[:,:,1] = monetaryCost[:,:,1] + priceTrainFixedMonth[index] * 12 #train (monthly fare)
                monetaryCost[:,:,2] = monetaryCost[:,:,2] + priceFixedVehiculeMonth[index] * 12 #private car
                monetaryCost[:,:,3] = monetaryCost[:,:,3] + priceTaxiFixedMonth[index] * 12 #minibus-taxi
                monetaryCost[:,:,4] = monetaryCost[:,:,4] + priceBusFixedMonth[index] * 12 #bus
            trans_monetaryCost[:,:,:,index] = copy.deepcopy(monetaryCost)

        #### STEP 3: COMPUTE PROBA TO WORK IN C, EXPECTED INCOME AND EXPECTED NB OF
        #RESIDENTS OF INCOME GROUP I WORKING IN C


        costTime = (timeOutput * param["timeCost"]) / (60 * numberHourWorkedPerDay) #en h de transport par h de travail
        costTime[np.isnan(costTime)] = 10 ** 2
        param_lambda = param["lambda"].squeeze()

        incomeNetOfCommuting = np.zeros((param["nb_of_income_classes"], len(grid.dist), len(yearTraffic)))
        averageIncome = np.zeros((param["nb_of_income_classes"], len(grid.dist), len(yearTraffic)))
        modalShares = np.zeros((len(job.incomeCentersInit), param["nb_of_income_classes"], len(grid.dist), 5, len(yearTraffic)))
        ODflows = np.zeros((len(job.incomeCentersInit), param["nb_of_income_classes"], len(grid.dist), len(yearTraffic)))
    
        #income
        income_group = interp1d(job.year - param["baseline_year"], np.transpose(job.averageIncomeGroup))
        income_group = income_group(int(yearTraffic[index]))
        incomeGroup = np.matlib.repmat(income_group, len(grid.dist), 1)
        #income in 2011
        income_group_ref = interp1d(job.year - param["baseline_year"], np.transpose(job.averageIncomeGroup))
        income_group_ref = income_group_ref(int(yearTraffic[0]))
        incomeGroupRef = np.matlib.repmat(income_group_ref, len(grid.dist), 1)
        #income centers
        incomeCenters = job.incomeCentersInit * incomeGroup[0, :] / incomeGroupRef[0, :]
    
        #switch to hourly
        monetaryCost = trans_monetaryCost * annualToHourly #en coût par heure
        monetaryCost[np.isnan(monetaryCost)] = 10**3 * annualToHourly
        incomeCenters = incomeCenters * annualToHourly
        
        xInterp = grid.horiz_coord
        yInterp = grid.vert_coord
            
        #whichCenters = incomeCenters > -100000
        #incomeCentersGroup = incomeCenters[whichCenters]
        incomeCentersGroup = incomeCenters
           
        #Transport costs and employment allocation (cout par heure)
        transportCostModes = np.array(param["household_size"])[None, :, None, None] * (np.repeat(monetaryCost[:, np.newaxis, :, :], 4, axis=1) + (np.repeat(costTime[:, np.newaxis, :, :], 4, axis=1) * np.repeat(np.repeat(incomeCentersGroup[:, :, np.newaxis], 24014, axis=2)[:, :, :, np.newaxis], 5, axis = 3)))
        #&(185, 4, 24014, 5)
        #Value max is to prevent the exp to diverge to infinity (in matlab: exp(800) = Inf)
        valueMax = (np.min(param_lambda * transportCostModes, axis = 3) - 500) #-500
        
        #Modal shares
        #modalShares[whichCenters,:,:,j,index] = np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2))  / np.repeat(np.nansum(np.exp(-param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2)[:, :, np.newaxis], 5, axis=2)
        modalShares[:,:,:,:,index] = np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, :, np.newaxis], 5, axis=3))  / np.repeat(np.nansum(np.exp(-param_lambda * transportCostModes + np.repeat(valueMax[:, :, :, np.newaxis], 5, axis=3)), 3)[:, :, :, np.newaxis], 5, axis=3)
        
        #Transport costs
        #transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2) - valueMax))
        #transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes + np.repeat(valueMax[:, :, np.newaxis], 5, axis=2)), 2) - valueMax))
        transportCost = - 1 /param_lambda * (np.log(np.nansum(np.exp(- param_lambda * transportCostModes), 3)))
        
        #minIncome is also to prevent diverging exponentials
        minIncome = np.nanmax(param_lambda * (np.repeat(incomeCentersGroup[:, :, np.newaxis], 24014, 2)) - transportCost) - 700
            
        #OD flows
        #ODflows[whichCenters,:,j] = np.exp(param_lambda * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1)) - transportCost) - minIncome) / np.transpose(np.matlib.repmat(np.nansum(np.exp(param_lambda * (np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1)) - transportCost) - minIncome), 1), 24014, 1))
        #ODflows[whichCenters,:,j, index] = np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome), 0)
        ODflows[:,:,:,index] = np.exp(param_lambda * ((np.repeat(incomeCentersGroup[:, :, np.newaxis], 24014, 2)) - transportCost) - minIncome) / np.nansum(np.exp(param_lambda * ((np.repeat(incomeCentersGroup[:, :, np.newaxis], 24014, 2)) - transportCost) - minIncome), 0)
                
        #Income net of commuting (correct formula)
        #incomeNetOfCommuting[j,:, index] = 1 /param_lambda * (np.log(np.nansum(np.exp(param_lambda * ((np.transpose(np.matlib.repmat(incomeCentersGroup, 24014, 1))) - transportCost) - minIncome), 0)) + minIncome)
        incomeNetOfCommuting[:,:, index] = 1 /param_lambda * (np.log(np.nansum(np.exp(param_lambda * (np.repeat(incomeCentersGroup[:, :, np.newaxis], 24014, 2) - transportCost) - minIncome), 0)) + minIncome)
    
        #Average income earned by a worker
        averageIncome[:,:, index] = np.nansum(ODflows[:,:,:,index] * (np.repeat(incomeCentersGroup[:, :, np.newaxis], 24014, 2)), 0)

        incomeNetOfCommuting = incomeNetOfCommuting / annualToHourly
        averageIncome = averageIncome / annualToHourly 

        self.distanceOutput = distanceOutput
        self.monetaryCost = trans_monetaryCost
        self.timeCost = costTime
        self.incomeNetOfCommuting = incomeNetOfCommuting
        self.modalShares = modalShares
        self.ODflows = ODflows
        self.averageIncome = averageIncome
        self.yearTransport = yearTraffic + param["baseline_year"]
        self.timeOutput = timeOutput 
