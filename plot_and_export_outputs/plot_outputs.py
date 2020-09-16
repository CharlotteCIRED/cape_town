# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:08:43 2020

@author: Charlotte Liotta
"""

def grid_map(x, grid):
    ''' Plot the results on the grid '''
    
    map = plt.scatter(grid.horiz_coord, 
                      grid.vert_coord, 
                      s=None,
                      c=x,
                      cmap = 'Reds',
                      marker='.')
    #plt.colorbar(map)
    plt.show()

def sp_map(x, data_courbe_SP_X, data_courbe_SP_Y):
    map = plt.scatter(data_courbe_SP_X, 
                      data_courbe_SP_Y, 
                      s=None,
                      c=x,
                      cmap = 'Reds',
                      marker='.')
    plt.colorbar(map)
    plt.show()
    
# %% Plot average income 2011

grid_map(households_data.income_grid_2011, grid)

# %% Plot housing 2011

#data
grid_map(households_data.formal_grid_2011, grid)
grid_map(households_data.informal_grid_2011, grid)
grid_map(households_data.backyard_grid_2001, grid)
grid_map(households_data.GV_count_RDP, grid)
#simul

grid_map(initialState_householdsHousingType[0], grid)
grid_map(initialState_householdsHousingType[1], grid)
grid_map(initialState_householdsHousingType[2], grid)
grid_map(initialState_householdsHousingType[3], grid)

# %% Plot median sale price for 2001 and 2011

sp_map(households_data.sale_price_SP[1], households_data.X_SP_2011, households_data.Y_SP_2011)

# %% Plot housing density and dwelling sizes

sp_map(households_data.spDwellingSize, households_data.X_SP_2011, households_data.Y_SP_2011)
grid_map(households_data.gridFormalDensityHFA, grid)

grid_map(initialState_dwellingSize[0], grid)
grid_map(initialState_dwellingSize[1], grid)
grid_map(initialState_dwellingSize[2], grid)
grid_map(initialState_dwellingSize[3], grid)


# %% Plot employment centers

plt.scatter(grid.horiz_coord, grid.vert_coord, s=None, marker='.', c = 'grey')
plt.scatter(job.xCenter, job.yCenter, c= 'red')

#Amenities
grid_map(land.amenities, grid)

#Transport
grid_map(trans.distanceOutput[0, :, 0], grid)
grid_map(trans.monetaryCost[0, :, 3, 3], grid)
grid_map(trans.timeCost[0, :, 3], grid)
grid_map(trans.incomeNetOfCommuting[2, :, 1], grid)
grid_map(trans.ODflows[0, :, 3, 3], grid)
grid_map(trans.timeOutput[0, :, 0], grid)


grid_map(trans.averageIncome[1, :, 0], grid)


#Simulation

grid_map(initialState_householdsCenter[0], grid)
grid_map(initialState_householdsCenter[1], grid)
grid_map(initialState_householdsCenter[2], grid)
grid_map(initialState_householdsCenter[3], grid)

grid_map(initialState_rent[0], grid)
grid_map(initialState_rent[1], grid)
grid_map(initialState_rent[2], grid)
grid_map(initialState_rent[3], grid)




#SOLVER
#plt.scatter(range(0,index_t), erreur[0:index_t, 0])
#plt.scatter(range(0,index_t), erreur[0:index_t, 1])
#plt.scatter(range(0,index_t), erreur[0:index_t, 2]) #pb ?
#plt.scatter(range(0,index_t), erreur[0:index_t, 3]) #pb ?
#plt.scatter(range(0,index_t), erreur[0:index_t, 4])
#plt.scatter(range(0,index_t), erreur[0:index_t, 5]) #pb ?
#plt.scatter(range(0,index_t), erreur[0:index_t, 6]) #pn ?
#plt.scatter(range(0,index_t), erreur[0:index_t, 7]) 
#plt.scatter(range(0,index_t), erreur[0:index_t, 8])
#plt.scatter(range(0,index_t), erreur[0:index_t, 9]) #pb
#plt.scatter(range(0,index_t), erreur[0:index_t, 10])
#plt.scatter(range(0,index_t), erreur[0:index_t, 11]) #pb
#plt.scatter(range(0,index_t), erreur[0:index_t, 12])
#plt.scatter(range(0,index_t), erreur[0:index_t, 13])
#plt.scatter(range(0,index_t), erreur[0:index_t, 14]) #PB
#plt.scatter(range(0,index_t), erreur[0:index_t, 15]) #pb
#plt.scatter(range(0,index_t), erreur[0:index_t, 16]) 
#plt.scatter(range(0,index_t), erreur[0:index_t, 17]) #pb

#plt.scatter(range(0,index_t), val_moy[0:index_t])

