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
                      marker='.')
    plt.colorbar(map)
    plt.show()

def sp_map(x, data_courbe_SP_X, data_courbe_SP_Y):
    map = plt.scatter(data_courbe_SP_X, 
                      data_courbe_SP_Y, 
                      s=None,
                      c=x,
                      marker='.')
    plt.colorbar(map)
    plt.show()
    
# %% Plot average income 2011

grid_map(data_courbe_income_grid, grid)
sp_map(data_courbe_income_SP, data_courbe_SP_X, data_courbe_SP_Y)

# %% Plot housing 2011

grid_map(data_courbe_informal_backyard_grid, grid)
grid_map(data_courbe_informal_settlement_grid, grid)
grid_map(data_courbe_formal_grid, grid)

sp_map(data_courbe_SP_informal_backyard, data_courbe_SP_X, data_courbe_SP_Y)
sp_map(data_courbe_SP_informal_settlement, data_courbe_SP_X, data_courbe_SP_Y)
sp_map(data_courbe_SP_total_dwellings, data_courbe_SP_X, data_courbe_SP_Y)

# %% Plot housing 2001

grid_map(data_courbe_formal_2001_grid, grid)
grid_map(data_courbe_backyard_2001_grid, grid)
grid_map(data_courbe_informal_2001_grid, grid)

sp_map(data_courbe_SP_2001_backyard, data_courbe_SP_2001_X, data_courbe_SP_2001_Y)
sp_map(data_courbe_SP_2001_formal, data_courbe_SP_2001_X, data_courbe_SP_2001_Y)
sp_map(data_courbe_SP_2001_informal, data_courbe_SP_2001_X, data_courbe_SP_2001_Y)

# %% Plot median sale price for 2001 and 2011

sp_map(data_courbe_SP_price[0], data_courbe_SP_X, data_courbe_SP_Y)
sp_map(data_courbe_SP_price[1], data_courbe_SP_X, data_courbe_SP_Y)

# %% Plot housing density and dwelling sizes

grid_map(data_courbe_DENS_HFA_formal_grid, grid) #Nombre de m2 de logements formels par km2
grid_map(data_courbe_DENS_HFA_informal_grid, grid) #Nombre de m2 de logements informels par km2
grid_map(data_courbe_DENS_DU_grid, grid) #Nombre de logements formets par km2
grid_map(data_courbe_DU_Size_grid, grid) #Taille moyenne des logements formels

# %% Plot floor factor, share urbanised and dwelling sizes

sp_map(data_courbe_SP_floor_factor, data_courbe_SP_X, data_courbe_SP_Y)
sp_map(data_courbe_SP_share_urbanised, data_courbe_SP_X, data_courbe_SP_Y)
sp_map(data_courbe_SP_dwelling_size, data_courbe_SP_X, data_courbe_SP_Y)

# %% Plot subsidized housing

grid_map(data_courbe_GV_count_RDP, grid)
grid_map(data_courbe_GV_area_RDP, grid)

# %% Plot employment centers

plt.scatter(grid.coord_horiz, grid.coord_vert, s=None, marker='.', c = 'grey')
plt.scatter(poly.Jx, poly.Jy, c= 'red')
label = ['Epping', 'Claremont', 'Bellville', 'CBD', 'Table View']
indexX = np.unique(poly.Jx, return_index=True)[1]
coordX = [poly.Jx[index] for index in sorted(indexX)]
indexY = np.unique(poly.Jy, return_index=True)[1]
coordY = [poly.Jy[index] for index in sorted(indexY)]
for i, txt in enumerate(label):
    plt.annotate(txt, (coordX[i], coordY[i]))

grid_map(data_courbe.income_grid, grid)
plt.scatter(Xou / 1000, You / 1000)

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

