# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:41:16 2020

@author: Charlotte Liotta
"""

#Pers per income class
grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP_intersect.csv', sep = ';')  
income_class_grid = np.zeros((len(grid.dist), 4))  
for index in range(0, len(grid.dist)): 
    intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])
    for i in range(0, len(intersect)): 
        if len(households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]]) != 0:  
            income_class_grid[index] = income_class_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]] / sum(grid_intersect.Area[grid_intersect.SP_CODE == intersect[i]])


### 1. Claus' database

#flood_plain = pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_50yr_and_100yr_grid.xlsx')
#flood_plain.area = flood_plain.area / 250000
#grid_flood = pd.DataFrame(list(range(1, 24015)), columns = {'ID'})
#grid_flood['flood_area'] = 0
#for i in range(1, 24015):
#    if i in flood_plain['ID'].values:
#        grid_flood.flood_area[grid_flood.ID == i] = np.squeeze(flood_plain.area[flood_plain.ID == i])

#grid_flood['safe'] = 1 - grid_flood['flood_area']
#flood_prone_formal = np.sum(grid_flood['flood_area'] * households_data.formal_grid_2011)
#safe_formal = np.sum(grid_flood['safe'] * households_data.formal_grid_2011)
#flood_prone_subsidized = np.sum(grid_flood['flood_area'] * households_data.GV_count_RDP)
#safe_subsidized = np.sum(grid_flood['safe'] * households_data.GV_count_RDP)
#flood_prone_informal = np.sum(grid_flood['flood_area'] * households_data.informal_settlement_grid_2011)
#safe_informal = np.sum(grid_flood['safe'] * households_data.informal_settlement_grid_2011)
#flood_prone_backyard = np.sum(grid_flood['flood_area'] * households_data.informal_backyard_grid_2011)
#safe_backyard = np.sum(grid_flood['safe'] * households_data.informal_backyard_grid_2011)

#income_class_flood = np.sum((np.transpose(np.matlib.repmat(grid_flood['flood_area'], 4, 1)) * income_class_grid), 0)

#### 2. FATHOM database

df = np.squeeze(pd.read_excel('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FU_20yr.xlsx'))

sum(df.prop_flood_prone) * 0.25
sum(df.flood_depth * df.prop_flood_prone / sum(df.prop_flood_prone))

flood_prone_formal = np.sum(df['prop_flood_prone'] * households_data.formal_grid_2011) / sum(households_data.formal_grid_2011)
flood_prone_subsidized = np.sum(df['prop_flood_prone'] * households_data.GV_count_RDP) / sum(households_data.GV_count_RDP)
flood_prone_informal = np.sum(df['prop_flood_prone'] * households_data.informal_settlement_grid_2011) / sum(households_data.informal_settlement_grid_2011)
flood_prone_backyard = np.sum(df['prop_flood_prone'] * households_data.informal_backyard_grid_2011) / sum(households_data.informal_backyard_grid_2011)

income_class_flood = np.sum((np.transpose(np.matlib.repmat(df['prop_flood_prone'], 4, 1)) * income_class_grid), 0) / np.sum(income_class_grid, 0)