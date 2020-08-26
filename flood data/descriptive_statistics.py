# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:41:16 2020

@author: Charlotte Liotta
"""

floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr', 
          'FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
### FLOODS
stats = pd.DataFrame(columns = ['flood', 'flood_prone_area', 'average_flood_depth'])
for flood in floods:
    type_flood = copy.deepcopy(flood)
    flood = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/" + flood + ".xlsx"))
    stats = stats.append({'flood': type_flood, 'flood_prone_area': sum(flood['prop_flood_prone']) * 0.25, 'average_flood_depth': sum(flood['flood_depth'] * flood['prop_flood_prone'] / sum(flood['prop_flood_prone']))}, ignore_index = True)   
stats.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/descriptive_statistics.xlsx")

### FLOODS PER HOUSING TYPES
stats_per_housing_type = pd.DataFrame(columns = ['flood',
                                                 'fraction_formal_in_flood_prone_area', 'fraction_subsidized_in_flood_prone_area', 'fraction_informal_in_flood_prone_area', 'fraction_backyard_in_flood_prone_area',
                                                 'flood_depth_formal', 'flood_depth_subsidized', 'flood_depth_informal', 'flood_depth_backyard'])
for flood in floods:
    type_flood = copy.deepcopy(flood)
    flood = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/" + flood + ".xlsx"))
    stats_per_housing_type = stats_per_housing_type.append({'flood': type_flood, 
                                                            'fraction_formal_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * households_data.formal_grid_2011) / sum(households_data.formal_grid_2011), 
                                                            'fraction_subsidized_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * households_data.GV_count_RDP) / sum(households_data.GV_count_RDP),
                                                            'fraction_informal_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * households_data.informal_grid_2011) / sum(households_data.informal_grid_2011), 
                                                            'fraction_backyard_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * households_data.backyard_grid_2011) / sum(households_data.backyard_grid_2011),
                                                            'flood_depth_formal': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * households_data.formal_grid_2011)  / sum(flood['prop_flood_prone'] * households_data.formal_grid_2011)),
                                                            'flood_depth_subsidized': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * households_data.GV_count_RDP)  / sum(flood['prop_flood_prone'] * households_data.GV_count_RDP)),
                                                            'flood_depth_informal': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * households_data.informal_grid_2011)  / sum(flood['prop_flood_prone'] * households_data.informal_grid_2011)),
                                                            'flood_depth_backyard': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * households_data.backyard_grid_2011)  / sum(flood['prop_flood_prone'] * households_data.backyard_grid_2011))}, ignore_index = True)   
stats_per_housing_type.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/descriptive_statistics_per_housing_type.xlsx")

### FLOODS PER INCOME CLASS
grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP_intersect.csv', sep = ';')  
income_class_grid = np.zeros((len(grid.dist), 4))  
for index in range(0, len(grid.dist)): 
    intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])
    for i in range(0, len(intersect)): 
        if len(households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]]) != 0:  
            income_class_grid[index] = income_class_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]] / sum(grid_intersect.Area[grid_intersect.SP_CODE == intersect[i]])
#Alternativement: people_income_group

stats_per_income_class = pd.DataFrame(columns = ['flood',
                                                 'fraction_class1_in_flood_prone_area', 'fraction_class2_in_flood_prone_area', 'fraction_class3_in_flood_prone_area', 'fraction_class4_in_flood_prone_area',
                                                 'flood_depth_class1', 'flood_depth_class2', 'flood_depth_class3', 'flood_depth_class4'])
for flood in floods:
    type_flood = copy.deepcopy(flood)
    flood = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/" + flood + ".xlsx"))
    stats_per_income_class = stats_per_income_class.append({'flood': type_flood, 
                                                            'fraction_class1_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,0]) / sum(income_class_grid[:,0]), 
                                                            'fraction_class2_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,1]) / sum(income_class_grid[:,1]),
                                                            'fraction_class3_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,2]) / sum(income_class_grid[:,2]), 
                                                            'fraction_class4_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,3]) / sum(income_class_grid[:,3]),
                                                            'flood_depth_class1': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,0])  / sum(flood['prop_flood_prone'] * income_class_grid[:,0])),
                                                            'flood_depth_class2': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,1])  / sum(flood['prop_flood_prone'] * income_class_grid[:,1])),
                                                            'flood_depth_class3': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,2])  / sum(flood['prop_flood_prone'] * income_class_grid[:,2])),
                                                            'flood_depth_class4': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,3])  / sum(flood['prop_flood_prone'] * income_class_grid[:,3]))}, ignore_index = True)   
stats_per_income_class.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/descriptive_statistics_per_income_class.xlsx")

### DAMAGES FOR FORMAL PRIVATE AND SUBSIDIZED

structural_damages_small_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0479, 0.1312, 0.1795, 0.3591, 1, 1])
structural_damages_medium_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.083, 0.2273, 0.3083, 0.62, 1, 1])
structural_damages_large_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0799, 0.2198, 0.2997, 0.5994, 1, 1])
content_damages = interp1d([0, 0.1, 0.3, 0.6, 1.2, 1.5, 2.4, 10], [0, 0.06, 0.15, 0.35, 0.77, 0.95, 1, 1])

informal_structure_cost = 4000
backyard_structure_cost = 4000
subsidized_structure_cost = 0
formal_structure_cost = ((param["coeff_A"] * param["coeff_b"] * etat_initial_rent1[0] /  (param["interest_rate"] + param["depreciation_rate"])) ** (1/param["coeff_a"])) * households_data.DU_Size_grid

content_cost = 7395

damages = pd.DataFrame(columns = ['flood',
                                  'informal_structure_damages',
                                  'backyard_structure_damages',
                                  'backyard_content_damages',
                                  'informal_content_damages',
                                  'backyard_content_damages',
                                  'subsidized_content_damages'])
for flood in floods:
    type_flood = copy.deepcopy(flood)
    flood = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/" + flood + ".xlsx"))
    informal_structure_damages = sum(households_data.informal_grid_2011 * flood["prop_flood_prone"] * informal_structure_cost * structural_damages_medium_houses(flood['flood_depth']))
    backyard_structure_damages = sum(households_data.backyard_grid_2011 * flood["prop_flood_prone"] * backyard_structure_cost * structural_damages_medium_houses(flood['flood_depth']))
    informal_content_damages = sum(households_data.informal_grid_2011 * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
    backyard_content_damages = sum(households_data.backyard_grid_2011 * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
    subsidized_content_damages = sum(households_data.GV_count_RDP * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
    damages = damages.append({'flood': type_flood,
                              'informal_structure_damages': informal_structure_damages,
                              'backyard_structure_damages': backyard_structure_damages,
                              'informal_content_damages': informal_content_damages,
                              'backyard_content_damages': backyard_content_damages,
                              'subsidized_content_damages': subsidized_content_damages}, ignore_index = True)
damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/damages.xlsx")
