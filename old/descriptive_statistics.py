# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:41:16 2020

@author: Charlotte Liotta
"""

# %% Floods data

#floods = ['P_5yr', 'P_10yr', 'P_20yr', 'P_50yr', 'P_75yr', 'P_100yr', 'P_200yr', 'P_250yr', 'P_500yr', 'P_1000yr', 
 #         'FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']

path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"

# %% Compute general stats on floods

stats = compute_general_stats(floods, path_data)
stats.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/descriptive_statistics.xlsx")


fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.
width = 0.4

stats.flood_prone_area.plot(kind='bar', color='red', ax=ax, width=width, position=1)
stats.average_flood_depth.plot(kind='bar', color='blue', ax=ax2, width=width, position=0)

ax.set_ylabel('Flood-prone area (km2)')
ax2.set_ylabel('Average flood depth (m)')
ax.legend(bbox_to_anchor=(0.4, 1))
ax2.legend(bbox_to_anchor=(0.4525, 0.9))
ax.set_xticks(np.arange(len(stats.flood)))
ax.set_xticklabels(["5 years", "10 years", "20 years", "50 years", "75 years", "100 years", "200 years", "500 years", "1000 years"], rotation = 45)

plt.show()

# %% Compute stats on floods per housing type

#etat_initial_people_housing_type
stats_per_housing_type = compute_stats_per_housing_type(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011)
stats_per_housing_type.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/descriptive_statistics_per_housing_type.xlsx")

# %% Compute stats on floods per income classes

grid_intersect = pd.read_csv('./2. Data/Basile data/grid_SP_intersect.csv', sep = ';')  
income_class_grid = np.zeros((len(grid.dist), 4))  
for index in range(0, len(grid.dist)): 
    intersect = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grid.ID[index]])
    for i in range(0, len(intersect)): 
        if len(households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]]) != 0:  
            income_class_grid[index] = income_class_grid[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grid.ID[index]) & (grid_intersect.SP_CODE == intersect[i])]) * households_data.income_n_class_SP_2011[households_data.Code_SP_2011 == intersect[i]] / sum(grid_intersect.Area[grid_intersect.SP_CODE == intersect[i]])
#Alternativement: people_income_group

stats_per_income_class = compute_stats_per_income_class(floods, path_data, income_class_grid)
stats_per_income_class.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/descriptive_statistics_per_income_class.xlsx")

# %% Damages

#Hypotheses
structural_damages_small_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0479, 0.1312, 0.1795, 0.3591, 1, 1])
structural_damages_medium_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.083, 0.2273, 0.3083, 0.62, 1, 1])
structural_damages_large_houses = interp1d([0, 0.1, 0.6, 1.2, 2.4, 6, 10], [0, 0.0799, 0.2198, 0.2997, 0.5994, 1, 1])
content_damages = interp1d([0, 0.1, 0.3, 0.6, 1.2, 1.5, 2.4, 10], [0, 0.06, 0.15, 0.35, 0.77, 0.95, 1, 1])

informal_structure_cost = 4000
backyard_structure_cost = 4000
subsidized_structure_cost = 0
# Between R 300 000 and R 700 000.

content_cost = 7395

#Case 1 : 2011 - data

count_formal = households_data.formal_grid_2011 - households_data.GV_count_RDP
count_formal[count_formal < 0] = 0

dwelling_size = SP_to_grid_2011_1(households_data.spDwellingSize, households_data.Code_SP_2011, grid)
formal_structure_cost = ((param["coeff_A"] * param["coeff_b"] * simul1_rent[0, 0, :] /  (param["interest_rate"] + param["depreciation_rate"])) ** (1/param["coeff_a"])) * dwelling_size

damages = compute_damages(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011, formal_structure_cost, 0, informal_structure_cost, backyard_structure_cost, content_cost, structural_damages_medium_houses, content_damages)
damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/damages_04092020_2011_data.xlsx")

#Case 2 : 2011 - simul1
formal_structure_cost = ((param["coeff_A"] * param["coeff_b"] * simul1_rent[0, 0, :] /  (param["interest_rate"] + param["depreciation_rate"])) ** (1/param["coeff_a"])) * simul1_dwellingSize[0, 0, :]
damages = compute_damages(floods, path_data, simul1_householdsHousingType[0, 0, :], simul1_householdsHousingType[0, 3, :], simul1_householdsHousingType[0, 2, :], simul1_householdsHousingType[0, 1, :], formal_structure_cost, 0, informal_structure_cost, backyard_structure_cost, content_cost, structural_damages_medium_houses, content_damages)
damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/damages_04092020_2011_simul1.xlsx")

#Case 3 : 2011 - simul2
formal_structure_cost = ((param["coeff_A"] * param["coeff_b"] * simul2_rent[0, 0, :] /  (param["interest_rate"] + param["depreciation_rate"])) ** (1/param["coeff_a"])) * simul2_dwellingSize[0, 0, :]
damages = compute_damages(floods, path_data, simul2_householdsHousingType[0, 0, :], simul2_householdsHousingType[0, 3, :], simul2_householdsHousingType[0, 2, :], simul2_householdsHousingType[0, 1, :], formal_structure_cost, 0, informal_structure_cost, backyard_structure_cost, content_cost, structural_damages_medium_houses, content_damages)
damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/damages_04092020_2011_simul2.xlsx")

#Case 4 : 2040 - simul1
formal_structure_cost = ((param["coeff_A"] * param["coeff_b"] * simul1_rent[29, 0, :] /  (param["interest_rate"] + param["depreciation_rate"])) ** (1/param["coeff_a"])) * simul1_dwellingSize[29, 0, :]

damages = compute_damages(floods, path_data, simul1_householdsHousingType[29, 0, :], simul1_householdsHousingType[29, 3, :], simul1_householdsHousingType[29, 2, :], simul1_householdsHousingType[29, 1, :], formal_structure_cost, 0, informal_structure_cost, backyard_structure_cost, content_cost, structural_damages_medium_houses, content_damages)
damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/damages_04092020_2040_simul1.xlsx")

#Case 5 : 2040 - simul2
formal_structure_cost = ((param["coeff_A"] * param["coeff_b"] * simul2_rent[28, 0, :] /  (param["interest_rate"] + param["depreciation_rate"])) ** (1/param["coeff_a"])) * simul2_dwellingSize[28, 0, :]
damages = compute_damages(floods, path_data, simul2_householdsHousingType[28, 0, :], simul2_householdsHousingType[28, 3, :], simul2_householdsHousingType[28, 2, :], simul2_householdsHousingType[28, 1, :], formal_structure_cost, 0, informal_structure_cost, backyard_structure_cost, content_cost, structural_damages_medium_houses, content_damages)
damages.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/damages_04092020_2040_simul2.xlsx")

### Enquête sur le fait que le modèle de Basile sous-estime les dégâts IS/IB
print(sum(count_formal))
print(sum(households_data.GV_count_RDP))
print(sum(households_data.informal_grid_2011))
print(sum(households_data.backyard_grid_2011))

np.sum(simul1_householdsHousingType[0, :, :], 1)

stats_data_2011 = compute_stats_per_housing_type(floods, path_data, count_formal, households_data.GV_count_RDP, households_data.informal_grid_2011, households_data.backyard_grid_2011)
stats_simul_2011 = compute_stats_per_housing_type(floods, path_data, simul1_householdsHousingType[0, 0, :], simul1_householdsHousingType[0, 3, :], simul1_householdsHousingType[0, 2, :], simul1_householdsHousingType[0, 1, :])

np.max(stats_data_2011, 0)[1:5] * np.array([sum(count_formal), sum(households_data.GV_count_RDP), sum(households_data.informal_grid_2011), sum(households_data.backyard_grid_2011)])
# %% Maps

data = pd.DataFrame([grid.horiz_coord, grid.vert_coord, data_5y.flood_depth])
data = np.transpose(data)
data_contour_5yr = pd.pivot(data, index='X', columns = "Y", values = 'flood_depth')

data = pd.DataFrame([grid.horiz_coord, grid.vert_coord, data_20y.flood_depth])
data = np.transpose(data)
data_contour_20yr = pd.pivot(data, index='X', columns = "Y", values = 'flood_depth')

data = pd.DataFrame([grid.horiz_coord, grid.vert_coord, data_100y.flood_depth])
data = np.transpose(data)
data_contour_100yr = pd.pivot(data, index='X', columns = "Y", values = 'flood_depth')

data = pd.DataFrame([grid.horiz_coord, grid.vert_coord, data_1000y.flood_depth])
data = np.transpose(data)
data_contour_1000yr = pd.pivot(data, index='X', columns = "Y", values = 'flood_depth')

plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
grid_map(formal_p1, grid)
plt.axis('off')
plt.title('Formal housing')
plt.clim(0, 1000)
plt.contour(data_contour.index, data_contour.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
plt.subplot(2, 2, 2)  # 1 ligne, 2 colonnes, sous-figure 2
grid_map(subsidized_p1, grid)
plt.axis('off')
plt.title('Subsidized housing')
plt.clim(0, 1000)
plt.contour(data_contour.index, data_contour.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
plt.subplot(2, 2, 3)  # 1 ligne, 2 colonnes, sous-figure 2
grid_map(informal_p1, grid)
plt.axis('off')
plt.title('Informal settlements')
plt.clim(0, 1000)
plt.contour(data_contour.index, data_contour.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
plt.subplot(2, 2, 4)  # 1 ligne, 2 colonnes, sous-figure 2
grid_map(backyard_p1, grid)
plt.axis('off')
plt.title('Backyarding')
plt.clim(0, 1000)
plt.contour(data_contour.index, data_contour.columns, np.transpose(data_contour_100yr), levels = [0.05], linewidths = 0.5)
plt.show()

data_5y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_5yr.xlsx"))
data_20y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_20yr.xlsx"))
data_50y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_50yr.xlsx"))
data_100y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_100yr.xlsx"))
data_1000y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_1000yr.xlsx"))

plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
grid_map(data_5y.flood_depth, grid)
plt.axis('off')
plt.title('5 years')
plt.clim(0, 1.5)

plt.subplot(2, 2, 2) # 1 ligne, 2 colonnes, sous-figure 1
grid_map(data_20y.flood_depth, grid)
plt.axis('off')
plt.title('20 years')
plt.clim(0, 1.5)

plt.subplot(2, 2, 3) # 1 ligne, 2 colonnes, sous-figure 1
grid_map(data_50y.flood_depth, grid)
plt.axis('off')
plt.title('50 years')
plt.clim(0, 1.5)

plt.subplot(2, 2, 4) # 1 ligne, 2 colonnes, sous-figure 1
grid_map(data_100y.flood_depth, grid)
plt.axis('off')
plt.title('100 years')
plt.clim(0, 1.5)


######## FUNCTIONS ###############

def compute_general_stats(floods, path_data):
    stats = pd.DataFrame(columns = ['flood', 'flood_prone_area', 'average_flood_depth'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        stats = stats.append({'flood': type_flood, 'flood_prone_area': sum(flood['prop_flood_prone']) * 0.25, 'average_flood_depth': sum(flood['flood_depth'] * flood['prop_flood_prone'] / sum(flood['prop_flood_prone']))}, ignore_index = True)   
    return stats
    
def compute_stats_per_housing_type(floods, path_data, nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard):
    stats_per_housing_type = pd.DataFrame(columns = ['flood',
                                                     'fraction_formal_in_flood_prone_area', 'fraction_subsidized_in_flood_prone_area', 'fraction_informal_in_flood_prone_area', 'fraction_backyard_in_flood_prone_area',
                                                     'flood_depth_formal', 'flood_depth_subsidized', 'flood_depth_informal', 'flood_depth_backyard'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        stats_per_housing_type = stats_per_housing_type.append({'flood': type_flood, 
                                                                'fraction_formal_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_formal) / sum(nb_households_formal), 
                                                                'fraction_subsidized_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_subsidized) / sum(nb_households_subsidized),
                                                                'fraction_informal_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_informal) / sum(nb_households_informal), 
                                                                'fraction_backyard_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * nb_households_backyard) / sum(nb_households_backyard),
                                                                'flood_depth_formal': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_formal)  / sum(flood['prop_flood_prone'] * nb_households_formal)),
                                                                'flood_depth_subsidized': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_subsidized)  / sum(flood['prop_flood_prone'] * nb_households_subsidized)),
                                                                'flood_depth_informal': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_informal)  / sum(flood['prop_flood_prone'] * nb_households_informal)),
                                                                'flood_depth_backyard': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * nb_households_backyard)  / sum(flood['prop_flood_prone'] * nb_households_backyard))}, ignore_index = True)   
    return stats_per_housing_type

def compute_stats_per_income_class(floods, path_data, income_class_grid):
    stats_per_income_class = pd.DataFrame(columns = ['flood',
                                                     'fraction_class1_in_flood_prone_area', 'fraction_class2_in_flood_prone_area', 'fraction_class3_in_flood_prone_area', 'fraction_class4_in_flood_prone_area',
                                                     'flood_depth_class1', 'flood_depth_class2', 'flood_depth_class3', 'flood_depth_class4'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        stats_per_income_class = stats_per_income_class.append({'flood': type_flood, 
                                                                'fraction_class1_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,0]) / sum(income_class_grid[:,0]), 
                                                                'fraction_class2_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,1]) / sum(income_class_grid[:,1]),
                                                                'fraction_class3_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,2]) / sum(income_class_grid[:,2]), 
                                                                'fraction_class4_in_flood_prone_area': np.sum(flood['prop_flood_prone'] * income_class_grid[:,3]) / sum(income_class_grid[:,3]),
                                                                'flood_depth_class1': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,0])  / sum(flood['prop_flood_prone'] * income_class_grid[:,0])),
                                                                'flood_depth_class2': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,1])  / sum(flood['prop_flood_prone'] * income_class_grid[:,1])),
                                                                'flood_depth_class3': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,2])  / sum(flood['prop_flood_prone'] * income_class_grid[:,2])),
                                                                'flood_depth_class4': sum(flood['flood_depth'] * (flood['prop_flood_prone'] * income_class_grid[:,3])  / sum(flood['prop_flood_prone'] * income_class_grid[:,3]))}, ignore_index = True)   
    return stats_per_income_class

def compute_damages(floods, path_data,
                    nb_households_formal, nb_households_subsidized, nb_households_informal, nb_households_backyard,
                    formal_structure_cost, subsidized_structure_cost, informal_structure_cost, backyard_structure_cost,
                    content_cost,
                    structural_damages, content_damages):
    
    damages = pd.DataFrame(columns = ['flood',
                                      'formal_structure_damages',
                                      'subsidized_structure_damages',
                                      'informal_structure_damages',
                                      'backyard_structure_damages',
                                      'formal_content_damages',
                                      'subsidized_content_damages',
                                      'informal_content_damages',
                                      'backyard_content_damages'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        formal_structure_damages = sum(nb_households_formal * flood["prop_flood_prone"] * formal_structure_cost * structural_damages(flood['flood_depth']))
        subsidized_structure_damages = sum(nb_households_subsidized * flood["prop_flood_prone"] * subsidized_structure_cost * structural_damages(flood['flood_depth']))
        informal_structure_damages = sum(nb_households_informal * flood["prop_flood_prone"] * informal_structure_cost * structural_damages(flood['flood_depth']))
        backyard_structure_damages = sum(nb_households_backyard * flood["prop_flood_prone"] * backyard_structure_cost * structural_damages(flood['flood_depth']))
        formal_content_damages = sum(nb_households_formal * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
        subsidized_content_damages = sum(nb_households_subsidized * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
        informal_content_damages = sum(nb_households_informal * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
        backyard_content_damages = sum(nb_households_backyard * flood["prop_flood_prone"] * content_cost * content_damages(flood['flood_depth']))
        damages = damages.append({'flood': type_flood,
                                  'formal_structure_damages': formal_structure_damages,
                                  'subsidized_structure_damages': subsidized_structure_damages,
                                  'informal_structure_damages': informal_structure_damages,
                                  'backyard_structure_damages': backyard_structure_damages,
                                  'formal_content_damages': formal_content_damages,
                                  'informal_content_damages': informal_content_damages,
                                  'backyard_content_damages': backyard_content_damages,
                                  'subsidized_content_damages': subsidized_content_damages}, ignore_index = True)
    
    return damages


### Graph annualized damages
damages = pd.DataFrame()
damages["Structure"] = [13561467.56,	0,	227094.41,	128988.39]
damages["Contents"] = [4205710.17,	271839.77,	657065.83,	325482.85]
fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
width = 0.4
damages.Structure.plot(kind='bar', color='red', ax=ax, width=width, position=1)
damages.Contents.plot(kind='bar', color='blue', ax=ax, width=width, position=0)
ax.set_ylabel('Annualized flood damages (R)')
ax.legend(bbox_to_anchor=(0.4, 1))
ax.set_xticks(np.arange(len(damages.Structure)))
ax.set_xticklabels(["Formal", "Subsidized", "Informal", "Backyarding"], rotation = 0)
plt.show()