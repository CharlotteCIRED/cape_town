# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 11:27:09 2020

@author: Charlotte Liotta
"""

#### GENERAL STATS ON FLOODS

name = "17092020_general_stats_flood"
os.mkdir('C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/' + name)

# %% Floods data

floods = ['FD_5yr', 'FD_10yr', 'FD_20yr', 'FD_50yr', 'FD_75yr', 'FD_100yr', 'FD_200yr', 'FD_250yr', 'FD_500yr', 'FD_1000yr']
path_data = "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/"

data_5y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_5yr.xlsx"))
data_20y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_20yr.xlsx"))
data_50y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_50yr.xlsx"))
data_100y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_100yr.xlsx"))
data_1000y = np.squeeze(pd.read_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_1000yr.xlsx"))


# %% Compute general stats on floods

stats = compute_general_stats(floods, path_data)
stats.to_excel("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/descriptive_statistics.xlsx')

# %% Graph on flood_prone areas and flood depth

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

plt.savefig("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/flood_area_and_depth.png')
plt.close()

# %% Graph with the different types of floods

plt.subplot(2, 2, 1) # 1 ligne, 2 colonnes, sous-figure 1
plt.scatter(grid.horiz_coord, 
            grid.vert_coord, 
            s=None,
            c=data_5y.flood_depth,
            cmap = 'Reds',
            marker='.')
plt.colorbar(map)
plt.axis('off')
plt.title('5 years')
plt.clim(0, 1.5)

plt.subplot(2, 2, 2) # 1 ligne, 2 colonnes, sous-figure 1
plt.scatter(grid.horiz_coord, 
            grid.vert_coord, 
            s=None,
            c=data_20y.flood_depth,
            cmap = 'Reds',
            marker='.')
plt.colorbar(map)
plt.title('20 years')
plt.axis('off')
plt.clim(0, 1.5)

plt.subplot(2, 2, 3) # 1 ligne, 2 colonnes, sous-figure 1
plt.scatter(grid.horiz_coord, 
            grid.vert_coord, 
            s=None,
            c=data_50y.flood_depth,
            cmap = 'Reds',
            marker='.')
plt.colorbar(map)
plt.title('50 years')
plt.axis('off')
plt.clim(0, 1.5)

plt.subplot(2, 2, 4) # 1 ligne, 2 colonnes, sous-figure 1
plt.scatter(grid.horiz_coord, 
            grid.vert_coord, 
            s=None,
            c=data_100y.flood_depth,
            cmap = 'Reds',
            marker='.')
plt.colorbar(map)
plt.title('100 years')
plt.axis('off')
plt.clim(0, 1.5)

plt.savefig("C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/" + name + '/flood_depth.png')
plt.close()


def compute_general_stats(floods, path_data):
    stats = pd.DataFrame(columns = ['flood', 'flood_prone_area', 'average_flood_depth'])
    for flood in floods:
        type_flood = copy.deepcopy(flood)
        flood = np.squeeze(pd.read_excel(path_data + flood + ".xlsx"))
        stats = stats.append({'flood': type_flood, 'flood_prone_area': sum(flood['prop_flood_prone']) * 0.25, 'average_flood_depth': sum(flood['flood_depth'] * flood['prop_flood_prone'] / sum(flood['prop_flood_prone']))}, ignore_index = True)   
    return stats