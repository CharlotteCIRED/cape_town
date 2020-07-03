# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 10:08:43 2020

@author: Charlotte Liotta
"""

def grid_map(x, grid):
    ''' Plot the results on the grid '''
    
    map = plt.scatter(grid.coord_horiz, 
                      grid.coord_vert, 
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
