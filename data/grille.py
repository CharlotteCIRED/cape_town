# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:15:48 2020

@author: Charlotte Liotta
"""

import pandas as pd

class SimulGrid:
    """ Class definig a grid with:
        - ID
        - coord_horiz
        - coord_vert
        - xcentre, ycentre
        - dist """
    
    def __init__(self, ID=0, horiz_coord=0, vert_coord=0, x_center=0, y_center=0, dist=0):
        
        self.ID = 0
        self.horiz_coord = 0
        self.vert_coord = 0
        self.x_center = 0
        self.y_center = 0
        self.dist = 0
        
    def create_grid(self):
        """Create a n*n grid with center in 0"""
        
        grid = pd.read_csv('./2. Data/Basile data/grid_NEDUM_Cape_Town_500.csv', sep = ';')

        #Coordonnées réelle en South African CRS : CAPE_NO_19 en km.
        horiz_coord = grid.X/1000
        vert_coord = grid.Y/1000
        x_center = -53267.944572790904/1000
        y_center = -3754855.1309322729/1000

        #Distance de chacun des noeuds au centre-ville
        dist = (((horiz_coord - x_center) ** 2) + ((vert_coord - y_center) ** 2)) ** 0.5
        
        self.ID = grid.ID
        self.horiz_coord = horiz_coord
        self.vert_coord = vert_coord
        self.x_center = x_center
        self.y_center = y_center
        self.dist = dist

    def __repr__(self):
        return "Grid:\n  coord_X: {}\n  coord_Y: {}\n  distance_centre: {}\n  area: {}".format(
                self.ID, self.horiz_coord, self.vert_coord, self.x_center, self.y_center, self.dist)