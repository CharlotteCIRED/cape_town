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
    
    def __init__(self, ID=0, coord_horiz=0, coord_vert=0, xcentre=0, ycentre=0, dist=0):
        
        self.ID = 0
        self.coord_horiz = 0
        self.coord_vert = 0
        self.xcentre = 0
        self.ycentre = 0
        self.dist = 0
        
    def create_grid(self):
        """Create a n*n grid with center in 0"""
        
        grid = pd.read_csv('./2. Data/Basile data/grid_NEDUM_Cape_Town_500.csv', sep = ';')

        #Coordonnées réelle en South African CRS : CAPE_NO_19 en km.
        coord_horiz = grid.X/1000
        coord_vert = grid.Y/1000
        xcentre = -53267.944572790904/1000
        ycentre = -3754855.1309322729/1000

        #Distance de chacun des noeuds au centre-ville
        dist = (((coord_horiz - xcentre) ** 2) + ((coord_vert - ycentre) ** 2)) ** 0.5
        
        self.ID = grid.ID
        self.coord_horiz = coord_horiz
        self.coord_vert = coord_vert
        self.xcentre = xcentre
        self.ycentre = ycentre
        self.dist = dist

    def __repr__(self):
        return "Grid:\n  coord_X: {}\n  coord_Y: {}\n  distance_centre: {}\n  area: {}".format(
                self.ID, self.coord_horiz, self.coord_vert, self.xcentre, self.ycentre, self.dist)