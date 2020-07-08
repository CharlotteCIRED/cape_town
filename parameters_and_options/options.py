# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:49:51 2020

@author: Charlotte Liotta
"""

def choice_options():
    
    option = {"init" : 1} #vaut 1 si on recalcule l'état initial, et 0 si on le charge
    option["Resol_stat"] = 0 #vaut 1 si on calcule juste une série d'équilibres
    option["Euler"] = 1
    option["iter_calc"] = 100 #nombre d'itérations entre 2 pas de temps si on utilise la méthode d'Euler pour la résolution de l'équation différentielle   
    option["Evolution"] = 0 #paramètres de la simulation
    option["OPTION_STATIQUE"] = 1 #Vaut 1 si on exécute le statique (dans acclimat, ça doit être 0)   
    option["Solveur"] = 'Lite' #type de solveur
    option["iter_calc_lite"] = 1 #type de solveur
    option["POPULATION_BASSE"] = 0 #Vaut 1 si on lance le scénario INSEE population basse    
    option["MODE_FIXE"] = 0 #Vaut 1 si on fixe le trafic à son niveau de 2010 (coûts)
    option["LOGIT"] = 1 #Vaut 1 si on a une relation logistique qui guide le choix du trans.mode de transport (continuum = pas tout ou rien dans une cellule)
    option["GRILLE_NEW"] = 0
    option["PETITE_GRILLE"] = 0  
    option["regression_amenity"] = 1
    option["future_construction_RDP"] = 1   
    option["double_storey_shacks"] = 0
    option["polycentric"] = 1
    
    print('*** Options imported succesfully ***')
    
    return option