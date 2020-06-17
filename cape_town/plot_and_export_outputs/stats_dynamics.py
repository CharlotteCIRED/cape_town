# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:17:47 2020

@author: Charlotte Liotta
"""

def compute_stat_finales_Cape_Town(macro,option,trans,land,grille,param,poly,simulation):

    #Inputs
    interest_rate_ici = np.zeros(len(simulation.T), 1)
    coeff_land_ici = np.zeros(len(simulation.T), size(land.coeff_land, 1), size(land.coeff_land, 2))
    for i in range(0, len(simulation.T)):
        interest_rate_ici[i] = interest_rate(macro, simulation.T[i])
        coeff_land_ici[i,:,:] = coeff_land_evol(land, option, param, simulation.T[i])

    stat_dynamics.real_people[:,:] = sum(simulation.people_housing_type,2)

    radius_CBD = grille.dist < 6
    rent_formal(:,:) = simulation.rent(:,1,:)
    housing_formal(:,:) = simulation.housing(:,1,:)
    people_formal(:,:) = simulation.people_housing_type(:,1,:) 
    people_RDP(:,:) = simulation.people_housing_type(:,4,:)  
    people_backyard(:,:) = simulation.people_housing_type(:,2,:)  
    people_settlement(:,:) = simulation.people_housing_type(:,3,:) 
    hous_formal(:,:) = simulation.hous(:,1,:)

    #Prices
    stat_dynamics.avg_rent_formal_CBD = np.nansum(people_formal(:,radius_CBD) * rent_formal(:,radius_CBD),2) / sum(people_formal(:,radius_CBD),2)
    stat_dynamics.avg_price_per_land_formal_CBD = np.nansum(people_formal(:,radius_CBD) * rent_formal(:,radius_CBD) * housing_formal(:,radius_CBD) / 1000000 /
                                                         np.matlib.repmat((param["depreciation_h"] + ppval(macro.spline_notaires, simulation.T) / 100), 1, size(people_formal(:,radius_CBD), 2))
                                                         , 2) / sum(people_formal(:,radius_CBD), 2)

    #Urban footprint
    stat_dynamics.urban_footprint = sum(stat_dynamics.real_people > 10, 2) #in sqm
    
    #Average prices
    quel_avg_price[:] = (sum(simulation.people_housing_type[1,:,:], 2) > 10)
    stat_dynamics.avg_price_per_house_formal = np.nansum(rent_formal(:, quel_avg_price) * hous_formal(:, quel_avg_price) / np.matlib.repmat(param["depreciation_h"] + ppval(macro.spline_notaires, simulation.T), 1, size(rent_formal(:,quel_avg_price),2)) / 100,2) /sum(quel_avg_price)
    stat_dynamics.weighted_avg_price_per_house_formal = np.nansum(people_formal * rent_formal * hous_formal / np.matlib.repmat(param["depreciation_h"] + ppval(macro.spline_notaires, simulation.T), 1, size(rent_formal,2)) / 100,2) / sum(people_formal, 2)

    #Affordability  
    stat_dynamics.avg_income = ppval(macro.spline_revenu, simulation.T) * ppval(macro.spline_inflation, simulation.T) / ppval(macro.spline_inflation, 2011 - param["year_begin"])
    stat_dynamics.affordability_index = stat_dynamics.avg_price_per_house_formal / stat_dynamics.avg_income
    stat_dynamics.weighted_affordability_index = stat_dynamics.weighted_avg_price_per_house_formal / stat_dynamics.avg_income

    #Total by housing types
    stat_dynamics.total_backyard(:) = sum(simulation.people_housing_type(:,2,:), 3) 
    stat_dynamics.total_settlement(:) = sum(simulation.people_housing_type(:,3,:), 3) 
    stat_dynamics.total_private(:) = sum(simulation.people_housing_type(:,1,:),3) 
    stat_dynamics.total_RDP(:) = sum(simulation.people_housing_type(:,4,:) ,3) 
    stat_dynamics.total_informal(:) = stat_dynamics.total_backyard + stat_dynamics.total_settlement
    stat_dynamics.total_formal_inc_RDP(:) = stat_dynamics.total_private + stat_dynamics.total_RDP
    stat_dynamics.total_population = stat_dynamics.total_formal_inc_RDP + stat_dynamics.total_informal

    #Average distance to the CBD
    stat_dynamics.avg_distance_to_CBD = sum(stat_dynamics.real_people(:,:) * np.matlib.repmat(grille.dist, len(simulation.T), 1), 2) / sum(stat_dynamics.real_people(:,:), 2)

    people_travaille_poor(:,:) = sum(simulation.people_travaille(:,1:2,:), 2)
    people_travaille_rich(:,:) = sum(simulation.people_travaille(:,3:4,:), 2)
    stat_dynamics.avg_distance_to_CBD_poor = sum(people_travaille_poor * np.matlib.repmat(grille.dist, len(simulation.T),1), 2) / sum(people_travaille_poor,2)
    stat_dynamics.avg_distance_to_CBD_rich = sum(people_travaille_rich * np.matlib.repmat(grille.dist, len(simulation.T),1), 2) / sum(people_travaille_rich,2)

    stat_dynamics.avg_distance_to_CBD_RDP = sum(people_RDP * np.matlib.repmat(grille.dist, len(simulation.T),1), 2) / sum(people_RDP,2)
    stat_dynamics.avg_distance_to_CBD_backyard = sum(people_backyard * repmat(grille.dist, len(simulation.T),1), 2) / sum(people_backyard,2)

    return stat_dynamics