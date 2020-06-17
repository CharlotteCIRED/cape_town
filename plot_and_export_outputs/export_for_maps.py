# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:58:38 2020

@author: Charlotte Liotta
"""

function export_for_maps_initial_benchmark(grille, land, etat_initial, stat_initial, data_courbe, name)
% Function that creates the export as csv that can then be imported in R to
% display simulation maps

table_export_simul = table(grille.ID, ...
                     sum(stat_initial.people_income_group .* data_courbe.household_size_group')', ...
                     sum(etat_initial.people_housing_type(:,:))', ...
                     (etat_initial.people_housing_type(1,:) + etat_initial.people_housing_type(4,:))', ...
                     etat_initial.people_housing_type(2,:)', ...
                     etat_initial.people_housing_type(3,:)', ...
                     stat_initial.revenu_moy', ...
                     etat_initial.hous1(1,:)', ...
                     sum(land.coeff_land)');
table_export_simul.Properties.VariableNames = {'ID_grid' ...
                                         'total_people' ...
                                         'total_households' ...
                                         'total_formal' ...
                                         'total_backyard' ...
                                         'total_settlement' ...
                                         'income' ...
                                         'dwelling_size_formal' ...
                                         'coeff_land_total'};
                                     
                                     
table_export_data = table(grille.ID, ...
                          data_courbe.nb_rich_grid' + data_courbe.nb_poor_grid',...
                          data_courbe.formal_grid',...
                          data_courbe.informal_backyard_grid',...
                          data_courbe.informal_settlement_grid',...
                          data_courbe.income_grid',...
                          data_courbe.DU_Size_grid',...
                          sum(land.coeff_land)');
table_export_data.Properties.VariableNames = {'ID_grid' ...
                                         'total_households' ...
                                         'total_formal' ...
                                         'total_backyard' ...
                                         'total_settlement' ...
                                         'income' ...
                                         'dwelling_size_formal' ...
                                         'coeff_land_total'};

writetable(table_export_simul, strcat('./', name, '/maps/initial_state_simul.csv'))
writetable(table_export_data, strcat('./', name, '/maps/initial_state_data_benchmark.csv'))


end
© 2020 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About