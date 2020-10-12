# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:58:37 2020

@author: Charlotte Liotta
"""

function [a_tracer]=trace_courbes_formal(grille, etat_initial, stat_initiales, macro, data_courbe, land, param, poly, option,t, name)


land.coeff_land = coeff_land_evol(land,option,param,t);

global slash

%% 1er cadran: densité de population


close all

%Data for year 2011
X_vrai = grille.dist; %(~isnan(data_courbe.hous_cible));
Y_vrai = (data_courbe.nb_rich_grid + data_courbe.nb_poor_grid) ./ (grille.delta_d)^2;

%Simulation
X_simul = grille.dist; %(~isnan(data_courbe.hous_cible));
Y_simul = sum(etat_initial.people_housing_type,1) ./ (grille.delta_d)^2;

trace_jolie_courbe(X_vrai,Y_vrai,X_simul,Y_simul, 'Distance to the center (km)', 'Housholds density (hh/km²)',1,1, 2);
ylim([0 2000])
xlim([0 40])

sauvegarde(strcat('.', slash, name, slash, 'density'))


%% deuxième cadran loyer seulement où on a les points de données

close all

data_courbe.X_loyer=data_courbe.X_price;
data_courbe.Y_loyer=data_courbe.Y_price;

price_simul = etat_initial.rent1(1,:).*etat_initial.housing1(1,:)./1000000./(param.depreciation_h + interest_rate(macro, t));
price_simul_coord = griddata(grille.coord_horiz, grille.coord_vert, double(price_simul), data_courbe.X_loyer, data_courbe.Y_loyer);

X_vrai1 = sqrt((data_courbe.X_loyer-grille.xcentre).^2+(data_courbe.Y_loyer-grille.ycentre).^2);
Y_vrai1 = data_courbe.SP_price(2,:);
X_simul1 = X_vrai1;
Y_simul1 = price_simul_coord;

trace_jolie_courbe(X_vrai1,Y_vrai1,X_simul1,Y_simul1, 'Distance to the center (km)', 'Price (R/m² of land)',1,1, 2);
ylim([0 18000])
xlim([0 40])

sauvegarde(strcat('.', slash, name, slash, 'price'))


%% troisieme cadran taille des logements

close all

X_vrai2 = grille.dist;
Y_vrai2 = data_courbe.DU_Size_grid;
X_simul2 = grille.dist; 
Y_simul2 = (etat_initial.hous1(1,:).*etat_initial.people_housing_type(1,:) + etat_initial.hous1(4,:).*etat_initial.people_housing_type(4,:))./(etat_initial.people_housing_type(1,:) + etat_initial.people_housing_type(4,:));

trace_jolie_courbe(X_vrai2',Y_vrai2',X_simul2',Y_simul2', 'Distance to the center (km)', 'Dwelling size (m2)', data_courbe.nb_rich_grid + data_courbe.nb_poor_grid, sum(stat_initiales.people1vrai,1), 3);
ylim([0 200])
xlim([0 40])

sauvegarde(strcat('.', slash, name, slash, 'dwelling_size'))


%% quatrieme cadran revenus moyens par subplace

close all

X_vrai2 = grille.dist;
Y_vrai2 = data_courbe.income_grid;
X_simul2 = grille.dist;
Y_simul2 = stat_initiales.revenu_moy;

trace_jolie_courbe(X_vrai2', Y_vrai2', X_simul2', Y_simul2', 'Distance to the center (km)', 'Household annual income (R)', data_courbe.nb_rich_grid + data_courbe.nb_poor_grid, sum(stat_initiales.people1vrai,1), 6);
xlim([0 40])

sauvegarde(strcat('.', slash, name, slash, 'household_annual_income'))

%% Modes of transportation

close all

Simul = stat_initiales.frac_mode.*100;
Data = [7.8 14.8 39.5+0.7 16 8] ./ (7.8+14.8+39.5+0.7+16+8).*100;
labels = {'Walking','Train','Private Vehicle','Minibus-Taxi','Bus'};
b = bar([Data; Simul]', 1);
b(1).FaceColor = [0.3 0.3 0.3];
b(2).FaceColor = [0.3 0.94 0.3];
lgd = legend(b, {'Data', 'Simulation'});
lgd.FontSize = 16;
lgd.FontName = 'Arial';
set(gca,'xticklabel',labels, 'FontSize', 14, 'FontName', 'Arial');
ylabel('% of travel trips to jobs', 'FontSize', 16, 'FontName','Arial');
set(gcf,'units','points','position',[50,50,700,320])

sauvegarde(strcat('.', slash, name, slash, 'modal_distribution'))



end


function sauvegarde(nom)
    format_d='-dpng ';
    format2='.png';
            
    print('-f1', format_d, [nom,format2]);
    saveas(1,[nom,'.fig']);

end