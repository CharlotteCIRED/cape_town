# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:58:38 2020

@author: Charlotte Liotta
"""

function [a_tracer] = trace_courbes_informal_abs(grille, etat_initial, stat_initiales, data_courbe, land, param, macro, poly, option,t, name)

global slash

% Graphs with the number of dwellings per housing types as a function of
% distance to the CBD

land.coeff_land = coeff_land_evol(land,option,param,t);

close all

formal = [sum(stat_initiales.people1vrai(1,:) + stat_initiales.people1vrai(4,:)); sum(data_courbe.formal_grid(data_courbe.limit_Cape_Town))];
formal_without_RDP = [sum(stat_initiales.people1vrai(1,:)); sum(data_courbe.formal_grid(data_courbe.limit_Cape_Town)) - sum(stat_initiales.people1vrai(4,:))];
backyard = [sum(stat_initiales.people1vrai(2,:)); sum(data_courbe.informal_backyard_grid(data_courbe.limit_Cape_Town))];
settlement = [sum(stat_initiales.people1vrai(3,:)); sum(data_courbe.informal_settlement_grid(data_courbe.limit_Cape_Town))];
total = [sum(sum(stat_initiales.people1vrai)); sum(data_courbe.formal_grid(data_courbe.limit_Cape_Town) + data_courbe.informal_backyard_grid(data_courbe.limit_Cape_Town) + data_courbe.informal_settlement_grid(data_courbe.limit_Cape_Town))];

T = [formal, backyard, settlement, total];
T_errors = {T(1,:)./T(2,:) - 1};
Tnew = [table(T); T_errors];

tab = uitable('Data',Tnew{:,:},'ColumnName',{'Formal', 'Backard', 'Settlement', 'Total'},...
    'RowName',{'Simulation'; 'Data'; 'Error'}, 'Units', 'Normalized', 'Position',[0.5, 0.5, 1, 1]);



%% 1er cadran backard

close all

%Year 2011
X_vrai = grille.dist;
Y_vrai = data_courbe.informal_backyard_grid.*data_courbe.limit_Cape_Town;
%Simulation
X_simul = grille.dist; %(~isnan(data_courbe.hous_cible));
Y_simul = etat_initial.people_housing_type(2,:);

prop_simul = round(backyard(1)./total(1).*1000)./1000 .* 100;
prop_data = round(backyard(2)./total(2).*1000)./1000 .* 100;

trace_jolie_courbe_abs(X_vrai,Y_vrai,X_simul,Y_simul, 'Distance to the center (km)', {'Sum of household'; 'backyarding'}, prop_data, prop_simul);
ylim([0 90000])
xlim([0 40])

sauvegarde(strcat('.', slash, name, slash,'total_informal_backyard'))


%% deuxième cadran settlement

close all

%Year 2011
X_vrai = grille.dist;
Y_vrai = data_courbe.informal_settlement_grid .* data_courbe.limit_Cape_Town;
%Simulation
X_simul = grille.dist; %(~isnan(data_courbe.hous_cible));
Y_simul = etat_initial.people_housing_type(3,:);

prop_simul = round(settlement(1)./total(1) .*1000)./1000 .* 100;
prop_data = round(settlement(2)./total(2) .*1000)./1000 .* 100;

trace_jolie_courbe_abs(X_vrai,Y_vrai,X_simul,Y_simul, 'Distance to the center (km)', {'Sum of household'; 'in informal settlements'}, prop_data, prop_simul);
ylim([0 90000])
xlim([0 40])

sauvegarde(strcat('.', slash, name, slash, 'total_informal_settlement'))



%% troisieme cadran densité de logement formel privately developed

close all

% Data for Year 2011
X_vrai = grille.dist;
Y_vrai = data_courbe.formal_grid.*data_courbe.limit_Cape_Town - etat_initial.people_housing_type(4,:);
% Simulation
X_simul = grille.dist; 
Y_simul = etat_initial.people_housing_type(1,:);

prop_simul = round(formal_without_RDP(1)./total(1).*1000)./1000 .* 100;
prop_data = round(formal_without_RDP(2)./total(2).*1000)./1000 .* 100;


trace_jolie_courbe_abs(X_vrai,Y_vrai,X_simul,Y_simul, 'Distance to the center (km)', {'Sum of hh in formal housing' ; 'privately developed'}, prop_data, prop_simul);
ylim([0 90000])
xlim([0 40])

sauvegarde(strcat('.',  slash, name, slash, 'total_formal'))


%% Graph of total population per housing type

close all

Simul = sum(etat_initial.people_housing_type, 2);
Simul = Simul ./ sum(Simul) .* 100;
Data = [sum(data_courbe.formal_grid .* data_courbe.limit_Cape_Town) - ppval(macro.spline_RDP, t); 
        sum(data_courbe.informal_backyard_grid .* data_courbe.limit_Cape_Town); 
        sum(data_courbe.informal_settlement_grid .* data_courbe.limit_Cape_Town); 
        ppval(macro.spline_RDP, t)];
Data = Data ./ sum(Data) .* 100;
labels = {'Formal private', 'Backyarding', 'Informal settlements', 'Formal RDP/BNG (estimated)'};
labels = cellfun(@(x) strrep(x,' ','\newline'), labels,'UniformOutput',false);
b = bar([Data, Simul], 1);
b(1).FaceColor = [0.3 0.3 0.3];
b(2).FaceColor = [0.3 0.94 0.3];
lgd = legend(b, {'Data (2011 Census)', 'Simulation'});
lgd.FontSize = 16;
lgd.FontName = 'Arial';
set(gca,'xticklabel',labels, 'FontSize', 14, 'FontName', 'Arial');
xlabel('Housing type','FontSize',16, 'FontName','Arial', 'FontWeight','bold');
ylabel('% of households', 'FontSize', 16, 'FontName','Arial', 'FontWeight','bold');
set(gcf,'units','points','position',[50,50,700,320])

sauvegarde(strcat('.', slash, name, slash, 'housing_type_bar'))


%% Table of housing type per income group

people_center_housing = sum(etat_initial.people1,3);
% Aggregation on the income groups
people_income_housing = zeros(param.multiple_class, 4);
for i = 1:param.multiple_class
    people_income_housing(i,:) = sum(people_center_housing(:,poly.class(1,:) == i),2)';
end

% Reorder
idx = [1 4 2 3];
people_income_housing = people_income_housing(:,idx);

% Create plot
b = bar(people_income_housing,...
    0.6,'stacked','LineWidth', 1);
b(1).FaceColor = [0.3 0.94 0.3];
b(2).FaceColor = [0.8 0.94 0.2];
b(3).FaceColor = [0.94 0.6 0.2];
b(4).FaceColor = [0.94 0.3 0.3];
lgd = legend('Formal', 'RDP/BNG', 'Backyarding', 'Informal settlements');
lgd.FontSize = 14;
set(gca,'TickDir','out','FontSize',14);
%set(gca,'xticklabel',labels, 'FontSize', 16, 'FontName', 'Arial');
xlabel('Income group','FontSize',16, 'FontName','Arial', 'FontWeight','bold');
ylabel('Number of households','FontSize',16, 'FontName','Arial', 'FontWeight','bold');
ylim([0 600000])

sauvegarde(strcat('.', slash, name, slash, 'housing_type_bar_simul'))



% Create plot for the data 
% Data from the Census
people_income_housing_data =  [124166, 57259, 320969;    
                               21294, 16247, 256227;
                               1400, 999, 124224;
                               1316, 451, 143916];
idx = [3,2,1];
people_income_housing_data = people_income_housing_data(:,idx);

people_income_housing_data_temp = zeros(4,4);
people_income_housing_data_temp(:,1:3) = people_income_housing_data;
people_income_housing_data_temp(1,4) = people_income_housing_data_temp(1,1);
people_income_housing_data_temp(1,1) = 0;
idx = [1 4 2 3];
people_income_housing_data_temp = people_income_housing_data_temp(:,idx);


b = bar(people_income_housing_data_temp,...
    0.6,'stacked','LineWidth',0.7);
b(1).FaceColor = [0.3 0.94 0.3];
b(2).FaceColor = [0.8 0.94 0.2];
b(3).FaceColor = [0.94 0.6 0.2];
b(4).FaceColor = [0.94 0.3 0.3];
lgd = legend('Formal', 'RDP/BNG', 'Backyarding', 'Informal settlements');
lgd.FontSize = 14;
set(gca,'TickDir','out','FontSize',14);
%set(gca,'xticklabel',labels, 'FontSize', 16, 'FontName', 'Arial');
xlabel('Income group','FontSize',16, 'FontName','Arial', 'FontWeight','bold');
ylabel('Number of households','FontSize',16, 'FontName','Arial', 'FontWeight','bold');
ylim([0 600000])

sauvegarde(strcat('.', slash, name, slash, 'housing_type_bar_data'))


end

function []=trace_jolie_courbe_abs(X_vrai,Y_vrai,X_simul,Y_simul, Xname, Yname, prop_data, prop_simul)

size_pour_font = 15;
size_pour_font_legend = 14;

step = 2;

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');

x_data = [0:step:max(X_vrai)];
x_simul = [0:step:max(X_simul)];
a2_data = zeros(1, length(x_data));
a2_simul = zeros(1, length(x_simul));

% Sum data
for i = 1 : length(x_data) - 1
    a2_data(i) = nansum(Y_vrai(X_vrai > x_data(i) & X_vrai < x_data(i + 1)));
end

% Sum model
for i = 1 : length(x_data) - 1
    a2_simul(i) = nansum(Y_simul(X_simul > x_simul(i) & X_simul < x_simul(i + 1)));
end

p1 = plot(x_data, a2_data, 'Linestyle',':','Color',[0 0 1],'Linewidth',3,'DisplayName', sprintf('Data: %g %% of total population',prop_data));
p2 = plot(x_simul, a2_simul, 'Linestyle','-','Color',[0 1 0],'Linewidth',3,'DisplayName', sprintf('Simulation: %g %% of total population',prop_simul));

ylabel(Yname, 'FontSize', size_pour_font, 'FontName','Arial', 'FontWeight','bold')
xlabel(Xname, 'FontSize', size_pour_font, 'FontName','Arial', 'FontWeight','bold')

set(axes1,'FontName','Arial','FontSize',size_pour_font,'FontWeight','bold');

legend1 = legend([p1, p2]);
set(legend1,...
    'FontSize',size_pour_font_legend,...
    'FontName','Arial');

xlim([0 70])

hold off

end

function sauvegarde(nom)
    format_d='-dpng ';
    format2='.png';
            
    print('-f1', format_d, [nom,format2]);
    saveas(1,[nom,'.fig']);

end