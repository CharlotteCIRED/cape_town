# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:27:47 2020

@author: Charlotte Liotta
"""

def amenity_calibration_parameters_v3(grille,param, macro, poly, option, trans, data_courbe, land, t_amenity):

    #Year for which we estimate parameters
    t_trafic_amenity = t_amenity - param.year_begin;

    #Preparation of data (at the SP_level)

    #Income
    amenity.revenu = data_courbe.income_SP;  
    revenu_ref = ppval(macro.spline_revenu,t_trafic_amenity);
    revenu_max = max(amenity.revenu);

    #Transport
    amenity.reliable_transport = griddata(grille.coord_horiz,grille.coord_vert,trans.reliable,data_courbe.X_price,data_courbe.Y_price);
    #Transport by income class
    cout_generalise = double(prix2_polycentrique3(trans.t_transport,trans.cout_generalise,param,t_trafic_amenity));
    for i = 1:param.multiple_class:
        trans_SP(i,:) = griddata(grille.coord_horiz, grille.coord_vert,cout_generalise(i,:),data_courbe.SP_X, data_courbe.SP_Y);

    #Average transport cost (using average income per SP)
    param2 = param;
    param2.multiple_class = 12 #to assign several household classes to each job center
    param2.taille_menage_transport = [1 1 1 1 1 1 2 2 2 2 2 2] #household size for transport costs, ? calibrer sur des donn?es r?elles
    param2.income_distribution = [1 2 3 4 5 6 7 8 9 10 11 12];
    poly2 = import_emploi_CAPE_TOWN2(grille, param2, option, macro, data_courbe, [t_trafic_amenity, t_trafic_amenity + 10]);
    trans2 = charges_temps_polycentrique_CAPE_TOWN_3(option,grille,macro,param2,poly2,t_trafic_amenity,trans);
    cout_generalise2 = double(prix2_polycentrique3(trans2.t_transport,trans2.cout_generalise,param,t_trafic_amenity));
    for i = 1:param2.multiple_class:
        trans_SP2(i,:) = griddata(grille.coord_horiz, grille.coord_vert,cout_generalise2(i,:),data_courbe.SP_X, data_courbe.SP_Y);

    amenity.trans_avg = zeros(1,length(amenity.revenu));
    for i=1:length(data_courbe.income_SP)
        amenity.trans_avg(i) = interp1(interp1(poly2.annee, poly2.avg_inc(:,:), t_amenity)', trans_SP2(:,i), data_courbe.income_SP(i));

    #Income class
    amenite_class = ones(1,length(amenity.revenu));
    for i=1:length(amenite_class)
        for j=2:param.multiple_class
            if amenity.revenu(i)>data_courbe.limit(j-1) 
                amenite_class(i)=j;
        amenity.revenu_class(i) = interp1(poly.annee, poly.avg_inc(:,amenite_class(i)), t_amenity);
        amenity.trans(i) = trans_SP(amenite_class(i),i)';


    #Income net of transportation costs
    net_income = data_courbe.income_SP' - amenity.trans_avg;

    #Price
    price = interp1(data_courbe.year_price, data_courbe.SP_price, t_amenity);
    distance = sqrt((data_courbe.X_price-grille.xcentre).^2+(data_courbe.Y_price-grille.ycentre).^2); %inutile, juste pour des courbes de contr?le

    #Dwelling size and density HFA
    amenite_dwelling_size = data_courbe.SP_dwelling_size;
    amenite_dwelling_size(amenite_dwelling_size > 1000) = NaN;
    amenite_construction = data_courbe.SP_formal_dens_HFA./param.coeff_landmax;
    amenite_construction(amenite_construction > 2) = NaN;
    amenite_rent_HFA = price .* (param.depreciation_h + interest_rate(macro, t_amenity - param.year_begin))./amenite_construction;


    #Calibration of the parameters of the utility function

    #When we estimate the basic need as the minimum
    amenity.basic_q = min(amenite_dwelling_size(data_courbe.SP_informal_backyard + data_courbe.SP_informal_settlement < 1));

    #What SP do we keep? 
    quel_general = (amenity.reliable_transport>0.95) & (amenity.trans_avg < 0.6*data_courbe.income_SP') & (net_income./amenite_rent_HFA > 100) & (price > 0)...
                                                        & (amenite_class >= 2) & (amenite_rent_HFA > quantile(amenite_rent_HFA, 0.1)) & (amenite_rent_HFA < quantile(amenite_rent_HFA, 0.9))...
                                                        & (amenite_construction > 0.01) & (data_courbe.SP_2011_distance' < 15);

    net_income_reg = net_income(quel_general);
    amenite_rent_HFA_reg = amenite_rent_HFA(quel_general);
    amenite_dwelling_size_reg = amenite_dwelling_size(quel_general)';

    #Model
    model = fitlm(net_income_reg./amenite_rent_HFA_reg - amenity.basic_q, amenite_dwelling_size_reg - amenity.basic_q, 'Intercept', false);
    amenity.coeff_beta = model.Coefficients.Estimate(1);
    
    #Calibration of amenities in each location

    #Estimation of the ratio of utilities for each income class and the one above
    residu = abs(amenity.coeff_beta.^amenity.coeff_beta.*(1 - amenity.coeff_beta).^(1 - amenity.coeff_beta).*...
                 (amenity.revenu_class - amenity.trans - amenity.basic_q.*amenite_rent_HFA))./(amenite_rent_HFA).^(amenity.coeff_beta);
    ratio_utility = zeros(1, length(param.multiple_class) - 1);
    utility_normalized = ones(1, length(amenite_class));
    for i = 1:param.multiple_class - 1
        residual_poor = residu((amenite_class == i) & ((amenity.revenu_class - amenity.trans - amenity.basic_q.*amenite_rent_HFA) > 0));
        residual_rich = residu((amenite_class == i + 1) & ((amenity.revenu_class - amenity.trans - amenity.basic_q.*amenite_rent_HFA) > 0));
        [~,m] = min(abs(residual_poor - quantile(residual_poor, .1)));
        [~,M] = min(abs(residual_rich - quantile(residual_rich, .9)));
        ratio_utility(i) = residual_rich(M)./residual_poor(m);
        ratio_utility_u1 = prod(ratio_utility);
        utility_normalized(amenite_class == i + 1) = ratio_utility_u1;

    #Estimation of A/u
    residual_reg = amenity.coeff_beta.*log(abs(amenite_rent_HFA)) - ...
        log(abs(amenity.coeff_beta.^amenity.coeff_beta.*(1 - amenity.coeff_beta).^(1 - amenity.coeff_beta).*...
                (amenity.revenu_class - amenity.trans - amenity.basic_q.*amenite_rent_HFA))) + ...
        log(utility_normalized);
    residual_reg(amenite_rent_HFA <= 0) = NaN;
    residual_reg((amenity.revenu_class - amenity.trans - amenity.basic_q.*amenite_rent_HFA) <= 0) = NaN;

    quel = (amenity.reliable_transport==1) & (~isnan(amenite_rent_HFA));


    #Import of the amenity files at the SP level (for the regression)
    importfile([path_nedum,'SP_amenities.csv']);

    #Airport cones
    airport_cone2=airport_cone;
    airport_cone2(airport_cone==55) = 1;
    airport_cone2(airport_cone==60) = 1;
    airport_cone2(airport_cone==65) = 1;    
    airport_cone2(airport_cone==70) = 1;
    airport_cone2(airport_cone==75) = 1;
    
    #Distance to RDP houses
    dist_RDP = 2;
    if dist_RDP ~= 2 
        matrix_distance = ((repmat(grille.coord_horiz, length(data_courbe.SP_X), 1) - repmat(data_courbe.SP_X, 1, length(grille.coord_horiz))).^2 ... 
                           + (repmat(grille.coord_vert, length(data_courbe.SP_Y), 1) - repmat(data_courbe.SP_Y, 1, length(grille.coord_vert))).^2)...
                            < dist_RDP ^ 2;
        SP_distance_RDP = (double(land.RDP_houses_estimates > 5) * (matrix_distance)')' > 1;
    else
        load(strcat('.', slash, 'precalculations', slash, 'SP_distance_RDP'))

    table_regression = table(residual_reg',...
        distance_distr_parks < 2, distance_ocean < 2,...
        distance_world_herit < 2, distance_urban_herit < 2, distance_UCT < 2,...
        airport_cone2, log(1+slope), distance_train < 2, distance_protected_envir < 2, log(1+SP_distance_RDP), distance_power_station < 2);
    table_reg = table_regression(quel,:);
    table_reg.Properties.VariableNames = {'residu' 'distance_distr_parks' 'distance_ocean' 'distance_world_herit' 'distance_urban_herit' 'distance_UCT' 'airport_cone2' 'slope' 'distance_train' 'distance_protected_envir' 'RDP_proximity' 'distance_power_station'};
    model_spec = 'residu ~ distance_distr_parks + distance_ocean + distance_urban_herit + airport_cone2 + slope + distance_protected_envir + RDP_proximity'; %+ distance_power_station';
    model_amenity = fitglm(table_reg,model_spec);
    
    #Import of the amenity files at the grid level for the extrapolation
    importfile([path_nedum,'grid_amenity.csv']);

    #Airport cones
    airport_cone2=airport_cone;
    airport_cone2(airport_cone==55) = 1;
    airport_cone2(airport_cone==60) = 1;
    airport_cone2(airport_cone==65) = 1;
    airport_cone2(airport_cone==70) = 1;
    airport_cone2(airport_cone==75) = 1;

    #Distance to RDP housing
    if dist_RDP ~= 2
        matrix_distance = ((repmat(grille.coord_horiz,length(grille.coord_horiz),1) - repmat(grille.coord_horiz',1,length(grille.coord_horiz))).^2 ... 
                                                                                             + (repmat(grille.coord_vert,length(grille.coord_horiz),1) - repmat(grille.coord_vert',1,length(grille.coord_horiz))).^2)...
                                                                                                                                                                < dist_RDP ^ 2;
        grid_distance_RDP = (double(land.RDP_houses_estimates > 5) * (matrix_distance)')' > 1;
    else 
        load(strcat('.', slash, 'precalculations', slash, 'grid_distance_RDP'))

    #Estimation
    table_predictors = table(distance_distr_parks < 2, distance_ocean < 2,...
                             distance_urban_herit < 2,...
                             airport_cone2, log(1+slope), distance_protected_envir < 2, log(1 + grid_distance_RDP));

    amenity.estimated_amenities = exp(amenity.coeff_beta.*model_amenity.Coefficients.Estimate(2:8)'*table2array(table_predictors)');    
    
    print(sprintf('Regression on the exogenous amenities - Rquared = %d',model_amenity.Rsquared.Ordinary));
    amenity.model_amenity = model_amenity;
    amenity.utility = exp(model_amenity.Coefficients.Estimate(1)).*[1 ratio_utility];

    #Calibration of the construction function
    #Formula: RH = A.b.H^b

    amenite_construction = data_courbe.SP_formal_dens_HFA ./ param.coeff_landmax;

    quel_construct = (amenite_construction > 0.05) & (amenite_construction < 3) &...
        (data_courbe.SP_2011_distance' < 20) & (amenite_class > 1) &...
         (amenite_rent_HFA > quantile(amenite_rent_HFA, 0.1)) & (amenite_rent_HFA < quantile(amenite_rent_HFA, 0.9));
    model_construction = fitlm(log(price(quel_construct)), log(amenite_construction(quel_construct).*1000000));

    amenity.coeff_b = model_construction.Coefficients.Estimate(2);
    amenity.coeff_a = 1 - amenity.coeff_b;
    amenity.coeff_grandA = (1./amenity.coeff_b.^amenity.coeff_b) .* exp(model_construction.Coefficients.Estimate(1).* amenity.coeff_b) ;
    amenity.model_construction = model_construction;


    #Calibration of the construction function for 2001

    total_2001_RDP = ppval(macro.spline_RDP, 2001 - param.year_begin);
    grid_2001_private_housing = max(0, data_courbe.formal_2001_grid - data_courbe.GV_count_RDP./sum(data_courbe.GV_count_RDP).*total_2001_RDP);
    formal_2001_SP_2011 = griddata(grille.coord_horiz, grille.coord_vert, grid_2001_private_housing, data_courbe.SP_X, data_courbe.SP_Y);

    quel = (formal_2001_SP_2011' > 0) & ...
            (data_courbe.SP_dwelling_size > 0) & (data_courbe.SP_dwelling_size < 500) &...
            (data_courbe.SP_2011_distance' < 20);

    model_constr_2001 = fitlm(log(data_courbe.SP_price(1,quel)), log(formal_2001_SP_2011(quel)'.*data_courbe.SP_dwelling_size(quel)));

    amenity.coeff_b_2001 = model_constr_2001.Coefficients.Estimate(2);
    amenity.coeff_a_2001 = 1 - amenity.coeff_b_2001;
    amenity.coeff_grandA_2001 = exp(model_constr_2001.Coefficients.Estimate(1))./(amenity.coeff_b_2001^amenity.coeff_b_2001);

    return amenity
