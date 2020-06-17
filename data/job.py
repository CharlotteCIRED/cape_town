# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:47:04 2020

@author: Charlotte Liotta
"""

def import_emploi_CAPE_TOWN2(grille, param, option, macro, data_courbe, t):

    [X,Y,TZ2013,Area,diss,Zones,BY5Origins,BY5Destina,PTODOrigin,PTODDestin,emploi_TAZ,emploi_T_1,emploi_T_2,emploi_T_3,emploi_T_4,emploi_T_5,emploi_T_6,emploi_T_7,emploi_T_8,job_total,job_dens,Ink1,Ink2,Ink3,Ink4,job_tot] ...
    = importfile_TAZ(strcat(path_nedum, 'TAZ_amp_2013_proj_centro2.csv'))

    load emploi.mat
    job_total = emploi';
    emploi_T_5 = emploi_T_5bis';
    emploi_T_6 = emploi_T_6bis';
    emploi_T_7 = emploi_T_7bis';
    emploi_T_8 = emploi_T_8bis';

    TAZ2013_centre = importfile_centers(strcat(path_nedum, 'TAZ_final.csv'));
    listou = TAZ2013_centre;
    Xou = zeros(1,length(listou));
    You = zeros(1,length(listou));
    corresp = zeros(1,length(listou));
    increment = 1:length(TZ2013);
    for index1 = 1:length(listou)
        Xou(index1) = X(TZ2013 == listou(index1));
        You(index1) = Y(TZ2013 == listou(index1));
        corresp(index1) = increment(TZ2013 == listou(index1));
               
    J_data = [Ink1./3 Ink1./3 Ink1./3 Ink2./3 Ink2./3 Ink2./3 Ink3./3 Ink3./3 Ink3./3 Ink4./3 Ink4./3 Ink4./3];

    poly.code_emploi_init = TZ2013;
    poly.code_emploi_init_simple = TZ2013;
    ID_centre = TZ2013;
    XCoord = X;
    YCoord = Y;


    #Total number of households per class

    year_income_distribution = param.year_begin + t;
    if length(t) == 1
        total_bracket = [ones(length(t),1), ppval(macro.spline_pop_inc_distribution, t)'];
        avg_inc_bracket = [zeros(length(t),1), ppval(macro.spline_inc_distribution, t)'];
    else
        total_bracket = [ones(length(t),1), ppval(macro.spline_pop_inc_distribution, t)];
        avg_inc_bracket = [zeros(length(t),1), ppval(macro.spline_inc_distribution, t)];

    avg_inc_class = zeros(length(year_income_distribution), param.multiple_class);
    total_class = zeros(length(year_income_distribution), param.multiple_class);
 
    #total income distribution in the city
    for j=1:param.multiple_class:
        total_class(:,j) = sum(total_bracket(:, param.income_distribution == j), 2);
        avg_inc_class(:,j) = sum(avg_inc_bracket(:,param.income_distribution==j).*total_bracket(:, param.income_distribution==j), 2)./total_class(:,j);

    #Selection of employment centers

    poly.code_emploi_init = zeros(1, param.multiple_class*length(ID_centre));
    Jx = zeros(1, param.multiple_class*length(ID_centre));
    Jy = zeros(1,param.multiple_class*length(ID_centre));
    Jval1 = zeros(length(year_income_distribution),param.multiple_class*length(ID_centre));
    avg_inc = zeros(length(year_income_distribution), param.multiple_class*length(ID_centre));
    class = zeros(length(year_income_distribution), param.multiple_class*length(ID_centre));


    #Duplication of employment centers for the several income groups

    for i = 1:length(ID_centre)
        for j = 1:param.multiple_class
            poly.code_emploi_init(1, param.multiple_class*(i-1)+j) = ID_centre(i);
            Jx(param.multiple_class*(i-1)+j) = XCoord(i)/1000;
            Jy(param.multiple_class*(i-1)+j) = YCoord(i)/1000;
            Jval1(:, param.multiple_class*(i-1)+j) = repmat(sum(J_data(i,param.income_distribution==j)), length(year_income_distribution), 1);
            avg_inc(:,param.multiple_class*(i-1)+j) = avg_inc_class(:,j);
            class(:,param.multiple_class*(i-1)+j) = j;

    ID_centre_poly = 1:length(poly.code_emploi_init);

    #Selection of employment centers to keep

    if option.polycentric==1
    
        poly.quel = false(1, length(poly.code_emploi_init));

        #Manual sorting 
        poly.quel(poly.code_emploi_init == 5101) = true; % CBD
        poly.quel(poly.code_emploi_init == 2002) = true; % Bellville
        poly.quel(poly.code_emploi_init == 1201) = true; % Epping
        poly.quel(poly.code_emploi_init == 1553) = true; % Claremont
        poly.quel(poly.code_emploi_init == 3509) = true; % Sommerset West
        poly.quel(poly.code_emploi_init == 5523) = true; % Table View + Century City

        poly.quel(Jval1(1,:) <= 0) = false;

    elif option.polycentric == 0:
    
        poly.quel = false(1, length(poly.code_emploi_init));
        poly.quel(poly.code_emploi_init == 5101) = true; %CBD

    #Rescale of number of jobs after selection
    #Rescale to include for each center all the jobs located in a defined buffer zone

    distance_buffer = 4;
    Jval_temp = Jval1(:,poly.quel);
    sum_class_j = zeros(length(year_income_distribution), param.multiple_class);
    Jdistance = sqrt((Jx - Jx(poly.quel)').^2 + (Jy - Jy(poly.quel)').^2);

    for i = 1:length(year_income_distribution)
        for j = 1:param.multiple_class
            Jval_i = Jval1(i,:);
            class_i = class(i,:);
            poly_class_i = class(i,poly.quel);
            Jval_temp(i, poly_class_i == j) = Jval_i(class_i == j) * (Jdistance(poly_class_i == j,class_i == j)' < distance_buffer);
            sum_class_j(i,j) = sum(Jval_temp(i, poly_class_i == j));

    #Remove the employment centers that are not significant enough
    ID_centre_poly_quel = ID_centre_poly(poly.quel);
    quel_temp = ones(1,sum(poly.quel));
    for j = 1:param.multiple_class
        quel_temp(Jval_temp(1, poly_class_i == j) ./ sum_class_j(1,j) < 0.1) = 0;

    quel_temp = logical(quel_temp);
    Jval_temp = Jval_temp(:,quel_temp);
    ID_centre_poly_remove = ID_centre_poly_quel(quel_temp == 0);
    poly.quel(ismember(ID_centre_poly, ID_centre_poly_remove)) = 0;
    poly.Jx = Jx(poly.quel);
    poly.Jy = Jy(poly.quel);
    poly.class = class(:,poly.quel);
    poly.avg_inc = avg_inc(:,poly.quel);

    #Rescale to keep the correct global income distribution

    sum_class_quel = zeros(length(year_income_distribution), param.multiple_class);
    Jval2 = zeros(size(Jval_temp));
    
    for j = 1:param.multiple_class
    
        sum_class_quel(:,j) = sum(Jval_temp(:, poly.class(1,:) == j), 2);
        nb_centre_quel = sum(poly.class(1,:) == j);
    
        Jval2(:,poly.class(1,:) == j)...
            = Jval_temp(:,poly.class(1,:) == j) .*...
                repmat(total_class(:, j)./sum_class_quel(:, j), 1, nb_centre_quel);
        
    #Export 
    annee_Jval = year_income_distribution; 
    poly.total_hh_class = total_class;

    poly.annee = annee_Jval;

    increment = 1:length(poly.avg_inc(1,:));
    poly.referencement = increment;
    poly.garde = poly.quel;

    poly.code_emploi_poly = poly.code_emploi_init(poly.quel);
    poly.avg_inc = avg_inc(:,poly.quel);
    poly.class = class(:,poly.quel);

    increment = 1:length(TZ2013);
    corresp = zeros(1,length(poly.Jx));
    for index1 = 1:length(poly.Jx)
        corresp(index1) = increment(TZ2013==poly.code_emploi_poly(index1));

    poly.corresp = corresp';

    poly.Jval_pour_garde = Jval1;
    poly.Jval = Jval2;

    #Who can live in informal / formal settlements
    poly.formal = [1 1 1 1];
    poly.backyard = [1 1 0 0];
    poly.settlement = [1 1 0 0];

    return poly

def importfile_TAZ(filename):
    startRow = 2
    endRow = inf

    formatSpec = '%f%f%f%f%f%f%f%f%f%f%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%[^\n\r]';

    fileID = fopen(filename,'r')

    dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(1)-1, 'ReturnOnError', false);
    for block=2:length(startRow)
        frewind(fileID);
        dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'HeaderLines', startRow(block)-1, 'ReturnOnError', false);
        for col=1:length(dataArray)
            dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    fclose(fileID);

    X = dataArray{:, 1};
    Y = dataArray{:, 2};
    TZ2013 = dataArray{:, 3};
    Area = dataArray{:, 4};
    diss = dataArray{:, 5};
    Zones = dataArray{:, 6};
    BY5Origins = dataArray{:, 7};
    BY5Destina = dataArray{:, 8};
    PTODOrigin = dataArray{:, 9};
    PTODDestin = dataArray{:, 10};
    emploi_TAZ = dataArray{:, 11};
    emploi_T_1 = dataArray{:, 12};
    emploi_T_2 = dataArray{:, 13};
    emploi_T_3 = dataArray{:, 14};
    emploi_T_4 = dataArray{:, 15};
    emploi_T_5 = dataArray{:, 16};
    emploi_T_6 = dataArray{:, 17};
    emploi_T_7 = dataArray{:, 18};
    emploi_T_8 = dataArray{:, 19};
    job_total = dataArray{:, 20};
    job_dens = dataArray{:, 21};
    Ink1 = dataArray{:, 22};
    Ink2 = dataArray{:, 23};
    Ink3 = dataArray{:, 24};
    Ink4 = dataArray{:, 25};
    job_tot = dataArray{:, 26};

    return X,Y,TZ2013,Area,diss,Zones,BY5Origins,BY5Destina,PTODOrigin,PTODDestin,emploi_TAZ,emploi_T_1,emploi_T_2,emploi_T_3,emploi_T_4,emploi_T_5,emploi_T_6,emploi_T_7,emploi_T_8,job_total,job_dens,Ink1,Ink2,Ink3,Ink4,job_tot
