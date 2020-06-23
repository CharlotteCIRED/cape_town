# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 11:08:34 2020

@author: Charlotte Liotta
"""

def data_SP_vers_grille(data_SP, data_courbe_SP_Code, grille):  
    grid_intersect = pd.read_csv('./2. Data/grid_SP_intersect.csv', sep = ';')   
    data_grille = np.zeros(len(grille.dist))   
    for index in range(0, len(grille.dist)):       
        ici = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grille.ID[index]])
        area_exclu = 0       
        for i in range(0, len(ici)):     
            if len(data_SP[data_courbe_SP_Code == ici[i]]) == 0:                      
                area_exclu = area_exclu + sum(grid_intersect.Area[(grid_intersect.ID_grille == grille.ID[index]) & (grid_intersect.SP_CODE == ici[i])])
            else:
                data_grille[index] = data_grille[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grille.ID[index]) & (grid_intersect.SP_CODE == ici[i])]) * data_SP[data_courbe_SP_Code == ici[i]]       
        if area_exclu > 0.9 * sum(grid_intersect.Area[grid_intersect.ID_grille == grille.ID[index]]):
            data_grille[index] = np.nan           
        else:
            data_grille[index] = data_grille[index] / (sum(grid_intersect.Area[grid_intersect.ID_grille == grille.ID[index]]) - area_exclu)
                
    return data_grille

def data_SP_vers_grille_alternative(data_SP, data_courbe_SP_Code, grille):
    
    grid_intersect = pd.read_csv('./2. Data/grid_SP_intersect.csv', sep = ';')  
    data_grille = np.zeros(len(grille.dist))   
    for index in range(0, len(grille.dist)): 
        ici = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grille.ID[index]])
        for i in range(0, len(ici)): 
            if len(data_SP[data_courbe_SP_Code == ici[i]]) != 0:  
                data_grille[index] = data_grille[index] + sum(grid_intersect.Area[(grid_intersect.ID_grille == grille.ID[index]) & (grid_intersect.SP_CODE == ici[i])]) * data_SP[data_courbe_SP_Code == ici[i]] / sum(grid_intersect.Area[grid_intersect.SP_CODE == ici[i]])

    return data_grille



def data_SP_2001_vers_grille_alternative(data_SP, data_courbe_SP_2001_Code, grille):
    #pour des variables extensives
    grid_intersect = pd.read_csv('./2. Data/grid_SP2001_intersect.csv', sep = ';') 
    data_grille = np.zeros(len(grille.dist)) 
    for index in range(0, len(grille.dist)):
        ici = np.unique(grid_intersect.SP_CODE[grid_intersect.ID_grille == grille.ID[index]])     
        for i in range(0, len(ici)): 
            if len(data_SP[data_courbe_SP_2001_Code == ici[i]]) != 0:
                data_grille[index] = data_grille[index] + sum(grid_intersect.area_intersection[(grid_intersect.ID_grille == grille.ID[index]) & (grid_intersect.SP_CODE == ici[i])]) * data_SP[data_courbe_SP_2001_Code == ici[i]] / sum(grid_intersect.area_intersection[grid_intersect.SP_CODE == ici[i]])
    return data_grille

def data_CensusSAL_vers_grille(data_SAL, data_courbe_SAL_Code_conversion, grille):
    #to transform data at the Census 2011 SAL level to data at the grid level
    grid_intersect = pd.read_csv('./2. Data/grid_SAL_intersect.csv', sep = ';') 
    data_grille = np.zeros(len(grille.dist))

    for index in range (0, len(grille.dist)):
        ici = np.unique(grid_intersect.OBJECTID_1[grid_intersect.ID_grille == grille.ID[index]])
        if len(ici) == 0:
            data_grille[index] = np.nan
        else:
            for i in range(0, len(ici)):
                if len(data_SAL[data_courbe_SAL_Code_conversion == ici[i]]) > 0:
                    data_grille[index] = data_grille[index] + sum(grid_intersect.Area_inter[(grid_intersect.ID_grille == grille.ID[index]) & (grid_intersect.OBJECTID_1 == ici[i])]) * data_SAL[data_courbe_SAL_Code_conversion == ici[i]]
            if sum(grid_intersect.Area_inter[grid_intersect.ID_grille == grille.ID[index]]) < 150000:
                data_grille[index] = np.nan
            else:
                data_grille[index] = data_grille[index] / (sum(grid_intersect.Area_inter[grid_intersect.ID_grille == grille.ID[index]]))
    return data_grille


def import_data_SAL_landuse(grille):
    importfile([path_nedum,'SAL_EA_inters_data_landuse.csv'])

    urb = Collective_living_quarters+Formal_residential+Informal_residential;
    non_urb = Commercial+Farms+Industrial+Informal_residential+Parks_and_recreation+Small_Holdings+Vacant;

    return urb /(urb+non_urb)

def data_TAZ_vers_grille(data_TAZ, data_courbe, grille):
    importfile([path_nedum,'grid_TAZ_intersect.csv'])

    data_grille=zeros(1,length(grille.dist))
    for index=1:length(grille.dist):
    
        ici=unique(TZ2015(ID_grille==grille.ID(index)));
        area_exclu=0;
        for i=1:length(ici):
            if isempty(data_TAZ(data_courbe.TAZ_Code==ici(i))):
                area_exclu=area_exclu + sum(Area_int(ID_grille==grille.ID(index) & TZ2015==ici(i)));
            else:
            data_grille(index)=data_grille(index)+sum(Area_int(ID_grille==grille.ID(index) & TZ2015==ici(i)))...
            *data_TAZ(data_courbe.TAZ_Code==ici(i));
        if area_exclu>0.8*sum(Area_int(ID_grille==grille.ID(index))):
            data_grille(index)=NaN;
        else
            data_grille(index)=data_grille(index)/(sum(Area_int(ID_grille==grille.ID(index)))-area_exclu);
    return data_grille


def data_TAZ_vers_grille_alternative(data_TAZ, data_courbe, grille):
    #fonction alternative qui permet le passage de données par TZ aux données par point de
    #grille. Par rapport à l'autre, celle-là compte le nombre de logement par
    #carreau de grille, puis divise par 1000000km². Doit être avec la variable
    #"extensive" (sommable). 

    importfile([path_nedum,'grid_TAZ_intersect.csv'])

    data_grille=zeros(1,length(grille.dist));

    for index=1:length(grille.dist),
    
        ici=unique(TZ2015(ID_grille==grille.ID(index)));
        for i=1:length(ici)
            if ~isempty(data_TAZ(data_courbe.TAZ_Code==ici(i)))
                data_grille(index)=data_grille(index)+sum(Area_int(ID_grille==grille.ID(index) & TZ2015==ici(i)))...
                *data_TAZ(data_courbe.TAZ_Code==ici(i))/sum(Area_int(TZ2015==ici(i)));
    return data_grille



def moyenne(entree, nombre):

    zot1=reshape(entree, 100,100);
    zot=smooth2b(zot1,nombre);
    zot=reshape(zot,1,size(zot,1)*size(zot,2))
    return zot

def matrice_moyenne
    #calcule la matrice par laquelle il faut multiplier pour moyenner
    #**** EXPLICATION ****
    #                       taille du vecteur moyenné
    #                           <---->
    #                   ^   xxxxxxxxxxxxxxx
    #  "sortie" taille  |   xxxxxxxxxxxxxxx
    #        grille     |   xxxxxxxxxxxxxxx
    #        normale    |   xxxxxxxxxxxxxxx
    #                   |   xxxxxxxxxxxxxxx
    #   grille normale  v   xxxxxxxxxxxxxxx
    #     <---->            xxxxxxxxxxxxxxx
    #   xxxxxxxxxxxxxxxxxxx ooooooooooooooo  Moyenne obtenue



    #definition de la matrise "sortie"à la main
    Xi=[1:50,1:50];
    Yi=[1:2:100,[1:2:100]+1];
    for i=0:48,
        Xi_en_plus=[[1:100]+50*i,[1:100]+50*i];
        Yi_en_plus=[[1:2:200]+200*(i+1)-100,[1:2:200]+1+200*(i+1)-100];
        Xi=[Xi,Xi_en_plus];
        Yi=[Yi,Yi_en_plus];
        
    i=49;
    Xi_en_plus=[[1:50]+50*i,[1:50]+50*i];
    Yi_en_plus=[[1:2:100]+200*(i+1)-100,[1:2:100]+1+200*(i+1)-100];
    Xi=[Xi,Xi_en_plus];
    Yi=[Yi,Yi_en_plus];
    Si=ones(size(Xi))*0.25;
    return sparse(Yi,Xi,Si)

def pour_moyenne_logit(coeff_logit,prix):
    #coeff c'est le facteur dans le calcul et prix est le vecteur des prix

    AAA=exp(-coeff_logit.*prix)
    return -log(sum(AAA,3))+0.5772

def logit(coeff_logit,prix, nbre_mode)
    #coeff c'est le facteur dans le calcul et prix est le vecteur des prix

    AAA=exp(-coeff_logit.*prix);
    BBB=sum(AAA,3);

    CCC=zeros(size(AAA));
    for i = 1:nbre_mode
        CCC(:,:,i)=BBB;
    return AAA./CCC

def import_donnees_voiture(poly, grille);
    #import des données de temps de transport pour les véhicules particuliers
    M = csvread([path_nedum,'grille_simple.csv'],1,0);
    load([path_nedum,'resultx3.mat']);
    X_grille_simple=M(1:375,5)./1000;
    Y_grille_simple=M(1:375,4)./1000;

    distance_car=zeros(length(poly.referencement),length(grille.dist));
    duration_car=zeros(length(poly.referencement),length(grille.dist));

    for i=1:length(poly.referencement)
        distance_car(i,:)=griddata(X_grille_simple,Y_grille_simple,distance2(poly.code_emploi_poly(i),1:375),grille.coord_horiz,grille.coord_vert);
        duration_car(i,:)=griddata(X_grille_simple,Y_grille_simple,duration_in_traffic(poly.code_emploi_poly(i),1:375),grille.coord_horiz,grille.coord_vert);

    distance_car=distance_car./1000; %en km
    duration_car=duration_car./60;   %en minutes

    return distance_car, duration_car

def griddata_hier(a,b,c,x,y):
    test = ~np.isnan(c)
    surface_linear = scatteredInterpolant(a(test), b(test), c(test), 'linear', 'linear')
    surface_nearest = scatteredInterpolant(a(test), b(test), c(test), 'linear', 'nearest')

    #If the extrapolated data is lower than the nearest neighboor, we take
    #the nearest neighboor
    return max(surface_linear(x', y'), surface_nearest(x',y'))