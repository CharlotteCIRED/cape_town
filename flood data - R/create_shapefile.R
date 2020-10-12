grid = readOGR("C:/Users/Charlotte Liotta/Desktop/cape_town/data_Cape_Town/data_maps/grid_reference_500.shp")
library(rmatio)
scenario = read.mat("C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/Modèle/projet_le_cap/simulations - 201907.mat")

s1 = data.frame((scenario$initialState$householdsHousingType[[1]]))
s1 = as.data.frame(t(as.matrix(s1)))
formal = s1$V1
backyard = s1$V2
informal = s1$V3
subsidized = s1$V4

grid$formal_s1<- formal
grid$backyard_s1<- backyard
grid$informal_s1<- informal
grid$subsidized_s1<- subsidized

formal_data = scenario$data$gridFormal[[1]]
informal_data = scenario$data$gridInformalSettlement[[1]]
backyard_data = scenario$data$gridInformalBackyard[[1]]

grid$formal_data<- as.vector(formal_data)
grid$backyard_data<- backyard_data
grid$informal_data<- informal_data

grid$erreur_informal = informal_data - informal
grid$erreur_backyard = backyard_data - backyard

writeOGR(grid, "C:/Users/Charlotte Liotta/Desktop/cape_town/4. Sorties/SIMUL_CHARLOTTE", driver = "ESRI Shapefile", layer = "error")

sum(grid@data$formal_data)



simul_charlotte = read.csv("C:/Users/Charlotte Liotta/Desktop/cape_town/hhtype.csv", header = FALSE)
grid$formal_charlotte = simul_charlotte$V1
grid$backyard_charlotte = simul_charlotte$V2
grid$informal_charlotte = simul_charlotte$V3
grid$subsidized_charlotte = simul_charlotte$V4