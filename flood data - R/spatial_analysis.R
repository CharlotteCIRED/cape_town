library(rgdal)
library(tidyverse)
library(openxlsx)
library("sf")
library("dplyr")
library(rgeos)
library(raster)

grid = readOGR("C:/Users/Charlotte Liotta/Desktop/Cape Town - pour Charlotte/data_Cape_Town/data_maps/grid_reference_500.shp")

FD_5yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_5yr.xlsx')
FD_10yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_10yr.xlsx')
FD_20yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_20yr.xlsx')
FD_50yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_50yr.xlsx')
FD_75yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_75yr.xlsx')
FD_100yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_100yr.xlsx')
FD_200yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_200yr.xlsx')
FD_250yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_250yr.xlsx')
FD_500yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_500yr.xlsx')
FD_1000yr = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/FATHOM/FD_1000yr.xlsx')

#1. Roads in flood-prone areas

#Download data on roads
road = readOGR("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/GIS/CT_Roads.shp")

#aggregate the length of roads per grid cell
road <- spTransform(road, CRS(proj4string(grid)))
road_sf <- st_as_sf(road)
grid_sf <- st_as_sf(grid)
int = st_intersection(road_sf, grid_sf)
int$len = st_length(int)
grid_sf$Id = 1:nrow(grid_sf)
join = st_join(grid_sf, int)
out = group_by(join, Id) %>%
  summarize(length = sum(len))
filter(out, is.na(length))
mutate(out, length = ifelse(is.na(length), 0, length))
out$length = ifelse(is.na(out$length), 0, out$length)

#define flood-prone areas
depth = 0.25
area = 0.4
FD_5yr$flood_prone = ifelse((FD_5yr$flood_depth > depth) & (FD_5yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(out$length, by = list(FD_5yr$flood_prone), FUN = sum)
FD_20yr$flood_prone = ifelse((FD_20yr$flood_depth > depth) & (FD_20yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(out$length, by = list(FD_20yr$flood_prone), FUN = sum)
FD_100yr$flood_prone = ifelse((FD_100yr$flood_depth > depth) & (FD_100yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(out$length, by = list(FD_100yr$flood_prone), FUN = sum)

#2. IS

#Import Claus' data on informal settlements
IS = readOGR("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Land occupation/Land occupation risk/Occupation_risk.shp")

#Intersect with grid and extract area
IS@data$id = 1
union_is <- gUnaryUnion(IS, id = IS@data$id)
union_is <- spTransform(union_is, CRS(proj4string(grid)))
union_is <- gBuffer(union_is, byid=TRUE, width=0)
pi <- intersect(grid, union_is)
pi$area <- area(pi)
aggregate(area~grid, data=pi, FUN=sum)

#Export
df = data.frame(grid@data$ID)
df = merge(df, pi@data, by.x = "grid.data.ID", by.y = "ID", all = T)
df = df[, c("grid.data.ID", "area")]
df$area = ifelse(is.na(df$area), 0, df$area)
write.csv(df, "C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Land occupation/informal_settlements_risk.csv")

#3. Constraints and flood-prone areas

land_use = read.csv('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Basile data/grid_NEDUM_Cape_Town_500.csv', sep = ';')

#Zone inondable = constraints?
depth = 0.5
area = 0.20

FD_5yr$flood_prone = ifelse((FD_5yr$flood_depth > depth) & (FD_5yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(df$area, by = list(FD_5yr$flood_prone), FUN = sum)
FD_20yr$flood_prone = ifelse((FD_20yr$flood_depth > depth) & (FD_20yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(df$area, by = list(FD_20yr$flood_prone), FUN = sum)
FD_100yr$flood_prone = ifelse((FD_100yr$flood_depth > depth) & (FD_100yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(df$area, by = list(FD_100yr$flood_prone), FUN = sum)

FD_5yr$flood_prone = ifelse((FD_5yr$flood_depth > depth) & (FD_5yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(land_use$informal, by = list(FD_5yr$flood_prone), FUN = sum)
FD_20yr$flood_prone = ifelse((FD_20yr$flood_depth > depth) & (FD_20yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(land_use$informal, by = list(FD_20yr$flood_prone), FUN = sum)
FD_100yr$flood_prone = ifelse((FD_100yr$flood_depth > depth) & (FD_100yr$prop_flood_prone > area), TRUE, FALSE)
aggregate(land_use$informal, by = list(FD_100yr$flood_prone), FUN = sum)
#IS en zone inondable ?
#Backyard en zone inondable?
