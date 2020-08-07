library(rgdal)
library(sf)
library(tidyverse)
library(openxlsx)
library(rgeos)

flood_plain = readOGR("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb", layer = "FLOODPLAINS")
flood_plain$FL_TYPE[flood_plain$FLDL_DESC == "Prinskasteel 100y Floodline"] = "100yr"
flood_plain_20yr = flood_plain[flood_plain$FL_TYPE == '20yr',]
flood_plain_50yr = flood_plain[flood_plain$FL_TYPE == '50yr',]
flood_plain_100yr = flood_plain[flood_plain$FL_TYPE == '100yr',]
flood_plain_20yr_and_50yr = gUnion(gBuffer(flood_plain_50yr, byid = T, width = 0), gBuffer(flood_plain_20yr, byid = T, width = 0))
flood_plain_20yr_50yr_and_100yr = gUnion(flood_plain_20yr_and_50yr, gBuffer(flood_plain_100yr, byid = T, width = 0))
gArea(flood_plain_20yr)
gArea(gBuffer(flood_plain_50yr, byid = T, width = 0))
gArea(gBuffer(flood_plain_100yr, byid = T, width = 0))
gArea(flood_plain_20yr_and_50yr)
gArea(flood_plain_20yr_50yr_and_100yr)
#flood_plain = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb", layer = "FLOODPLAINS")
grid = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/data_Cape_Town/data_maps/grid_reference_500.shp")
flood_plain_20yr = st_as_sf(gBuffer(flood_plain_20yr, byid = T, width = 0))
flood_plain_50yr = st_as_sf(gBuffer(flood_plain_50yr, byid = T, width = 0))
flood_plain_100yr = st_as_sf(gBuffer(flood_plain_100yr, byid = T, width = 0))
flood_plain_20yr_and_50yr = st_as_sf(flood_plain_20yr_and_50yr, byid = T, width = 0)
flood_plain_20yr_50yr_and_100yr = st_as_sf(flood_plain_20yr_50yr_and_100yr, byid = T, width = 0)

sum(st_area(flood_plain_20yr))
sum(st_area(flood_plain_50yr))
sum(st_area(flood_plain_100yr))
sum(st_area(flood_plain_20yr_and_50yr))
sum(st_area(flood_plain_20yr_50yr_and_100yr))

st_crs(flood_plain_20yr) = st_crs(grid)
st_crs(flood_plain_50yr) = st_crs(grid)
st_crs(flood_plain_100yr) = st_crs(grid)
st_crs(flood_plain_20yr_and_50yr) = st_crs(grid)
st_crs(flood_plain_20yr_50yr_and_100yr) = st_crs(grid)

flood_plain_20yr_grid <- as_tibble(st_intersection(st_buffer(grid, 0), flood_plain_20yr))
flood_plain_20yr_grid$area <- st_area(flood_plain_20yr_grid$geometry)
sum(flood_plain_20yr_grid$area)
flood_plain_50yr_grid <- as_tibble(st_intersection(st_buffer(grid, 0), flood_plain_50yr))
flood_plain_50yr_grid$area <- st_area(flood_plain_50yr_grid$geometry)
sum(flood_plain_50yr_grid$area)
flood_plain_100yr_grid <- as_tibble(st_intersection(st_buffer(grid, 0), flood_plain_100yr))
flood_plain_100yr_grid$area <- st_area(flood_plain_100yr_grid$geometry)
sum(flood_plain_100yr_grid$area)
flood_plain_20yr_and_50yr_grid <- as_tibble(st_intersection(st_buffer(grid, 0), flood_plain_20yr_and_50yr))
flood_plain_20yr_and_50yr_grid$area <- st_area(flood_plain_20yr_and_50yr_grid$geometry)
sum(flood_plain_20yr_and_50yr_grid$area)
flood_plain_20yr_50yr_and_100yr_grid <- as_tibble(st_intersection(st_buffer(grid, 0), flood_plain_20yr_50yr_and_100yr))
flood_plain_20yr_50yr_and_100yr_grid$area <- st_area(flood_plain_20yr_50yr_and_100yr_grid$geometry)
sum(flood_plain_20yr_50yr_and_100yr_grid$area)

st_write(flood_plain_20yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_grid.shp')
st_write(flood_plain_50yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_50yr_grid.shp')
st_write(flood_plain_100yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_100yr_grid.shp')
st_write(flood_plain_20yr_and_50yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_and_50yr_grid.shp')
st_write(flood_plain_20yr_50yr_and_100yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_50yr_and_100yr_grid.shp')
flood_plain_20yr_grid = flood_plain_20yr_grid[, c("ID", "area")]
flood_plain_50yr_grid = flood_plain_50yr_grid[, c("ID", "area")]
flood_plain_100yr_grid = flood_plain_100yr_grid[, c("ID", "area")]
flood_plain_20yr_and_50yr_grid = flood_plain_20yr_and_50yr_grid[, c("ID", "area")]
flood_plain_20yr_50yr_and_100yr_grid = flood_plain_20yr_50yr_and_100yr_grid[, c("ID", "area")]
write.xlsx(flood_plain_20yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_grid.xlsx')
write.xlsx(flood_plain_50yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_50yr_grid.xlsx')
write.xlsx(flood_plain_100yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_100yr_grid.xlsx')
write.xlsx(flood_plain_20yr_and_50yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_and_50yr_grid.xlsx')
write.xlsx(flood_plain_20yr_50yr_and_100yr_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_20yr_50yr_and_100yr_grid.xlsx')


###
df = read.xlsx('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_grid.xlsx')
df$area = as.numeric(df$area)
#Verifier que, si il y a 20 years, il y a aussi 50 years et 100 years
`%notin%` <- Negate(`%in%`)
subset_20 = df[df$FL_TYPE == "20yr",]
subset_50 = df[df$FL_TYPE == "50yr",]
subset_100 = df[df$FL_TYPE == "100yr",]
print(subset_20$ID[subset_20$ID %notin% subset_50$ID]) #Négligeable
print(subset_20$ID[subset_20$ID %notin% subset_100$ID]) #Négligeable
summary(subset_50$area[subset_50$ID %notin% subset_100$ID]) #Pas négligeable

df2 = subset_50[(subset_50$ID %notin% subset_100$ID),]
df2$FL_TYPE = "100yr"
df2 <- rbind(df, df2)

df$FL_TYPE[df$FLDL_DESC == "Prinskasteel 100y Floodline"] = "100yr"
df2$FL_TYPE[df2$FLDL_DESC == "Prinskasteel 100y Floodline"] = "100yr"

sum(df$area[df$FL_TYPE == "20yr"])
sum(df$area[df$FL_TYPE == "50yr"])
sum(df$area[df$FL_TYPE == "100yr"])
sum(df2$area[df$FL_TYPE == "100yr"])

spatial_df = st_read('C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_grid.shp')
plot(spatial_df)
