library(rgdal)
library(sf)
library(tidyverse)
library(openxlsx)

###
flood_plain = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb", layer = "FLOODPLAINS")
grid = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/data_Cape_Town/data_maps/grid_reference_500.shp")
st_crs(flood_plain) = st_crs(grid)
flood_grid <- as_tibble(st_intersection(st_buffer(grid, 0), st_buffer(flood_plain, 0)))
flood_grid$area <- st_area(flood_grid$geometry)
st_write(flood_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_grid.shp')
flood_grid = flood_grid[, c("ID", "OBJECTID", "FL_TYPE", "FLDL_DESC", "area")]
write.xlsx(flood_grid, 'C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/flood_plain_grid.xlsx')


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