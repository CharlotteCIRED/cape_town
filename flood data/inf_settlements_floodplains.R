library(rgdal)
library(sf)
library(tidyverse)
library(openxlsx)

### Informal settlements 2013

flood_plain = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb", layer = "FLOODPLAINS")
inf_2013 = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb", layer = "INF_DWELLINGS_2013")
inf_2013$ID = 1:nrow(inf_2013)
inf_per_floodplain_2013 = st_intersection(inf_2013, flood_plain) 
inf_per_floodplain_2013_20yr= inf_per_floodplain_2013[inf_per_floodplain_2013$FL_TYPE == '20yr',]
inf_per_floodplain_2013_50yr= inf_per_floodplain_2013[inf_per_floodplain_2013$FL_TYPE == '50yr',]
inf_per_floodplain_2013_100yr= inf_per_floodplain_2013[inf_per_floodplain_2013$FL_TYPE == '100yr',]
`%notin%` <- Negate(`%in%`)
sum(inf_per_floodplain_2013_50yr$ID %in% inf_per_floodplain_2013_100yr$ID)
sum(inf_per_floodplain_2013_100yr$ID %notin% inf_per_floodplain_2013_50yr$ID)

### Informal settlements 2020

inf_2020 = st_read("C:/Users/Charlotte Liotta/Desktop/cape_town/2. Data/Cape_Town_data/WBUS2_data.gdb", layer = "INF_DWELLINGS_2020")
inf_2020$ID = 1:nrow(inf_2020)
inf_per_floodplain_2020 = st_intersection(inf_2020, flood_plain) 
inf_per_floodplain_2020_20yr= inf_per_floodplain_2020[inf_per_floodplain_2020$FL_TYPE == '20yr',]
inf_per_floodplain_2020_50yr= inf_per_floodplain_2020[inf_per_floodplain_2020$FL_TYPE == '50yr',]
inf_per_floodplain_2020_100yr= inf_per_floodplain_2020[inf_per_floodplain_2020$FL_TYPE == '100yr',]
`%notin%` <- Negate(`%in%`)
sum(inf_per_floodplain_2020_20yr$ID %in% inf_per_floodplain_2013_50yr$ID)
sum(inf_per_floodplain_2020_50yr$ID %in% inf_per_floodplain_2013_100yr$ID)
sum(inf_per_floodplain_2020_20yr$ID %in% inf_per_floodplain_2013_100yr$ID)
sum(inf_per_floodplain_2020_100yr$ID %in% inf_per_floodplain_2013_50yr$ID)
sum(inf_per_floodplain_2020_50yr$ID %in% inf_per_floodplain_2013_20yr$ID)

### Informal settlements per floodplain

inf_per_floodplain_2013$count = 1
df = aggregate(inf_per_floodplain_2013$count,
               by = list(inf_per_floodplain_2013$FLDL_DESC),
               FUN = sum)

inf_per_floodplain_2020$count = 1
df2 = aggregate(inf_per_floodplain_2020$count,
                by = list(inf_per_floodplain_2020$FLDL_DESC),
                FUN = sum)