# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:21:35 2020

@author: Charlotte Liotta
"""
import pandas as pd
import feather

# %% 2011

#Import 2011 data
jan11 = pd.read_excel('./2. Data./Floods - Service requests/2011/Jan11.ods', engine = "odf", header=4)
feb11 = pd.read_excel('./2. Data./Floods - Service requests/2011/feb11.ods', engine = "odf", header=4)
mar11 = pd.read_excel('./2. Data./Floods - Service requests/2011/mar11.ods', engine = "odf", header=4)
apr11 = pd.read_excel('./2. Data./Floods - Service requests/2011/apr11.ods', engine = "odf", header=4)
may11 = pd.read_excel('./2. Data./Floods - Service requests/2011/may11.ods', engine = "odf", header=4)
jun11 = pd.read_excel('./2. Data./Floods - Service requests/2011/jun11.ods', engine = "odf", header=4)
jul11 = pd.read_excel('./2. Data./Floods - Service requests/2011/jul11.ods', engine = "odf", header=4)
aug11 = pd.read_excel('./2. Data./Floods - Service requests/2011/aug11.ods', engine = "odf", header=4)
sep11 = pd.read_excel('./2. Data./Floods - Service requests/2011/sep11.ods', engine = "odf", header=4)
oct11 = pd.read_excel('./2. Data./Floods - Service requests/2011/oct11.ods', engine = "odf", header=4)
nov11 = pd.read_excel('./2. Data./Floods - Service requests/2011/nov11.ods', engine = "odf", header=4)
dec11 = pd.read_excel('./2. Data./Floods - Service requests/2011/dec11.ods', engine = "odf", header=4)

#Append and save
df11 = pd.DataFrame().append(jan11).append(feb11).append(mar11).append(apr11).append(may11).append(jun11).append(jul11).append(aug11).append(sep11).append(oct11).append(nov11).append(dec11)
feather.write_dataframe(df11, './2. Data./Floods - Service requests/datasets/df11.feather')

# %% 2012

#Import 2012 data
jan12 = pd.read_excel('./2. Data./Floods - Service requests/2012/jan12.ods', engine = "odf", header=4)
feb12 = pd.read_excel('./2. Data./Floods - Service requests/2012/feb12.ods', engine = "odf", header=4)
mar12 = pd.read_excel('./2. Data./Floods - Service requests/2012/mar12.ods', engine = "odf", header=4)
apr12 = pd.read_excel('./2. Data./Floods - Service requests/2012/apr12.ods', engine = "odf", header=4)
may12 = pd.read_excel('./2. Data./Floods - Service requests/2012/may12.ods', engine = "odf", header=4)
jun12 = pd.read_excel('./2. Data./Floods - Service requests/2012/jun12.ods', engine = "odf", header=4)
jul12 = pd.read_excel('./2. Data./Floods - Service requests/2012/jul12.ods', engine = "odf", header=4)
aug12 = pd.read_excel('./2. Data./Floods - Service requests/2012/aug12.ods', engine = "odf", header=4)
sep12 = pd.read_excel('./2. Data./Floods - Service requests/2012/sep12.ods', engine = "odf", header=4)
oct12 = pd.read_excel('./2. Data./Floods - Service requests/2012/oct12.ods', engine = "odf", header=4)
nov12 = pd.read_excel('./2. Data./Floods - Service requests/2012/nov12.ods', engine = "odf", header=4)
dec12 = pd.read_excel('./2. Data./Floods - Service requests/2012/dec12.ods', engine = "odf", header=4)

#Append and save
df12 = pd.DataFrame().append(jan12).append(feb12).append(mar12).append(apr12).append(may12).append(jun12).append(jul12).append(aug12).append(sep12).append(oct12).append(nov12).append(dec12)
feather.write_dataframe(df12, './2. Data./Floods - Service requests/datasets/df12.feather')

# %% 2013

#Import 2013 data
jan13 = pd.read_excel('./2. Data./Floods - Service requests/2013/01-13.ods', engine = "odf", header=4)
feb13 = pd.read_excel('./2. Data./Floods - Service requests/2013/02-13.ods', engine = "odf", header=4)
mar13 = pd.read_excel('./2. Data./Floods - Service requests/2013/03-13.ods', engine = "odf", header=4)
apr13 = pd.read_excel('./2. Data./Floods - Service requests/2013/04-13.ods', engine = "odf", header=4)
may13 = pd.read_excel('./2. Data./Floods - Service requests/2013/05-13.ods', engine = "odf", header=4)
jun13 = pd.read_excel('./2. Data./Floods - Service requests/2013/06-13.ods', engine = "odf", header=4)
jul13 = pd.read_excel('./2. Data./Floods - Service requests/2013/07-13.ods', engine = "odf", header=4)
aug13 = pd.read_excel('./2. Data./Floods - Service requests/2013/08-13.ods', engine = "odf", header=4)
sep13 = pd.read_excel('./2. Data./Floods - Service requests/2013/09-13.ods', engine = "odf", header=4)
oct13 = pd.read_excel('./2. Data./Floods - Service requests/2013/10-13.ods', engine = "odf", header=4)
nov13 = pd.read_excel('./2. Data./Floods - Service requests/2013/11-13.ods', engine = "odf", header=4)
dec13 = pd.read_excel('./2. Data./Floods - Service requests/2013/12-13.ods', engine = "odf", header=4)

#Append and save
df13 = pd.DataFrame().append(jan13).append(feb13).append(mar13).append(apr13).append(may13).append(jun13).append(jul13).append(aug13).append(sep13).append(oct13).append(nov13).append(dec13)
feather.write_dataframe(df13, './2. Data./Floods - Service requests/datasets/df13.feather')

# %% 2014

#Import 2014 data
jan14 = pd.read_excel('./2. Data./Floods - Service requests/2014/01-14.ods', engine = "odf", header=4)
feb14 = pd.read_excel('./2. Data./Floods - Service requests/2014/02-14.ods', engine = "odf", header=4)
mar14 = pd.read_excel('./2. Data./Floods - Service requests/2014/03-14.ods', engine = "odf", header=4)
apr14 = pd.read_excel('./2. Data./Floods - Service requests/2014/04-14.ods', engine = "odf", header=4)
may14 = pd.read_excel('./2. Data./Floods - Service requests/2014/05-14.ods', engine = "odf", header=4)
jun14 = pd.read_excel('./2. Data./Floods - Service requests/2014/06-14.ods', engine = "odf", header=4)
jul14 = pd.read_excel('./2. Data./Floods - Service requests/2014/07-14.ods', engine = "odf", header=4)
aug14 = pd.read_excel('./2. Data./Floods - Service requests/2014/08-14.ods', engine = "odf", header=4)
sep14 = pd.read_excel('./2. Data./Floods - Service requests/2014/09-14.ods', engine = "odf", header=4)
oct14 = pd.read_excel('./2. Data./Floods - Service requests/2014/10-14.ods', engine = "odf", header=4)
nov14 = pd.read_excel('./2. Data./Floods - Service requests/2014/11-14.ods', engine = "odf", header=4)
dec14 = pd.read_excel('./2. Data./Floods - Service requests/2014/12-14.ods', engine = "odf", header=4)

#Append and save
df14 = pd.DataFrame().append(jan14).append(feb14).append(mar14).append(apr14).append(may14).append(jun14).append(jul14).append(aug14).append(sep14).append(oct14).append(nov14).append(dec14)
feather.write_dataframe(df14, './2. Data./Floods - Service requests/datasets/df14.feather')

# %% 2015

#Import 2015 data
jan15 = pd.read_excel('./2. Data./Floods - Service requests/2015/01-15.ods', engine = "odf", header=4)
feb15 = pd.read_excel('./2. Data./Floods - Service requests/2015/02-15.ods', engine = "odf", header=4)
mar15 = pd.read_excel('./2. Data./Floods - Service requests/2015/03-15.ods', engine = "odf", header=4)
apr15 = pd.read_excel('./2. Data./Floods - Service requests/2015/04-15.ods', engine = "odf", header=4)
may15 = pd.read_excel('./2. Data./Floods - Service requests/2015/05-15.ods', engine = "odf", header=4)
jun15 = pd.read_excel('./2. Data./Floods - Service requests/2015/06-15.ods', engine = "odf", header=4)
jul15 = pd.read_excel('./2. Data./Floods - Service requests/2015/07-15.ods', engine = "odf", header=4)
aug15 = pd.read_excel('./2. Data./Floods - Service requests/2015/08-15.ods', engine = "odf", header=4)
sep15 = pd.read_excel('./2. Data./Floods - Service requests/2015/09-15.ods', engine = "odf", header=4)
oct15 = pd.read_excel('./2. Data./Floods - Service requests/2015/10-15.ods', engine = "odf", header=4)
nov15 = pd.read_excel('./2. Data./Floods - Service requests/2015/11-15.ods', engine = "odf", header=4)
dec15 = pd.read_excel('./2. Data./Floods - Service requests/2015/12-15.ods', engine = "odf", header=4)

#Append and save
df15 = pd.DataFrame().append(jan15).append(feb15).append(mar15).append(apr15).append(may15).append(jun15).append(jul15).append(aug15).append(sep15).append(oct15).append(nov15).append(dec15)
feather.write_dataframe(df15, './2. Data./Floods - Service requests/datasets/df15.feather')

# %% 2016

#Import 2016 data
jan16 = pd.read_excel('./2. Data./Floods - Service requests/2016/01-16.ods', engine = "odf", header=4)
feb16 = pd.read_excel('./2. Data./Floods - Service requests/2016/02-16.ods', engine = "odf", header=4)
mar16 = pd.read_excel('./2. Data./Floods - Service requests/2016/03-16.ods', engine = "odf", header=4)
apr16 = pd.read_excel('./2. Data./Floods - Service requests/2016/04-16.ods', engine = "odf", header=4)
may16 = pd.read_excel('./2. Data./Floods - Service requests/2016/05-16.ods', engine = "odf", header=4)
jun16 = pd.read_excel('./2. Data./Floods - Service requests/2016/06-16.ods', engine = "odf", header=4)
jul16 = pd.read_excel('./2. Data./Floods - Service requests/2016/07-16.ods', engine = "odf", header=4)
aug16 = pd.read_excel('./2. Data./Floods - Service requests/2016/15-08-16.ods', engine = "odf", header=4)

#Append and save
df16 = pd.DataFrame().append(jan16).append(feb16).append(mar16).append(apr16).append(may16).append(jun16).append(jul16).append(aug16)
feather.write_dataframe(df16, './2. Data./Floods - Service requests/datasets/df16.feather')

# %% 2017

#Import 2017 data
jan17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests January 2017.ods', engine = "odf", header=4)
feb17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests February 2017.ods', engine = "odf", header=4)
mar17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests March 2017.ods', engine = "odf", header=4)
apr17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests April 2017.ods', engine = "odf", header=4)
may17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests May 2017.ods', engine = "odf", header=4)
jun17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests June 2017.ods', engine = "odf", header=4)
jul17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests July 2017.ods', engine = "odf", header=4)
aug17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests August 2017.ods', engine = "odf", header=4)
sep17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests September 2017.ods', engine = "odf", header=4)
oct17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests October 2017.ods', engine = "odf", header=4)
nov17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests November 2017.ods', engine = "odf", header=4)
dec17 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2017 to December 2017/Service requests December 2017.ods', engine = "odf", header=4)

#Append and save
df17 = pd.DataFrame().append(jan17).append(feb17).append(mar17).append(apr17).append(may17).append(jun17).append(jul17).append(aug17).append(sep17).append(oct17).append(nov17).append(dec17)
feather.write_dataframe(df17, './2. Data./Floods - Service requests/datasets/df17.feather')

# %% 2018

#Import 2018 data
jan18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests January 2018.ods', engine = "odf", header=4)
feb18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests February 2018.ods', engine = "odf", header=4)
mar18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests March 2018.ods', engine = "odf", header=4)
apr18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests April 2018.ods', engine = "odf", header=4)
may18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests May 2018.ods', engine = "odf", header=4)
jun18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests June 2018.ods', engine = "odf", header=4)
jul18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests July 2018.ods', engine = "odf", header=4)
aug18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests August 2018.ods', engine = "odf", header=4)
sep18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests September 2018.ods', engine = "odf", header=4)
oct18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests October 2018.ods', engine = "odf", header=4)
nov18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests November 2018.ods', engine = "odf", header=4)
dec18 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2018 to December 2018/Service requests December 2018.ods', engine = "odf", header=4)

#Append and save
df18 = pd.DataFrame().append(jan18).append(feb18).append(mar18).append(apr18).append(may18).append(jun18).append(jul18).append(aug18).append(sep18).append(oct18).append(nov18).append(dec18)
feather.write_dataframe(df18, './2. Data./Floods - Service requests/datasets/df18.feather')

# %% 2019

#Import 2019 data
jan19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests January 2019.ods', engine = "odf", header=4)
feb19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests February 2019.ods', engine = "odf", header=4)
mar19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests March 2019.ods', engine = "odf", header=4)
apr19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests April 2019.ods', engine = "odf", header=4)
may19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests May 2019.ods', engine = "odf", header=4)
jun19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests June 2019.ods', engine = "odf", header=4)
jul19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests July 2019.ods', engine = "odf", header=4)
aug19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests August 2019.ods', engine = "odf", header=4)
sep19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests September 2019.ods', engine = "odf", header=4)
oct19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests October 2019.ods', engine = "odf", header=4)
nov19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests November 2019.ods', engine = "odf", header=4)
dec19 = pd.read_excel('./2. Data./Floods - Service requests/Service requests January 2019 to December 2019/Service requests December 2019.ods', engine = "odf", header=4)

#Append and save
df19 = pd.DataFrame().append(jan19).append(feb19).append(mar19).append(apr19).append(may19).append(jun19).append(jul19).append(aug19).append(sep19).append(oct19).append(nov19).append(dec19)
feather.write_dataframe(df19, './2. Data./Floods - Service requests/datasets/df19.feather')




df11bis = feather.read_dataframe('./2. Data./Floods - Service requests/datasets/df11.feather')