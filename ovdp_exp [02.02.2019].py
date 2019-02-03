import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize as optimization

#for linear model
import math
import datetime
import statsmodels.api as sma                          #for linear model
import statsmodels.formula.api as sm                   #for linear model

#for validation
from sklearn.model_selection import train_test_split

os.chdir("D:/USERS/ROMAN/WORK/Python/Macro motel for OVDP")
#list(moodys_scale_2017)     # names()

moodys_2005 = pd.read_excel("moodys_exp.xlsx", sheetname="m2005")
moodys_2006 = pd.read_excel("moodys_exp.xlsx", sheetname="m2006")
moodys_2007 = pd.read_excel("moodys_exp.xlsx", sheetname="m2007")
moodys_2008 = pd.read_excel("moodys_exp.xlsx", sheetname="m2008")
moodys_2009 = pd.read_excel("moodys_exp.xlsx", sheetname="m2009")
moodys_2010 = pd.read_excel("moodys_exp.xlsx", sheetname="m2010")
moodys_2011 = pd.read_excel("moodys_exp.xlsx", sheetname="m2011")
moodys_2012 = pd.read_excel("moodys_exp.xlsx", sheetname="m2012")
moodys_2013 = pd.read_excel("moodys_exp.xlsx", sheetname="m2013")
moodys_2014 = pd.read_excel("moodys_exp.xlsx", sheetname="m2014")
moodys_2015 = pd.read_excel("moodys_exp.xlsx", sheetname="m2015")
moodys_2016 = pd.read_excel("moodys_exp.xlsx", sheetname="m2016")
moodys_2017 = pd.read_excel("moodys_exp.xlsx", sheetname="m2017")

################################################################################################ 2005_year
moodys_scale_2005 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2005"])

moodys_scale_2005["Year_1"] = np.nan
moodys_scale_2005["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2005.iloc[0, 1] = moodys_2005.iloc[0, 1]
moodys_scale_2005.iloc[1, 1] = moodys_2005.iloc[1, 1]
moodys_scale_2005.iloc[2, 1] = moodys_2005.iloc[1, 1]
moodys_scale_2005.iloc[3, 1] = moodys_2005.iloc[1, 1]
moodys_scale_2005.iloc[4, 1] = moodys_2005.iloc[2, 1]
moodys_scale_2005.iloc[5, 1] = moodys_2005.iloc[2, 1]
moodys_scale_2005.iloc[6, 1] = moodys_2005.iloc[2, 1]
moodys_scale_2005.iloc[7, 1] = moodys_2005.iloc[3, 1]
moodys_scale_2005.iloc[8, 1] = moodys_2005.iloc[3, 1]
moodys_scale_2005.iloc[9, 1] = moodys_2005.iloc[3, 1]
moodys_scale_2005.iloc[11, 1] = moodys_2005.iloc[4, 1]
moodys_scale_2005.iloc[14, 1] = moodys_2005.iloc[5, 1]
moodys_scale_2005.iloc[18, 1] = moodys_2005.iloc[6, 1]

moodys_scale_2005drop = moodys_scale_2005.dropna()

y = moodys_scale_2005drop["Year_1"]
x = moodys_scale_2005drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2005["pred"] = np.exp(a + b * moodys_scale_2005["x"])

moodys_scale_2005_for_print = moodys_scale_2005[["Rating_2005", "Year_1", "pred"]]
#moodys_scale_2005_for_print.to_csv("moodys_scale_2005_for_print_exp.csv", index=False)

# plot_2005
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2005")
ax.grid()
ax.plot(moodys_scale_2005["x"], moodys_scale_2005["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2005["x"], moodys_scale_2005["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2005["x"], moodys_scale_2005["Rating_2005"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2006_year
moodys_scale_2006 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2006"])

moodys_scale_2006["Year_1"] = np.nan
moodys_scale_2006["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2006.iloc[0, 1] = moodys_2006.iloc[0, 1]
moodys_scale_2006.iloc[1, 1] = moodys_2006.iloc[1, 1]
moodys_scale_2006.iloc[2, 1] = moodys_2006.iloc[1, 1]
moodys_scale_2006.iloc[3, 1] = moodys_2006.iloc[1, 1]
moodys_scale_2006.iloc[4, 1] = moodys_2006.iloc[2, 1]
moodys_scale_2006.iloc[5, 1] = moodys_2006.iloc[2, 1]
moodys_scale_2006.iloc[6, 1] = moodys_2006.iloc[2, 1]
moodys_scale_2006.iloc[7, 1] = moodys_2006.iloc[3, 1]
moodys_scale_2006.iloc[8, 1] = moodys_2006.iloc[3, 1]
moodys_scale_2006.iloc[9, 1] = moodys_2006.iloc[3, 1]
moodys_scale_2006.iloc[11, 1] = moodys_2006.iloc[4, 1]
moodys_scale_2006.iloc[14, 1] = moodys_2006.iloc[5, 1]
moodys_scale_2006.iloc[18, 1] = moodys_2006.iloc[6, 1]

moodys_scale_2006drop = moodys_scale_2006.dropna()

y = moodys_scale_2006drop["Year_1"]
x = moodys_scale_2006drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2006["pred"] = np.exp(a + b * moodys_scale_2006["x"])

moodys_scale_2006_for_print = moodys_scale_2006[["Rating_2006", "Year_1", "pred"]]
#moodys_scale_2006_for_print.to_csv("moodys_scale_2006_for_print_exp.csv", index=False)

# plot_2006
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2006")
ax.grid()
ax.plot(moodys_scale_2006["x"], moodys_scale_2006["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2006["x"], moodys_scale_2006["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2006["x"], moodys_scale_2006["Rating_2006"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2007_year
moodys_scale_2007 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2007"])

moodys_scale_2007["Year_1"] = np.nan
moodys_scale_2007["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2007.iloc[0, 1] = moodys_2007.iloc[0, 1]
moodys_scale_2007.iloc[1, 1] = moodys_2007.iloc[1, 1]
moodys_scale_2007.iloc[2, 1] = moodys_2007.iloc[1, 1]
moodys_scale_2007.iloc[3, 1] = moodys_2007.iloc[1, 1]
moodys_scale_2007.iloc[4, 1] = moodys_2007.iloc[2, 1]
moodys_scale_2007.iloc[5, 1] = moodys_2007.iloc[2, 1]
moodys_scale_2007.iloc[6, 1] = moodys_2007.iloc[2, 1]
moodys_scale_2007.iloc[7, 1] = moodys_2007.iloc[3, 1]
moodys_scale_2007.iloc[8, 1] = moodys_2007.iloc[3, 1]
moodys_scale_2007.iloc[9, 1] = moodys_2007.iloc[3, 1]
moodys_scale_2007.iloc[11, 1] = moodys_2007.iloc[4, 1]
moodys_scale_2007.iloc[14, 1] = moodys_2007.iloc[5, 1]
moodys_scale_2007.iloc[18, 1] = moodys_2007.iloc[6, 1]

moodys_scale_2007drop = moodys_scale_2007.dropna()

y = moodys_scale_2007drop["Year_1"]
x = moodys_scale_2007drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2007["pred"] = np.exp(a + b * moodys_scale_2007["x"])

moodys_scale_2007_for_print = moodys_scale_2007[["Rating_2007", "Year_1", "pred"]]
#moodys_scale_2007_for_print.to_csv("moodys_scale_2007_for_print_exp.csv", index=False)

# plot_2007
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2007")
ax.grid()
ax.plot(moodys_scale_2007["x"], moodys_scale_2007["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2007["x"], moodys_scale_2007["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2007["x"], moodys_scale_2007["Rating_2007"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2008_year
moodys_scale_2008 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2008"])

moodys_scale_2008["Year_1"] = np.nan
moodys_scale_2008["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2008.iloc[0, 1] = moodys_2008.iloc[0, 1]
moodys_scale_2008.iloc[1, 1] = moodys_2008.iloc[1, 1]
moodys_scale_2008.iloc[2, 1] = moodys_2008.iloc[1, 1]
moodys_scale_2008.iloc[3, 1] = moodys_2008.iloc[1, 1]
moodys_scale_2008.iloc[4, 1] = moodys_2008.iloc[2, 1]
moodys_scale_2008.iloc[5, 1] = moodys_2008.iloc[2, 1]
moodys_scale_2008.iloc[6, 1] = moodys_2008.iloc[2, 1]
moodys_scale_2008.iloc[7, 1] = moodys_2008.iloc[3, 1]
moodys_scale_2008.iloc[8, 1] = moodys_2008.iloc[3, 1]
moodys_scale_2008.iloc[9, 1] = moodys_2008.iloc[3, 1]
moodys_scale_2008.iloc[11, 1] = moodys_2008.iloc[4, 1]
moodys_scale_2008.iloc[14, 1] = moodys_2008.iloc[5, 1]
moodys_scale_2008.iloc[18, 1] = moodys_2008.iloc[6, 1]

moodys_scale_2008drop = moodys_scale_2008.dropna()

y = moodys_scale_2008drop["Year_1"]
x = moodys_scale_2008drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2008["pred"] = np.exp(a + b * moodys_scale_2008["x"])

moodys_scale_2008_for_print = moodys_scale_2008[["Rating_2008", "Year_1", "pred"]]
#moodys_scale_2008_for_print.to_csv("moodys_scale_2008_for_print_exp.csv", index=False)

# plot_2008
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2008")
ax.grid()
ax.plot(moodys_scale_2008["x"], moodys_scale_2008["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2008["x"], moodys_scale_2008["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2008["x"], moodys_scale_2008["Rating_2008"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2009_year
moodys_scale_2009 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2009"])

moodys_scale_2009["Year_1"] = np.nan
moodys_scale_2009["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2009.iloc[0, 1] = moodys_2009.iloc[0, 1]
moodys_scale_2009.iloc[1, 1] = moodys_2009.iloc[1, 1]
moodys_scale_2009.iloc[2, 1] = moodys_2009.iloc[1, 1]
moodys_scale_2009.iloc[3, 1] = moodys_2009.iloc[1, 1]
moodys_scale_2009.iloc[4, 1] = moodys_2009.iloc[2, 1]
moodys_scale_2009.iloc[5, 1] = moodys_2009.iloc[2, 1]
moodys_scale_2009.iloc[6, 1] = moodys_2009.iloc[2, 1]
moodys_scale_2009.iloc[7, 1] = moodys_2009.iloc[3, 1]
moodys_scale_2009.iloc[8, 1] = moodys_2009.iloc[3, 1]
moodys_scale_2009.iloc[9, 1] = moodys_2009.iloc[3, 1]
moodys_scale_2009.iloc[11, 1] = moodys_2009.iloc[4, 1]
moodys_scale_2009.iloc[14, 1] = moodys_2009.iloc[5, 1]
moodys_scale_2009.iloc[18, 1] = moodys_2009.iloc[6, 1]

moodys_scale_2009drop = moodys_scale_2009.dropna()

y = moodys_scale_2009drop["Year_1"]
x = moodys_scale_2009drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2009["pred"] = np.exp(a + b * moodys_scale_2009["x"])

moodys_scale_2009_for_print = moodys_scale_2009[["Rating_2009", "Year_1", "pred"]]
#moodys_scale_2009_for_print.to_csv("moodys_scale_2009_for_print_exp.csv", index=False)

# plot_2009
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2009")
ax.grid()
ax.plot(moodys_scale_2009["x"], moodys_scale_2009["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2009["x"], moodys_scale_2009["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2009["x"], moodys_scale_2009["Rating_2009"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2010_year
moodys_scale_2010 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2010"])

moodys_scale_2010["Year_1"] = np.nan
moodys_scale_2010["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2010.iloc[0, 1] = moodys_2010.iloc[0, 1]
moodys_scale_2010.iloc[1, 1] = moodys_2010.iloc[1, 1]
moodys_scale_2010.iloc[2, 1] = moodys_2010.iloc[1, 1]
moodys_scale_2010.iloc[3, 1] = moodys_2010.iloc[1, 1]
moodys_scale_2010.iloc[4, 1] = moodys_2010.iloc[2, 1]
moodys_scale_2010.iloc[5, 1] = moodys_2010.iloc[2, 1]
moodys_scale_2010.iloc[6, 1] = moodys_2010.iloc[2, 1]
moodys_scale_2010.iloc[7, 1] = moodys_2010.iloc[3, 1]
moodys_scale_2010.iloc[8, 1] = moodys_2010.iloc[3, 1]
moodys_scale_2010.iloc[9, 1] = moodys_2010.iloc[3, 1]
moodys_scale_2010.iloc[11, 1] = moodys_2010.iloc[4, 1]
moodys_scale_2010.iloc[14, 1] = moodys_2010.iloc[5, 1]
moodys_scale_2010.iloc[18, 1] = moodys_2010.iloc[6, 1]

moodys_scale_2010drop = moodys_scale_2010.dropna()

y = moodys_scale_2010drop["Year_1"]
x = moodys_scale_2010drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2010["pred"] = np.exp(a + b * moodys_scale_2010["x"])

moodys_scale_2010_for_print = moodys_scale_2010[["Rating_2010", "Year_1", "pred"]]
#moodys_scale_2010_for_print.to_csv("moodys_scale_2010_for_print_exp.csv", index=False)

# plot_2010
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2010")
ax.grid()
ax.plot(moodys_scale_2010["x"], moodys_scale_2010["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2010["x"], moodys_scale_2010["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2010["x"], moodys_scale_2010["Rating_2010"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2011_year
moodys_scale_2011 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2011"])

moodys_scale_2011["Year_1"] = np.nan
moodys_scale_2011["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2011.iloc[0, 1] = moodys_2011.iloc[0, 1]
moodys_scale_2011.iloc[1, 1] = moodys_2011.iloc[1, 1]
moodys_scale_2011.iloc[2, 1] = moodys_2011.iloc[1, 1]
moodys_scale_2011.iloc[3, 1] = moodys_2011.iloc[1, 1]
moodys_scale_2011.iloc[4, 1] = moodys_2011.iloc[2, 1]
moodys_scale_2011.iloc[5, 1] = moodys_2011.iloc[2, 1]
moodys_scale_2011.iloc[6, 1] = moodys_2011.iloc[2, 1]
moodys_scale_2011.iloc[7, 1] = moodys_2011.iloc[3, 1]
moodys_scale_2011.iloc[8, 1] = moodys_2011.iloc[3, 1]
moodys_scale_2011.iloc[9, 1] = moodys_2011.iloc[3, 1]
moodys_scale_2011.iloc[11, 1] = moodys_2011.iloc[4, 1]
moodys_scale_2011.iloc[14, 1] = moodys_2011.iloc[5, 1]
moodys_scale_2011.iloc[18, 1] = moodys_2011.iloc[6, 1]

moodys_scale_2011drop = moodys_scale_2011.dropna()

y = moodys_scale_2011drop["Year_1"]
x = moodys_scale_2011drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2011["pred"] = np.exp(a + b * moodys_scale_2011["x"])

moodys_scale_2011_for_print = moodys_scale_2011[["Rating_2011", "Year_1", "pred"]]
#moodys_scale_2011_for_print.to_csv("moodys_scale_2011_for_print_exp.csv", index=False)

# plot_2011
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2011")
ax.grid()
ax.plot(moodys_scale_2011["x"], moodys_scale_2011["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2011["x"], moodys_scale_2011["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2011["x"], moodys_scale_2011["Rating_2011"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2012_year
moodys_scale_2012 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2012"])

moodys_scale_2012["Year_1"] = np.nan
moodys_scale_2012["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2012.iloc[0, 1] = moodys_2012.iloc[0, 1]
moodys_scale_2012.iloc[1, 1] = moodys_2012.iloc[1, 1]
moodys_scale_2012.iloc[2, 1] = moodys_2012.iloc[1, 1]
moodys_scale_2012.iloc[3, 1] = moodys_2012.iloc[1, 1]
moodys_scale_2012.iloc[4, 1] = moodys_2012.iloc[2, 1]
moodys_scale_2012.iloc[5, 1] = moodys_2012.iloc[2, 1]
moodys_scale_2012.iloc[6, 1] = moodys_2012.iloc[2, 1]
moodys_scale_2012.iloc[7, 1] = moodys_2012.iloc[3, 1]
moodys_scale_2012.iloc[8, 1] = moodys_2012.iloc[3, 1]
moodys_scale_2012.iloc[9, 1] = moodys_2012.iloc[3, 1]
moodys_scale_2012.iloc[11, 1] = moodys_2012.iloc[4, 1]
moodys_scale_2012.iloc[14, 1] = moodys_2012.iloc[5, 1]
moodys_scale_2012.iloc[18, 1] = moodys_2012.iloc[6, 1]

moodys_scale_2012drop = moodys_scale_2012.dropna()

y = moodys_scale_2012drop["Year_1"]
x = moodys_scale_2012drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2012["pred"] = np.exp(a + b * moodys_scale_2012["x"])

moodys_scale_2012_for_print = moodys_scale_2012[["Rating_2012", "Year_1", "pred"]]
#moodys_scale_2012_for_print.to_csv("moodys_scale_2012_for_print_exp.csv", index=False)

# plot_2012
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2012")
ax.grid()
ax.plot(moodys_scale_2012["x"], moodys_scale_2012["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2012["x"], moodys_scale_2012["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2012["x"], moodys_scale_2012["Rating_2012"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2013_year
moodys_scale_2013 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2013"])

moodys_scale_2013["Year_1"] = np.nan
moodys_scale_2013["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2013.iloc[0, 1] = moodys_2013.iloc[0, 1]
moodys_scale_2013.iloc[1, 1] = moodys_2013.iloc[1, 1]
moodys_scale_2013.iloc[2, 1] = moodys_2013.iloc[1, 1]
moodys_scale_2013.iloc[3, 1] = moodys_2013.iloc[1, 1]
moodys_scale_2013.iloc[4, 1] = moodys_2013.iloc[2, 1]
moodys_scale_2013.iloc[5, 1] = moodys_2013.iloc[2, 1]
moodys_scale_2013.iloc[6, 1] = moodys_2013.iloc[2, 1]
moodys_scale_2013.iloc[7, 1] = moodys_2013.iloc[3, 1]
moodys_scale_2013.iloc[8, 1] = moodys_2013.iloc[3, 1]
moodys_scale_2013.iloc[9, 1] = moodys_2013.iloc[3, 1]
moodys_scale_2013.iloc[11, 1] = moodys_2013.iloc[4, 1]
moodys_scale_2013.iloc[14, 1] = moodys_2013.iloc[5, 1]
moodys_scale_2013.iloc[18, 1] = moodys_2013.iloc[6, 1]

moodys_scale_2013drop = moodys_scale_2013.dropna()

y = moodys_scale_2013drop["Year_1"]
x = moodys_scale_2013drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2013["pred"] = np.exp(a + b * moodys_scale_2013["x"])

moodys_scale_2013_for_print = moodys_scale_2013[["Rating_2013", "Year_1", "pred"]]
#moodys_scale_2013_for_print.to_csv("moodys_scale_2013_for_print_exp.csv", index=False)

# plot_2013
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2013")
ax.grid()
ax.plot(moodys_scale_2013["x"], moodys_scale_2013["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2013["x"], moodys_scale_2013["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2013["x"], moodys_scale_2013["Rating_2013"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2014_year
moodys_scale_2014 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2014"])

moodys_scale_2014["Year_1"] = np.nan
moodys_scale_2014["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2014.iloc[0, 1] = moodys_2014.iloc[0, 1]
moodys_scale_2014.iloc[1, 1] = moodys_2014.iloc[1, 1]
moodys_scale_2014.iloc[2, 1] = moodys_2014.iloc[1, 1]
moodys_scale_2014.iloc[3, 1] = moodys_2014.iloc[1, 1]
moodys_scale_2014.iloc[4, 1] = moodys_2014.iloc[2, 1]
moodys_scale_2014.iloc[5, 1] = moodys_2014.iloc[2, 1]
moodys_scale_2014.iloc[6, 1] = moodys_2014.iloc[2, 1]
moodys_scale_2014.iloc[7, 1] = moodys_2014.iloc[3, 1]
moodys_scale_2014.iloc[8, 1] = moodys_2014.iloc[3, 1]
moodys_scale_2014.iloc[9, 1] = moodys_2014.iloc[3, 1]
moodys_scale_2014.iloc[11, 1] = moodys_2014.iloc[4, 1]
moodys_scale_2014.iloc[14, 1] = moodys_2014.iloc[5, 1]
moodys_scale_2014.iloc[18, 1] = moodys_2014.iloc[6, 1]

moodys_scale_2014drop = moodys_scale_2014.dropna()

y = moodys_scale_2014drop["Year_1"]
x = moodys_scale_2014drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2014["pred"] = np.exp(a + b * moodys_scale_2014["x"])

moodys_scale_2014_for_print = moodys_scale_2014[["Rating_2014", "Year_1", "pred"]]
#moodys_scale_2014_for_print.to_csv("moodys_scale_2014_for_print_exp.csv", index=False)

# plot_2014
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2014")
ax.grid()
ax.plot(moodys_scale_2014["x"], moodys_scale_2014["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2014["x"], moodys_scale_2014["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2014["x"], moodys_scale_2014["Rating_2014"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2015_year
moodys_scale_2015 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2015"])

moodys_scale_2015["Year_1"] = np.nan
moodys_scale_2015["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2015.iloc[0, 1] = moodys_2015.iloc[0, 1]
moodys_scale_2015.iloc[1, 1] = moodys_2015.iloc[1, 1]
moodys_scale_2015.iloc[2, 1] = moodys_2015.iloc[1, 1]
moodys_scale_2015.iloc[3, 1] = moodys_2015.iloc[1, 1]
moodys_scale_2015.iloc[4, 1] = moodys_2015.iloc[2, 1]
moodys_scale_2015.iloc[5, 1] = moodys_2015.iloc[2, 1]
moodys_scale_2015.iloc[6, 1] = moodys_2015.iloc[2, 1]
moodys_scale_2015.iloc[7, 1] = moodys_2015.iloc[3, 1]
moodys_scale_2015.iloc[8, 1] = moodys_2015.iloc[3, 1]
moodys_scale_2015.iloc[9, 1] = moodys_2015.iloc[3, 1]
moodys_scale_2015.iloc[11, 1] = moodys_2015.iloc[4, 1]
moodys_scale_2015.iloc[14, 1] = moodys_2015.iloc[5, 1]
moodys_scale_2015.iloc[18, 1] = moodys_2015.iloc[6, 1]

moodys_scale_2015drop = moodys_scale_2015.dropna()

y = moodys_scale_2015drop["Year_1"]
x = moodys_scale_2015drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2015["pred"] = np.exp(a + b * moodys_scale_2015["x"])

moodys_scale_2015_for_print = moodys_scale_2015[["Rating_2015", "Year_1", "pred"]]
#moodys_scale_2015_for_print.to_csv("moodys_scale_2015_for_print_exp.csv", index=False)

# plot_2015
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2015")
ax.grid()
ax.plot(moodys_scale_2015["x"], moodys_scale_2015["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2015["x"], moodys_scale_2015["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2015["x"], moodys_scale_2015["Rating_2015"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2016_year
moodys_scale_2016 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2016"])

moodys_scale_2016["Year_1"] = np.nan
moodys_scale_2016["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2016.iloc[0, 1] = moodys_2016.iloc[0, 1]
moodys_scale_2016.iloc[1, 1] = moodys_2016.iloc[1, 1]
moodys_scale_2016.iloc[2, 1] = moodys_2016.iloc[1, 1]
moodys_scale_2016.iloc[3, 1] = moodys_2016.iloc[1, 1]
moodys_scale_2016.iloc[4, 1] = moodys_2016.iloc[2, 1]
moodys_scale_2016.iloc[5, 1] = moodys_2016.iloc[2, 1]
moodys_scale_2016.iloc[6, 1] = moodys_2016.iloc[2, 1]
moodys_scale_2016.iloc[7, 1] = moodys_2016.iloc[3, 1]
moodys_scale_2016.iloc[8, 1] = moodys_2016.iloc[3, 1]
moodys_scale_2016.iloc[9, 1] = moodys_2016.iloc[3, 1]
moodys_scale_2016.iloc[11, 1] = moodys_2016.iloc[4, 1]
moodys_scale_2016.iloc[14, 1] = moodys_2016.iloc[5, 1]
moodys_scale_2016.iloc[18, 1] = moodys_2016.iloc[6, 1]

moodys_scale_2016drop = moodys_scale_2016.dropna()

y = moodys_scale_2016drop["Year_1"]
x = moodys_scale_2016drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2016["pred"] = np.exp(a + b * moodys_scale_2016["x"])

moodys_scale_2016_for_print = moodys_scale_2016[["Rating_2016", "Year_1", "pred"]]
#moodys_scale_2016_for_print.to_csv("moodys_scale_2016_for_print_exp.csv", index=False)

# plot_2016
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2016")
ax.grid()
ax.plot(moodys_scale_2016["x"], moodys_scale_2016["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2016["x"], moodys_scale_2016["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2016["x"], moodys_scale_2016["Rating_2016"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################ 2017_year
moodys_scale_2017 = pd.DataFrame([["Aaa"], ["Aa1"], ["Aa2"], ["Aa3"], ["A1"], ["A2"], ["A3"],
                                  ["Baa1"], ["Baa2"], ["Baa3"], ["Ba1"], ["Ba2"], ["Ba3"], ["B1"], ["B2"], ["B3"],
                                  ["Caa1"], ["Caa2"], ["Caa3"], ["Ca"], ["C"]], columns = ["Rating_2017"])

moodys_scale_2017["Year_1"] = np.nan
moodys_scale_2017["x"] = np.linspace(start = 1, stop = 21, num = 21)

moodys_scale_2017.iloc[0, 1] = moodys_2017.iloc[0, 1]
moodys_scale_2017.iloc[1, 1] = moodys_2017.iloc[1, 1]
moodys_scale_2017.iloc[2, 1] = moodys_2017.iloc[1, 1]
moodys_scale_2017.iloc[3, 1] = moodys_2017.iloc[1, 1]
moodys_scale_2017.iloc[4, 1] = moodys_2017.iloc[2, 1]
moodys_scale_2017.iloc[5, 1] = moodys_2017.iloc[2, 1]
moodys_scale_2017.iloc[6, 1] = moodys_2017.iloc[2, 1]
moodys_scale_2017.iloc[7, 1] = moodys_2017.iloc[3, 1]
moodys_scale_2017.iloc[8, 1] = moodys_2017.iloc[3, 1]
moodys_scale_2017.iloc[9, 1] = moodys_2017.iloc[3, 1]
moodys_scale_2017.iloc[11, 1] = moodys_2017.iloc[4, 1]
moodys_scale_2017.iloc[14, 1] = moodys_2017.iloc[5, 1]
moodys_scale_2017.iloc[18, 1] = moodys_2017.iloc[6, 1]

moodys_scale_2017drop = moodys_scale_2017.dropna()

y = moodys_scale_2017drop["Year_1"]
x = moodys_scale_2017drop["x"]
start = np.array([0, 0])

def f(x, a, b):
    fx = np.exp(a + b * x)
    return fx

model = optimization.curve_fit(f, x, y, start)
arra = model[0]
a = arra[0]
b = arra[1]

moodys_scale_2017["pred"] = np.exp(a + b * moodys_scale_2017["x"])

moodys_scale_2017_for_print = moodys_scale_2017[["Rating_2017", "Year_1", "pred"]]
#moodys_scale_2017_for_print.to_csv("moodys_scale_2017_for_print_exp.csv", index=False)

# plot_2017
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Ratings")
ax.set_ylabel("PD, %")
ax.set_title("Rating 2017")
ax.grid()
ax.plot(moodys_scale_2017["x"], moodys_scale_2017["Year_1"]*100, 'bo', label = 'Moody"s rate')
ax.plot(moodys_scale_2017["x"], moodys_scale_2017["pred"]*100, 'g-', label = 'smoothing')
plt.xticks(moodys_scale_2017["x"], moodys_scale_2017["Rating_2017"], rotation='horizontal')
ax.legend(loc=0)

################################################################################################

#linear model

base = pd.read_excel("moodys_exp.xlsx", sheetname="baze_ukrstat", skiprows=0)
#base = base[(base["date"]>=datetime.date(2006,4,1)) & (base["date"]<datetime.date(2019,1,1))]

base = base[(base["date"]>=datetime.date(2010,7,1)) & (base["date"]<datetime.date(2019,1,1))]    #test1
base = base[["date", "Rate", "PD_Moodys",
             "nomGDPkk", "nomGDPkk_lag3", "nomGDPkk_lag6", "nomGDPkk_lag9", "nomGDPkk_lag12",
             "realGDPkk", "realGDPkk_lag3", "realGDPkk_lag6", "realGDPkk_lag9", "realGDPkk_lag12"]]
corr_matrix = base.corr(method='pearson')
corr_matrix[["PD_Moodys"]]
corr_matrix[["PD_Moodys"]][1:].abs().max()                      #0.899    34obs.
#math.sqrt(corr_matrix[["PD_Moodys"]][1:].abs().max())

x = base[["realGDPkk_lag3"]]
y = base[["PD_Moodys"]]
x = sma.add_constant(x)
lm2 = sm.OLS(y,x).fit()
lm2.summary()
lm2.rsquared                                                    #0.80
lm2.params


base["DR_pred"] = lm2.predict(x)
#base.to_csv("base_pred2Y.csv", index=False)

##log#######################################
#base["PD_Moodys_log"] = np.log(base["PD_Moodys"])
#x = base[["realGDPkk_lag3"]]
#y = base[["PD_Moodys_log"]]
#x = sma.add_constant(x)
#lm2 = sm.OLS(y,x).fit()
#lm2.summary()
#lm2.rsquared                                                    #0.62
#base["DR_pred_log"] = lm2.predict(x)
#base["DR_pred_log"] = np.exp(base["DR_pred_log"])
############################################

#plot------------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)
ax.set_xlabel("Dates")
ax.set_ylabel("Default Rate")
ax.set_title("Default Rates")
ax.plot(base["date"], base["PD_Moodys"], "b", label = 'PD_Moodys')           #fact
ax.plot(base["date"], base["DR_pred"], "g", label = 'pred')                  #pred
fig.autofmt_xdate()                                                          #norm x_axis
ax.legend(loc=0)
#ax.grid()
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])  #persentage format
fig.show()
#----------------------------------------------------------------------------------
#np.random.choice([1, 2, 3, 4], size=10, replace=True, p=None)
###################################################################################

# VALIDATION    points-out-of-time

#fold1
base_I_train = base[:26]
base_I_testt = base[26:]

x1 = base_I_train[["realGDPkk_lag3"]]
y1 = base_I_train[["PD_Moodys"]]
x1 = sma.add_constant(x1)
lm21 = sm.OLS(y1,x1).fit()
lm21.summary()
lm21.rsquared                      #0.819
lm21.params

x11 = base_I_testt[["realGDPkk_lag3"]]
x11 = sma.add_constant(x11)
base_I_testt["DR_pred_f1"] = lm21.predict(x11)

base_I_testt[["PD_Moodys", "DR_pred_f1"]].corr()**2       #R2 == 0.105521

###################################################################################

# VALIDATION    bootstrap
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

dat = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


train, test = train_test_split(base, test_size = 0.30, random_state = 1111)



















