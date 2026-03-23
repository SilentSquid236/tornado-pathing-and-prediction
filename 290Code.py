#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:24:35 2026

@author: silentsquid236
"""

import pandas as pd
import numpy as np
import math
import os, sys, glob
import matplotlib.pyplot as plt

# Pulling the CSV directly from the web
tDF = pd.read_csv("https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv")

# Filter for the 30-year period (1993-2023) and magnitudes >= 3
# Note: Used the '&' operator to combine conditions so it returns a single DataFrame
fTDF = tDF[(tDF['yr'] >= 1993) & (tDF['yr'] <= 2023) & (tDF['mag'] >= 3)].dropna(subset=['mag', 'len'])

# Extracting data (using 'path_len' instead of 'len' to protect Python's built-in len() function)
path_len = fTDF["len"]
mag = fTDF["mag"]

# Calculate the regression line variables (b = slope, a = intercept)
b, a = np.polyfit(mag, path_len, 1)
r = mag.corr(path_len)
print(r)

# Generate points to draw the regression line
mag_vals = np.array([3, 4, 5]) 
predicted_len = b * mag_vals + a

fig = plt.figure(figsize = (10,10))
plt.scatter(mag, path_len)
plt.xlabel("Tornado Magnitude (EF Scale)", fontweight = "bold", fontsize = 24)
plt.ylabel("Path Length (Miles)", fontweight = "bold", fontsize = 24)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
plt.title('Magnitude vs Path Length (1993-2023)', fontweight = "bold", fontsize = 26)


plt.plot(mag_vals, predicted_len, linewidth = 4, label="Regression: y = %1.3fx + " %b + "%1.3f"%a)

plt.legend(loc = "upper left", fontsize = 22)

# Adjusted limits to fit EF3-EF5 magnitudes and their standard path lengths
plt.xlim(2.5, 5.5)
plt.ylim(0, 160)
plt.grid()
plt.savefig("Magnitude_vs_PathLength_93_23.png", dpi = 300)