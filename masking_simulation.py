#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:09:21 2018

@author: edf
"""

#%%
###############################################################################
#                               PACKAGE IMPORTS                               # 
############################################################################### 

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

#%%
###############################################################################
#                            DATA GENERATOR FUNCTION                          # 
############################################################################### 

def GenerateData(seed, n, mu, sigma):
    np.random.seed(seed)
    data = np.random.multivariate_normal(mu, sigma, (n), check_valid='ignore')
    data[:,0] = np.round(data[:,0])
    data[data[:,0] <= 0] = 1
    data[:,1] = np.power(100, data[:,1])
    print('Data Generated!')
    return data

#%%
###############################################################################
#                            MASKING CHECK                                    # 
############################################################################### 

def IsMasked(data, userCat):
    masked = False
    subset = data[data[:,0] == userCat,:]
    threshold = 0.15 * subset[:,1].sum()
    for j in range(subset.shape[0]):
        if (subset.shape[0] < 15):
            masked = True
        elif (subset[j,1] > threshold):
            masked = True
    return masked

#%%
###############################################################################
#                       LARGE USER EXTRACTION FUNCTION                        # 
############################################################################### 

def ExtractLargeUser(data, largeUserCat):
    lucat = list(largeUserCat.keys())[0]
    ind = data[:,0] != lucat
    subset = data[ind,:]
    threshold = 0.15 * subset[:,1].sum()
    for i in range(data.shape[0]):
        if (data[i,0] != lucat) & (data[i,1] > threshold):
            data[i,0] = lucat
            break
    return data

#%%
###############################################################################
#                 RECURSIVE LARGE USER EXTRACTION FUNCTION                    # 
############################################################################### 
    
def RecursiveExtractLargeUser(data, categories, userCat, largeUserCat):
    while IsMasked(data, userCat):
        data = ExtractLargeUser(data, largeUserCat)
        print('Large ' + categories[userCat] + ' User Extracted!')
    return data

#%%
###############################################################################
#          PER CATEGORY LARGE USER EXTRACTION INTERATION FUNCTION             # 
############################################################################### 

def LargeUserCategoryExtraction(data, categories, largeUserCat):
    data = data[data[:,1].argsort()[::-1]]
    for userCat in categories.keys():
        ind = data[:,0] == userCat
        subset_data = data[ind]
        if IsMasked(subset_data, userCat):
            subset_data = RecursiveExtractLargeUser(subset_data, 
                                                    categories, 
                                                    userCat, 
                                                    largeUserCat)
            data[ind] = subset_data
    return data

#%%
###############################################################################
#                           EXTRACT FILL USER FUNCTION                        # 
############################################################################### 
    
def ExtractFillUser(data, largeUserCat):
    lucat = list(largeUserCat.keys())[0]
    for i in range(data.shape[0]):
        if (data[i,0] != lucat):
            userCat = data[i,0]
            data[i,0] = lucat
            break
    return data, userCat

#%%
###############################################################################
#                   RECURSIVE EXTRACT FILL USER FUNCTION                      # 
############################################################################### 

def RecursiveExtractFillUser(data, categories, largeUserCat):
    lucat = list(largeUserCat.keys())[0]
    while IsMasked(data, lucat):
        data, userCat = ExtractFillUser(data, largeUserCat)  
        print(categories[userCat] + ' User Assigned to Fill Large User Category!')
    return data
    
#%%
###############################################################################
#                 PER CATEGORY EXTRACT FILL FUNCTION                          # 
############################################################################### 

def LargeUserCategoryFill(data, categories, largeUserCat):
    data = data[data[:,1].argsort()[::-1]]
    lucat = list(largeUserCat.keys())[0]
    if IsMasked(data, lucat):
        data = RecursiveExtractFillUser(data, categories, largeUserCat)
    return data

#%%
###############################################################################
#                          EXTRACT AND FILL FUNCTION                          # 
############################################################################### 
    
def ExtractAndFill(data, categories, largeUserCat):
    data = LargeUserCategoryExtraction(data, categories, largeUserCat)
    data = LargeUserCategoryFill(data, categories, largeUserCat)
    return data

#%%
###############################################################################
#                           DATA PLOT FUNCTION                                # 
############################################################################### 
    
def DataPlot(data, categories, largeUserCat):
    cmap = 'jet'
    lucat = list(largeUserCat.keys())[0]
    ludata = data[data[:,0] == lucat,:]
    data = data[data[:,0] != lucat,:]
    c = data[:,1]
    plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_title('Consumption by User Category')
    ax0.scatter(data[:,0], data[:,1], c=c, cmap=cmap)
    ax0.set_xticks(list(categories.keys()))
    ax0.set_xticklabels(list(categories.values()),rotation=-45)
    ax0.grid(True)
    ax0.set_ylabel('Consumption (kwh)')
    ax0.set_xlabel('User Categories')
    if ludata.shape[0] == 0:
        c = 0
        ax1.scatter(lucat, 0, c=c, cmap=cmap)
        ax1.set_ylim(0,data[:,1].max())
    else:
        c = ludata[:,1]
        ax1.scatter(ludata[:,0], ludata[:,1], c=c, cmap=cmap)
        ax1.set_ylim(0,ludata[:,1].max())
    ax1.set_xticks([lucat])
    ax1.set_xticklabels(['large_users'],rotation=-45)
    ax1.grid(True)
    plt.show()
    print('')
    return 

#%%
###############################################################################
#                           DATA STATISTICS FUNCTION                          # 
############################################################################### 
    
def DataStatistics(data, categories, largeUserCat):
    print('\n---Data Statistics---\n')
    cat = copy.deepcopy(categories) 
    cat.update(largeUserCat)
    totalCounts = data.shape[0]
    totalUsage = data[:,1].sum()
    for c in list(cat.keys()):
        ind = data[:,0] == c
        catCounts = ind.sum() 
        catUsage = data[ind,1].sum()
        pctCount = np.multiply(np.divide(catCounts,totalCounts), 100.0)
        pctUsage = np.multiply(np.divide(catUsage, totalUsage), 100.0)
        print(cat[c] + ':')
        print('\tMasked = ' + str(IsMasked(data[ind], c)))
        print('\tAccounts = ' + '{:.2f}'.format(pctCount) + '%' + 
              ' (' +'{:,.0f}'.format(catCounts) + ' accounts)')
        print('\tUsage = ' + '{:.2f}'.format(pctUsage) + '%' + 
              ' (' + '{:,.0f}'.format(catUsage) + ' kwh)\n')
    return

#%%
###############################################################################
#                           DATA DIAGNOSTICS FUNCTION                         # 
############################################################################### 
    
def DataDiagnostics(data, categories, largeUserCat):
    print('\n---Data Diagnostics---')
    DataPlot(data, categories, largeUserCat)
    DataStatistics(data, categories, largeUserCat)
    return 

#%%
###############################################################################
#                             SET SIMULATION PARAMETERS                       # 
############################################################################### 

seed = 123456                       # Random number seed
n = 100000                          # Simulation realizations
mu = [2.0, 2.0]                     # Mean vector
sigma = [[1.0, 0.5], 
         [0.5, 0.01]]               # Covariance Matrix
categories = {1.0: 'single_family',
              2.0: 'multi_family',
              3.0: 'condo',
              4.0: 'other',
              5.0: 'commercial',
              6.0: 'industrial'}    # Account usetypes for masking 
largeUserCat = {7.0: 'large_users'} # Designation of large user category

#%%
###############################################################################
#               SIMULATE DATA AND PERFORM INITIAL DIAGNOSTICS                 # 
############################################################################### 

print('\n-----Generate Initial Data-----\n')
data = GenerateData(seed, n, mu, sigma)
DataDiagnostics(data, categories, largeUserCat)

#%%
###############################################################################
#               CLEAN OUT LARGE USERS AND PERFORM DIAGNOSTICS                 # 
############################################################################### 

print('\n-----Extract Large Users -----\n')
clean = LargeUserCategoryExtraction(data, categories, largeUserCat)
DataDiagnostics(clean, categories, largeUserCat)

#%%
###############################################################################
#               FILL OUT LARGE USERS AND PERFORM DIAGNOSTICS                  # 
############################################################################### 

print('\n-----Fill Large User Group-----\n')
clean_fill = LargeUserCategoryFill(clean, categories, largeUserCat)
DataDiagnostics(clean_fill, categories, largeUserCat)