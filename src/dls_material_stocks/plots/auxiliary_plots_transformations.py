# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 08:34:27 2025

@author: jstreeck
"""



'''#################################################################

       Description:
        This script creates Figures S4, S5, S6, S7, S8 in the 
        supplementary information of the article related to 
        the repository.
        
       Note:
        You might need to adjust the paths to material intensites
        and 
                
   #################################################################'''



# load packages
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import re


#paths
base_path = os.getcwd()
main_path = os.path.dirname(base_path)

# Add module folder to sys.path
module_path = os.path.join(main_path, 'load')
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.join(main_path, 'analysis')
if module_path not in sys.path:
    sys.path.append(module_path)

# load functions
from DLS_load_data import load_country_correspondence_dict, load_population_2015, load_DLS_2015, load_regional_diets, load_population
from DLS_functions_v3 import read_excel_into_dict, create_nested_dict, expand_nested_dict_to_df, prepare_df_for_concat,\
                                   calc_functions_scale, calc_food_flows_kcal_cap, conv_food_flows_kg_cap, calc_food_flows_kg_scale,\
                                   calc_indir_stock_intens_cap, calc_indir_stock_scale, regionalize_material_demand_ecoinvent, regionalize_material_demand_ecoinvent_coal


input_path = os.path.join(main_path, "input/") # path to model input data
path_to_MIs = os.path.join(input_path, "MI_comparison/") # path to material intensity data for figure S4
path_to_popgdp = os.path.join(input_path,"MISO2/") # path to population and GDP data
path_to_results = os.path.join(main_path, "final_results") # path to model results
results_filename = "results_data_supplement_final_Aug425.xlsx" # model results filename
hhSize_path = '2021_Kikstra/DLS_household_size_version0_1_rc4_20240307.csv' # path to household size input data
modal_share_path = 'mobility/DLS_modal_share_version0_1_rc4_20240307.csv' # path to modal share input data
reg_diets_path = 'food/dietary-composition-by-country_OURWORLDINDATA.csv' # path to dietary share input data
pickles_path = os.path.join(main_path, "pickles/") # path to pickled bottom up DLS stocks from pickle_runs_different_settings

### load DLS thresholds per dimension (current national/regional practices)
DLS_stocks_thresh = pd.read_pickle(os.path.join(pickles_path, "DLS_stocks_thresh_current_2025-05-09.pkl"))





''' ############### 1 ################
      Compare material intensities
             for figure S4
    ##################################'''



## Nutrition

# load data & rename
nutrition = pd.read_excel(os.path.join(path_to_MIs, "indirect_stock_intensities_nutrition_v1_pork_modified_subset.xlsx"))

nutrition_rename =    {"market for swine for slaughtering, live weight" : "swine for slaughtering, live weight (modified)",
     " market for butter, from cow milk" : "butter, from cow milk",
     "market for sheep for slaughtering, live weight" : "sheep for slaughtering, live weight",
     "market for cashew" : "cashew",
     "market for red meat, live weight" : "red meat",
     "market for palm oil, refined" : "palm oil, refined",
     "market for cow milk" : "cow milk",
     "market for almond" : "almond",
     "market for frozen fish sticks, hake" : "frozen fish sticks, hake",
     "market for chicken for slaughtering, live weight" : "chicken for slaughtering, live weight",
     "market for jatropha seed" : "jatropha seed",
     "market for wheat flour mix" : "wheat flour mix",
     "market for tomato, fresh grade" : "tomato, fresh grade",
     "market for maize flour" : "maize flour",
     "market for rye grain" : "rye grain",
     "market for peanut" : "peanut",
     "market for apple" : "apple",
     "market for chickpea" : "chickpea",
     "market for oat grain" : "oat grain",
     "market for sugar, from sugarcane" : "sugar, from sugarcane",
     "market for barley grain" : "barley grain",
     "market for rice, non-basmati" : "rice, non-basmati",
     "market for onion" : "onion",
     "market for potato" : "potato",
     "market for tomato, processing grade" : "tomato, processing grade",
     "market for barley grain, organic" : "barley grain, organic",
     "market for millet" : "millet",
     " market for small pelagic fish, fresh" : "small pelagic fish, fresh",
     "market for demersal fish, fresh" : "demersal fish, fresh",
     "market for orange, fresh grade" : "orange, fresh grade",
     "market for banana" : "banana",
     "market for sugar beet" : "sugar beet",
     "market for sugarcane" : "sugarcane"}

nutrition = nutrition.replace(nutrition_rename)

# Sort the DataFrame by the 'do.call..rbind...summary.' column
# calculate stock per kilocalorie provided
nutrition['stock/kcal'] = nutrition['stock/FU'] / nutrition['kcal/kg']
nutrition= nutrition[['names(summary)','stock/kcal']]
nutrition_sorted = nutrition.sort_values(by='stock/kcal', ascending=False)
# Plotting
plt.figure(figsize=(10, 8))
plt.barh(nutrition_sorted['names(summary)'], nutrition_sorted['stock/kcal'], color='skyblue')
plt.xlabel('kg material stocks per kcal/year')
plt.title('Material stock intensity of different foods')
plt.gca().invert_yaxis()  # To have the highest values at the top
plt.tight_layout()
plt.show()



## Mobility

# load data & rename
mobility = pd.read_excel(os.path.join(path_to_MIs, "indirect_stock_intensities_mobility_v1_subset.xlsx"))

mobility_rename = {
    "market […] passenger car, medium size, petrol, EURO 5_GLO_occup1.3":"passenger car, petrol, EURO 5_GLO_occup1.3",
    "market […] passenger car, medium size, petrol, EURO 5_GLO_occup1.7":"passenger car, petrol, EURO 5_GLO_occup1.7",
    "market […] passenger car, medium size, petrol, EURO 5_GLO_occup2.3":"passenger car, petrol, EURO 5_GLO_occup2.3",
    "market for transport, passenger car, medium size, diesel, EURO 5_GLO":" passenger car, diesel, EURO 5_GLO",
    "market for transport, passenger car_RoW":"passenger car_RoW",
    "market for transport, passenger car_RER":"passenger car_RER",
    "market for transport, passenger car, electric_GLO":"passenger car, electric_GLO",
    "market for transport, passenger train_GLO":"passenger train_GLO",
    "market for transport, trolleybus_GLO":"trolleybus_GLO",
    "market for transport, trolleybus1_GLO":"trolleybus1_GLO",
    "market for transport, regular bus_GLO":"regular bus_GLO",
    "market for transport, regular bus1_GLO":"regular bus1_GLO",
    "market for transport, passenger, electric scooter_GLO":"electric scooter_GLO",
    "market for transport, passenger, motor scooter_GLO":"motor scooter_GLO",
    "market for transport, passenger, electric bicycle_GLO":"electric bicycle_GLO",
    "market for transport, passenger, bicycle_GLO":"bicycle_GLO"}

mobility = mobility.replace(mobility_rename)
mobility = mobility[~mobility.iloc[:,0].isin(["regular bus1_GLO","trolleybus1_GLO" ])]
# calculate stock required per person-kilometer driven
mobility['stock/pkm'] = mobility['stock/FU'] / mobility['occupants/FU']
# Sort the DataFrame by the 'do.call..rbind...summary.' column
mobility_sorted = mobility.sort_values(by='stock/pkm', ascending=False)

# Plotting
plt.figure(figsize=(10, 8))
plt.barh(mobility_sorted['names(summary)'], mobility_sorted['stock/pkm'], color='skyblue')
plt.xlabel('kg material stocks per person-km/year')
plt.title('Material stock intensity of different mobility options')
plt.gca().invert_yaxis()  # To have the highest values at the top
plt.tight_layout()
plt.show()



## Buildings

# load data & rename
buildings = pd.read_excel(os.path.join(path_to_MIs, "Haberl_MI_buildings_global.xlsx"),sheet_name='total out')

buildings_rename = {
    "RS - R5.2OECD":"Single-family, OECD", 
    "RS - OECD_NA":"Single-family, North America",
    "RS - JPN":"Single-family, Japan",
    "RS - R5.2ASIA":"Single-family, Asia",
    "RS - CHN":"Single-family, China",
    "RS - R5.2LAM":"Single-family, Latin America",
    "RS - R5.2MAF":"Single-family, Middle East & Africa",
    "RS - R5.2REF":"RS - R5.2REF",
    "RM - R5.2OECD":"Multi-family, OECD",
    "RM - OECD_NA":"Multi-family,North America",
    "RM - JPN":"Multi-family, Japan",
    "RM - R5.2ASIA":"Multi-family, Asia",
    "RM - CHN":"Multi-family, China",
    "RM - R5.2LAM":"Multi-family, Latin America",
    "RM - R5.2MAF":"Multi-family, Middle East & Africa",
    "RM - R5.2REF":"RM - R5.2REF",
    "NR - R5.2OECD":"Non-residential, OECD",
    "NR - OECD_NA":"Non-residential, North America",
    "NR - JPN":"Non-residential, Japan",
    "NR - R5.2ASIA":"Non-residential, Asia",
    "NR - CHN":"Non-residential, China",
    "NR - R5.2LAM":"Non-residential, Latin America",
    "NR - R5.2MAF":"Non-residential, Middle East & Africa",
    "NR - R5.2REF":"NR - R5.2REF" }

buildings = buildings.replace(buildings_rename)
buildings = buildings[~buildings.iloc[:,0].isin(["RS - R5.2REF", "RM - R5.2REF", "NR - R5.2REF"])]

# plot for housing
housing = buildings.iloc[:14,:]
housing_sorted = housing.sort_values(by='MI per sqm', ascending=False)
plt.figure(figsize=(10, 8))
plt.barh(housing_sorted["type"], housing_sorted['MI per sqm'], color='skyblue')
plt.xlabel('kg material stocks per person-km')
plt.title('Material stock intensity of different mobility options')
plt.gca().invert_yaxis()  # To have the highest values at the top
plt.tight_layout()
plt.show()

# plot for nonresidential
nonres = buildings.iloc[14:,:]
nonres_sorted = nonres.sort_values(by='MI per sqm', ascending=False)
plt.figure(figsize=(10, 8))
plt.barh(nonres_sorted["type"], nonres_sorted['MI per sqm'], color='skyblue')
plt.xlabel('kg material stocks per person-km')
plt.title('Material stock intensity of different mobility options')
plt.gca().invert_yaxis()  # To have the highest values at the top
plt.tight_layout()
plt.show()



### ALL TO MULTIPLOT

# Create a 2x2 grid for the subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Material stock intensity of different foods
axs[0, 0].barh(nutrition_sorted['names(summary)'], nutrition_sorted['stock/kcal'], color='skyblue')
axs[0, 0].set_title('(a) Material stock intensity of different foods')
axs[0, 0].invert_yaxis()  # To have the highest values at the top
axs[0, 0].set_xlabel('kg material stocks per kcal/year')
# Plot 2: Material stock intensity of different mobility options
axs[0, 1].barh(mobility_sorted['names(summary)'], mobility_sorted['stock/pkm'], color='skyblue')
axs[0, 1].set_title('(b) Material stock intensity of different mobility options')
axs[0, 1].invert_yaxis()  # To have the highest values at the top
axs[0, 1].set_xlabel('kg material stocks per person-km/year')
# Plot 3: Material stock intensity of different housing options
axs[1, 0].barh(housing_sorted["type"], housing_sorted['MI per sqm'], color='skyblue')
axs[1, 0].set_title('(c) Material stock intensity of different housing options')
axs[1, 0].invert_yaxis()  # To have the highest values at the top
axs[1, 0].set_xlabel('kg material stocks per squaremeter')
# Plot 4: Material stock intensity of different non-residential options
axs[1, 1].barh(nonres_sorted["type"], nonres_sorted['MI per sqm'], color='skyblue')
axs[1, 1].set_title('(d) Material stock intensity of different non-residential options')
axs[1, 1].invert_yaxis()  # To have the highest values at the top
axs[1, 1].set_xlabel('kg material stocks per squaremeter')
# Adjust the layout so the plots do not overlap
plt.tight_layout()
# Show the combined plot
plt.show()





''' ############### 2 ################
      Compare relation of material 
      stocks to GDP for figure S5
    ##################################'''


## PREPARE DATA to plot material stock thresholds (at current national/regional practices) per capita of countries against GDP 

# calculate GDP/cap from source data
pop = pd.read_excel(os.path.join(path_to_popgdp, "MISO2_population.xlsx"), sheet_name='values', index_col=0)
gdp = pd.read_excel(os.path.join(path_to_popgdp, "MISO2_GDP.xlsx"), sheet_name='wb_ppp_$2021', index_col=0)
gdp.columns = gdp.columns.astype(int)
gdp_cap_2015 = (gdp[2015]/ pop[2015])*1e6 #$2021/capta

# total stocks vs. GDP
# load stock thresholds and format
stock_thresh_cap = pd.read_excel(os.path.join(path_to_results,results_filename), sheet_name='Fig1bc_stock_distr_countr', index_col=0)
stock_thresh_cap = stock_thresh_cap[ 'DLS material stock threshold current practices']
stock_thresh_cap =  stock_thresh_cap[:-3]
stock_vs_gdp = pd.concat([stock_thresh_cap, gdp_cap_2015],axis=1)
# drop all which have zero GDP or missing entries
stock_vs_gdp = stock_vs_gdp.dropna()
stock_vs_gdp = stock_vs_gdp[stock_vs_gdp[2015]>0]


# DLS stock by DLS dimension vs GDP - prepare data

## FORMAT & PARTIALLY HARMONIZE SYSTEM SCOPE & LABELS IN BOTTOM-UP AND TOP-DOWN RESULTS, remove materials as stocks in their own right
stocks_of_materials = [1610,1621,1622,1623,1701,1702,1709,2011,2013,2022,2029,2030,2219,2220,2310,2391,2392,2393,
                       2394,2395,2396,2399,2410,2420,2431,2432,2511,2512,2591,2592,2599]
DLS_stocks_thresh_noMats = DLS_stocks_thresh[~DLS_stocks_thresh.index.get_level_values(5).isin(stocks_of_materials)]
DLS_stocks_thresh_noMats.reset_index(inplace=True)

#remove materials which are not related to stocks (fertilizer, pesticides, soap, basic chemicals, chemical products n.e.c., man-made fibres)
DLS_stocks_thresh_noMats = DLS_stocks_thresh_noMats[~(DLS_stocks_thresh_noMats.material.isin([2011,2012,2021,2023,2029,2030,'2011','2012','2021','2023','2029','2030']))]

# drop materials which cannot be identified according to type
DLS_stocks_thresh_noMats = DLS_stocks_thresh_noMats[~DLS_stocks_thresh_noMats.material.isin(['other_materials'])]

#harmonize DLS dimension labels
aggr_direct_indirect = {'clothing_i':'clothing', 'communication_i':'communication',
                        'educ_i':'education', 'education_i':'education', 'health_i':'health',
                        'hh_appliance_i':'hh_appliance', 'housing_i': 'housing',
                        'transport_i':'transport'}
aggr_dim = { 'communication':'Socialization', 'clothing':'Shelter', 'cond_i':'Shelter', 'education':'Socialization', 'health':'Health',
       'hh_appliance':'Nutrition', 'housing':'Shelter', 'nutrition_i':'Nutrition', 'sanitation_i':'Health', 'transport':'Mobility',
       'water_i':'Health'}
DLS_stocks_thresh_noMats.set_index(['region', 'year', 'dimension', 'stock_type', 'product', 'stock',
       'material', 'unit'],inplace=True)
DLS_stocks_thresh_noMats = DLS_stocks_thresh_noMats.rename(aggr_direct_indirect).groupby(['region','dimension']).sum()
DLS_stocks_thresh_noMats = DLS_stocks_thresh_noMats.rename(aggr_dim ).groupby(['region','dimension']).sum()
DLS_stocks_thresh_noMats.sum() /1e9

## obtain population data to calculate per capita stocks
## define main paths 
country_correspondence_filename = 'country_correspondence.xlsx'
# country corespondence MISO2 (Wiedernhofer et al., in prep.) & DLS data (Kikstra et al., 2021 updated)                    
country_correspondence, country_correspondence_dict = load_country_correspondence_dict(country_correspondence_filename, input_path)

# load population data (Wiedernhofer et al., 2024) with Sudan & South Sudan aggregated to match to sytem boundaries of DLS indicators (Kikstra et al., 2021/2024)  
MISO2_population_2015, MISO2_population_2015_subset = load_population_2015(filename='MISO2/MISO2_population.xlsx', path_name=input_path, sheet_name='values', country_correspondence=country_correspondence)
MISO2_population =  load_population(filename='MISO2/MISO2_population.xlsx', path_name=input_path, sheet_name='values', country_correspondence=country_correspondence)

#prepare population data to calculate PER CAPITA stocks
MISO2_2015_pop_country_subset = MISO2_population_2015.reset_index()[MISO2_population_2015.reset_index()['region'].isin(DLS_stocks_thresh_noMats.reset_index()['region'].unique())].set_index('region')
# calculate per capita stoc
DLS_stocks_thresh_noMats_country_cap = DLS_stocks_thresh_noMats.reset_index().merge(MISO2_2015_pop_country_subset.reset_index(), how='left', on='region')
DLS_stocks_thresh_noMats_country_cap['value_cap'] = DLS_stocks_thresh_noMats_country_cap['value'] / DLS_stocks_thresh_noMats_country_cap['2015']
DLS_stocks_thresh_noMats_country_cap = DLS_stocks_thresh_noMats_country_cap[['region','dimension','value_cap']]

DLS_stocks_thresh_noMats_country_cap.value_cap = DLS_stocks_thresh_noMats_country_cap.value_cap/1e3


## PLOT material stock thresholds (at current national/regional practices) per capita of countries against GDP 

### total stock requirements for DLS thresholds vs. GDP
# Create scatter plot
plt.figure(figsize=(10,6))
plt.scatter(stock_vs_gdp [2015], stock_vs_gdp ['DLS material stock threshold current practices'],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(  stock_vs_gdp[2015],stock_vs_gdp['DLS material stock threshold current practices'])
trendline = slope *  stock_vs_gdp[2015] + intercept
plt.plot(stock_vs_gdp[2015], trendline, color='red')
# Add R² text to the plot (adjust the x and y positions as needed)
plt.text(400,50, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
plt.title('Relationship between material stocks to reach DLS threshold and GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('Material stocks per capita to reach DLS threshold')
plt.grid(True)
plt.show()


### Mobility stock requirements for DLS thresholds vs. GDP
DLS_stocks_thresh_noMats_country_cap_mob = DLS_stocks_thresh_noMats_country_cap[DLS_stocks_thresh_noMats_country_cap.dimension == 'Mobility']
mobStock_vs_gdp = pd.concat([DLS_stocks_thresh_noMats_country_cap_mob.set_index('region'), gdp_cap_2015],axis=1).dropna().drop(['Qatar'])
plt.scatter(mobStock_vs_gdp [2015], mobStock_vs_gdp['value_cap'],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(mobStock_vs_gdp[2015], mobStock_vs_gdp.iloc[:,1].astype(float))
trendline = slope * mobStock_vs_gdp [2015] + intercept
plt.plot(mobStock_vs_gdp [2015], trendline, color='red')
# Add R² text to the plot (adjust the x and y positions as needed)
plt.text(400,12, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
plt.title('Relationship between material stocks to reach DLS MOBILITY threshold and GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('Material stocks per capita to reach DLS threshold')
plt.grid(True)
plt.show()


### Shelter stock requirements for DLS thresholds vs. GDP
DLS_stocks_thresh_noMats_country_cap_shelt = DLS_stocks_thresh_noMats_country_cap[DLS_stocks_thresh_noMats_country_cap.dimension == 'Shelter']
sheltStock_vs_gdp = pd.concat([DLS_stocks_thresh_noMats_country_cap_shelt.set_index('region'), gdp_cap_2015],axis=1).dropna().drop(['Qatar'])
plt.scatter(sheltStock_vs_gdp [2015], sheltStock_vs_gdp['value_cap'],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(sheltStock_vs_gdp[2015], sheltStock_vs_gdp.iloc[:,1].astype(float))
trendline = slope * sheltStock_vs_gdp [2015] + intercept
plt.plot(sheltStock_vs_gdp [2015], trendline, color='red')
# Add R² text to the plot (adjust the x and y positions as needed)
plt.text(400,30, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
# plt.plot(mobStock_vs_gdp [2015], trendline, color='red')
plt.title('Relationship between material stocks to reach DLS SHELTER threshold and GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('Material stocks per capita to reach DLS threshold')
plt.grid(True)
plt.show()

### Nutrition stock requirements for DLS thresholds vs. GDP
DLS_stocks_thresh_noMats_country_cap_nutr = DLS_stocks_thresh_noMats_country_cap[DLS_stocks_thresh_noMats_country_cap.dimension == 'Nutrition']
nutrStock_vs_gdp = pd.concat([DLS_stocks_thresh_noMats_country_cap_nutr.set_index('region'), gdp_cap_2015],axis=1).dropna().drop(['Qatar'])
plt.scatter(nutrStock_vs_gdp [2015], nutrStock_vs_gdp['value_cap'],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(nutrStock_vs_gdp[2015], nutrStock_vs_gdp.iloc[:,1].astype(float))
trendline = slope * nutrStock_vs_gdp [2015] + intercept
plt.plot(nutrStock_vs_gdp [2015], trendline, color='red')
# Add R² text to the plot (adjust the x and y positions as needed)
plt.text(400,8, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
# plt.plot(mobStock_vs_gdp [2015], trendline, color='red')
plt.title('Relationship between material stocks to reach DLS NUTRITION threshold and GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('Material stocks per capita to reach DLS threshold')
plt.grid(True)
plt.show()


##### ASSEMBLE IN MULTIPLOT

# Create subplots with a 2x2 grid (4 plots)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout(pad=5.0)

# Adjusting the label and title font size
label_size = 12
title_size = 14

# Total stock requirements for DLS thresholds vs. GDP
ax = axs[0, 0]
ax.scatter(stock_vs_gdp[2015], stock_vs_gdp['DLS material stock threshold current practices'], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(stock_vs_gdp[2015], stock_vs_gdp['DLS material stock threshold current practices'])
trendline = slope * stock_vs_gdp[2015] + intercept
ax.plot(stock_vs_gdp[2015], trendline, color='red')
ax.text(400, 50, s=f'R² = {r_value**2:.2f}', fontsize=label_size, color='red')
ax.set_title('(a) DLS Total Stock Requirements vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Material stocks for DLS threshold (tons/capita)', fontsize=label_size)
ax.grid(True)

# Mobility stock requirements for DLS thresholds vs. GDP
ax = axs[0, 1]
ax.scatter(mobStock_vs_gdp[2015], mobStock_vs_gdp['value_cap'], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(mobStock_vs_gdp[2015], mobStock_vs_gdp.iloc[:,1].astype(float))
trendline = slope * mobStock_vs_gdp[2015] + intercept
ax.plot(mobStock_vs_gdp[2015], trendline, color='red')
ax.text(400,12, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
ax.set_title('(b) DLS Mobility Stock Requirements vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Material stocks for DLS threshold (tons/capita)', fontsize=label_size)
ax.grid(True)

# Shelter stock requirements for DLS thresholds vs. GDP
ax = axs[1, 0]
ax.scatter(sheltStock_vs_gdp[2015], sheltStock_vs_gdp['value_cap'], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(sheltStock_vs_gdp[2015], sheltStock_vs_gdp.iloc[:,1].astype(float))
trendline = slope * sheltStock_vs_gdp[2015] + intercept
ax.plot(sheltStock_vs_gdp[2015], trendline, color='red')
ax.text(400,30, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
ax.set_title('(c) DLS Shelter Stock Requirements vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Material stocks for DLS threshold (tons/capita)', fontsize=label_size)
ax.grid(True)

# Nutrition stock requirements for DLS thresholds vs. GDP
ax = axs[1, 1]
ax.scatter(nutrStock_vs_gdp[2015], nutrStock_vs_gdp['value_cap'], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(nutrStock_vs_gdp[2015], nutrStock_vs_gdp.iloc[:,1].astype(float))
trendline = slope * nutrStock_vs_gdp[2015] + intercept
ax.plot(nutrStock_vs_gdp[2015], trendline, color='red')
ax.text(400,8, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
ax.set_title('(d) DLS Nutrition Stock Requirements vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Material stocks for DLS threshold (tons/capita)', fontsize=label_size)
ax.grid(True)

# Adjust the overall layout for better spacing
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Show the plot
plt.show()





''' ############### 3 ################
      Compare relation of practices 
      to GDP for figure S6
    ##################################'''

# Mobility
# load modal shares for cars (2015)
modal_shares = pd.read_csv((os.path.join(input_path, modal_share_path))).rename(columns={'iso':'region'})
#rename regions according to MISO2 country labels and drop all which are not in MISO2 country list
for key, value in country_correspondence_dict.items():
    modal_shares['region'].replace({key : value}, inplace=True)
modal_shares = modal_shares[modal_shares.region.isin(country_correspondence.MISO2.to_list())]
modal_shares.set_index(['region'], inplace=True)
modal_vs_gdp = pd.concat([modal_shares, gdp_cap_2015],axis=1)
modal_vs_gdp = modal_vs_gdp.dropna()
# drop Qatar because GDP very high and appears as outlier in plot
modal_vs_gdp = modal_vs_gdp.drop(['Qatar'])
plt.scatter(modal_vs_gdp [2015], modal_vs_gdp.iloc[:,0],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(modal_vs_gdp [2015], modal_vs_gdp.iloc[:,0])
trendline = slope * modal_vs_gdp [2015] + intercept
plt.plot(modal_vs_gdp [2015], trendline, color='red')

plt.title('Relationship light-duty vehicle modal share to GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('light-duty vehicle modal share')
plt.grid(True)
plt.show()


# Nutrition
# load and format regional diet data (2015)
reg_diets_act_shares_final = load_regional_diets(filename=reg_diets_path, path_name=input_path,  country_correspondence=country_correspondence, country_correspondence_dict = country_correspondence_dict)
reg_diets_act_shares_final = reg_diets_act_shares_final[reg_diets_act_shares_final.index.get_level_values(1)==2015]
reg_diets_act_shares_meat = reg_diets_act_shares_final[['Meat, sheep and goat | 00002732 || Food available for consumption | 0664pc || kilocalories per day per capita',
'Meat, pig | 00002733 || Food available for consumption | 0664pc || kilocalories per day per capita',
'Meat, poultry | 00002734 || Food available for consumption | 0664pc || kilocalories per day per capita',
'Meat, beef | 00002731 || Food available for consumption | 0664pc || kilocalories per day per capita']]
reg_diets_act_shares_meat_sum = reg_diets_act_shares_meat.sum(axis=1)
reg_diets_act_shares_meat_sum.index = reg_diets_act_shares_meat_sum.index.get_level_values(0)
meat_vs_gdp = pd.concat([reg_diets_act_shares_meat_sum, gdp_cap_2015],axis=1)
meat_vs_gdp = meat_vs_gdp.dropna()
# drop Qatar because GDP very high and appears as outlier in plot
meat_vs_gdp = meat_vs_gdp.drop(['Qatar'])
plt.scatter(meat_vs_gdp [2015], meat_vs_gdp.iloc[:,0],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(meat_vs_gdp [2015], meat_vs_gdp.iloc[:,0])
trendline = slope * meat_vs_gdp [2015] + intercept
plt.plot(meat_vs_gdp [2015], trendline, color='red')
plt.title('Relationship meat calorific intake shares to GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('meat intake shares on total calorific intake')
plt.grid(True)
plt.show()

# Housing
# load household size data (2015)
hh_size = pd.read_csv((os.path.join(input_path, hhSize_path))).rename(columns={'iso':'region'})
#rename regions according to MISO2 country labels and drop all which are not in MISO2 country list
for key, value in country_correspondence_dict.items():
    hh_size['region'].replace({key : value}, inplace=True)
hh_size = hh_size[hh_size.region.isin(country_correspondence.MISO2.to_list())]
hh_size.set_index(['region'], inplace=True)
hh_vs_gdp = pd.concat([hh_size, gdp_cap_2015],axis=1)
hh_vs_gdp = hh_vs_gdp.dropna()
# drop Qatar because GDP very high and appears as outlier in plot
hh_vs_gdp = hh_vs_gdp.drop(['Qatar'])
plt.scatter(hh_vs_gdp [2015], hh_vs_gdp.iloc[:,0],color='blue', alpha=0.6)
# Perform linear regression to get the slope, intercept, and R² value
slope, intercept, r_value, p_value, std_err = linregress(hh_vs_gdp [2015], hh_vs_gdp.iloc[:,0])
trendline = slope * hh_vs_gdp [2015] + intercept
plt.plot(hh_vs_gdp [2015], trendline, color='red')

plt.title('Relationship between household size and GDP')
plt.xlabel('$2021 PPP GDP/capita')
plt.ylabel('household size (capita/household)')
plt.grid(True)
plt.show()



### Assemble to MULTIPLOT

# Create subplots with a 2x2 grid (last panel will be empty)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.tight_layout(pad=5.0)

# Adjusting the label and title font size
label_size = 12
title_size = 14

### Plot 1: Modal shares for cars (2015)
# Assuming modal_shares and gdp_cap_2015 are loaded previously
ax = axs[0, 1]
ax.scatter(modal_vs_gdp[2015], modal_vs_gdp.iloc[:, 0], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(modal_vs_gdp[2015], modal_vs_gdp.iloc[:, 0])
trendline = slope * modal_vs_gdp[2015] + intercept
ax.plot(modal_vs_gdp[2015], trendline, color='red')
ax.text(400,0.7, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
ax.set_title('(b) Light-duty Vehicle Modal Share vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Light-duty vehicle modal share', fontsize=label_size)
ax.grid(True)

### Plot 2: Dietary shares (2015)
# Assuming reg_diets_act_shares_final and gdp_cap_2015 are loaded previously
ax = axs[1, 1]
ax.scatter(meat_vs_gdp[2015], meat_vs_gdp.iloc[:, 0], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(meat_vs_gdp[2015], meat_vs_gdp.iloc[:, 0])
trendline = slope * meat_vs_gdp[2015] + intercept
ax.plot(meat_vs_gdp[2015], trendline, color='red')
ax.text(400,0.2, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
ax.set_title('(d) Meat Calorific Intake Share vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Meat intake share of total calorific intake', fontsize=label_size)
ax.grid(True)

### Plot 3: Household size & GDP (2015)
# Assuming hh_vs_gdp is loaded previously
ax = axs[1, 0]
ax.scatter(hh_vs_gdp[2015], hh_vs_gdp.iloc[:, 0], color='blue', alpha=0.6)
slope, intercept, r_value, p_value, std_err = linregress(hh_vs_gdp[2015], hh_vs_gdp.iloc[:, 0])
trendline = slope * hh_vs_gdp[2015] + intercept
ax.plot(hh_vs_gdp[2015], trendline, color='red')
ax.text(400,7, s=f'R² = {r_value**2:.2f}', fontsize=12, color='red')
ax.set_title('(c) Household Size vs. GDP', fontsize=title_size)
ax.set_xlabel('$2021 PPP GDP/capita', fontsize=label_size)
ax.set_ylabel('Household size (capita/household)', fontsize=label_size)
ax.grid(True)

# Leave the last plot empty (panel d)
axs[0, 0].axis('off')

# Adjust the overall layout for better spacing
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Show the plot
plt.show()





''' ############### 4 ################
      Plot effect of material efficiency 
      measures to DLS stock thresholds 
      for figure S7 & S8
    ##################################'''
    
    
## plot as % changes from base case (current/converged)

results_filename = "Streeck_supplementary_data1.xlsx"

df = pd.read_excel(os.path.join(main_path,results_filename),sheet_name="DLS_stock_thresh_efficiency", skiprows=2)

# Assuming first column is Region
regions = df.iloc[:, 0]
data = df.set_index(df.columns[0])

# Find columns that include 'current' and 'converged'
current_cols = [col for col in data.columns if 'current' in col.lower()]

#for fixed col order
current_cols = ['current',
 'current_lightWood',
 'current_woodBased',
 'current_hhSizeHigh',
 'current_hhSizeMedium',
 'current_lightweight',
 'current_vegan',
 'current_vegetarian',
 'current_lowMeat',
 'current_lowCarLowDemand',
 'current_Evs',
 'current_B2DS',
 'current_lowCar',
 'current_combined',
 'converged_combined',
 'current_hhSizeLow',
 'current_RE',]

current_cols_conv_combined = [col for col in data.columns if 'current' in col.lower()] + ['converged_combined']
converged_cols = [col for col in data.columns if 'converged' in col.lower()]

# Helper to clean column names (remove dates and underscores)
def clean_label(label):
    label = re.sub(r'\d{4}[-/]\d{2}[-/]\d{2}', '', label)
    label = re.sub(r'\d{2}[-/]\d{2}[-/]\d{4}', '', label)
    label = label.replace('_', '')
    return label.strip()

# Region full names dictionary
region_full_names = {
    'AFR': 'Subs. Africa',
    'CPA': 'Centrally planned Asia',
    'EEU': 'E. Europe',
    'FSU': 'Former Soviet Union',
    'LAM': 'Latin America',
    'MEA': 'N. Africa & Middle East',
    'NAM': 'North America',
    'PAO': 'JP, AU, NZ',
    'PAS': 'Pacific Asia',
    'SAS': 'South Asia',
    'WEU': 'W. Europe'
}


def plot_clustered_horizontal_bars(data, cols, title, base_keyword):
    n_regions = len(data)
    y = np.arange(n_regions) 
    
    # Prepare cleaned labels
    cleaned_labels = [clean_label(col) for col in cols]
    
    plt.figure(figsize=(14, 20))
    
    # Bar and color settings
    bar_height = 0.9 / len(cols)  # Reduced bar height to reduce space
    n_cols = len(cols)
    cmap = plt.cm.get_cmap('tab20b', n_cols)
    colors = [cmap(i) for i in range(n_cols)]

    # Build label -> color mapping
    label_color_dict = {clean_label(col): colors[i] for i, col in enumerate(cols)}

    # Store y-offsets for horizontal line placement
    bar_offsets = []

    for idx, region in enumerate(data.index):
        region_values = data.loc[region, cols]
        
        # Find base value (the column which includes 'current' or 'converged')
        base_value = region_values[base_keyword]
        
        # Calculate % change from base
        percent_changes = 100 * (region_values - base_value) / base_value
        
        # Sort bars per region (largest to smallest % change)
        sorted_pairs = sorted(zip(percent_changes.values, cols), reverse=True)
        
        for i, (pct_change, col) in enumerate(sorted_pairs):
            clean_col = clean_label(col)
            color = label_color_dict[clean_col]
            offset =  y[idx] + (i - len(cols)/2)*bar_height + bar_height *5.5/12*n_cols
            
            # Plot bar
            plt.barh(offset, pct_change, height=bar_height, color=color, zorder=3)
            
            clean_col = clean_col.replace(base_keyword, '').strip()
            # Plot value + cleaned label
            if pct_change != 0:
                plt.text(
                    pct_change + (1 if pct_change >= 0 else -1),
                    offset,
                    f"{clean_col} {pct_change:.1f}%",
                    va='center',
                    ha='left' if pct_change >= 0 else 'right',
                    fontsize=7
                )
        
        # Store the last offset for drawing the horizontal line at the fringe of columns
        bar_offsets.append(y[idx] + (len(cols) - 0.5) * bar_height)

    # Draw horizontal lines between column groups at the fringes
    for y_offset in bar_offsets:
        plt.axhline(y_offset, color='black', linewidth=0.8, zorder=2)

    # Set full region names as y-tick labels
    full_region_names = [region_full_names.get(region, region) for region in data.index]
    plt.yticks(y+0.35/12*n_cols, full_region_names, fontsize=10)

    # Create single legend
    handles = [plt.Rectangle((0,0),1,1, color=label_color_dict[clean_label(col)]) for col in cols if col != base_keyword]
    
    
    labels = [clean_label(col) for col in cols if col != base_keyword]
    labels = [clean_label(col).replace(base_keyword, '').strip() 
          for col in cols if col != base_keyword]
    
    for col in cols:
        print(f"{clean_label(col)}")
    
    rename_dict = {
    'currentB2DS': 'IEAB2DS transp. modes',
    'currentEVs':'vehicles electric',
    'currenthhSizeHigh':'large households',
    'currenthhSizeLow':'small households',
    'currenthhSizeMedium':'median househ. size',
    'currentlightweight':'light construction',
    'currentwoodBased':'wood construction',
    'currentlightWood':'light&wood construction',
    'currentlowCar':'low car',
    'currentlowCarlowDemand':'low pkm,low car',
    'currentlowMeat':'low meat',
    'currentRE':'renewable energy',
    'currentvegetarian':'vegetarian',
    'currentvegan':'vegan',
    'currentcombined':'combined: lightwood, low pkm/car,\n vegan, renewable, EVs',
    'convergedcombined': 'combined+median househ.size',
     #'convergedcombined':'combined: lightwood, low pkm/car,\n vegan, renewable, EVs',
    'convergedEvs':'vehicles electric',
    'convergedhhSizeHigh':'large households',
    'convergedhhSizeLow':'small households',
    'convergedlightweight':'light construction',
    'convergedwoodBased':'wood construction',
    'convergedlightWood':'light&wood construction',
    'convergedlowCarLowDemand':'low pkm,low car',
    'convergedlowCar':'low car',
    'convergedRE':'renewable energy',
    'convergedvegan':'vegan',
    'convergedvegetarian':'vegetarian'}
    
    #Combined scenario with vegan diet, low car low demand, EVs, renewable energy, 
    #                   and lightweight & wood-based buildings (excludes converged scenario changes)
                       
                       
    labels = [
    rename_dict.get(clean_label(col), clean_label(col)).replace(base_keyword, '').strip()
    for col in cols if col != base_keyword]
    
    
    
    
    plt.legend(handles, labels, ncols=1, loc= 'upper right',fontsize=10)
    
    plt.title(title)
    plt.xlabel('Percentage change from base scenario (%)')
    plt.grid(axis='x', linestyle='--', zorder=0)
    plt.axvline(0, color='black', linewidth=0.8)  # Add zero line
    plt.tight_layout() 
    
    # Get current y-axis limits
    plt.ylim(plt.ylim()[0] + 0.5, plt.ylim()[1] - 0.5)
    plt.savefig('Streeck_ED_Fig1_' + base_keyword + '.tif', format='tif', dpi =300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
plot_clustered_horizontal_bars(data[current_cols], current_cols, 'Effect of practice and technology changes on DLS stock thresholds – relative to current national/regional practices', base_keyword='current')

# Plot for 'converged'
plot_clustered_horizontal_bars(data[converged_cols], converged_cols, 'Effect of practice and technology changes on DLS stock thresholds – relative to converged practices', base_keyword='converged')




''' ############### 5 ################
     load DLE data and write all to 
     electricity use to simulate 
     renewable energy scenario
    ##################################'''


# path_to_DLE = os.path.join(input_path, "2021_Kikstra/") 

# files = ["DLE_threshold_input_opcon_elecnonelec_dimensions_default", "DLE_threshold_input_opcon_elecnonelec_dimensions_median_hh_size"]
# for file in files:
#     DLE_default = pd.read_csv(os.path.join(path_to_DLE, file + '.csv'))
#     DLE_default.set_index(['iso', 'elec', 'variable', 'unit.energy'], inplace=True)
#     DLE_fuelsElect = DLE_default.groupby(['iso',  'variable', 'unit.energy']).sum()
#     DLE_default[['thres.energy', 'thres.energy.op', 'thres.energy.conrep']] = 0
#     #DLE_default[DLE_default.index.get_level_values(1)=="elec"] = DLE_fuelsElect
#     # Loop over each country (iso)
#     for iso in DLE_default.index.get_level_values('iso').unique():
#         # Loop over each (variable, unit.energy) in the grouped result
#         for (variable, unit_energy), row in DLE_fuelsElect.loc[iso].iterrows():
#             # Build full index for the target location in DLE_default
#             idx = (iso, 'elec', variable, unit_energy)
            
#             # Update all matching columns (you can specify only certain columns if needed)
#             DLE_default.loc[idx] = row.values
#     DLE_default.to_csv(file + '_onlyELECTRICITY' + '.csv')        