# -*- coding: utf-8 -*-
"""
@author: jstreeck

"""

'''############################################################

       Description:
           
        This script specifies the functions to run regression
        of economy-wide material stock data on DLS mean index,
        as described in section SI 1.3.4 in the supplementary
        associated to this repository.
        
        Run the script with the variables
         - DLS_data_path 
         - MISO2_MFA_data_pat
        defined as LOW / HIGH in lines 35 - 41 to test sensitivity.
        
   ############################################################'''
   
   
   
import os
import sys
import pandas as pd

#paths
main_path = os.path.dirname(os.getcwd())
# data paths
input_path = os.path.join(main_path,'input')

# main data paths - default case
DLS_data_path = '2021_Kikstra/DLS_dataproducts_combined_default.csv' 
MISO2_MFA_data_path = 'MISO2/MISO2_global_v1_enduse_stockGasEoL_1950_2016.csv' 
country_correspondence_filename = 'country_correspondence.xlsx'

# LOW case (for sensitivity analysis)
# DLS_data_path = '2021_Kikstra/DLS_dataproducts_combined_low.csv' 
# MISO2_MFA_data_path = 'MISO2/MISO2_global_v1_enduse_Lifetimes_Low_scenario_stockGasEoL_1950_2016.csv' 

# # HIGH case (for sensitivity analysis)
# DLS_data_path = '2021_Kikstra/DLS_dataproducts_combined_high.csv' 
# MISO2_MFA_data_path = 'MISO2/MISO2_global_v1_enduse_Lifetimes_High_scenario_stockGasEoL_1950_2016.csv' 

# Add module folder to sys.path
module_path = os.path.join(main_path, 'module')
if module_path not in sys.path:
    sys.path.append(module_path)
    
from EDITS_DLS_load_data_v1 import load_country_correspondence_dict, load_population_2015, load_ew_MFA_data,load_DLS_2015
from EDITS_DLS_functions_v3 import singleplot_predef_data_satur_bounded, singleplot_predef_data_satur_bounded_popweight





''' ##############################################
        1 - LOAD MISO2 ECONOMY-WIDE MATERIAL STOCK 
        DATA AND PREPARE FOR REGRESSIONS 
    ################################################ '''
    
# country corespondence MISO2 (Wiedernhofer et al., 2024) & DLS data (Kikstra et al., 2021/2024)                    
country_correspondence, country_correspondence_dict = load_country_correspondence_dict(country_correspondence_filename , input_path)

# load population and material stock data (Wiedernhofer et al., 2024)
MISO2_population_2015, MISO2_population_2015_subset = load_population_2015(filename='MISO2/MISO2_population.xlsx', path_name=input_path, sheet_name='values', country_correspondence=country_correspondence)
MISO2_stocks_GAS_2015 = load_ew_MFA_data(filename=MISO2_MFA_data_path, path_name=input_path,  country_correspondence=country_correspondence)
MISO2_stocks_GAS_2015.reset_index(inplace = True)
MISO2_stocks_2015 = MISO2_stocks_GAS_2015[MISO2_stocks_GAS_2015['name'] == 'S10_stock_enduse'][['region', 'name', 'material', 'sector','2015']]      

## drop aggregates in road and civil engineering basecourses from MISO2_stocks_2015 and MISO2_stocks_GAS_2015 because too uncertain
MISO2_stocks_2015 = MISO2_stocks_2015[~((MISO2_stocks_2015.material.isin(['aggr_virgin','aggr_downcycl'])) & (MISO2_stocks_2015.sector == 'Roads'))]  
MISO2_stocks_2015 = MISO2_stocks_2015[~((MISO2_stocks_2015.material.isin(['aggr_virgin','aggr_downcycl'])) & (MISO2_stocks_2015.sector == 'Civil_engineering_except_roads'))]                                  

# sum: stock total per country
MISO2_stocks_2015_total = MISO2_stocks_2015.set_index(['region', 'name', 'material', 'sector']).groupby('region').sum()
# sum: stock total per country and end-use
MISO2_stocks_2015_total_endUse = MISO2_stocks_2015.set_index(['region', 'name', 'material', 'sector']).groupby(['region','sector']).sum()
# sum: stock total per country and material
MISO2_stocks_2015_total_material = MISO2_stocks_2015.set_index(['region', 'name', 'material', 'sector']).groupby(['region','material']).sum()

# total per capita stocks by country
MISO2_stocks_2015_total_cap = MISO2_stocks_2015_total/MISO2_population_2015_subset
#per capita stocks per region, end-use and material for 2015
MISO2_stocks_2015_grouped = MISO2_stocks_2015.set_index(['region','name', 'material', 'sector']).groupby(['region', 'material'])
intermed_list = []
for group_key, group_data in MISO2_stocks_2015_grouped:
    df = (MISO2_stocks_2015_grouped.get_group(group_key)/MISO2_population_2015_subset[MISO2_population_2015_subset.index == group_key[0]])
    intermed_list.append(df)
MISO2_stocks_2015_cap = pd.concat(intermed_list)
assert(abs((MISO2_stocks_2015_total_cap - MISO2_stocks_2015_cap.groupby('region').sum()).sum().values) < 0.00001)
# pivot
MISO2_stocks_2015_cap_piv = MISO2_stocks_2015_cap.reset_index().pivot(index=['region', 'sector'],\
                                                                              columns='material', values='2015')
MISO2_stocks_2015_cap_piv_noEndUse = MISO2_stocks_2015_cap_piv.groupby(['region']).sum()

MISO2_stocks_2015_cap_4mats = MISO2_stocks_2015_cap.copy()
MISO2_stocks_2015_cap_4mats = MISO2_stocks_2015_cap_4mats.reset_index().pivot(index=['region', 'sector'],\
                                                                              columns='material', values='2015') 

#harmonize materials to 4 groups
MISO2_stocks_2015_cap_4mats['biomass'] = MISO2_stocks_2015_cap_4mats[['paper','wood']].sum(axis=1)
MISO2_stocks_2015_cap_4mats['fossils'] = MISO2_stocks_2015_cap_4mats[['bitumen','plastic']].sum(axis=1)
MISO2_stocks_2015_cap_4mats['metals'] = MISO2_stocks_2015_cap_4mats[['aluminum', 'iron_steel', 'lead', 'manganese',
'metals_other', 'nickel', 'zinc', 'chromium', 'copper','tin']].sum(axis=1)
#! for minerals, we do not include  'aggr_downcycl', 'aggr_virgin' in foundations
MISO2_stocks_2015_cap_4mats['minerals'] = MISO2_stocks_2015_cap_4mats[['asphalt', 'bricks', 'cement', 'concrete',
 'glass_cont', 'glass_flat','aggr_virgin','aggr_downcycl']].sum(axis=1)
MISO2_stocks_2015_cap_4mats = MISO2_stocks_2015_cap_4mats[['biomass','fossils','metals','minerals']]

MISO2_stocks_2015_cap_4mats_total = MISO2_stocks_2015_cap_4mats.groupby('region').sum()





''' ##############################################
        2 - LOAD DLS INDICATOR DATA 
             AND PREPARE FOR REGRESSIONS 
    ################################################ '''
    
#load DLS data (Kikstra et al., 2025 updated)   
DLS_data_path = DLS_data_path
DLS_2015_funct_prov, DLS_2015_thresh = load_DLS_2015(filename=DLS_data_path, path_name=input_path,  country_correspondence=country_correspondence, country_correspondence_dict = country_correspondence_dict)

# calculate: DLS REACHED [%] (in percent)
DLS_reached_perc = pd.DataFrame(DLS_2015_funct_prov / DLS_2015_thresh)
DLS_reached_perc.index = DLS_reached_perc.index.droplevel([2])
DLS_reached_perc_housing = DLS_reached_perc[DLS_reached_perc.index.get_level_values('variable') == 'Housing|total']
DLS_reached_perc_housing.index = DLS_reached_perc_housing.index.droplevel([1])

# calculate DLS index mean_all =  the mean over all DLS dimensions
DLS_reached_index = pd.DataFrame()
DLS_reached_index['mean_all'] = DLS_reached_perc.groupby(['region']).mean()





''' ########################################
         3 - OTHER PREPARATIONS
    #########################################'''


R11_country_correspondence = pd.read_excel((os.path.join(input_path, country_correspondence_filename)))[['MISO2','R11']]
#subset sample removing WEU, NA, PAO based on R11 to yield subset for Global South
R11_country_correspondence_South = R11_country_correspondence[~R11_country_correspondence['R11'].isin(['WEU', 'NAM', 'PAO', 'EEU', 'FSU'])] 
#remove island states and small oil-rich countries + OECD
island_oil_states = ['Maldives','Fiji','Cape Verde','Trinidad and Tobago',\
                      'Dominican Republic','Mauritius','Bahamas','Reunion',\
                       'Martinique','Bahrain','Saudi Arabia', 'Hong Kong SAR',\
                           'United Arab Emirates', 'Qatar','Brunei','Singapore',\
                               'Kuwait','Guadeloupe', 'Oman','Libya', 'South Korea',\
                                   'Israel']
    
def prepare_labels(x,country_correspondence_dict):
    labels=x.copy().reset_index()
    for key, value in country_correspondence_dict.items():
        labels['region'].replace({value:key}, inplace=True)
    labels = labels.region.to_list()
    return labels




#remove island states and small oil-rich countries + OECD

''' ########################################
        4 - PLOTS: DLS_index (mean_all) vs. 
        MISO2 ECONOMY-WIDE STOCKS (TOTAL)
    #########################################'''
    
    

# without modification
x = MISO2_stocks_2015_cap_4mats_total.sum(axis=1)
y_select = DLS_reached_index.sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = 'DLS_index vs. TOTAL stocks (no aggr.)'
x_label = '[ton/cap]'
y_label = 'DLS_index (mean all)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
title = 'DLS_index vs. TOTAL stocks (no aggr., pop.weighted)'
singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)



#remove DLS_reached >95%
# without modification
x = MISO2_stocks_2015_cap_4mats_total.sum(axis=1)
#y_select = DLS_reached_index
y_select = y_select[y_select < 0.95].dropna().sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = 'DLS_index vs. TOTAL stocks (no aggr., subsampe: <95%DLS)'
x_label = '[ton/cap]'
y_label = 'DLS_index (mean all)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)



# modified: remove Global North Countries, island & oil states
# proof: subsample hardly different to full sample
x = MISO2_stocks_2015_cap_4mats_total.sum(axis=1)
y_select = DLS_reached_index
y_select = y_select[y_select.index.isin(R11_country_correspondence_South.MISO2.to_list())]
y_select = y_select[~y_select.index.isin(island_oil_states)] 
x = x[x.index.isin(y_select.index.to_list())]
title = 'DLS_index vs. TOTAL stocks (no aggr., subsample)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)



''' ########################################
        5 - PLOTS: DLS_index (mean_all) vs. 
        MISO2 ECONOMY-WIDE STOCKS 
        (BY FOUR MATERIAL CATEGORIES)
    #########################################'''



# BIOMASS

x = MISO2_stocks_2015_cap_4mats_total['biomass']
y_select = DLS_reached_index.sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(a) DLS_index vs. BIOMASS stocks'
x_label = '[ton/cap]'
y_label = 'DLS_index (mean all)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs. BIOMASS stocks (pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)

#remove outliers
x = MISO2_stocks_2015_cap_4mats_total['biomass'][MISO2_stocks_2015_cap_4mats_total['biomass'] < 15]
y_select = y_select[y_select.index.isin(x.index)].sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(b) DLS_index vs. BIOMASS stocks (outliers removed)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs. BIOMASS stocks (outliers removed, pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)



# FOSSILS

x = MISO2_stocks_2015_cap_4mats_total['fossils']
y_select = DLS_reached_index.sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(c) DLS_index vs. FOSSILS stocks'
x_label = '[ton/cap]'
y_label = 'DLS_index (mean all)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs. FOSSILS stocks (pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)

#remove outliers
x = MISO2_stocks_2015_cap_4mats_total['fossils'][MISO2_stocks_2015_cap_4mats_total['fossils'] < 4]
y_select = y_select[y_select.index.isin(x.index)].sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(d) DLS_index vs. FOSSILS stocks (outliers removed)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs.  FOSSILS stocks (outliers removed, pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)



# METALS

x = MISO2_stocks_2015_cap_4mats_total['metals']
y_select = DLS_reached_index.sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(e) DLS_index vs. METALS stocks'
x_label = '[ton/cap]'
y_label = 'DLS_index (mean all)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs. METALS stocks (pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)

#remove outliers
x = MISO2_stocks_2015_cap_4mats_total['metals'][MISO2_stocks_2015_cap_4mats_total['metals'] < 25]
y_select = y_select[y_select.index.isin(x.index)].sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(f) DLS_index vs. METALS stocks (outliers removed)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs. METALS stocks (outliers removed, pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)



# MINERALS
x = MISO2_stocks_2015_cap_4mats_total['minerals']
y_select = DLS_reached_index.sort_index()
x = x[x.index.isin(y_select.index.to_list())].sort_index()
pop = MISO2_population_2015_subset[MISO2_population_2015_subset.index.isin(y_select.index.to_list())].sort_index()
title = '(g) DLS_index vs. MINERALS stocks'
x_label = '[ton/cap]'
y_label = 'DLS_index (mean all)'
labels = prepare_labels(x,country_correspondence_dict)
singleplot_predef_data_satur_bounded(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, labels)
# title = 'DLS_index vs. MINERALS stocks (pop.weighted)'
# singleplot_predef_data_satur_bounded_popweight(x.values.flatten(), y_select.values.flatten(), y_label, x_label, title, pop.values.flatten() , labels)