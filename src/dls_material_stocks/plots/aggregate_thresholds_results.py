# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 18:05:27 2025

@author: jstreeck
"""


'''#################################################################

       Description:
        This script assembles the DLS stock thresholds from the
        result files created with the script run_from_pickle_v1
        - which are located in a specified folder - into data
        frames for the global, regional and country-level, in order
        to compare them.
        
       Note:
        You might need to adjust the paths.
        Paths can be adjusted to include different sets of result scenarios.
                
   #################################################################'''


import os
import pandas as pd


# paths
main_path = os.getcwd()
parent_path = os.path.dirname(main_path)
parent_path = os.path.dirname(parent_path)
parent_path = os.path.dirname(parent_path)
input_path = os.path.join(parent_path,'input')

# Path to the folder with Excel files
#folder_path = os.path.join(parent_path,'output/DLS stock threshold divergence decomp') # only includes results from DLS stock threshold decomposition (see SI section 2.4)
folder_path = os.path.join(parent_path,'output') # uncomment if wanna include all result outputs



### REGIONAL



# Prepare an empty list to store the DataFrames
dfs_threshold = []
dfs_converge = []
file_names = []

# Loop through all Excel files in the folder
for file in os.listdir(folder_path):
    # Skip non-Excel files
    try:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path = os.path.join(folder_path, file)
            file_name = os.path.splitext(file)[0]  # Name without extension
            
            # Remove the undesired substring from the file name
            clean_name = file_name.replace("results_data_supplement_", "")
            
            # Read the relevant sheet with appropriate index columns
            df = pd.read_excel(file_path, sheet_name="Fig2a_map_reg_stocks_dim", engine='openpyxl',
                               index_col=[0, 1, 2])
    
            # Extract the relevant columns and rename with cleaned name
            threshold = df["DLSstock_threshold"].rename(clean_name)
            dfs_threshold.append(threshold)
    
            # Check if the converge column exists
            if "DLSstock_threshold_converge" in df.columns:
                converge = df["DLSstock_threshold_converge"].rename(clean_name)
                dfs_converge.append(converge)
            else:
                print(f"Note: 'DLSstock_threshold_converge' not found in {clean_name}")
            
            file_names.append(clean_name)
            
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")
    
# Combine all threshold columns into a single DataFrame
df_threshold_all_reg = pd.concat(dfs_threshold, axis=1) # by world region and DLS dimension
df_threshold_all_reg_tot = df_threshold_all_reg.groupby('region').sum() # by world region with DLS dimensions summed

# Combine all converge columns into a single DataFrame if any were found
df_converge_all_reg = pd.concat(dfs_converge, axis=1) if dfs_converge else None




## GLOBAL



# Prepare an empty list to store the DataFrames
dfs_threshold_glob = []
file_names_glob = []

# Loop through all Excel files in the folder
for file in os.listdir(folder_path):
    # Skip non-Excel files
    try:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path_glob = os.path.join(folder_path, file)
            file_name_glob = os.path.splitext(file)[0]  # Name without extension
            
            # Remove the undesired substring from the file name
            clean_name = file_name_glob.replace("results_data_supplement_", "")
        
            # Read the relevant sheet with appropriate index columns
            df_glob = pd.read_excel(file_path_glob, sheet_name="global_av_DLSstock_thresh", engine='openpyxl',
                               index_col=[0, 1])
            
            # Extract the relevant columns
            threshold_glob = df_glob["DLSstock_threshold"].rename(clean_name)
            #print(threshold_glob )
            dfs_threshold_glob.append(threshold_glob)
            file_names_glob.append(clean_name)
            
    except Exception as e:
        print(f"Skipping {file_name_glob} due to error: {e}")
    
# Combine all threshold columns into a single DataFrame
df_threshold_all_glob = pd.concat(dfs_threshold_glob, axis=1)




## COUNTRIES



# Prepare an empty list to store the DataFrames
dfs_threshold_countr = []
file_names_countr = []

# Loop through all Excel files in the folder
for file in os.listdir(folder_path):
    # Skip non-Excel files
    try:
        if file.endswith(".xlsx") or file.endswith(".xls"):
            file_path_countr = os.path.join(folder_path, file)
            file_name_countr = os.path.splitext(file)[0]  # Name without extension
            
            # Remove the undesired substring from the file name
            clean_name = file_name_countr.replace("results_data_supplement_", "")
        
            # Read the relevant sheet with appropriate index columns
            df_countr= pd.read_excel(file_path_countr, sheet_name="Fig2bc_stock_distr_countr", engine='openpyxl',
                               index_col=[0, 1, 2])
            
            # Extract the relevant columns
            threshold_countr= df_countr["(c) DLS material stocks, threshold"].rename(clean_name)
            #print(threshold_countr)
            dfs_threshold_countr.append(threshold_countr)
            file_names_countr.append(clean_name)
            
    except Exception as e:
        print(f"Skipping {file_name_countr} due to error: {e}")
    
# Combine all threshold columns into a single DataFrame
df_threshold_all_countr= pd.concat(dfs_threshold_countr, axis=1)

df_threshold_all_countr_tot = df_threshold_all_countr.groupby('region').sum()
