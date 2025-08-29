import traceback
import os
import pandas as pd
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
# Set non-interactive backend before importing pyplot. This prevents pyplot from opening plots
# if you want to interactively explore those, comment out this line
from dls_material_stocks.analysis.analysis_flexible_v1 import run_analysis
import json

"""DLS Material Stocks Analysis Runner

This script calculates the final results and creates the figures in manuscript / 
supplementary and the results data output file for different scenario cases.

The script runs analysis on scenarios with pickled DLS stock data using a metadata-driven 
approach. Scenarios are organized into categories:

1. Base cases: current national/regional practices & converged practices
2. Current practices with material efficiency variations (diet, mobility, housing, etc.)
3. Converged practices with material efficiency variations  
4. Decomposition analysis scenarios to isolate parameter effects
5. Sensitivity analysis scenarios for robustness testing

Variables:
    current_DLS_stocks_prov: Currently existing DLS stocks (2015)
    converge_DLS_stocks_thresh: Thresholds of DLS stocks to reach DLS for all
    activate_converge: Lever to indicate whether converged practices analysis should
                      be calculated in addition to current practices (inactive for None)

Created on Tue Mar 25 17:39:33 2025
@author: jstreeck
"""


def load_scenario_metadata(data_path, metadata_filename="scenario_outputs.json"):
    """Load scenario metadata from JSON file."""
    metadata_path = Path(data_path) / metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def run_scenario_analysis(
    data_path,
    DLS_stocks_thresh_converge=None,
    converge=None,
    scenario_names=None,
    output_suffix="",
    MISO2_MFA_data_path=None,
    provided_path=None,
    threshold_path=None,
    save_results=None,
    conv_gap_mode=False,
):
    """Run analysis on scenarios from pickled data and metadata.
    If not provided, MISO2_MFA_data_path, providd_path and threshold_path to pickled files are assumed to be present in the config."""

    metadata = load_scenario_metadata(data_path)

    if scenario_names is None:
        scenarios_to_run = list(metadata.keys())
    else:
        scenarios_to_run = scenario_names

    for scenario_key in scenarios_to_run:
        if scenario_key not in metadata:
            print(f"Scenario {scenario_key} not found in metadata")
            continue

        scenario_info = metadata[scenario_key]

        if "error" in scenario_info:
            print(f"Skipping {scenario_key} - had creation errors")
            continue

        print(f"Running analysis for scenario: {scenario_key}")

        provided_path = Path(scenario_info["output_files"]["provided"] + ".pkl")
        threshold_path = Path(scenario_info["output_files"]["threshold"] + ".pkl")

        DLS_stocks_prov = pd.read_pickle(provided_path)
        DLS_stocks_thresh = pd.read_pickle(threshold_path)

        if DLS_stocks_thresh_converge is not None:
            DLS_stocks_thresh_converge = pd.read_pickle(
                str(DLS_stocks_thresh_converge) + ".pkl"
            )

        # TODO the .pkl suffix should be added in creation, not here

        analysis_output_path = Path(
            output_path, f"results_data_supplement_{scenario_key}{output_suffix}.xlsx"
        )

        if MISO2_MFA_data_path is None:
            MISO2_MFA_data_path = scenario_info["input_paths"]["MISO2_stock_Gas_EoL"]

        try:
            run_analysis(
                scenario_name=scenario_key,
                DLS_stocks_prov=DLS_stocks_prov,
                DLS_stocks_thresh=DLS_stocks_thresh,
                activate_converge=converge,
                DLS_stocks_thresh_converge=DLS_stocks_thresh_converge,
                input_paths_dict=scenario_info["input_paths"],
                output_path=analysis_output_path,
                MISO2_MFA_data_path=MISO2_MFA_data_path,
                conv_gap_mode=conv_gap_mode,
                save_results=save_results
            )
            print(f"Analysis completed for {scenario_key}")
        except Exception as e:
            print(f"Error in analysis for {scenario_key}: {str(e)}")
            print(traceback.format_exc())
            

# use this function if only interested to calculate thresholds (avoids negative gaps if thresholds lower than current provision in scenarios)
def run_scenario_analysis_threshOnly(
    data_path,
    DLS_stocks_thresh_converge=None,
    converge=None,
    scenario_names=None,
    output_suffix="",
    MISO2_MFA_data_path=None,
    provided_path=None,
    threshold_path=None,
    save_results=None,
    conv_gap_mode=False,
):
    """Run analysis on scenarios from pickled data and metadata.
    If not provided, MISO2_MFA_data_path, providd_path and threshold_path to pickled files are assumed to be present in the config."""

    metadata = load_scenario_metadata(data_path)

    if scenario_names is None:
        scenarios_to_run = list(metadata.keys())
    else:
        scenarios_to_run = scenario_names

    for scenario_key in scenarios_to_run:
        if scenario_key not in metadata:
            print(f"Scenario {scenario_key} not found in metadata")
            continue

        scenario_info = metadata[scenario_key]

        if "error" in scenario_info:
            print(f"Skipping {scenario_key} - had creation errors")
            continue

        print(f"Running analysis for scenario: {scenario_key}")

        provided_path = Path(scenario_info["output_files"]["provided"] + ".pkl")
        threshold_path = Path(scenario_info["output_files"]["threshold"] + ".pkl")

        DLS_stocks_prov = pd.read_pickle(provided_path)
        DLS_stocks_thresh = pd.read_pickle(threshold_path)

        if DLS_stocks_thresh_converge is not None:
            DLS_stocks_thresh_converge = pd.read_pickle(
                str(DLS_stocks_thresh_converge) + ".pkl"
            )

        # TODO the .pkl suffix should be added in creation, not here

        analysis_output_path = Path(
            output_path, f"results_data_supplement_{scenario_key}{output_suffix}.xlsx"
        )

        if MISO2_MFA_data_path is None:
            MISO2_MFA_data_path = scenario_info["input_paths"]["MISO2_stock_Gas_EoL"]

        try:
            run_analysis(
                scenario_name=scenario_key,
                DLS_stocks_prov=DLS_stocks_thresh,
                DLS_stocks_thresh=DLS_stocks_thresh,
                activate_converge=converge,
                DLS_stocks_thresh_converge=DLS_stocks_thresh_converge,
                input_paths_dict=scenario_info["input_paths"],
                output_path=analysis_output_path,
                MISO2_MFA_data_path=MISO2_MFA_data_path,
                conv_gap_mode=conv_gap_mode,
                save_results=save_results
            )
            print(f"Analysis completed for {scenario_key}")
        except Exception as e:
            print(f"Error in analysis for {scenario_key}: {str(e)}")
            print(traceback.format_exc())


def get_data_path(meta_data_path, scenario_name, data_name="threshold"):
    if data_name not in ["threshold", "provided"]:
        raise ValueError(f"Uknown type of data: {data_name}")
    metadata = load_scenario_metadata(meta_data_path)
    scenario_info = metadata[scenario_name]
    return Path(scenario_info["output_files"][data_name])


if __name__ == "__main__":
    print("Calculating scenarios from pickled data")
    project_root = Path(__file__).parent.parent
    input_path = Path(project_root, "input")
    output_path = Path(project_root, "output")
    os.makedirs(output_path, exist_ok=True)
    scenarios_data_path = Path(output_path, "scenario_data")




    """##################### 0 ####################################
    Calculate the trajectory of DLS gap closure for the main case
                    'converged practices'
    ############################################################"""

    # # EXISTING DLSS: current practices, DLSS THRESHOLD: converged practices
    # # in this special case, the DLS stock gaps can be negative, because converged
    # # practices are more material-efficient, than current ones (and thus the DLSS
    # # thresholds lower than currently existing DLSS stocks);
    # # using the script run_analysis_convTime implies special handling of these
    # # negatives, only to produce the trajectories for figure5c in main manuscript
    
    
    ## currently yields same results as normal  run?
    
    converge_DLS_stocks_thresh = get_data_path(
        scenarios_data_path, "converged", "threshold"
    )

    # run_scenario_analysis(
    #     data_path=scenarios_data_path,
    #     scenario_names=["current"],
    #     threshold_path=converge_DLS_stocks_thresh,
    #     converge=None,
    #     DLS_stocks_thresh_converge=converge_DLS_stocks_thresh,
    #     output_suffix="gapConverged",
    #     conv_gap_mode=True,
    #     save_results=["Cover","Fig5c_closeGap_Glob_curr", "Fig5c_closeGap_Reg_curr"]
    # )
    
    run_scenario_analysis(
        data_path=scenarios_data_path,
        scenario_names=["current"],
        threshold_path=converge_DLS_stocks_thresh,
        converge=None,
        output_suffix="gapConverged",
        conv_gap_mode=True,
        save_results=["Cover","Fig5c_closeGap_Glob_curr", "Fig5c_closeGap_Reg_curr"]
    )



    """##################### 1 #########################################
   Calculate the two base cases: current national/regional practices 
   & converged practices
   #################################################################"""
   
   
   
    ### Calculate DLSS with CURRENT practices + converged threshold estimate on top
    run_scenario_analysis(
        data_path=scenarios_data_path,
        scenario_names=["current"],
        output_suffix="_converged",
        DLS_stocks_thresh_converge=converge_DLS_stocks_thresh,
        converge="yes",
    )



    ### Calculate DLSS with CONVERGED practices ONLY FOR THRESHOLD EXTRACTION
    run_scenario_analysis_threshOnly(
        scenarios_data_path,
        scenario_names=["converged"],
        DLS_stocks_thresh_converge=None,
        converge=None,
        save_results=["Cover", "Fig2a_map_reg_stocks_dim", "Fig2bc_stock_distr_countr", "global_av_DLSstock_thresh"]
    )



    # Calculate DLSS for CURRENT practices with MATERIAL EFFICIENCY
    # Includes diet variations, mobility options, energy systems, building materials, and household sizes
    current_scenarios = [
        "current_lowmeat",  # Low meat diet scenario
        "current_vegetarian",  # Vegetarian diet scenario
        "current_vegan",  # Vegan diet scenario
        "current_b2ds",  # IEA B2DS scenario modal shares for mobility
        "current_lowcar",  # Low car use
        "current_low_cardemand",  # Low car use & demand reduction (90% motorized transport)
        "current_evs",  # Electric light-duty vehicles for cars & two/three wheelers
        "current_re",  # Renewable energy (100% electricity, renewable stock intensities)
        "current_lightweight",  # Buildings light weighting
        "current_woodbased",  # Wood-based buildings
        "current_lightwood",  # Light weighting & wood-based buildings
        "current_hhsizehigh",  # High household size (maximum of 2015 value)
        "current_hhsizemedium",  # Medium household size (median of 2015 value)
        "current_hhsizelow",  # Low household size (lowest of 2015 value)
        "current_combined",  # Combined: vegan diet, low car low demand, EVs, renewable energy, lightweight & wood-based buildings
    ]

    run_scenario_analysis_threshOnly(scenarios_data_path, scenario_names=current_scenarios, save_results=["Cover", "Fig2a_map_reg_stocks_dim", "Fig2bc_stock_distr_countr", "global_av_DLSstock_thresh"])



    # Calculate DLSS for CONVERGED practices with MATERIAL EFFICIENCY
    # Uses median household size, B2DS modal shares, and low-meat diet as base, with additional variations
    converged_scenarios = [
        "converged_vegt",  # Vegetarian diet scenario
        "converged_vegan",  # Vegan diet scenario
        "converged_lowcar",  # Low car use (B2DS modal shares)
        "converged_lowcardemand",  # Low car use & demand reduction (90% motorized transport)
        "converged_evs",  # Electric light-duty vehicles for cars & two/three wheelers
        "converged_re",  # Renewable energy (100% electricity, renewable stock intensities)
        "converged_lightweight",  # Buildings light weighting
        "converged_woodbased",  # Wood-based buildings
        "converged_lightwood",  # Light weighting & wood-based buildings
        "converged_hhsizehigh",  # High household size (maximum of 2015 value)
        "converged_hhsizelow",  # Low household size (lowest of 2015 value)
        "converged_combined",  # Combined: vegan diet, low car low demand, EVs, renewable energy, lightweight & wood-based buildings
    ]

    run_scenario_analysis_threshOnly(scenarios_data_path, scenario_names=converged_scenarios, save_results=["Cover", "Fig2a_map_reg_stocks_dim", "Fig2bc_stock_distr_countr", "global_av_DLSstock_thresh"])



    ''' #################################################   3   #############################################################
        ###### DECOMPOSE DIFFERENCES IN REGIONAL AND COUNTRY DLSS THRESHOLDs -  FROM CURRENT NATIONAL / REGIONAL THRESHOLDS #
        ##################################################################################################################### '''


    # DECOMPOSITION ANALYSIS - Isolate specific parameter effects
    # Sets various parameters equal across all countries to identify sources of regional differences
    # Note: Some decomposition effects already covered in converged scenario (median household size, B2DS mobility, low meat diets)
    decomposition_scenarios = [
        "current_pkm_same",  # Equal transport demand (person-kilometers) across countries
        "current_energy_same",  # Equal energy parameters (stock intensities, fuel shares, processes) across countries
        "current_educ_same",  # Equal education shares (schoolgoing population) across countries
        "current_bmi_same",  # Equal building material intensities across countries
        "current_ecoinv_same",  # Equal ecoinvent processes (indirect material intensities) across countries
        "current_arch_same",  # Equal building archetype shares across countries
        "current_vocc",  # Equal vehicle occupancy rates across countries
        "current_same_combined",  # Combined decomposition with all global averages applied
        "current_same_combined_not_school",  # Combined decomposition excluding education normalization
    ]

    run_scenario_analysis_threshOnly(scenarios_data_path, scenario_names=decomposition_scenarios, save_results=["Cover", "Fig2a_map_reg_stocks_dim", "Fig2bc_stock_distr_countr", "global_av_DLSstock_thresh"])



    ''' ##############################  4   ##################################
        ###################### SENSITIVITY ANALYSES ##########################
        ###################################################################### '''
    # DLS thresholds +/-25%, EW-MFA stock lifetimes +/-30%, and alternative building material intensities


    ### 1a. DLS THRESHOLDS PLUS 25%
    converge_DLS_stocks_thresh_plus = get_data_path(
        scenarios_data_path, "convergedDLSplus", "threshold"
    )

    run_scenario_analysis(
        scenarios_data_path,
        scenario_names=["currentDLSplus"],
        DLS_stocks_thresh_converge=converge_DLS_stocks_thresh_plus,
        converge="yes",
    )
    

    ### 1b. DLS THRESHOLDS LESS 25%
    converge_DLS_stocks_thresh_less = get_data_path(
        scenarios_data_path, "convergedDLSless", "threshold"
    )

    run_scenario_analysis(
        scenarios_data_path,
        scenario_names=["currentDLSless"],
        DLS_stocks_thresh_converge=converge_DLS_stocks_thresh_less,
        converge="yes",
    )



    ### 2a. EW-MFA MISO2 stock LIFETIME LESS 30%
      
    # current practices for existing and threshold stocks + converged threshold estimate on top
    converge_DLS_stocks_thresh = get_data_path(
        scenarios_data_path, scenario_name="converged", data_name="threshold"
    )
    
    run_scenario_analysis(
        scenarios_data_path,
        scenario_names=["current"],
        DLS_stocks_thresh_converge=converge_DLS_stocks_thresh,
        converge="yes",
        output_suffix="_converged_LTLess",
        MISO2_MFA_data_path=Path(input_path, "MISO2", "MISO2_global_v1_enduse_Lifetimes_Low_scenario_stockGasEoL_1950_2016.csv"),
    )


    ### 2b. EW-MFA MISO2 stock LIFETIME PLUS 30%

    # current practices for existing and threshold stocks + converged threshold estimate on top
    converge_DLS_stocks_thresh = get_data_path(
        scenarios_data_path, scenario_name="converged", data_name="threshold"
    )
    
    run_scenario_analysis(
        scenarios_data_path,
        scenario_names=["current"],
        DLS_stocks_thresh_converge=converge_DLS_stocks_thresh,
        converge="yes",
        output_suffix="_converged_LTLplus",
        MISO2_MFA_data_path=Path(input_path, "MISO2", "MISO2_global_v1_enduse_Lifetimes_High_scenario_stockGasEoL_1950_2016.csv"),
    )



    # ''' ### 3 - Residential building MIs Pauliuk et al. (2021) ### '''

    # current national / regional practices
    run_scenario_analysis(scenarios_data_path, scenario_names=["current_recc_mi"])
    
    # converged practices
    converged_recc_mi_thres = get_data_path(
        scenarios_data_path, "converged_recc_mi", "threshold"
    )

    run_scenario_analysis_threshOnly(
        scenarios_data_path,
        scenario_names=["converged_recc_mi"],
        save_results=["Cover", "Fig2a_map_reg_stocks_dim", "Fig2bc_stock_distr_countr", "global_av_DLSstock_thresh"]
    )
    
    
    

