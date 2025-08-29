"""Calculate bottom-up DLS material stocks for different scenario cases.
@author: jstreeck
"""

from pathlib import Path
from copy import deepcopy
import json
from datetime import datetime
import traceback

from dls_material_stocks.analysis.DLS_materials_bottomUp_flexible_v1 import (
    calc_bottomUp_DLSS,
)
from dls_material_stocks.analysis.DLS_materials_bottomUp_flexible_v1_RECCmi import (
    calc_bottomUp_DLSS_RECCmi,
)
from dls_material_stocks.scenarios.scenario_config import (
    get_scenario_config,
    update_sensitivity,
)

def create_scenario_data(
    scenarios_to_run,
    input_path,
    sensitivity=None,
    creation_function=calc_bottomUp_DLSS,
    output_paths=None,
    metadata_file="scenario_outputs.json",
):
    metadata_path = Path(output_path, metadata_file)
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    for scenario in scenarios_to_run:

        scenario_key = scenario
        
        try:
            config = deepcopy(get_scenario_config(scenario, output_path, input_path))

            if config is None:
                print(f"No scenario with name {scenario} found")
                continue

            if sensitivity is not None:
                update_sensitivity(config, sensitivity)

            if output_paths is not None:
                config["output_paths"] = output_paths[scenario]

            print(f"Starting to calculate {scenario_key}")

            creation_function(
                input_paths_dict=config["input_paths"],
                output_paths_dict=config["output_paths"],
                sensitivity_factors=config["sensitivity"],
                change_DLS_thresholds=config["change_DLS_thresholds"],
                process_change=config["process_change"],
            )

            metadata[scenario_key] = {
                "scenario_name": scenario,
                "sensitivity": config["sensitivity"],
                "output_files": {
                    "provided": str(config["output_paths"]["provided"]),
                    "threshold": str(config["output_paths"]["threshold"]),
                },
                "created_at": datetime.now().isoformat(),
                "creation_function": creation_function.__name__,
                "input_paths": {k: str(v) for k, v in config["input_paths"].items()},
                "change_DLS_thresholds": str(config["change_DLS_thresholds"]),
            }
        except FileNotFoundError as e:
            print(repr(e))
            traceback.print_exc()
            
            metadata[scenario_key] = {
                "scenario_name": scenario,
                "created_at": "Not created, missing inputs",
                "error": repr(e)
            }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        print(f"Finished calculating {scenario_key}")
        

def create_scenario_data_recc_mi(
    scenarios_to_run,
    input_path,
    sensitivity=None,
    creation_function=calc_bottomUp_DLSS_RECCmi,
    output_paths=None,
    metadata_file="scenario_outputs.json",
):
    metadata_path = Path(output_path, metadata_file)
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    for scenario in scenarios_to_run:

        scenario_key = scenario
        
        try:
            config = deepcopy(get_scenario_config(scenario, output_path, input_path))

            if config is None:
                print(f"No scenario with name {scenario} found")
                continue

            if sensitivity is not None:
                update_sensitivity(config, sensitivity)

            if output_paths is not None:
                config["output_paths"] = output_paths[scenario]

            print(f"Starting to calculate {scenario_key}")

            creation_function(
                input_paths_dict=config["input_paths"],
                output_paths_dict=config["output_paths"],
                sensitivity_factors=config["sensitivity"],
                change_DLS_thresholds=config["change_DLS_thresholds"],
                process_change=config["process_change"],
            )

            metadata[scenario_key] = {
                "scenario_name": scenario,
                "sensitivity": config["sensitivity"],
                "output_files": {
                    "provided": str(config["output_paths"]["provided"]),
                    "threshold": str(config["output_paths"]["threshold"]),
                },
                "created_at": datetime.now().isoformat(),
                "creation_function": creation_function.__name__,
                "input_paths": {k: str(v) for k, v in config["input_paths"].items()},
                "change_DLS_thresholds": str(config["change_DLS_thresholds"]),
            }
        except FileNotFoundError as e:
            print(repr(e))
            traceback.print_exc()
            
            metadata[scenario_key] = {
                "scenario_name": scenario,
                "created_at": "Not created, missing inputs",
                "error": repr(e)
            }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        print(f"Finished calculating {scenario_key}")


project_root = Path(__file__).parent.parent
input_path = Path(project_root, "input")
output_path = Path(project_root, "output", "scenario_data")



"""##################### 2A ##########################################
        Calculate DLSS for CURRENT practices with MATERIAL EFFICIENCY
   ###################################################################

    Available scenario categories:

- current: Base scenario with current national/regional practices
- current_lowmeat: Current practices with low-meat diet scenario
- current_vegetarian: Current practices with vegetarian diet scenario  
- current_vegan: Current practices with vegan diet scenario
- current_b2ds: Current practices with IEA B2DS scenario modal shares for mobility
- current_lowcar: Current practices with low car use
- current_low_cardemand: Current practices with low car use & demand reduction (90% motorized transport)
- current_evs: Current practices with electric light-duty vehicles for cars & two/three wheelers
- current_re: Current practices with renewable energy (100% electricity, renewable stock intensities)
- current_lightweight: Current practices with buildings light weighting
- current_woodbased: Current practices with wood-based buildings  
- current_lightwood: Current practices with buildings light weighting & wood-based
- current_hhsizehigh: Current practices with high household size (maximum of 2015 value)
- current_hhsizemedium: Current practices with medium household size (median of 2015 value)
- current_hhsizelow: Current practices with low household size (lowest of 2015 value)
- current_combined: Combined scenario with vegan diet, low car low demand, EVs, renewable energy, 
                   and lightweight & wood-based buildings (excludes converged scenario changes)
 """

scenarios_to_run = [
    "current",
    "current_lowmeat",
    "current_vegetarian",
    "current_vegan",
    "current_b2ds",
    "current_lowcar",
    "current_low_cardemand",
    "current_evs",
    "current_re",
    "current_lightweight",
    "current_woodbased",
    "current_lightwood",
    "current_hhsizehigh",
    "current_hhsizemedium",
    "current_hhsizelow",
    "current_combined",
]

create_scenario_data(scenarios_to_run, input_path=input_path)



"""##################### 2B ############################################
        Calculate DLSS for CONVERGED practices with MATERIAL EFFICIENCY
   #####################################################################

    Available converged scenario categories:

- converged: Base converged scenario with median household size, B2DS modal shares, and low-meat diet
- converged_vegt: Converged practices with vegetarian diet scenario
- converged_vegan: Converged practices with vegan diet scenario
- converged_lowcar: Converged practices with low car use (B2DS modal shares)
- converged_lowcardemand: Converged practices with low car use & demand reduction (90% motorized transport)
- converged_evs: Converged practices with electric light-duty vehicles for cars & two/three wheelers
- converged_re: Converged practices with renewable energy (100% electricity, renewable stock intensities)
- converged_lightweight: Converged practices with buildings light weighting
- converged_woodbased: Converged practices with wood-based buildings
- converged_lightwood: Converged practices with buildings light weighting & wood-based
- converged_hhsizehigh: Converged practices with high household size (maximum of 2015 value)
- converged_hhsizelow: Converged practices with low household size (lowest of 2015 value)  
- converged_combined: Combined converged scenario with vegan diet, low car low demand, EVs, renewable energy,
                     and lightweight & wood-based buildings
 """

converged_scenarios_to_run = [
    "converged",
    "converged_vegt",
    "converged_vegan",
    "converged_lowcar",
    "converged_lowcardemand",
    "converged_evs",
    "converged_re",
    "converged_lightweight",
    "converged_woodbased",
    "converged_lightwood",
    "converged_hhsizehigh",
    "converged_hhsizelow",
    "converged_combined",
]

create_scenario_data(converged_scenarios_to_run, input_path=input_path)



"""##################### 2C ############################################
     DECOMPOSITION ANALYSIS SCENARIOS - Isolate specific parameter effects
   #####################################################################

    Available decomposition analysis scenario categories:

- current_pkm_same: Current practices with equal transport demand (person-kilometers) across all countries
- current_energy_same: Current practices with equal energy parameters (stock intensities, fuel shares, processes) across countries
- current_educ_same: Current practices with equal education shares (schoolgoing population) across all countries
- current_bmi_same: Current practices with equal building material intensities across all countries
- current_ecoinv_same: Current practices with equal ecoinvent processes (indirect material intensities) across countries
- current_arch_same: Current practices with equal building archetype shares across all countries
- current_vocc: Current practices with equal vehicle occupancy rates across all countries
- current_same_combined: Combined decomposition scenario with all global averages applied
- current_same_combined_not_school: Combined decomposition scenario excluding education normalization
 """

decomposition_scenarios_to_run = [
    "current_pkm_same",
    "current_energy_same",
    "current_educ_same",
    "current_bmi_same",
    "current_ecoinv_same",
    "current_arch_same",
    "current_vocc",
    "current_same_combined",
    "current_same_combined_not_school",
]

create_scenario_data(decomposition_scenarios_to_run, input_path=input_path)



"""##################### 3 ############################################
        SENSITIVITY ANALYSIS SCENARIOS - Test parameter robustness
   #####################################################################

    Dynamic sensitivity analysis using existing scenario configurations:
    - Reuses 'current' and 'converged' base scenarios with modified DLS thresholds
 """

create_scenario_data(
    scenarios_to_run=["currentDLSplus", "convergedDLSplus"],
    input_path=input_path,
    sensitivity=1.25,
)

create_scenario_data(
    scenarios_to_run=["currentDLSless", "convergedDLSless"],
    input_path=input_path,
    sensitivity=0.75,
)



""" ### EW-MFA MISO2 stock LIFETIME +/-30% - is adapted in script run_scenarios.py ###"""



""" ###  - Residential building MIs Pauliuk et al. (2021) ### """

create_scenario_data(
    scenarios_to_run=["current_recc_mi", "converged_recc_mi"],
    input_path=input_path,
    creation_function=calc_bottomUp_DLSS_RECCmi,
)

create_scenario_data(
    scenarios_to_run=["current_recc_mi", "converged_recc_mi"],
    input_path=input_path,
    creation_function=calc_bottomUp_DLSS_RECCmi,
)