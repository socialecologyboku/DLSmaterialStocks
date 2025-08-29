"""Scenario configurations for DLS material stocks analysis.

This module provides centralized configuration for all DLS (Decent Living Standards) 
material stock scenarios analyzed in the research. 

@author: jstreeck
"""

from pathlib import Path
from copy import deepcopy

def dls_paths(variant="default"):
    return {
        "DLS_data_path": Path("2021_Kikstra", f"DLS_dataproducts_combined_{variant}.csv"),
        "DLE_provided_path": Path("2021_Kikstra", f"DLE_current_opcon_elecnonelec_dimensions_{variant}.csv"),
        "DLE_threshold_path": Path("2021_Kikstra", f"DLE_threshold_input_opcon_elecnonelec_dimensions_{variant}.csv"),
    }

def dls_paths_renewable(variant="default"):
    return {
        "DLS_data_path": Path("2021_Kikstra", f"DLS_dataproducts_combined_{variant}.csv"),
        "DLE_provided_path": Path("2021_Kikstra", "DLE_current_opcon_elecnonelec_dimensions_default_onlyELECTRICITY.csv"),
        "DLE_threshold_path": Path("2021_Kikstra", f"DLE_threshold_input_opcon_elecnonelec_dimensions_{variant}_onlyELECTRICITY.csv"),
    }

base_paths = {
    "housing_archetypes_path": Path("housing", "RECC_2.4_residential_archetypes_country_shares.xlsx"),
    "housing_mat_intens_path": Path("housing", "Haberl_MI_buildings_global.xlsx"),
    "ecoinvent_housing_reg_mapping_path": Path("housing", "housing_country_ecoinvent_process_correspondence.xlsx"),
    "ecoinvent_mobility_reg_mapping_path": Path("mobility", "mobility_country_ecoinvent_process_correspondence.xlsx"),
    "education_shares_path": Path("education", "processed_schoolgoing_population_share.csv"),
    "electricity_indstocks_path": Path("2023_Velez", "indirect_stock_intensities_electricity_v1.xlsx"),
    "fuel_shares_path": Path("energy", "2024_ourworldindata_share-energy-source-sub.csv"),
    "fuel_shares_reg_mapping_path": Path("energy", "share_energy_source_country_correspondence.xlsx"),
    "ecoinvent_coal_reg_mapping_path": Path("energy", "fuels_country_ecoinvent_process_correspondence.xlsx"),
    "fuels_indstocks_path": Path("2023_Velez", "indirect_stock_intensities_fuels_v1.xlsx"),
}

base_mobility = {
    "modal_shares_path": Path("mobility", "DLS_modal_share_version0_1_rc4_20240307.csv"),
    "occupancy_path": Path("mobility", "vehicle_occupancy_rates_R11.xlsx"),
}

converged_mobility = {
    "modal_shares_path": Path("mobility", "DLS_modal_share_B2DS_noAir.csv"),
    "occupancy_path": Path("mobility", "vehicle_occupancy_rates_R11_median.xlsx"),
}

diet_paths = {
    "current": Path("food", "dietary-composition-by-country_OURWORLDINDATA.csv"),
    "lowmeat": Path("food", "dietary-composition-by-country_Scenario_LowMeat.csv"),
    "vegetarian": Path("food", "dietary-composition-by-country_Scenario_Vegt.csv"),
    "vegan": Path("food", "dietary-composition-by-country_Scenario_Vegan.csv"),
}

housing_materials = {
    "default": Path("housing", "Haberl_MI_buildings_global.xlsx"),
    "lightweight": Path("housing", "Haberl_MI_buildings_global_lightweight.xlsx"),
    "woodbased": Path("housing", "Haberl_MI_buildings_global_substitute.xlsx"),
    "lightwood": Path("housing", "Haberl_MI_buildings_global_light&subst.xlsx"),
}

modal_shares = {
    "default": Path("mobility", "DLS_modal_share_version0_1_rc4_20240307.csv"),
    "b2ds": Path("mobility", "DLS_modal_share_B2DS_noAir.csv"),
    "lowcar": Path("mobility", "DLS_modal_share_B2DS_noAir_lowCar.csv"),
    "lowcardemand": Path("mobility", "DLS_modal_share_B2DS_noAir_lowCarLowDemand.csv"),
}

occupancy_rates = {
    "default": Path("mobility", "vehicle_occupancy_rates_R11.xlsx"),
    "median": Path("mobility", "vehicle_occupancy_rates_R11_median.xlsx"),
}

mobility_mapping = {
    "default": Path("mobility", "mobility_country_ecoinvent_process_correspondence.xlsx"),
    "evs": Path("mobility", "mobility_country_ecoinvent_process_correspondence_EVs.xlsx"),
}

renewable_energy_paths = {
    "electricity_only_threshold_default": Path("energy", "DLE_threshold_input_opcon_elecnonelec_dimensions_default_onlyELECTRICITY.csv"),
    "electricity_only_threshold_median": Path("energy", "DLE_threshold_input_opcon_elecnonelec_dimensions_median_floorspace_onlyELECTRICITY.csv"),
    "renewable_electricity": Path("energy", "indirect_stock_intensities_electricity_onlyRenewable_v1.xlsx"),
}

threshold_paths = {
    "no_changes": Path("sensitivity", "no_changes.xlsx"),
    "global_transport": Path("sensitivity", "decompose_stock_threshold", "DLS_service_thresholds_globAV_Transport.xlsx"),
    "global_av": Path("sensitivity", "decompose_stock_threshold", "DLE_threshold_globAV.csv"),
}

education_paths = {
    "default": Path("education", "processed_schoolgoing_population_share.csv"),
    "global_av": Path("sensitivity", "decompose_stock_threshold", "processed_schoolgoing_population_share_globAV.csv"),
}

housing_decompose_paths = {
    "global_av_mi": Path("sensitivity", "decompose_stock_threshold", "Haberl_MI_buildings_globAV.xlsx"),
    "global_av_archetypes": Path("sensitivity", "decompose_stock_threshold", "residential_archetypes_shares_globAV.xlsx"),
}

ecoinvent_paths = {
    "housing_global_av": Path("sensitivity", "decompose_stock_threshold", "housing_country_ecoinvent_process_correspondence_globAV.xlsx"),
    "mobility_global_av": Path("sensitivity", "decompose_stock_threshold", "mobility_country_ecoinvent_process_correspondence_globAV.xlsx"),
    "fuel_global_av": Path("sensitivity", "decompose_stock_threshold", "fuels_country_ecoinvent_process_correspondence_globAV.xlsx"),
}

electricity_paths = {
    "default": Path("2023_Velez", "indirect_stock_intensities_electricity_v1.xlsx"),
    "puerto_rico": Path("sensitivity", "decompose_stock_threshold", "indirect_stock_intensities_electricity_PuertoRico.xlsx"),
}

fuel_share_paths = {
    "default": Path("energy", "share_energy_source_country_correspondence.xlsx"),
    "global_av": Path("sensitivity", "decompose_stock_threshold", "share_energy_source_country_correspondence_globAV.xlsx"),
}

recc_housing_path = Path("housing", "Pauliuk_2021_MI_buildings_global.xlsx")

base_sensitivity = {
    "sensitivity_factor_clothing": 1,
    "sensitivity_factor_sanitation_water": 1,
    "sensitivity_factor_health_education": 1,
}


# change hard-coded process names in case one wants to consder electric car and scooter
base_process_change = {"car": "m_car", "scooter": "m_motor_scooter_GLO"}
ev_process_change = {"car": "m_Ecar", "scooter": "m_electric_scooter_GLO"}

# enter desired global average DLS service thresholds if wanting to substitute them for default values (base = sensitivity/no_changes.xlsx)
base_change_path = Path("sensitivity", "no_changes.xlsx")

def get_scenario_config(scenario_name, output_path, input_path, sensitivity=1):
    scenarios = {
        "current": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_lowmeat": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_lowMeat"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_lowMeat"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_vegetarian": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["vegetarian"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_vegetarian"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_vegetarian"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_vegan": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["vegan"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_vegan"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_vegan"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_b2ds": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                "regional_diets_path": diet_paths["current"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["default"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_B2DS"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_B2DS"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_lowcar": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                "regional_diets_path": diet_paths["current"],
                "modal_shares_path": modal_shares["lowcar"],
                "occupancy_path": occupancy_rates["default"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_lowCar"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_lowCar"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_low_cardemand": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                "regional_diets_path": diet_paths["current"],
                "modal_shares_path": modal_shares["lowcardemand"],
                "occupancy_path": occupancy_rates["default"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_lowCarLowDemand"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_lowCarLowDemand"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_evs": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "ecoinvent_mobility_reg_mapping_path": mobility_mapping["evs"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_EVs"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_EVs"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": ev_process_change,
        },
        "current_re": {
            "input_paths": {
                **base_paths,
                **dls_paths_renewable("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "DLE_threshold_path": renewable_energy_paths["electricity_only_threshold_default"],
                "electricity_indstocks_path": renewable_energy_paths["renewable_electricity"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_RE"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_RE"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_lightweight": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "housing_mat_intens_path": housing_materials["lightweight"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_lightweight"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_lightweight"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_woodbased": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "housing_mat_intens_path": housing_materials["woodbased"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_woodBased"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_woodBased"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_lightwood": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "housing_mat_intens_path": housing_materials["lightwood"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_lightWood"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_lightWood"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_hhsizehigh": {
            "input_paths": {
                **base_paths,
                **dls_paths("min_floorspace"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_hhSizeHigh"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_hhSizeHigh"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_hhsizemedium": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_hhSizeMedium"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_hhSizeMedium"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_hhsizelow": {
            "input_paths": {
                **base_paths,
                **dls_paths("max_floorspace"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_hhSizeLow"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_hhSizeLow"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "current_combined": {
            "input_paths": {
                **base_paths,
                **dls_paths_renewable("default"),
                "regional_diets_path": diet_paths["vegan"],
                "modal_shares_path": modal_shares["lowcardemand"],
                "occupancy_path": occupancy_rates["default"],
                "ecoinvent_mobility_reg_mapping_path": mobility_mapping["evs"],
                "housing_mat_intens_path": housing_materials["lightwood"],
                "DLE_threshold_path": renewable_energy_paths["electricity_only_threshold_default"],
                "electricity_indstocks_path": renewable_energy_paths["renewable_electricity"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_current_combined"),
                "threshold": Path(output_path, "DLS_stocks_thresh_current_combined"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": ev_process_change,
        },
        
        #### CONVERGED SCENARIOS

        "converged": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                **converged_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_vegt": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                **converged_mobility,
                "regional_diets_path": diet_paths["vegetarian"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_vegetarian"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_vegetarian"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_vegan": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                **converged_mobility,
                "regional_diets_path": diet_paths["vegan"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_vegan"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_vegan"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_lowcar": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["lowcar"],
                "occupancy_path": occupancy_rates["median"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_lowCar"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_lowCar"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_lowcardemand": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["lowcardemand"],
                "occupancy_path": occupancy_rates["median"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_lowCarLowDemand"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_lowCarLowDemand"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_evs": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                **converged_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
                "ecoinvent_mobility_reg_mapping_path": mobility_mapping["evs"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_EVs"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_EVs"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": ev_process_change,
        },
        "converged_re": {
            "input_paths": {
                **base_paths,
                **dls_paths_renewable("median_hh_size"),
                **converged_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
                "DLE_threshold_path": renewable_energy_paths["electricity_only_threshold_median"],
                "electricity_indstocks_path": renewable_energy_paths["renewable_electricity"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_RE"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_RE"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_lightweight": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
                "housing_mat_intens_path": housing_materials["lightweight"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_lightweight"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_lightweight"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_woodbased": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
                "housing_mat_intens_path": housing_materials["woodbased"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_woodBased"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_woodBased"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_lightwood": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
                "housing_mat_intens_path": housing_materials["lightwood"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_lightWood"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_lightWood"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_hhsizehigh": {
            "input_paths": {
                **base_paths,
                **dls_paths("min_floorspace"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_hhSizeHigh"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_hhSizeHigh"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_hhsizelow": {
            "input_paths": {
                **base_paths,
                **dls_paths("max_floorspace"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_hhSizeLow"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_hhSizeLow"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": base_process_change,
        },
        "converged_combined": {
            "input_paths": {
                **base_paths,
                **dls_paths_renewable("median_hh_size"),
                "regional_diets_path": diet_paths["vegan"],
                "modal_shares_path": modal_shares["lowcardemand"],
                "occupancy_path": occupancy_rates["median"],
                "ecoinvent_mobility_reg_mapping_path": mobility_mapping["evs"],
                "housing_mat_intens_path": housing_materials["lightwood"],
                "DLE_threshold_path": renewable_energy_paths["electricity_only_threshold_median"],
                "electricity_indstocks_path": renewable_energy_paths["renewable_electricity"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_converged_combined"),
                "threshold": Path(output_path, "DLS_stocks_thresh_converged_combined"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": base_change_path,
            "process_change": ev_process_change,
        },

        #### DECOMPOSITION ANALYSIS SCENARIOS

        "current_pkm_same": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentPkmSame"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentPkmSame"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["global_transport"],
            "process_change": base_process_change,
        },

        # 2021_Kikstra/DLS_dataproducts_combined_median_hh_size.csv
        "current_energy_same": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "DLE_threshold_path": threshold_paths["global_av"],
                "electricity_indstocks_path": electricity_paths["puerto_rico"],
                "fuel_shares_reg_mapping_path": fuel_share_paths["global_av"],
                "ecoinvent_coal_reg_mapping_path": ecoinvent_paths["fuel_global_av"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentEnergySame"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentEnergySame"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "current_educ_same": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "education_shares_path": education_paths["global_av"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentEducSame"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentEducSame"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "current_bmi_same": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "housing_mat_intens_path": housing_decompose_paths["global_av_mi"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentBMISame"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentBMISame"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "current_ecoinv_same": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "ecoinvent_housing_reg_mapping_path": ecoinvent_paths["housing_global_av"],
                "ecoinvent_mobility_reg_mapping_path": ecoinvent_paths["mobility_global_av"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentEcoInvSame"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentEcoInvSame"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "current_arch_same": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "housing_archetypes_path": housing_decompose_paths["global_av_archetypes"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentArchSame"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentArchSame"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "current_vocc": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                "regional_diets_path": diet_paths["current"],
                "modal_shares_path": modal_shares["default"],
                "occupancy_path": occupancy_rates["median"],
                "ecoinvent_mobility_reg_mapping_path": mobility_mapping["default"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentVocc"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentVocc"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "current_same_combined": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
                "ecoinvent_mobility_reg_mapping_path": ecoinvent_paths["mobility_global_av"],
                "housing_archetypes_path": housing_decompose_paths["global_av_archetypes"],
                "housing_mat_intens_path": housing_decompose_paths["global_av_mi"],
                "ecoinvent_housing_reg_mapping_path": ecoinvent_paths["housing_global_av"],
                "education_shares_path": education_paths["global_av"],
                "electricity_indstocks_path": electricity_paths["puerto_rico"],
                "fuel_shares_reg_mapping_path": fuel_share_paths["global_av"],
                "ecoinvent_coal_reg_mapping_path": ecoinvent_paths["fuel_global_av"],
                "DLE_threshold_path": threshold_paths["global_av"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentSameCombined"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentSameCombined"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["global_transport"],
            "process_change": base_process_change,
        },
        "current_same_combined_not_school": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                "regional_diets_path": diet_paths["lowmeat"],
                "modal_shares_path": modal_shares["b2ds"],
                "occupancy_path": occupancy_rates["median"],
                "ecoinvent_mobility_reg_mapping_path": ecoinvent_paths["mobility_global_av"],
                "housing_archetypes_path": housing_decompose_paths["global_av_archetypes"],
                "housing_mat_intens_path": housing_decompose_paths["global_av_mi"],
                "ecoinvent_housing_reg_mapping_path": ecoinvent_paths["housing_global_av"],
                "education_shares_path": education_paths["default"],
                "electricity_indstocks_path": electricity_paths["puerto_rico"],
                "fuel_shares_reg_mapping_path": fuel_share_paths["global_av"],
                "ecoinvent_coal_reg_mapping_path": ecoinvent_paths["fuel_global_av"],
                "DLE_threshold_path": threshold_paths["global_av"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentSameCombinedNotSchool"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentSameCombinedNotSchool"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["global_transport"],
            "process_change": base_process_change,
        },

        #### RECC MATERIAL INTENSITY SCENARIOS

        "current_recc_mi": {
            "input_paths": {
                **base_paths,
                **dls_paths("default"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
                "housing_mat_intens_path": recc_housing_path,
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentRECCmi"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentRECCmi"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "converged_recc_mi": {
            "input_paths": {
                **base_paths,
                **dls_paths("median_hh_size"),
                **converged_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
                "housing_mat_intens_path": recc_housing_path,
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_convergedRECCmi"),
                "threshold": Path(output_path, "DLS_stocks_thresh_convergedRECCmi"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },

        #### DLS SENSITIVITY SCENARIOS - Using pre-processed input files

        "currentDLSplus": {
            "input_paths": {
                **base_paths,
                "DLS_data_path": Path("2021_Kikstra", "DLS_dataproducts_combined_default_high.csv"),
                "DLE_provided_path": Path("2021_Kikstra", "DLE_current_opcon_elecnonelec_dimensions_default_high.csv"),
                "DLE_threshold_path": Path("2021_Kikstra", "DLE_threshold_input_opcon_elecnonelec_dimensions_default_high.csv"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentDLSplus"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentDLSplus"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "currentDLSless": {
            "input_paths": {
                **base_paths,
                "DLS_data_path": Path("2021_Kikstra", "DLS_dataproducts_combined_default_low.csv"),
                "DLE_provided_path": Path("2021_Kikstra", "DLE_current_opcon_elecnonelec_dimensions_default_low.csv"),
                "DLE_threshold_path": Path("2021_Kikstra", "DLE_threshold_input_opcon_elecnonelec_dimensions_default_low.csv"),
                **base_mobility,
                "regional_diets_path": diet_paths["current"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_currentDLSless"),
                "threshold": Path(output_path, "DLS_stocks_thresh_currentDLSless"),
            },
            "sensitivity":base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },


        "convergedDLSplus": {
            "input_paths": {
                **base_paths,
                "DLS_data_path": Path("2021_Kikstra", "DLS_dataproducts_combined_median_hh_size_high.csv"),
                "DLE_provided_path": Path("2021_Kikstra", "DLE_current_opcon_elecnonelec_dimensions_median_hh_size_high.csv"),
                "DLE_threshold_path": Path("2021_Kikstra", "DLE_threshold_input_opcon_elecnonelec_dimensions_median_hh_size_high.csv"),
                **converged_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_convergedDLSplus"),
                "threshold": Path(output_path, "DLS_stocks_thresh_convergedDLSplus"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        },
        "convergedDLSless": {
            "input_paths": {
                **base_paths,
                "DLS_data_path": Path("2021_Kikstra", "DLS_dataproducts_combined_median_hh_size_low.csv"),
                "DLE_provided_path": Path("2021_Kikstra", "DLE_current_opcon_elecnonelec_dimensions_median_hh_size_low.csv"),
                "DLE_threshold_path": Path("2021_Kikstra", "DLE_threshold_input_opcon_elecnonelec_dimensions_median_hh_size_low.csv"),
                **converged_mobility,
                "regional_diets_path": diet_paths["lowmeat"],
            },
            "output_paths": {
                "provided": Path(output_path, "DLS_stocks_prov_convergedDLSless"),
                "threshold": Path(output_path, "DLS_stocks_thresh_convergedDLSless"),
            },
            "sensitivity": base_sensitivity,
            "change_DLS_thresholds": threshold_paths["no_changes"],
            "process_change": base_process_change,
        }
    }

    active_scenario = scenarios.get(scenario_name)
    if active_scenario is None:
        return None
    
    # these sheets are identical for all scenarios
    active_scenario["input_paths"]["cover"] = Path("cover_for_result_files.xlsx")
    active_scenario["input_paths"]["country_correspondence"] = Path("country_correspondence.xlsx")
    active_scenario["input_paths"]["MISO2_population"] = Path("MISO2", "MISO2_population.xlsx")
    active_scenario["input_paths"]["Kikstra_population"] = Path("2021_Kikstra", "Kikstra2021_population_etc.xlsx")
    active_scenario["input_paths"]["Velez_direct_stocks"] = Path("2023_Velez", "2023_Veléz-Henao_SI4_direct_indirect_stocks_modified.xlsx")
    active_scenario["input_paths"]["Velez_nutrition"] = Path("2023_Velez", "indirect_stock_intensities_nutrition_v1_pork_modified.xlsx")
    active_scenario["input_paths"]["Velez_communic"] = Path("2023_Velez", "indirect_stock_intensities_communication_v1.xlsx")
    active_scenario["input_paths"]["Velez_shelter"] = Path("2023_Velez", "indirect_stock_intensities_shelter_v1.xlsx")
    active_scenario["input_paths"]["Velez_mobility"] = Path("2023_Velez", "indirect_stock_intensities_mobility_v1.xlsx")
    active_scenario["input_paths"]["Velez_hygiene_clothing"] = Path("2023_Velez", "indirect_stock_intensities_hygiene_clothing_v1.xlsx")
    active_scenario["input_paths"]["Velez_hygiene_water"] = Path("2023_Velez", "indirect_stock_intensities_hygiene_water_v1.xlsx")
    active_scenario["input_paths"]["Velez_ecoinvent_target_sectors"] = Path("2023_Velez", "ecoinvent_target_sectors_materials_final.xlsx")
    
    active_scenario["input_paths"]["MISO2_clothing"] = Path("clothing", "clothing_material_needs.xlsx")
    active_scenario["input_paths"]["MISO2_sanitation"] = Path("sanitation_water", "sanitation_water_service_demand.xlsx")
    active_scenario["input_paths"]["MISO2_stock_Gas_EoL"] = Path("MISO2", "MISO2_global_v1_enduse_stockGasEoL_1950_2016.csv")

    active_scenario["input_paths"]["food_map"] = Path("food", "food_map_dietary_categories_to_ecoinvent.xlsx")
    active_scenario["input_paths"]["admin_boundaries"] = Path("ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")

    active_scenario = create_full_paths(active_scenario, input_path)
    
    check_inputs_present(active_scenario)

    return active_scenario

def update_sensitivity(config, new_value):
    config["sensitivity"] = {k: new_value for k, _ in config["sensitivity"].items()}

def create_full_paths(config, input_path):
    config = deepcopy(config)

    config["change_DLS_thresholds"] = Path(input_path, config["change_DLS_thresholds"])

    for var_name, path_to_file in config["input_paths"].items():
        full_file_path = Path(input_path, path_to_file)
        config["input_paths"][var_name] = full_file_path

    return config

def check_inputs_present(config):

    files_not_found = []

    for path_to_file in config["input_paths"].values():
        if not path_to_file.is_file():
            files_not_found.append(path_to_file)

    if len(files_not_found) > 0:
        raise FileNotFoundError(f"Could not find files: {files_not_found}")