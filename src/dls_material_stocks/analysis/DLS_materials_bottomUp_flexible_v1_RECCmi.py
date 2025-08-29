import pandas as pd
import numpy as np
from datetime import datetime
from ..load.DLS_load_data import (
    load_country_correspondence_dict,
    load_population_2015,
    load_DLS_2015,
    load_regional_diets,
)
from ..analysis.DLS_functions_v3 import (
    read_excel_into_dict,
    create_nested_dict,
    expand_nested_dict_to_df,
    prepare_df_for_concat,
    calc_functions_scale,
    calc_food_flows_kcal_cap,
    conv_food_flows_kg_cap,
    calc_food_flows_kg_scale,
    calc_indir_stock_intens_cap,
    calc_indir_stock_scale,
    regionalize_material_demand_ecoinvent,
    regionalize_material_demand_ecoinvent_coal,
    prep_DLS_func_for_scaling_single,
    align_indices_zero,
    calc_function_split,
    calc_direct_cap,
    calc_direct_scale,
    mult_singleCol_otherCols,
    scale_kg_cap,
    create_nested_dict_energy,
    adjust_index_fuel,
    create_nested_dict_fuel,
)

"""
@author: jstreeck

"""

"""############################################################

       Description:
           
        This script calculates the DLS material stocks currently
        existing (2015) and thresholds required to reach DLS for
        all by combining data on service levels, practices and
        material intensities.   
        
        In comparison to the script with the same name less
        suffix _RECCmi, here, material intensities for buildings
        are taken from Pauliuk et al. (2021) to test the sensitivity
        of results to building material intensity data.
        
   ############################################################"""

# thresholds per capita
# assumption that 1.6 // 2.48 m²/cap for good health care and education system (Veléz-Henao & Pauliuk 2023)
m2_cap_health = 1.6
m2_pupil_education = 10


def calc_bottomUp_DLSS_RECCmi(
    input_paths_dict,
    output_paths_dict,
    sensitivity_factors,
    change_DLS_thresholds,
    process_change,
):
    """#################
        load base data
    #################"""

    # country corespondence MISO2 (Wiedernhofer et al., in prep.) & DLS data (Kikstra et al., 2025 updated)
    country_correspondence, country_correspondence_dict = (
        load_country_correspondence_dict(input_paths_dict.get("country_correspondence"))
    )
    # load population data (Wiedernhofer et al.,2024)
    MISO2_population_2015, MISO2_population_2015_subset = load_population_2015(
        filename=input_paths_dict.get("MISO2_population"),
        sheet_name="values",
        country_correspondence=country_correspondence,
    )
    # load DLS data (Kikstra et al., 2025 updated)
    DLS_2015_funct_prov, DLS_2015_thresh = load_DLS_2015(
        filename=input_paths_dict.get("DLS_data_path"),
        country_correspondence=country_correspondence,
        country_correspondence_dict=country_correspondence_dict,
    )
    # load direct stock intensities from SI of Veléz-Henao & Pauliuk (2023)
    direct_stocks_intens = read_excel_into_dict(
        input_paths_dict.get("Velez_direct_stocks")
    )

    # define sensitivity factors for variables not from Kikstra et al. (2021/2024)
    sensitivity_factor_clothing = sensitivity_factors.get("sensitivity_factor_clothing")
    sensitivity_factor_sanitation_water = sensitivity_factors.get(
        "sensitivity_factor_sanitation_water"
    )
    sensitivity_factor_health_education = sensitivity_factors.get(
        "sensitivity_factor_health_education"
    )

    # read decent living energy (DLE) data for DLS provided and DLS threshold from Kikstra et al. (2025)
    DLE_provided = input_paths_dict.get("DLE_provided_path")
    DLE_threshold = input_paths_dict.get("DLE_threshold_path")

    # POTENTIALLY SET DLS THRESHOLDS TO GLOBAL AVERAGE SERVICE LEVELS for each region FOR SPECIFIC SERVICES as
    # specified in run_from_pickle_v1 (empty input maintains original values)
    global_average_thresholds = pd.read_excel(
        change_DLS_thresholds
    )
    DLS_2015_thresh = DLS_2015_thresh.reset_index()
    DLS_2015_thresh["value_new"] = DLS_2015_thresh["variable"].map(
        global_average_thresholds.set_index("variable")["weighted"]
    )
    DLS_2015_thresh["value_new"] = DLS_2015_thresh["value_new"].fillna(
        DLS_2015_thresh["value"]
    )
    DLS_2015_thresh = (
        DLS_2015_thresh[["region", "variable", "unit", "value_new"]]
        .rename(columns={"value_new": "value"})
        .set_index(["region", "variable", "unit"])
    )

    """###########################################################################
            BOTTOM-UP  quantification of material flows, direct/indirect stocks based on 
                   Kikstra et al. (2025) & Veléz-Henao & Pauliuk (2023)
       ###########################################################################"""

    """####################
        FOOD: DIRECT FLOWS 
       ####################"""
    # note! 2019 values at scale arecurrently calculated with 2015 population figures - not corrected yet as we don't use 2019'''

    # load and format regional diet data
    reg_diets_path = input_paths_dict.get("regional_diets_path")
    reg_diets_act_shares_final = load_regional_diets(
        filename=reg_diets_path,
        country_correspondence=country_correspondence,
        country_correspondence_dict=country_correspondence_dict,
    )

    # get DLS data for food requirement
    DLS_2015_thresh_nutr = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Nutrition"
    ]
    DLS_2015_funct_prov_nutr = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Nutrition"
    ]

    ###calculate direct food flows for DLS PROVIDED
    # calculate direct flows [kcal] of food per food item based on diet shares and kcal level per capita and day
    food_kcal_cap_DLS_curr = calc_food_flows_kcal_cap(
        reg_diets_act_shares_final, DLS_2015_funct_prov_nutr
    )
    # load data input for calorific values + bring in required data shape + convert to kg
    food_mappings = pd.read_excel(
        input_paths_dict.get("food_map")
    )
    food_mappings.set_index("diet categories", inplace=True)
    calorific_values = food_mappings["calorific value FAO 1"].reindex(
        food_kcal_cap_DLS_curr.columns[2:]
    )

    ### calculate direct food flows for DLS CURRENT PROVISION
    food_kg_cap_DLS_curr = conv_food_flows_kg_cap(
        food_kcal_cap_DLS_curr, calorific_values
    )
    # scale up to region per year
    food_ton_scale_DLS_curr = calc_food_flows_kg_scale(
        food_kg_cap_DLS_curr, MISO2_population_2015_subset
    )
    # sum for globe for 2015
    food_ton_scale_DLS_global_curr = food_ton_scale_DLS_curr[
        food_ton_scale_DLS_curr.index.get_level_values(1) == 2015
    ].sum(axis=0)

    ### calculate direct food flows for DLS THRESHOLD (operations equal to above for DLS provided)
    food_kcal_cap_DLS_thresh = calc_food_flows_kcal_cap(
        reg_diets_act_shares_final, DLS_2015_thresh_nutr
    )
    food_kg_cap_DLS_thresh = conv_food_flows_kg_cap(
        food_kcal_cap_DLS_thresh, calorific_values
    )
    food_ton_scale_DLS_thresh = calc_food_flows_kg_scale(
        food_kg_cap_DLS_thresh, MISO2_population_2015_subset
    )
    food_ton_scale_DLS_global_thresh = food_ton_scale_DLS_thresh[
        food_ton_scale_DLS_thresh.index.get_level_values(1) == 2015
    ].sum(axis=0)

    # rename food categories in results
    diets_rename = dict(
        zip(
            food_mappings.reset_index()["diet categories"],
            food_mappings["diet_categories_abbrev"],
        )
    )
    food_ton_scale_DLS_global_curr.rename(diets_rename, inplace=True)
    food_ton_scale_DLS_global_thresh.rename(diets_rename, inplace=True)

    """####################
        FOOD: IN-DIRECT STOCKS
       ####################"""

    # multiply direct food flows by indirect stock intensities per food product
    # Read all data sheets of material intensities into a dictionary of dataframes
    # data _pork_modified uses modified ecoinvent material intensities for pork meat because original values are implausibly high (cf. SI 1.2.2)
    nutrition_indstocks_path = input_paths_dict.get("Velez_nutrition")
    nutrition_indstocks_dict = read_excel_into_dict(nutrition_indstocks_path)
    del nutrition_indstocks_dict["summary"]
    for key in nutrition_indstocks_dict:
        nutrition_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)
        nutrition_indstocks_dict.get(key).columns = nutrition_indstocks_dict.get(
            key
        ).columns.astype(int)
    # to match food product categories of diets to ecoinvent process names in food_mappings-> read mapping data for common technology labels
    diet_ecoinv_dict = dict(
        zip(
            food_mappings.reset_index()["diet categories"],
            food_mappings["ecoinvent_abbrev1"],
        )
    )

    # DLS PROVIDED
    # multiply kg food supply by indirect stock intensities per food product,unit: kg/cap/day
    food_ind_stock_intens_cap_curr = calc_indir_stock_intens_cap(
        food_kg_cap_DLS_curr, diet_ecoinv_dict, nutrition_indstocks_dict
    )
    # scale to total population - food_ind_stock_intens_cap_curr as stock per kg/cap/day of food, MISO2_pop in 1000's: resulting unit: tons/region/stock_category/material
    food_ind_stock_scale_curr = calc_indir_stock_scale(
        food_ind_stock_intens_cap_curr, MISO2_population_2015_subset
    )
    # prepare to concatenate with other DLS dimensions
    food_indirect_stocks_prov = prepare_df_for_concat(
        df=food_ind_stock_scale_curr,
        unit="tons",
        stock_type="indirect",
        dimension="nutrition_i",
        stock="food_related",
    )
    food_indirect_stocks_prov_2015 = food_indirect_stocks_prov[
        food_indirect_stocks_prov.index.get_level_values(1).isin([2015])
    ]

    # DLS THRESHOLD (operations equal to above)
    food_ind_stock_intens_cap_thresh = calc_indir_stock_intens_cap(
        food_kg_cap_DLS_thresh, diet_ecoinv_dict, nutrition_indstocks_dict
    )
    food_ind_stock_scale_thresh = calc_indir_stock_scale(
        food_ind_stock_intens_cap_thresh, MISO2_population_2015_subset
    )
    food_indirect_stocks_thresh = prepare_df_for_concat(
        food_ind_stock_scale_thresh,
        unit="tons",
        stock_type="indirect",
        dimension="nutrition_i",
        stock="food_related",
    )
    food_indirect_stocks_thresh_2015 = food_indirect_stocks_thresh[
        food_indirect_stocks_thresh.index.get_level_values(1).isin([2015])
    ]

    """####################################
        STOVE & FRIDGE: IN-DIRECT STOCKS (including direct stocks)
       ####################################"""

    # select DLS indicators for cooking and fridge per capita
    DLS_2015_thresh_cook = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Appliance|clean_cooking_fuel"
    ]
    DLS_2015_funct_prov_cook = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Appliance|clean_cooking_fuel"
    ]
    DLS_2015_thresh_fridge = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Appliance|refrigerator"
    ]
    DLS_2015_funct_prov_fridge = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Appliance|refrigerator"
    ]

    # calculate DLS per region
    # DLS provided
    DLS_2015_funct_prov_cook_scale = calc_functions_scale(
        DLS_2015_funct_prov_cook, MISO2_population_2015_subset
    )
    DLS_2015_funct_prov_fridge_scale = calc_functions_scale(
        DLS_2015_funct_prov_fridge, MISO2_population_2015_subset
    )
    # DLS threshold
    DLS_2015_thresh_cook_scale = calc_functions_scale(
        DLS_2015_thresh_cook, MISO2_population_2015_subset
    )
    DLS_2015_thresh_fridge_scale = calc_functions_scale(
        DLS_2015_thresh_fridge, MISO2_population_2015_subset
    )

    # prepare dataframes for input into functions to calc stocks
    # DLS provided
    DLS_2015_funct_prov_cook_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_funct_prov_cook_scale, "cookstove"
    )
    DLS_2015_funct_prov_fridge_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_funct_prov_fridge_scale, "refrigerator"
    )
    # DLS threshold
    DLS_2015_thresh_cook_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_thresh_cook_scale, "cookstove"
    )
    DLS_2015_thresh_fridge_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_thresh_fridge_scale, "refrigerator"
    )

    # multiply with indirect stock intensities (use nutrition_indstocks_dict) and prepare to concat with other dimensions
    # cooking - DLS provided
    stove_indirect_stocks_prov_dict_scale = create_nested_dict(
        nutrition_indstocks_dict, DLS_2015_funct_prov_cook_scale
    )
    stove_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        stove_indirect_stocks_prov_dict_scale
    )
    stove_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    stove_indirect_stocks_prov = prepare_df_for_concat(
        stove_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="hh_appliance_i",
        stock="stove_related",
    )
    # cooking - DLS threshold
    stove_indirect_stocks_thresh_dict_scale = create_nested_dict(
        nutrition_indstocks_dict, DLS_2015_thresh_cook_scale
    )
    stove_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        stove_indirect_stocks_thresh_dict_scale
    )
    stove_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    stove_indirect_stocks_thresh = prepare_df_for_concat(
        stove_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="hh_appliance_i",
        stock="stove_related",
    )
    # fridge - DLS provided
    fridge_indirect_stocks_prov_dict_scale = create_nested_dict(
        nutrition_indstocks_dict, DLS_2015_funct_prov_fridge_scale
    )
    fridge_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        fridge_indirect_stocks_prov_dict_scale
    )
    fridge_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    fridge_indirect_stocks_prov = prepare_df_for_concat(
        fridge_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="hh_appliance_i",
        stock="fridge_related",
    )
    # fridge - DLS threshold
    fridge_indirect_stocks_thresh_dict_scale = create_nested_dict(
        nutrition_indstocks_dict, DLS_2015_thresh_fridge_scale
    )
    fridge_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        fridge_indirect_stocks_thresh_dict_scale
    )
    fridge_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    fridge_indirect_stocks_thresh = prepare_df_for_concat(
        fridge_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="hh_appliance_i",
        stock="fridge_related",
    )

    """####################################
        STOVE & FRIDGE: DIRECT STOCKS
       ####################################"""
    # we also calculate direct stocks here, and separate them from indirect stocks below - BUT these results are not used in paper (only total stocks)

    # subset direct stock intensities from SI of Veléz-Henao & Pauliuk (2023) for nutrition appliances
    nutrition_direct_stocks_intens = {
        k: v
        for k, v in direct_stocks_intens.items()
        if k in ("cookstove", "refrigerator")
    }

    # copy DLS indicators for cooking stoves and fridges from indirect stock calculations
    fridges_prov = DLS_2015_funct_prov_fridge_scale.copy()
    stoves_prov = DLS_2015_funct_prov_cook_scale.copy()
    fridges_thresh = DLS_2015_thresh_fridge_scale.copy()
    stoves_thresh = DLS_2015_thresh_cook_scale.copy()

    # prepare direct stock dictionary
    for key in nutrition_direct_stocks_intens:
        df_subset = nutrition_direct_stocks_intens[key][["Material", "kg/unit"]]
        df_subset.set_index("Material", inplace=True)
        nutrition_direct_stocks_intens[key] = df_subset

    # calc fridge and stove direct stocks and prepare for concatenation with other dimensions
    # fridge - DLS provided
    fridge_direct_stocks_prov_dict = create_nested_dict(
        nutrition_direct_stocks_intens, fridges_prov
    )
    fridge_direct_stocks_prov = expand_nested_dict_to_df(fridge_direct_stocks_prov_dict)
    fridge_direct_stocks_prov.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    fridge_direct_stocks_prov = prepare_df_for_concat(
        fridge_direct_stocks_prov,
        unit="kg",
        stock_type="direct",
        dimension="hh_appliance",
        stock="fridge",
    )
    # fridge - DLS threshold
    fridge_direct_stocks_thresh_dict = create_nested_dict(
        nutrition_direct_stocks_intens, fridges_thresh
    )
    fridge_direct_stocks_thresh = expand_nested_dict_to_df(
        fridge_direct_stocks_thresh_dict
    )
    fridge_direct_stocks_thresh.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    fridge_direct_stocks_thresh = prepare_df_for_concat(
        fridge_direct_stocks_thresh,
        unit="kg",
        stock_type="direct",
        dimension="hh_appliance",
        stock="fridge",
    )
    # cooking - DLS provided
    stove_direct_stocks_prov_dict = create_nested_dict(
        nutrition_direct_stocks_intens, stoves_prov
    )
    stove_direct_stocks_prov = expand_nested_dict_to_df(stove_direct_stocks_prov_dict)
    stove_direct_stocks_prov.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    stove_direct_stocks_prov = prepare_df_for_concat(
        stove_direct_stocks_prov,
        unit="kg",
        stock_type="direct",
        dimension="hh_appliance",
        stock="stove",
    )
    # cooking - DLS threshold
    stove_direct_stocks_thresh_dict = create_nested_dict(
        nutrition_direct_stocks_intens, stoves_thresh
    )
    stove_direct_stocks_thresh = expand_nested_dict_to_df(
        stove_direct_stocks_thresh_dict
    )
    stove_direct_stocks_thresh.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    stove_direct_stocks_thresh = prepare_df_for_concat(
        stove_direct_stocks_thresh,
        unit="kg",
        stock_type="direct",
        dimension="hh_appliance",
        stock="stove",
    )

    """########################################################
        STOVE & FRIDGE: INDIRECT STOCKS - REMOVE DIRECT STOCKS
       ########################################################"""

    # COOKING STOVE
    stove_direct_stocks_prov_DC = stove_direct_stocks_prov.copy()
    stove_direct_stocks_thresh_DC = stove_direct_stocks_thresh.copy()

    # highest value in direct stock intensities for material: 2410:Manufacture of basic iron and steel = stock 2750:Manufacture of domestic appliances
    # we assume that stoves accounted as 2750:Manufacture of domestic appliances and deduct direct stocks from this account
    stove_direct_stocks_prov_DC = (
        stove_direct_stocks_prov_DC.reset_index()
        .replace(
            {"direct": "indirect", "hh_appliance": "hh_appliance_i", "stove": 2750}
        )
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )
    stove_direct_stocks_thresh_DC = (
        stove_direct_stocks_thresh_DC.reset_index()
        .replace(
            {"direct": "indirect", "hh_appliance": "hh_appliance_i", "stove": 2750}
        )
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )
    # align indices for correct subtraction
    stove_direct_stocks_prov_DC_align = align_indices_zero(
        stove_direct_stocks_prov_DC, stove_indirect_stocks_prov
    )
    stove_direct_stocks_thresh_DC_align = align_indices_zero(
        stove_direct_stocks_thresh_DC, stove_indirect_stocks_thresh
    )

    # subtraction and check for negatives and correct sums
    stove_indirect_stocks_prov_DCfree = stove_indirect_stocks_prov.sub(
        stove_direct_stocks_prov_DC_align
    )
    stove_indirect_stocks_prov_DCfree.lt(0).any().any()
    stove_indirect_stocks_thresh_DCfree = (
        stove_indirect_stocks_thresh - stove_direct_stocks_thresh_DC_align
    )
    stove_indirect_stocks_thresh_DCfree.lt(0).any().any()
    assert (
        (
            stove_direct_stocks_prov_DC.sum().sum()
            + stove_indirect_stocks_prov_DCfree.sum().sum()
        )
        - stove_indirect_stocks_prov.sum().sum()
    ) < 1e-05
    assert (
        (
            stove_direct_stocks_thresh_DC.sum().sum()
            + stove_indirect_stocks_thresh_DCfree.sum().sum()
        )
        - stove_indirect_stocks_thresh.sum().sum()
    ) < 1e-05

    # FRIDGE
    fridge_direct_stocks_prov_DC = fridge_direct_stocks_prov.copy()
    fridge_direct_stocks_thresh_DC = fridge_direct_stocks_thresh.copy()

    # highest value in direct stock intensities for material: 2410:Manufacture of basic iron and steel = stock 2750:Manufacture of domestic appliances
    # we assume that fridges accounted as 2750:Manufacture of domestic appliances and deduct direct stocks from this account
    fridge_direct_stocks_prov_DC = (
        fridge_direct_stocks_prov_DC.reset_index()
        .replace(
            {"direct": "indirect", "hh_appliance": "hh_appliance_i", "fridge": 2750}
        )
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )
    fridge_direct_stocks_thresh_DC = (
        fridge_direct_stocks_thresh_DC.reset_index()
        .replace(
            {"direct": "indirect", "hh_appliance": "hh_appliance_i", "fridge": 2750}
        )
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )
    # align indiced for correct subtraction
    fridge_direct_stocks_prov_DC_align = align_indices_zero(
        fridge_direct_stocks_prov_DC, fridge_indirect_stocks_prov
    )
    fridge_direct_stocks_thresh_DC_align = align_indices_zero(
        fridge_direct_stocks_thresh_DC, fridge_indirect_stocks_thresh
    )

    # subtraction and check for negatives and correct sums
    fridge_indirect_stocks_prov_DCfree = fridge_indirect_stocks_prov.sub(
        fridge_direct_stocks_prov_DC_align
    )
    fridge_indirect_stocks_prov_DCfree.lt(0).any().any()
    fridge_indirect_stocks_thresh_DCfree = (
        fridge_indirect_stocks_thresh - fridge_direct_stocks_thresh_DC_align
    )
    fridge_indirect_stocks_thresh_DCfree.lt(0).any().any()
    assert (
        (
            fridge_direct_stocks_prov_DC.sum().sum()
            + fridge_indirect_stocks_prov_DCfree.sum().sum()
        )
        - fridge_indirect_stocks_prov.sum().sum()
    ) < 1e-05
    assert (
        (
            fridge_direct_stocks_thresh_DC.sum().sum()
            + fridge_indirect_stocks_thresh_DCfree.sum().sum()
        )
        - fridge_indirect_stocks_thresh.sum().sum()
    ) < 1e-05

    """####################################
        COMMUNICATION: INDIRECT STOCKS (including direct stocks)
       ####################################"""

    # obtain functions provided for mobile_telephone and TV
    DLS_2015_thresh_phone = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Appliance|mobile_telephone"
    ]
    DLS_2015_funct_prov_phone = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Appliance|mobile_telephone"
    ]
    DLS_2015_thresh_TV = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Appliance|television"
    ]
    DLS_2015_funct_prov_TV = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Appliance|television"
    ]

    # direct & indirect stocks
    # calculate total appliances per region
    DLS_2015_funct_prov_phone_scale = calc_functions_scale(
        DLS_2015_funct_prov_phone, MISO2_population_2015_subset
    )
    DLS_2015_funct_prov_TV_scale = calc_functions_scale(
        DLS_2015_funct_prov_TV, MISO2_population_2015_subset
    )
    DLS_2015_thresh_phone_scale = calc_functions_scale(
        DLS_2015_thresh_phone, MISO2_population_2015_subset
    )
    DLS_2015_thresh_TV_scale = calc_functions_scale(
        DLS_2015_thresh_TV, MISO2_population_2015_subset
    )

    # load indirect stock intensities data
    communic_indstocks_path = input_paths_dict.get("Velez_communic")
    communic_indstocks_dict = read_excel_into_dict(communic_indstocks_path)
    del communic_indstocks_dict["summary"]
    for key in communic_indstocks_dict:
        communic_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)
        communic_indstocks_dict.get(key).columns = communic_indstocks_dict.get(
            key
        ).columns.astype(int)

    # prepare dataframes for calculating material stocks (insert keys from communic dict)
    DLS_2015_funct_prov_phone_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_funct_prov_phone_scale, "consumer electronics, mobile de"
    )
    DLS_2015_funct_prov_TV_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_funct_prov_TV_scale, "television"
    )
    DLS_2015_thresh_phone_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_thresh_phone_scale, "consumer electronics, mobile de"
    )
    DLS_2015_thresh_TV_scale = prep_DLS_func_for_scaling_single(
        DLS_2015_thresh_TV_scale, "television"
    )

    # multiply with (in)direct stock intensities
    # phone DLS provided
    phone_indirect_stocks_prov_dict_scale = create_nested_dict(
        communic_indstocks_dict, DLS_2015_funct_prov_phone_scale
    )
    phone_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        phone_indirect_stocks_prov_dict_scale
    )
    phone_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    phone_indirect_stocks_prov = prepare_df_for_concat(
        phone_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="communication_i",
        stock="phone_related",
    )
    # phone DLS threshold
    phone_indirect_stocks_thresh_dict_scale = create_nested_dict(
        communic_indstocks_dict, DLS_2015_thresh_phone_scale
    )
    phone_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        phone_indirect_stocks_thresh_dict_scale
    )
    phone_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    phone_indirect_stocks_thresh = prepare_df_for_concat(
        phone_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="communication_i",
        stock="phone_related",
    )
    # TV DLS provided
    TV_indirect_stocks_prov_dict_scale = create_nested_dict(
        communic_indstocks_dict, DLS_2015_funct_prov_TV_scale
    )
    TV_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        TV_indirect_stocks_prov_dict_scale
    )
    TV_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    TV_indirect_stocks_prov = prepare_df_for_concat(
        TV_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="communication_i",
        stock="TV_related",
    )
    # TV DLS threshold
    TV_indirect_stocks_thresh_dict_scale = create_nested_dict(
        communic_indstocks_dict, DLS_2015_thresh_TV_scale
    )
    TV_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        TV_indirect_stocks_thresh_dict_scale
    )
    TV_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    TV_indirect_stocks_thresh = prepare_df_for_concat(
        TV_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="communication_i",
        stock="TV_related",
    )

    """####################################
        COMMUNICATION: DIRECT STOCKS
       ####################################"""

    # subset direct stock intensities from SI of Veléz-Henao & Pauliuk (2023) for communication appliances
    communication_direct_stocks_intens = {
        k: v for k, v in direct_stocks_intens.items() if k in ("phone", "television")
    }  #

    # appliances at scale
    phone_prov = DLS_2015_funct_prov_phone_scale.copy()
    phone_thresh = DLS_2015_thresh_phone_scale.copy()
    phone_prov.rename(
        columns={"consumer electronics, mobile de": "phone"}, inplace=True
    )
    phone_thresh.rename(
        columns={"consumer electronics, mobile de": "phone"}, inplace=True
    )

    # prepare direct stock dictionary
    for key in communication_direct_stocks_intens:
        df_subset = communication_direct_stocks_intens[key][["Material", "kg/unit"]]
        df_subset.set_index("Material", inplace=True)
        communication_direct_stocks_intens[key] = df_subset

    # calc direct stocks
    # phone DLS provided
    phone_direct_stocks_prov_dict = create_nested_dict(
        communication_direct_stocks_intens, phone_prov
    )
    phone_direct_stocks_prov = expand_nested_dict_to_df(phone_direct_stocks_prov_dict)
    phone_direct_stocks_prov.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    phone_direct_stocks_prov = prepare_df_for_concat(
        phone_direct_stocks_prov,
        unit="kg",
        stock_type="direct",
        dimension="communication",
        stock="mobile_phone",
    )
    # phone DLS threshold
    phone_direct_stocks_thresh_dict = create_nested_dict(
        communication_direct_stocks_intens, phone_thresh
    )
    phone_direct_stocks_thresh = expand_nested_dict_to_df(
        phone_direct_stocks_thresh_dict
    )
    phone_direct_stocks_thresh.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    phone_direct_stocks_thresh = prepare_df_for_concat(
        phone_direct_stocks_thresh,
        unit="kg",
        stock_type="direct",
        dimension="communication",
        stock="mobile_phone",
    )

    """########################################################
        COMMUNICATION: INDIRECT STOCKS - REMOVE DIRECT STOCKS
       ########################################################"""
    # we also calculate direct stocks here, and separate them from indirect stocks below - BUT these results are not used in paper (only total stocks)

    #! indirect stocks contain negatives atm because of non-matching categories of direct vs. indirect stock intensities
    # negatives in some stock categories for material 2393 (porcelain) but overall positive
    # overall negatives for 2599 because in direct stocks all 259: categories summed up
    # --> currently we do not make use of distinction of direct vs. indirect stocks - so the negatives disappear when adding both again

    # PHONE
    phone_direct_stocks_prov_DC = phone_direct_stocks_prov.copy()
    phone_direct_stocks_thresh_DC = phone_direct_stocks_thresh.copy()

    # highest value in direct stock intensities for material: 2011:Manufacture of basic chemicals = stock 2610:Manufacture of electronic components and boards
    # we assume that phones accounted as 2610:Manufacture of electronic components and boards and deduct direct stocks from this account
    # material 259: in direct but not indirect stocks - for this purpose simply assigning to 2591
    phone_direct_stocks_prov_DC = (
        phone_direct_stocks_prov_DC.reset_index()
        .replace(
            {
                "direct": "indirect",
                "communication": "communication_i",
                "mobile_phone": 2630,
                "259:": 2591,
            }
        )
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )
    phone_direct_stocks_thresh_DC = (
        phone_direct_stocks_thresh_DC.reset_index()
        .replace(
            {
                "direct": "indirect",
                "communication": "communication_i",
                "mobile_phone": 2630,
                "259:": 2591,
            }
        )
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )

    # rename index product in indirect stock dfs
    phone_indirect_stocks_prov = (
        phone_indirect_stocks_prov.reset_index()
        .replace({"consumer electronics, mobile de": "phone"})
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )
    phone_indirect_stocks_thresh = (
        phone_indirect_stocks_thresh.reset_index()
        .replace({"consumer electronics, mobile de": "phone"})
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
    )

    # align indiced for correct subtraction
    phone_direct_stocks_prov_DC_align = align_indices_zero(
        phone_direct_stocks_prov_DC, phone_indirect_stocks_prov
    )
    phone_direct_stocks_thresh_DC_align = align_indices_zero(
        phone_direct_stocks_thresh_DC, phone_indirect_stocks_thresh
    )

    # subtraction and check for negatives and correct sums
    phone_indirect_stocks_prov_DCfree = phone_indirect_stocks_prov.sub(
        phone_direct_stocks_prov_DC_align
    )
    phone_indirect_stocks_prov_DCfree.lt(0).any().any()
    phone_indirect_stocks_thresh_DCfree = (
        phone_indirect_stocks_thresh - phone_direct_stocks_thresh_DC_align
    )
    phone_indirect_stocks_thresh_DCfree.lt(0).any().any()
    assert (
        (
            phone_direct_stocks_prov_DC.sum().sum()
            + phone_indirect_stocks_prov_DCfree.sum().sum()
        )
        - phone_indirect_stocks_prov.sum().sum()
    ) < 1e-05
    assert (
        (
            phone_direct_stocks_thresh_DC.sum().sum()
            + phone_indirect_stocks_thresh_DCfree.sum().sum()
        )
        - phone_indirect_stocks_thresh.sum().sum()
    ) < 1e-05
    # check for negatives
    # phone_indirect_stocks_prov_DCfree.groupby('material').sum()
    # phone_indirect_stocks_prov.groupby('material').sum()
    # ps = phone_indirect_stocks_prov.groupby('stock').sum()
    # phone_direct_stocks_prov.groupby('material').sum()

    """####################################
            HOUSING: DIRECT STOCKS
       ####################################"""

    # 1) obtain DLS functions provided for housing
    DLS_2015_thresh_hous = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Housing|total"
    ]
    max(DLS_2015_thresh_hous.value)
    min(DLS_2015_thresh_hous.value)
    DLS_2015_funct_prov_hous = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Housing|total"
    ]

    # 2) load housing archetype split and material intensities and bring in desired format
    housing_archetypes = pd.read_excel(
        input_paths_dict.get("housing_archetypes_path")
    )
    housing_archetypes = pd.melt(housing_archetypes, id_vars=["region"])
    housing_archetypes.rename(
        columns={"variable": "stock type", "value": "split"}, inplace=True
    )
    # assumption: informal/RT have MI's such as SFH (curr. only few informal houses in USA, Canada + RT 1-2% elsewhere); RES 0  = conventional design
    housing_archetypes = housing_archetypes.replace(
        {
            "SFH": "SFH",
            "MFH": "MFH",
            "RT": "SFH",
            "informal": "SFH",
            "Oth_R32EU12-H": "R32EU12-H",
            "Oth_R32EU15": "R32EU15",
        }
    )
    housing_archetypes = housing_archetypes.groupby(["region", "stock type"]).sum()

    # load and rename material intensities
    housing_mat_intens = pd.read_excel(
        input_paths_dict.get("housing_mat_intens_path"),
        sheet_name="data_m2_out",
    )
    housing_mat_intens.rename(
        columns={
            "Unnamed: 0": "stock type",
            "Unnamed: 1": "region",
            "iron_steel": "iron_steel (kg/m²)",
            "copper": "copper (kg/m²)",
            "aluminum": "aluminum (kg/m²)",
            "other_metals": "other_metals (kg/m²)",
            "timber": "wood (kg/m²)",
            "other_biomass": "other_biomass (kg/m²)",
            "concrete": "concrete (kg/m²)",
            "cement": "cement (kg/m²)",
            "aggregates": "aggregates (kg/m²)",
            "clay_bricks": "bricks (kg/m²)",
            "mortar": "mortar (kg/m²)",
            "gypsum": "gypsum (kg/m²)",
            "ceramics": "ceramics (kg/m²)",
            "glass": "glass (kg/m²)",
            "other_minerals": "other_minerals (kg/m²)",
            "plastics": "plastics (kg/m²)",
            "bitumen": "bitumen (kg/m²)",
            "other_materials": "other_materials (kg/m²)",
        },
        inplace=True,
    )

    # 3) map archetypes and material intensities to MISO2 regions
    housing_mapping = pd.read_excel(
        input_paths_dict.get("country_correspondence"), sheet_name="MISO_SSP"
    )
    housing_mapping.rename(columns={"RECC2.4": "region"}, inplace=True)

    housing_archetypes_mapped = housing_mapping.merge(
        housing_archetypes.reset_index(), on=["region"]
    )
    housing_archetypes_mapped.rename(
        columns={"region": "RECC2.4", "MISO2_country": "region"}, inplace=True
    )
    housing_archetypes_mapped.set_index(
        ["region", "stock type", "SSP_5", "SSP_35", "RECC2.4", "Haberl_et_al"],
        inplace=True,
    )
    housing_mapping.rename(
        columns={"region": "RECC2.4", "Haberl_et_al": "region"}, inplace=True
    )
    housing_mat_intens_mapped = housing_mapping.merge(
        housing_mat_intens.reset_index(), on=["region"]
    )
    housing_mat_intens_mapped.rename(
        columns={
            "region": "Haberl_et_al",
            "MISO2_country": "region",
            "building_type": "stock type",
        },
        inplace=True,
    )

    # housing_mat_intens_mapped = housing_mat_intens_mapped  [['region', 'stock type', 'iron_steel (kg/m²)', 'copper (kg/m²)',
    #                                         'aluminum (kg/m²)', 'other_metals (kg/m²)', 'wood (kg/m²)', 'other_biomass (kg/m²)',
    #                                         'concrete (kg/m²)', 'cement (kg/m²)', 'aggregates (kg/m²)', 'bricks (kg/m²)',
    #                                         'mortar (kg/m²)', 'gypsum (kg/m²)', 'ceramics (kg/m²)','glass (kg/m²)',
    #                                         'other_minerals (kg/m²)', 'plastics (kg/m²)', 'bitumen (kg/m²)', 'other_materials (kg/m²)']]
    housing_mat_intens_mapped = housing_mat_intens_mapped[
        [
            "region",
            "stock type",
            "iron_steel (kg/m²)",
            "wood (kg/m²)",
            "concrete (kg/m²)",
            "cement (kg/m²)",
            "other_materials (kg/m²)",
        ]
    ]

    housing_mat_intens_mapped["stock type"].replace(
        {"RS": "SFH", "RM": "MFH"}, inplace=True
    )
    housing_archetypes_mapped.index.get_level_values(0).unique()  # 177
    housing_mat_intens_mapped.region.unique().shape  # 177

    # 4) calculate functions per archetype and material stocks for DLS provided (funct_prov) and DLS threshold
    # a) DLS provided: per capita
    DLS_2015_funct_prov_hous.rename(columns={"value": "functions"}, inplace=True)
    DLS_2015_funct_prov_hous.region.unique().shape  # 175
    housing_archetypes_mapped_regions = pd.DataFrame(
        housing_archetypes_mapped.index.get_level_values(0).unique()
    )
    DLS_2015_funct_prov_hous_regions = pd.DataFrame(
        DLS_2015_funct_prov_hous.region.unique()
    ).rename(columns={0: "region"})
    housing_archetypes_mapped_regions.equals(DLS_2015_funct_prov_hous_regions)
    equal_regions = pd.merge(
        housing_archetypes_mapped_regions,
        DLS_2015_funct_prov_hous_regions,
        how="inner",
        on=["region"],
    )
    housing_funct_archetype = calc_function_split(
        DLS_2015_funct_prov_hous, housing_archetypes_mapped, "hous_funct_prov"
    )
    housing_funct_archetype.region.unique().shape

    housing_funct_archetype_prov = housing_funct_archetype[
        ["region", "stock type", "hous_funct_prov"]
    ]
    housing_direct_prov_cap = calc_direct_cap(
        housing_funct_archetype_prov, housing_mat_intens_mapped, "hous_funct_prov"
    )
    housing_direct_prov_cap = housing_direct_prov_cap.iloc[:, 6:]

    # b) at scale (regions)
    housing_direct_prov_scale = calc_direct_scale(
        housing_direct_prov_cap, MISO2_population_2015_subset
    ).astype(float)
    housing_direct_prov_scale_unpiv = pd.melt(
        housing_direct_prov_scale.reset_index(),
        id_vars=["region", "stock type", "unit"],
        value_vars=housing_direct_prov_scale.columns,
        var_name="material",
        value_name="value",
    ).set_index(["region", "stock type", "unit", "material"])
    housing_direct_prov_scale_unpiv.index.rename(
        {"stock type": "product"}, inplace=True
    )
    housing_direct_stocks_prov = prepare_df_for_concat(
        housing_direct_prov_scale_unpiv,
        unit="tons",
        stock_type="direct",
        dimension="housing",
        stock="housing",
        year=2015,
        product="res_building",
    )

    # 5) calculate functions per archetype and material stocks for DLS_threshold (thresh)
    DLS_2015_thresh_hous.rename(columns={"value": "functions"}, inplace=True)
    housing_funct_archetype_thresh = calc_function_split(
        DLS_2015_thresh_hous, housing_archetypes_mapped, "hous_thresh"
    )
    housing_funct_archetype_thresh = housing_funct_archetype_thresh[
        ["region", "stock type", "hous_thresh"]
    ]
    housing_direct_cap_thresh = calc_direct_cap(
        housing_funct_archetype_thresh, housing_mat_intens_mapped, "hous_thresh"
    )
    housing_direct_cap_thresh = housing_direct_cap_thresh.iloc[:, 6:]

    # b) at scale (regions)
    housing_direct_thresh_scale = calc_direct_scale(
        housing_direct_cap_thresh, MISO2_population_2015_subset
    ).astype(float)
    housing_direct_thresh_scale_unpiv = pd.melt(
        housing_direct_thresh_scale.reset_index(),
        id_vars=["region", "stock type", "unit"],
        value_vars=housing_direct_thresh_scale.columns,
        var_name="material",
        value_name="value",
    ).set_index(["region", "stock type", "unit", "material"])
    housing_direct_thresh_scale_unpiv.index.rename(
        {"stock type": "product"}, inplace=True
    )
    housing_direct_stocks_thresh = prepare_df_for_concat(
        housing_direct_thresh_scale_unpiv,
        unit="tons",
        stock_type="direct",
        dimension="housing",
        stock="housing",
        year=2015,
        product="res_building",
    )

    """####################################
            HOUSING: IN-DIRECT STOCKS
       ####################################"""

    # multiply direct material demand for housing by indirect stock intensities per building material
    # read indirect stock intensities from ecoinvent based data and modify technology labels for matching
    shelter_indstocks_path = input_paths_dict.get("Velez_shelter")
    shelter_indstocks_dict = read_excel_into_dict(shelter_indstocks_path)
    del shelter_indstocks_dict["summary"]
    for key in shelter_indstocks_dict:
        shelter_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)

    # rename dict keys to match dataframe: Haberl et al. MIs
    shelter_key_rename = {
        "m_cement_port_RoW": "cement",
        "m_paper_RoW": "NA",
        "m_wood_dried_RER": "wood_RER",
        "m_wood_dried_RoW": "wood_RoW",
        "m_concrete_30MPa_AT": "concrete_AT",
        "m_concrete_30MPa_BR": "concrete_BR",
        "m_concrete_30MPa_ZA": "concrete_ZA",
        "m_concrete_30MPa_RoW": "concrete_RoW",
        "m_steel_unalloyed_GLO": "iron_steel",
        "m_clay_brick_GLO": "bricks",
        "m_copper_GLO": "copper",
        "m_flat_glass_coated_RER": "glass_RER",
        "m_flat_glass_coated_RoW": "glass_RoW",
        "m_alu_GLO": "aluminum",
        "m_plastic_pipes_GLO": "plastics",
    }

    # ecoinvent MIs: adjust units for above materials (from ecoinvent: 2316 kg/m³ concrete and 422.4 kg/m³ sawnwood)
    for old_key, new_key in shelter_key_rename.items():
        if old_key in shelter_indstocks_dict:
            shelter_indstocks_dict[new_key] = shelter_indstocks_dict.pop(old_key)
    for key in list(shelter_indstocks_dict.keys()):
        if key.startswith("concrete_"):
            shelter_indstocks_dict[key] = shelter_indstocks_dict[key] / 2316
        elif key.startswith("wood_"):
            shelter_indstocks_dict[key] = shelter_indstocks_dict[key] / 422.4

    # modify material demand dataframe (from direct stocks) and calc. respective indirect stock requirements to provide these materials
    # DLS provided - prepare dataframe for indirect material demand of housing
    housing_material_demand_prov = housing_direct_prov_scale.groupby("region").sum()

    # regionalize material demand to match to regional ecoinvent processes: define a list of materials to process and path for regional ecoinvent process lists per region
    materials = ["concrete", "wood"]
    path_ecoinvent_housing_reg = input_paths_dict.get("ecoinvent_housing_reg_mapping_path")
    housing_material_demand_prov_reg = regionalize_material_demand_ecoinvent(
        materials, housing_material_demand_prov, path_ecoinvent_housing_reg
    )
    housing_material_demand_prov = housing_material_demand_prov_reg.copy()
    housing_material_demand_prov.insert(0, "year", 2015)
    housing_material_demand_prov = housing_material_demand_prov.set_index(
        "year", append=True
    )
    housing_material_demand_prov.sum(axis=0)

    # multiply housing_material_demand (tons) with ind. stock intens. by material (kg/kg); result = tons
    housing_indirect_stocks_prov_dict = create_nested_dict(
        shelter_indstocks_dict, housing_material_demand_prov
    )
    housing_indirect_stocks_prov = expand_nested_dict_to_df(
        housing_indirect_stocks_prov_dict
    )
    housing_indirect_stocks_prov.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    housing_indirect_stocks_prov = prepare_df_for_concat(
        housing_indirect_stocks_prov,
        unit="tons",
        stock_type="indirect",
        dimension="housing_i",
        stock="housing_related",
        year=2015,
        product="housing_related",
    )
    housing_indirect_stocks_prov = (
        housing_indirect_stocks_prov.reset_index()
        .replace(
            {
                "wood_RER": "wood",
                "wood_RoW": "wood",
                "concrete_AT": "concrete",
                "concrete_BR": "concrete",
                "concrete_ZA": "concrete",
                "concrete_RoW": "concrete",
                "glass_RER": "glass",
                "glass_RoW": "glass",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )

    # DLS threshold - dataframe for indirect material demand of housing
    housing_material_demand_thresh = housing_direct_thresh_scale.groupby("region").sum()

    # regionalize material demand to match to regional ecoinvent processes
    housing_material_demand_thresh_reg = regionalize_material_demand_ecoinvent(
        materials, housing_material_demand_thresh, path_ecoinvent_housing_reg
    )
    housing_material_demand_thresh = housing_material_demand_thresh_reg
    housing_material_demand_thresh.insert(0, "year", 2015)
    housing_material_demand_thresh = housing_material_demand_thresh.set_index(
        "year", append=True
    )
    housing_material_demand_thresh.sum(axis=0)

    # multiply housing_material_demand (tons) with ind. stock intens. by material (kg/kg)
    # result = tons
    housing_indirect_stocks_thresh_dict = create_nested_dict(
        shelter_indstocks_dict, housing_material_demand_thresh
    )
    housing_indirect_stocks_thresh = expand_nested_dict_to_df(
        housing_indirect_stocks_thresh_dict
    )
    housing_indirect_stocks_thresh.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    housing_indirect_stocks_thresh = prepare_df_for_concat(
        housing_indirect_stocks_thresh,
        unit="tons",
        stock_type="indirect",
        dimension="housing_i",
        stock="housing_related",
        year=2015,
        product="housing_related",
    )
    housing_indirect_stocks_thresh = (
        housing_indirect_stocks_thresh.reset_index()
        .replace(
            {
                "wood_RER": "wood",
                "wood_RoW": "wood",
                "concrete_AT": "concrete",
                "concrete_BR": "concrete",
                "concrete_ZA": "concrete",
                "concrete_RoW": "concrete",
                "glass_RER": "glass",
                "glass_RoW": "glass",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )

    """###########################################################
            MOBILITY: IN-DIRECT STOCKS (including direct stocks)
       ###########################################################"""

    # 1) obtain functions provided for passenger kilometers
    DLS_2015_thresh_mob = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Transport"
    ]
    DLS_2015_funct_prov_mob = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Transport"
    ]
    max(DLS_2015_thresh_mob.value)
    min(DLS_2015_thresh_mob.value)

    # 2) load modal share data and format
    modal_shares_path = input_paths_dict.get("modal_shares_path")
    modal_shares = pd.read_csv(modal_shares_path).rename(
        columns={"iso": "region"}
    )
    # rename regions according to MISO2 country labels and drop all which are not in MISO2 country list
    for key, value in country_correspondence_dict.items():
        modal_shares["region"].replace({key: value}, inplace=True)
    modal_shares = modal_shares[
        modal_shares.region.isin(country_correspondence.MISO2.to_list())
    ]
    modal_shares.set_index(["region"], inplace=True)

    # 3) load indirect stock intensity dictionary for mobility
    mobility_indstocks_path = input_paths_dict.get("Velez_mobility")
    mobility_indstocks_dict = read_excel_into_dict(mobility_indstocks_path)
    del mobility_indstocks_dict["summary"]
    for key in mobility_indstocks_dict:
        mobility_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)
        mobility_indstocks_dict.get(key).columns = mobility_indstocks_dict.get(
            key
        ).columns.astype(int)
    # rename dict keys to match dataframe
    modal_shares.rename(
        columns={
            "rail": "m_train_GLO",
            "bus": "m_bus_GLO",
            "ldv": process_change.get("car"),
            "twothree": process_change.get("scooter"),
        },
        inplace=True,
    )
    modal_shares = pd.melt(modal_shares.reset_index(), id_vars=["region"]).set_index(
        ["region", "variable"]
    )
    modal_shares.rename(columns={"value": "split"}, inplace=True)

    # 4) calculate functions per modal share for DLS_reached (funct_prov)
    # DLS provided - per capita
    DLS_2015_funct_prov_mob.rename(columns={"value": "functions"}, inplace=True)
    DLS_2015_funct_prov_mob.set_index(["region", "variable", "unit"], inplace=True)
    mobility_funct_split = calc_function_split(
        DLS_2015_funct_prov_mob, modal_shares, "mob_funct_prov"
    )
    mobility_funct_split = mobility_funct_split.set_index(["region", "variable"])[
        ["mob_funct_prov"]
    ]
    mobility_funct_split = mobility_funct_split.reset_index().pivot(
        index="region", columns="variable", values="mob_funct_prov"
    )

    # adjust vehicle-km in motorized individual transport (car) for occupancy rates in mobility_funct_split to get to person-km in DLS data
    # read occupancy data and mapping of countries to R11
    occupancy_data_path = input_paths_dict.get("occupancy_path")
    occupancy_rate = pd.read_excel(
        occupancy_data_path
    ).rename(
        columns={
            "R11.region": "region",
            "occupancy_ldv": "p car, medium size, petrol",
            "occupancy_two": "passenger, motor scooter",
        }
    )
    occupancy_rate_dict_car = dict(
        zip(occupancy_rate.region, occupancy_rate["p car, medium size, petrol"])
    )
    R11_country_correspondence = pd.read_excel(
        input_paths_dict.get("country_correspondence")
    )
    R11_country_correspondence_dict = dict(
        zip(R11_country_correspondence.MISO2, R11_country_correspondence["R11"])
    )
    mobility_funct_split_occup = mobility_funct_split.copy()
    mobility_funct_split_occup.reset_index(inplace=True)
    mobility_funct_split_occup["R11"] = mobility_funct_split_occup["region"].map(
        R11_country_correspondence_dict
    )
    mobility_funct_split_occup["occupancy_car"] = mobility_funct_split_occup["R11"].map(
        occupancy_rate_dict_car
    )
    # mobility_funct_split_occup['occupancy_scoot'] = mobility_funct_split_occup['R11'].map(occupancy_rate_dict_scooter)
    mobility_funct_split_occup["vkm_car"] = (
        mobility_funct_split_occup[process_change.get("car")]
        / mobility_funct_split_occup["occupancy_car"]
    )
    mobility_funct_split_occup.rename(
        columns={
            process_change.get("car"): "pkm_car",
            "vkm_car": process_change.get("car"),
        },
        inplace=True,
    )
    mobility_funct_split_occup = mobility_funct_split_occup.set_index(["region"])[
        [
            process_change.get("car"),
            "m_train_GLO",
            process_change.get("scooter"),
            "m_bus_GLO",
        ]
    ]
    mobility_funct_split = mobility_funct_split_occup.copy()
    del mobility_funct_split_occup

    # regionalize mobility function demand to match to regional ecoinvent processes for car transport
    # Define a list of materials to process and path for regional ecoinvent process lists per region
    modes = [process_change.get("car")]
    path_ecoinvent_mobility_reg = input_paths_dict.get("ecoinvent_mobility_reg_mapping_path")
    mobility_funct_split_reg = regionalize_material_demand_ecoinvent(
        modes, mobility_funct_split, path_ecoinvent_mobility_reg
    )
    mobility_funct_split_reg.insert(0, "year", 2015)
    mobility_funct_split_reg = mobility_funct_split_reg.set_index("year", append=True)

    # multiply person-km demand with indirect stock intens. by material (kg/cap)

    # DLS provided - at scale
    # calculate total pkm per region
    DLS_2015_funct_prov_mob_scale = calc_functions_scale(
        DLS_2015_funct_prov_mob, MISO2_population_2015_subset
    )
    # calculate total pkm modal shares per region & format to multoply with indirect stock intensities
    mobility_funct_split_prov_scale = calc_function_split(
        DLS_2015_funct_prov_mob_scale, modal_shares, "mob_funct_prov"
    )
    mobility_funct_split_prov_scale = mobility_funct_split_prov_scale.set_index(
        ["region", "variable"]
    )[["mob_funct_prov"]]
    mobility_funct_split_prov_scale = (
        mobility_funct_split_prov_scale.reset_index().pivot(
            index="region", columns="variable", values="mob_funct_prov"
        )
    )
    mobility_funct_split_prov_scale.insert(0, "year", 2015)
    mobility_funct_split_prov_scale = mobility_funct_split_prov_scale.set_index(
        "year", append=True
    )

    # introduce occupancy rate for car
    mobility_funct_split_occup = mobility_funct_split_prov_scale.copy()
    mobility_funct_split_occup.reset_index(inplace=True)
    mobility_funct_split_occup["R11"] = mobility_funct_split_occup["region"].map(
        R11_country_correspondence_dict
    )
    mobility_funct_split_occup["occupancy_car"] = mobility_funct_split_occup["R11"].map(
        occupancy_rate_dict_car
    )
    mobility_funct_split_occup["vkm_car"] = (
        mobility_funct_split_occup[process_change.get("car")]
        / mobility_funct_split_occup["occupancy_car"]
    )
    mobility_funct_split_occup.rename(
        columns={
            process_change.get("car"): "pkm_car",
            "vkm_car": process_change.get("car"),
        },
        inplace=True,
    )
    mobility_funct_split_occup = mobility_funct_split_occup.set_index(["region"])[
        [
            process_change.get("car"),
            "m_train_GLO",
            process_change.get("scooter"),
            "m_bus_GLO",
        ]
    ]
    mobility_funct_split_prov_scale = mobility_funct_split_occup.copy()
    del mobility_funct_split_occup

    # regionalize material production processes
    mobility_funct_split_prov_scale_reg = regionalize_material_demand_ecoinvent(
        modes, mobility_funct_split_prov_scale, path_ecoinvent_mobility_reg
    )
    mobility_funct_split_prov_scale_reg.insert(0, "year", 2015)
    mobility_funct_split_prov_scale_reg = mobility_funct_split_prov_scale_reg.set_index(
        "year", append=True
    )
    mobility_indirect_stocks_prov_dict_scale = create_nested_dict(
        mobility_indstocks_dict, mobility_funct_split_prov_scale_reg
    )
    mobility_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        mobility_indirect_stocks_prov_dict_scale
    )
    mobility_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    mobility_indirect_stocks_prov = prepare_df_for_concat(
        mobility_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="transport_i",
        stock="transport_related",
        year=2015,
        product="transport_related",
    )
    mobility_indirect_stocks_prov.sum() / 1e9

    # DLS threshold - at scale
    # calculate total pkm per region
    DLS_2015_thresh_mob.rename(columns={"value": "functions"}, inplace=True)
    DLS_2015_thresh_mob.set_index(["region", "variable", "unit"], inplace=True)
    DLS_2015_thresh_mob_scale = calc_functions_scale(
        DLS_2015_thresh_mob, MISO2_population_2015_subset
    )
    # calculate total pkm modal shares per region & format to multoply with indirect stock intensities
    mobility_funct_split_thresh_scale = calc_function_split(
        DLS_2015_thresh_mob_scale, modal_shares, "mob_funct_prov"
    )
    mobility_funct_split_thresh_scale = mobility_funct_split_thresh_scale.set_index(
        ["region", "variable"]
    )[["mob_funct_prov"]]
    mobility_funct_split_thresh_scale = (
        mobility_funct_split_thresh_scale.reset_index().pivot(
            index="region", columns="variable", values="mob_funct_prov"
        )
    )
    mobility_funct_split_thresh_scale.insert(0, "year", 2015)
    mobility_funct_split_thresh_scale = mobility_funct_split_thresh_scale.set_index(
        "year", append=True
    )

    # introduce occupancy rate for car
    mobility_funct_split_occup = mobility_funct_split_thresh_scale.copy()
    mobility_funct_split_occup.reset_index(inplace=True)
    mobility_funct_split_occup["R11"] = mobility_funct_split_occup["region"].map(
        R11_country_correspondence_dict
    )
    mobility_funct_split_occup["occupancy_car"] = mobility_funct_split_occup["R11"].map(
        occupancy_rate_dict_car
    )
    mobility_funct_split_occup["vkm_car"] = (
        mobility_funct_split_occup[process_change.get("car")]
        / mobility_funct_split_occup["occupancy_car"]
    )
    mobility_funct_split_occup.rename(
        columns={
            process_change.get("car"): "pkm_car",
            "vkm_car": process_change.get("car"),
        },
        inplace=True,
    )
    mobility_funct_split_occup = mobility_funct_split_occup.set_index(["region"])[
        [
            process_change.get("car"),
            "m_train_GLO",
            process_change.get("scooter"),
            "m_bus_GLO",
        ]
    ]
    mobility_funct_split_thresh_scale = mobility_funct_split_occup.copy()
    del mobility_funct_split_occup

    # regionalize material production processes
    mobility_funct_split_thresh_scale_reg = regionalize_material_demand_ecoinvent(
        modes, mobility_funct_split_thresh_scale, path_ecoinvent_mobility_reg
    )
    mobility_funct_split_thresh_scale_reg.insert(0, "year", 2015)
    mobility_funct_split_thresh_scale_reg = (
        mobility_funct_split_thresh_scale_reg.set_index("year", append=True)
    )

    # multiply with indirect stock intensities (total pkm/region * kg_indirect_stocks/region) = kg_indirect_stock/region
    mobility_indirect_stocks_thresh_dict_scale = create_nested_dict(
        mobility_indstocks_dict, mobility_funct_split_thresh_scale_reg
    )
    mobility_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        mobility_indirect_stocks_thresh_dict_scale
    )
    mobility_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    mobility_indirect_stocks_thresh = prepare_df_for_concat(
        mobility_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="transport_i",
        stock="transport_related",
        year=2015,
        product="transport_related",
    )

    """####################################
            MOBILITY: DIRECT STOCKS
       ####################################"""
    # we also calculate direct stocks here, and separate them from indirect stocks below - BUT these results are not used in paper (only total stocks)

    # subset direct stock intensities from SI of Veléz-Henao & Pauliuk (2023) for mobility
    mobility_direct_stocks_intens = {
        k: v
        for k, v in direct_stocks_intens.items()
        if k in ("car", "bus", "train", "scooter")
    }
    # prepare direct stock dictionary to match with function demand
    for key in mobility_direct_stocks_intens:
        df_subset = mobility_direct_stocks_intens[key][["Material", "kg/pkm"]]
        df_subset.set_index("Material", inplace=True)
        mobility_direct_stocks_intens[key] = df_subset

    # DLS provided
    # obtain function demand at scale per mode of transport & align labels with mobility_direct_stocks_inten
    mobility_funct_demand_prov = mobility_funct_split_prov_scale.copy()
    mobility_funct_demand_prov.insert(0, "year", 2015)
    mobility_funct_demand_prov = mobility_funct_demand_prov.set_index(
        "year", append=True
    )
    mobility_funct_demand_prov.rename(
        columns={
            process_change.get("car"): "car",
            "m_train_GLO": "train",
            process_change.get("scooter"): "scooter",
            "m_bus_GLO": "bus",
        },
        inplace=True,
    )

    # calc mobility direct stocks
    mobility_direct_stocks_prov_dict = create_nested_dict(
        mobility_direct_stocks_intens, mobility_funct_demand_prov
    )
    mobility_direct_stocks_prov = expand_nested_dict_to_df(
        mobility_direct_stocks_prov_dict
    )
    mobility_direct_stocks_prov.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    # prepare to concat with other dimensions
    mobility_direct_stocks_prov = prepare_df_for_concat(
        mobility_direct_stocks_prov,
        unit="kg",
        stock_type="direct",
        dimension="transport",
        stock="motor_vehicles",
        year=2015,
        product="motor_vehicles",
    )
    mobility_direct_stocks_prov.sum() / 1e9

    # DLS threshold
    # obtain function demand at scale per mode of transport & align labels with mobility_direct_stocks_inten
    mobility_funct_demand_thresh = mobility_funct_split_thresh_scale.copy()
    mobility_funct_demand_thresh.insert(0, "year", 2015)
    mobility_funct_demand_thresh = mobility_funct_demand_thresh.set_index(
        "year", append=True
    )
    mobility_funct_demand_thresh.rename(
        columns={
            process_change.get("car"): "car",
            "m_train_GLO": "train",
            process_change.get("scooter"): "scooter",
            "m_bus_GLO": "bus",
        },
        inplace=True,
    )

    # calc mobility direct stocks
    mobility_direct_stocks_thresh_dict = create_nested_dict(
        mobility_direct_stocks_intens, mobility_funct_demand_thresh
    )
    mobility_direct_stocks_thresh = expand_nested_dict_to_df(
        mobility_direct_stocks_thresh_dict
    )
    mobility_direct_stocks_thresh.index.names = [
        "region,year",
        "product",
        "material",
        "unit",
    ]
    # prepare to concat with other dimensions
    mobility_direct_stocks_thresh = prepare_df_for_concat(
        mobility_direct_stocks_thresh,
        unit="kg",
        stock_type="direct",
        dimension="transport",
        stock="motor_vehicles",
        year=2015,
        product="motor_vehicles",
    )
    mobility_direct_stocks_thresh.sum() / 1e9

    """########################################################
        MOBILITY: INDIRECT STOCKS - REMOVE DIRECT STOCKS
       ########################################################"""
    #! indirect stocks contain negatives atm because of non-matching categories of direct vs. indirect stock intensities
    # --> currently we do not make use of distinction of direct vs. indirect stocks - so the negatives disappear when adding both again

    mobility_direct_stocks_prov_DC = mobility_direct_stocks_prov.copy().reset_index()
    mobility_direct_stocks_thresh_DC = (
        mobility_direct_stocks_thresh.copy().reset_index()
    )

    # assign direct stocks to indirect ISIC stock types for vehicles (2910-3092)
    conditions = [
        mobility_direct_stocks_thresh_DC["product"] == "car",
        mobility_direct_stocks_thresh_DC["product"] == "train",
        mobility_direct_stocks_thresh_DC["product"] == "bus",
        mobility_direct_stocks_thresh_DC["product"] == "scooter",
    ]

    # Values to assign based on conditions
    values = [2910, 3020, 2910, 3091]

    # Apply replacements
    mobility_direct_stocks_thresh_DC["stock"] = np.select(
        conditions, values, default=mobility_direct_stocks_thresh_DC["stock"]
    )
    mobility_direct_stocks_prov_DC["stock"] = np.select(
        conditions, values, default=mobility_direct_stocks_thresh_DC["stock"]
    )
    mobility_direct_stocks_thresh_DC = mobility_direct_stocks_thresh_DC.replace(
        {"direct": "indirect", "transport": "transport_i"}
    ).set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ]
    )
    mobility_direct_stocks_prov_DC = mobility_direct_stocks_prov_DC.replace(
        {"direct": "indirect", "transport": "transport_i"}
    ).set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ]
    )

    # rename index product in indirect stock dfs
    mobility_indirect_stocks_prov = (
        mobility_indirect_stocks_prov.reset_index()
        .replace(
            {
                process_change.get("scooter"): "scooter",
                process_change.get("car") + "_GLO": "car",
                process_change.get("car") + "_RoW": "car",
                process_change.get("car") + "_RER": "car",
                "m_bus_GLO": "bus",
                "m_train_GLO": "train",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )
    mobility_indirect_stocks_thresh = (
        mobility_indirect_stocks_thresh.reset_index()
        .replace(
            {
                process_change.get("scooter"): "scooter",
                process_change.get("car") + "_GLO": "car",
                process_change.get("car") + "_RoW": "car",
                process_change.get("car") + "_RER": "car",
                "m_bus_GLO": "bus",
                "m_train_GLO": "train",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )

    # align indices for correct subtraction
    mobility_direct_stocks_prov_DC_align = align_indices_zero(
        mobility_direct_stocks_prov_DC, mobility_indirect_stocks_prov
    )
    mobility_direct_stocks_thresh_DC_align = align_indices_zero(
        mobility_direct_stocks_thresh_DC, mobility_indirect_stocks_thresh
    )

    # subtraction and check for negatives and correct sums
    mobility_indirect_stocks_prov_DCfree = mobility_indirect_stocks_prov.sub(
        mobility_direct_stocks_prov_DC_align
    )
    mobility_indirect_stocks_prov_DCfree.lt(0).any().any()
    mobility_indirect_stocks_thresh_DCfree = (
        mobility_indirect_stocks_thresh - mobility_direct_stocks_thresh_DC_align
    )
    mobility_indirect_stocks_thresh_DCfree.lt(0).any().any()
    assert (
        (
            mobility_direct_stocks_prov_DC.sum().sum()
            + mobility_indirect_stocks_prov_DCfree.sum().sum()
        )
        - mobility_indirect_stocks_prov.sum().sum()
    ) < 1e-03
    assert (
        (
            mobility_direct_stocks_thresh_DC.sum().sum()
            + mobility_indirect_stocks_thresh_DCfree.sum().sum()
        )
        - mobility_indirect_stocks_thresh.sum().sum()
    ) < 1e-03
    # check for negatives
    # mobility_check = mobility_indirect_stocks_prov_DCfree.groupby(['region','material']).sum()
    # mobility_check_USA = mobility_check[mobility_check.index.get_level_values(0)=='United States of America']
    # mobility_check = mobility_indirect_stocks_prov_DCfree.groupby(['region','stock','material']).sum()
    # mobility_direct_stocks_prov.sum()/1e9

    """####################################
            CLOTHING: DIRECT STOCKS 
       ####################################"""
    # assuming no gap; multiply reference scenario for 'decent clothing' from Veléz-Henao & pauliuk with population figures (*1000 cause pop figures in 1000's)

    # load scenario on clothing requirements
    clothing_needs = pd.read_excel(input_paths_dict.get("MISO2_clothing"))
    clothing_needs.set_index(["product", "unit/year"], inplace=True)

    # introduce factor to adjust in sensitivity analysis
    clothing_needs = clothing_needs * sensitivity_factor_clothing

    # scale up to whole population and format
    clothing_region = {}
    for i in MISO2_population_2015_subset.index:
        clothing_region[i] = (
            MISO2_population_2015_subset[
                MISO2_population_2015_subset.index == i
            ].values[0][0]
            * clothing_needs
            * 1000
        )
    clothing = expand_nested_dict_to_df(clothing_region)
    clothing.reset_index(inplace=True)
    clothing[["product", "unit"]] = pd.DataFrame(
        clothing.level_1.tolist(), index=clothing.index
    )
    clothing.rename(columns={"level_0": "region", 0: "value"}, inplace=True)
    clothing = clothing[["region", "product", "unit", "value"]].set_index(
        ["region", "product", "unit"]
    )

    # subset to countries overlapping in DLS and MISO2 data
    clothing_subset_MISOregions = clothing[
        clothing.index.get_level_values(0).isin(
            DLS_2015_thresh.index.get_level_values(0).unique()
        )
    ]
    clothing_material_demand = clothing_subset_MISOregions[
        ~clothing_subset_MISOregions.index.get_level_values(1).isin(
            ["market for washing machine", "market group for electricity, low voltage"]
        )
    ]
    clothing_material_demand = (
        clothing_material_demand.reset_index()
        .replace({"kg/cap": "kg"})
        .set_index(["region", "product", "unit"])
    )
    clothing_material_demand.index.names = ["region", "material", "unit"]
    clothing_direct_stocks = prepare_df_for_concat(
        clothing_material_demand.copy(),
        unit="kg",
        stock_type="direct",
        dimension="clothing",
        stock="clothing",
        year=2015,
        product="clothing",
    )
    # results in kg per region

    """####################################
            CLOTHING: IN-DIRECT STOCKS
       ####################################"""
    # multiply by indirect stock intensities per clothing material

    # read material intensity data for common technology labels and modify for matching
    clothing_indstocks_path = input_paths_dict.get("Velez_hygiene_clothing")
    clothing_indstocks_dict = read_excel_into_dict(clothing_indstocks_path)
    del clothing_indstocks_dict["summary"]
    for key in clothing_indstocks_dict:
        clothing_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)

    # rename dict keys to match dataframe
    clothing_key_rename = {
        "textile, woven cotton": "market for textile, woven cotton",
        "textile, knit cotton": "market for textile, knit cotton",
        "textile, silk": "market for textile, silk",
        "textile, nonwoven polypropylene": "market for textile, nonwoven polypropylene",
        "textile, nonwoven polyester": "market for textile, nonwoven polyester",
        "textile, kenaf": "market for textile, kenaf",
        "textile, jute": "market for textile, jute",
        "synthetic rubber": "market for synthetic rubber",
        "sheep production, for wool": "sheep production, for wool",
    }
    for key, n_key in zip(clothing_key_rename.keys(), clothing_key_rename.values()):
        clothing_indstocks_dict[n_key] = clothing_indstocks_dict.pop(key)

    # prepare dataframe for indirect material demand of clothing
    clothing_material_demand_piv = clothing_material_demand.reset_index().pivot(
        index="region", columns="material", values="value"
    )
    clothing_material_demand_piv.insert(0, "year", 2015)
    clothing_material_demand_piv = clothing_material_demand_piv.set_index(
        "year", append=True
    )

    # mult. clothing_material_demand (kg) with ind. stock intens. by material (kg/kg); result = kg
    clothing_indirect_stocks_dict = create_nested_dict(
        clothing_indstocks_dict, clothing_material_demand_piv
    )
    clothing_indirect_stocks = expand_nested_dict_to_df(clothing_indirect_stocks_dict)
    clothing_indirect_stocks.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    clothing_indirect_stocks = prepare_df_for_concat(
        clothing_indirect_stocks,
        unit="kg",
        stock_type="indirect",
        dimension="clothing_i",
        stock="clothing_related",
        year=2015,
        product="clothing_related",
    )

    """#####################################################
            SANITATION, HYGIENE & WATER: IN-DIRECT STOCKS
       #####################################################"""

    # SANITATION FLOWS
    sanitation_needs = pd.read_excel(
        input_paths_dict.get("MISO2_sanitation"), sheet_name="sanitation"
    )
    sanitation_needs.set_index(["product", "unit/year"], inplace=True)

    # introduce factor to adjust in sensitivity analysis
    sanitation_needs = sanitation_needs * sensitivity_factor_sanitation_water

    sanitation_region = {}
    # scale up to total population
    for i in MISO2_population_2015_subset.index:
        sanitation_region[i] = (
            MISO2_population_2015_subset[
                MISO2_population_2015_subset.index == i
            ].values[0][0]
            * sanitation_needs
            * 1000
        )
    sanitation = expand_nested_dict_to_df(sanitation_region)
    sanitation.reset_index(inplace=True)
    sanitation[["product", "unit"]] = pd.DataFrame(
        sanitation.level_1.tolist(), index=sanitation.index
    )
    sanitation.rename(columns={"level_0": "region", 0: "value"}, inplace=True)
    sanitation = sanitation[["region", "product", "unit", "value"]].set_index(
        ["region", "product", "unit"]
    )

    # 1) obtain % fullfillment provided for SANITATION
    DLS_2015_funct_prov_sanit = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Sanitation"
    ]
    DLS_sanitation_prov = (
        sanitation.copy().reset_index().merge(DLS_2015_funct_prov_sanit, on="region")
    )
    # multiply % of functions provided per region with ref_scenario requirements from Johan (per cap)
    DLS_sanitation_prov["value"] = (
        DLS_sanitation_prov["value_x"] * DLS_sanitation_prov["value_y"]
    )
    DLS_sanitation_prov = DLS_sanitation_prov[["region", "product", "value"]]
    # threshold = 100% fulfillment just equals variable 'sanitation' for whole population
    DLS_sanitation_thresh = sanitation.copy()
    # need to subset to countries overlapping in DLS and MISO2 data because not automatically done by mult with DLS_data like for _prov
    DLS_sanitation_thresh = DLS_sanitation_thresh[
        DLS_sanitation_thresh.index.get_level_values(0).isin(
            DLS_2015_thresh.index.get_level_values(0).unique()
        )
    ]

    ## SANITATION INDIRECT STOCKS
    # read mapping data for common technology labels and modify for matching
    sanitation_water_indstocks_path = input_paths_dict.get("Velez_hygiene_water")
    sanitation_water_indstocks_dict = read_excel_into_dict(
        sanitation_water_indstocks_path
    )
    del sanitation_water_indstocks_dict["summary"]
    for key in sanitation_water_indstocks_dict:
        sanitation_water_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)

    # rename dict keys to match dataframe
    sanitation_key_rename = {
        "p for electricity, low voltage": "market group for electricity, low voltage",
        "p for natural gas, high pressur": "market group for natural gas, high pressure",
        "p for tap water": "market group for tap water",
        "tmws, open burning": "treatment of municipal solid waste, open burning",
        "tmws, unsanitary landfill, wet ": "treatment of municipal solid waste, unsanitary landfill, wet infiltration class (500mm)",
        "treatment of kitchen and garden": "treatment of kitchen and garden biowaste, home composting in heaps and containers",
        "tmsw, sanitary landfill": "treatment of municipal solid waste, sanitary landfill",
        "wastewater, from residence": "market for wastewater, from residence",
        "tmws, unsanitary landfill, open": "treatment of municipal solid waste, open dump, wet infiltration class (500mm)",
    }
    for key, n_key in zip(sanitation_key_rename.keys(), sanitation_key_rename.values()):
        sanitation_water_indstocks_dict[n_key] = sanitation_water_indstocks_dict.pop(
            key
        )

    # SANITATION DLS provided
    ## adjust unit/format and multiply with indirect stock intensity (units between service demand and ecoinvent functional unit must match)
    sanitation_service_demand_prov_piv = DLS_sanitation_prov.reset_index().pivot(
        index="region", columns="product", values="value"
    )
    sanitation_service_demand_prov_piv.insert(0, "year", 2015)
    sanitation_service_demand_prov_piv = sanitation_service_demand_prov_piv.set_index(
        "year", append=True
    )

    # # mult. sanitation_material_demand (kg/m3) with ind. stock intens. by material (kg/kg or kg/m3); result = kg
    sanitation_indirect_stocks_prov_dict = create_nested_dict(
        sanitation_water_indstocks_dict, sanitation_service_demand_prov_piv
    )
    sanitation_indirect_stocks_prov = expand_nested_dict_to_df(
        sanitation_indirect_stocks_prov_dict
    )
    sanitation_indirect_stocks_prov.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    sanitation_indirect_stocks_prov = prepare_df_for_concat(
        sanitation_indirect_stocks_prov,
        unit="kg",
        stock_type="indirect",
        dimension="sanitation_i",
        stock="sanitation_related",
        year=2015,
        product="sanitation_related",
    )
    # prepare_df_for_concat converts kg to tons
    sanitation_indirect_stocks_prov.sum() / 1e9

    # SANITATION DLS threshold (operations as above)
    sanitation_service_demand_thresh_piv = DLS_sanitation_thresh.reset_index().pivot(
        index="region", columns="product", values="value"
    )
    sanitation_service_demand_thresh_piv.insert(0, "year", 2015)
    sanitation_service_demand_thresh_piv = (
        sanitation_service_demand_thresh_piv.set_index("year", append=True)
    )
    sanitation_indirect_stocks_thresh_dict = create_nested_dict(
        sanitation_water_indstocks_dict, sanitation_service_demand_thresh_piv
    )
    sanitation_indirect_stocks_thresh = expand_nested_dict_to_df(
        sanitation_indirect_stocks_thresh_dict
    )
    sanitation_indirect_stocks_thresh.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    sanitation_indirect_stocks_thresh = prepare_df_for_concat(
        sanitation_indirect_stocks_thresh,
        unit="kg",
        stock_type="indirect",
        dimension="sanitation_i",
        stock="sanitation_related",
        year=2015,
        product="sanitation_related",
    )
    # prepare_df_for_concat converts kg to tons
    sanitation_indirect_stocks_thresh.sum() / 1e9

    # WATER FLOWS
    water_needs = pd.read_excel(
        input_paths_dict.get("MISO2_sanitation"), sheet_name="water"
    )
    water_needs.set_index(["product", "unit/year"], inplace=True)
    # introduce factor to adjust in sensitivity analysis
    water_needs = water_needs * sensitivity_factor_sanitation_water
    water_region = {}
    # scale up to total population
    for i in MISO2_population_2015_subset.index:
        water_region[i] = (
            MISO2_population_2015_subset[
                MISO2_population_2015_subset.index == i
            ].values[0][0]
            * water_needs
            * 1000
        )
    water = expand_nested_dict_to_df(water_region)
    water.reset_index(inplace=True)
    water[["product", "unit"]] = pd.DataFrame(water.level_1.tolist(), index=water.index)
    water.rename(columns={"level_0": "region", 0: "value"}, inplace=True)
    water = water[["region", "product", "unit", "value"]].set_index(
        ["region", "product", "unit"]
    )

    # 1) obtain % fullfillment provided for WATER
    DLS_water_prov = (
        water.copy().reset_index().merge(DLS_2015_funct_prov_sanit, on="region")
    )
    # multiply % of functions provided per region with ref_scenario requirements from Johan (per cap)
    DLS_water_prov["value"] = DLS_water_prov["value_x"] * DLS_water_prov["value_y"]
    DLS_water_prov = DLS_water_prov[["region", "product", "value"]]
    # threshold = 100% fulfillment just equals variable 'water' for whole population
    DLS_water_thresh = water.copy()
    # need to subset to countries overlapping in DLS and MISO2 data because not automatically done by mult with DLS_data like for _prov
    DLS_water_thresh = DLS_water_thresh[
        DLS_water_thresh.index.get_level_values(0).isin(
            DLS_2015_thresh.index.get_level_values(0).unique()
        )
    ]

    # use sanitation_water_indstocks_dict to calculate ind stocks of water provision
    # WATER DLS provided
    # # #prepare dataframe for indirect service demand of water
    water_service_demand_prov_piv = DLS_water_prov.reset_index().pivot(
        index="region", columns="product", values="value"
    )
    water_service_demand_prov_piv.insert(0, "year", 2015)
    water_service_demand_prov_piv = water_service_demand_prov_piv.set_index(
        "year", append=True
    )

    # # mult. water_material_demand (kg/m3) with ind. stock intens. by material (kg/kg or kg/m3), result = kg
    water_indirect_stocks_prov_dict = create_nested_dict(
        sanitation_water_indstocks_dict, water_service_demand_prov_piv
    )
    water_indirect_stocks_prov = expand_nested_dict_to_df(
        water_indirect_stocks_prov_dict
    )
    water_indirect_stocks_prov.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    water_indirect_stocks_prov = prepare_df_for_concat(
        water_indirect_stocks_prov,
        unit="kg",
        stock_type="indirect",
        dimension="water_i",
        stock="water_related",
        year=2015,
        product="water_related",
    )
    water_indirect_stocks_prov.sum() / 1e9

    # WATER DLS threshold (operations as above)
    water_service_demand_thresh_piv = DLS_water_thresh.reset_index().pivot(
        index="region", columns="product", values="value"
    )
    water_service_demand_thresh_piv.insert(0, "year", 2015)
    water_service_demand_thresh_piv = water_service_demand_thresh_piv.set_index(
        "year", append=True
    )
    water_indirect_stocks_thresh_dict = create_nested_dict(
        sanitation_water_indstocks_dict, water_service_demand_thresh_piv
    )
    water_indirect_stocks_thresh = expand_nested_dict_to_df(
        water_indirect_stocks_thresh_dict
    )
    water_indirect_stocks_thresh.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    water_indirect_stocks_thresh = prepare_df_for_concat(
        water_indirect_stocks_thresh,
        unit="kg",
        stock_type="indirect",
        dimension="water_i",
        stock="water_related",
        year=2015,
        product="water_related",
    )
    water_indirect_stocks_thresh.sum() / 1e9

    """##############################################
            EDUCATION & HEALTHCARE: DIRECT STOCKS
       ##############################################"""
    # calculate direct stocks for buildings required to provide healthcare and education
    # here based on the following assumptions:
    # DLS in Kikstra et al. 2025 are in $/cap/a
    # from these DLS gaps and thresholds we calculate the % of 'decent healthcare & education' provided,
    # we multiply this achievement by the floor area assumed in Veléz-Henao & Pauliuk (2023) for providing 'decent healthcare & education'

    # get non-residential building material stock intensities by region
    buildings_mat_intens_NR = housing_mat_intens[
        housing_mat_intens["building_type"] == "NR"
    ]

    # 1) obtain functions provided for HEALTHCARE

    # HEALTHCARE
    DLS_2015_thresh_health = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Health care"
    ].set_index(["region", "variable", "unit"])
    DLS_2015_funct_prov_health = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Health care"
    ].set_index(["region", "variable", "unit"])
    DLS_2015_funct_prov_health_rel = DLS_2015_funct_prov_health / DLS_2015_thresh_health
    DLS_2015_funct_prov_health_rel.reset_index(inplace=True)
    DLS_2015_funct_prov_health_rel.unit = "%"
    DLS_2015_funct_prov_health_rel.set_index(
        ["region", "variable", "unit"], inplace=True
    )
    DLS_funct_prov_health_m2cap = (
        DLS_2015_funct_prov_health_rel
        * m2_cap_health
        * sensitivity_factor_health_education
    )
    DLS_funct_prov_health_m2 = calc_functions_scale(
        DLS_funct_prov_health_m2cap, MISO2_population_2015_subset
    )

    # EDUCATION
    # as the eduction gaps/thresholds are in $/cap/year, we can simply sum primary + secondary education expenditures to obtain one indicator for education
    educ_primary_funct_prov = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Education|primary"
    ]
    educ_secondary_funct_prov = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Education|lower_secondary"
    ]
    educ_funct_prov = (
        pd.concat([educ_primary_funct_prov, educ_secondary_funct_prov])
        .set_index(["region", "variable", "unit"])
        .groupby("region")
        .sum()
    )

    educ_primary_thresh = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Education|primary"
    ]
    educ_secondary_thresh = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Education|lower_secondary"
    ]
    educ_thresh = (
        pd.concat([educ_primary_thresh, educ_secondary_thresh])
        .set_index(["region", "variable", "unit"])
        .groupby("region")
        .sum()
    )

    educ_funct_prov_rel = educ_funct_prov / educ_thresh

    # load share of schooling population for each country and calc m2 required for reachig threshold
    education_share_path = input_paths_dict.get("education_shares_path")
    educ_shares = pd.read_csv(education_share_path).rename(
        columns={"iso": "region"}
    )
    # rename regions according to MISO2 country labels and drop all which are not in MISO2 country list
    for key, value in country_correspondence_dict.items():
        educ_shares["region"].replace({key: value}, inplace=True)
    educ_shares = educ_shares[
        educ_shares.region.isin(country_correspondence.MISO2.to_list())
    ]
    educ_shares.set_index(["region"], inplace=True)
    educ_shares = pd.DataFrame(educ_shares["percentage_school_prim_ls"]).rename(
        columns={"percentage_school_prim_ls": "value"}
    )

    # multiply relative achievement of decent education with schoolgoing population share and floorspace requirements from Veléz-Henao & Pauliuk (2023)
    DLS_funct_prov_educ_m2cap = (
        educ_funct_prov_rel
        * educ_shares
        * m2_pupil_education
        * sensitivity_factor_health_education
    )
    DLS_funct_prov_educ_m2cap.insert(1, "variable", "education")
    DLS_funct_prov_educ_m2cap.insert(2, "unit", "%")
    DLS_funct_prov_educ_m2cap = DLS_funct_prov_educ_m2cap.set_index(
        ["variable", "unit"], append=True
    )
    DLS_funct_prov_educ_m2 = calc_functions_scale(
        DLS_funct_prov_educ_m2cap, MISO2_population_2015_subset
    )

    # concatenate HEALTH + EDUCATION in one dataframe
    DLS_funct_prov_educ_health_m2 = pd.concat(
        [DLS_funct_prov_educ_m2, DLS_funct_prov_health_m2]
    )

    # 2) map material intensities to achievement
    buildings_mapping = housing_mapping.copy().rename(
        columns={"region": "Haberl_et_al", "MISO2_country": "region"}
    )
    DLS_funct_prov_educ_health_m2_mapped = (
        DLS_funct_prov_educ_health_m2.reset_index().merge(
            buildings_mapping.reset_index(), how="left"
        )
    )
    DLS_funct_prov_educ_health_m2_mapped = DLS_funct_prov_educ_health_m2_mapped.merge(
        buildings_mat_intens_NR.rename(columns={"region": "Haberl_et_al"}),
        on=["Haberl_et_al"],
    )

    DLS_funct_prov_educ_health_m2_mapped = (
        DLS_funct_prov_educ_health_m2_mapped[
            [
                "region",
                "variable",
                "value",
                "iron_steel (kg/m²)",
                "wood (kg/m²)",
                "concrete (kg/m²)",
                "cement (kg/m²)",
                "other_materials (kg/m²)",
            ]
        ]
        .rename(columns={"variable": "stock type"})
        .set_index(["region", "stock type"])
    )

    # 3) calculate direct stocks by multiplying achievement with material intensities
    health_educ_direct_stocks_prov = mult_singleCol_otherCols(
        DLS_funct_prov_educ_health_m2_mapped, "value"
    )
    health_educ_direct_stocks_prov.columns = [
        col.replace(" (kg)", "") for col in health_educ_direct_stocks_prov.columns
    ]

    health_educ_direct_stocks_prov = DLS_funct_prov_educ_health_m2_mapped[
        ["iron_steel", "wood", "concrete", "cement", "other_materials"]
    ]

    health_educ_direct_stocks_prov_piv = health_educ_direct_stocks_prov.copy()
    health_educ_direct_stocks_prov = (
        health_educ_direct_stocks_prov.reset_index()
        .melt(id_vars=["region", "stock type"])
        .rename(columns={"variable": "material", "stock type": "dimension"})
        .set_index(["region", "dimension", "material"])
    )
    health_educ_direct_stocks_prov = prepare_df_for_concat(
        health_educ_direct_stocks_prov,
        unit="kg",
        stock_type="direct",
        dimension="education",
        stock="nonres_buildings",
        year=2015,
        product="nonres_buildings",
    )
    health_educ_direct_stocks_prov.sum() / 1e9

    # separate dimenions education and health
    educ_direct_stocks_prov = health_educ_direct_stocks_prov[
        health_educ_direct_stocks_prov.index.get_level_values(2) == "education"
    ]
    health_direct_stocks_prov = health_educ_direct_stocks_prov[
        health_educ_direct_stocks_prov.index.get_level_values(2) == "Health care"
    ]
    health_direct_stocks_prov.rename({"Health care": "health"}, inplace=True)

    # DLS threshold for education & health
    # calculate functions
    DLS_2015_thresh_health_rel = DLS_2015_funct_prov_health_rel.copy()
    DLS_2015_thresh_health_rel.value = 1
    DLS_thresh_health_m2cap = (
        DLS_2015_thresh_health_rel * m2_cap_health * sensitivity_factor_health_education
    )
    DLS_thresh_health_m2 = calc_functions_scale(
        DLS_thresh_health_m2cap, MISO2_population_2015_subset
    )
    educ_thresh_rel = educ_funct_prov_rel.copy()
    educ_thresh_rel.value = 1
    DLS_thresh_educ_m2cap = (
        educ_thresh_rel
        * educ_shares
        * m2_pupil_education
        * sensitivity_factor_health_education
    )
    DLS_thresh_educ_m2cap.insert(1, "variable", "education")
    DLS_thresh_educ_m2cap.insert(2, "unit", "%")
    DLS_thresh_educ_m2cap = DLS_thresh_educ_m2cap.set_index(
        ["variable", "unit"], append=True
    )
    DLS_thresh_educ_m2 = calc_functions_scale(
        DLS_thresh_educ_m2cap, MISO2_population_2015_subset
    )

    # concatenate HEALTH + EDUCATION in one dataframe
    DLS_thresh_educ_health_m2 = pd.concat([DLS_thresh_educ_m2, DLS_thresh_health_m2])

    # map material intensities to achievement
    buildings_mapping = housing_mapping.copy().rename(
        columns={"region": "Haberl_et_al", "MISO2_country": "region"}
    )
    DLS_thresh_educ_health_m2_mapped = DLS_thresh_educ_health_m2.reset_index().merge(
        buildings_mapping.reset_index(), how="left"
    )
    DLS_thresh_educ_health_m2_mapped = DLS_thresh_educ_health_m2_mapped.merge(
        buildings_mat_intens_NR.rename(columns={"region": "Haberl_et_al"}),
        on=["Haberl_et_al"],
    )

    DLS_thresh_educ_health_m2_mapped = (
        DLS_thresh_educ_health_m2_mapped[
            [
                "region",
                "variable",
                "value",
                "iron_steel (kg/m²)",
                "wood (kg/m²)",
                "concrete (kg/m²)",
                "cement (kg/m²)",
                "other_materials (kg/m²)",
            ]
        ]
        .rename(columns={"variable": "stock type"})
        .set_index(["region", "stock type"])
    )

    # calculate direct stocks by multiplying achievement with material intensities
    health_educ_direct_stocks_thresh = mult_singleCol_otherCols(
        DLS_thresh_educ_health_m2_mapped, "value"
    )
    health_educ_direct_stocks_thresh.columns = [
        col.replace(" (kg)", "") for col in health_educ_direct_stocks_thresh.columns
    ]

    health_educ_direct_stocks_thresh = DLS_thresh_educ_health_m2_mapped[
        ["iron_steel", "wood", "concrete", "cement", "other_materials"]
    ]

    health_educ_direct_stocks_thresh_piv = health_educ_direct_stocks_thresh.copy()
    health_educ_direct_stocks_thresh = (
        health_educ_direct_stocks_thresh.reset_index()
        .melt(id_vars=["region", "stock type"])
        .rename(columns={"variable": "material", "stock type": "dimension"})
        .set_index(["region", "dimension", "material"])
    )
    health_educ_direct_stocks_thresh
    health_educ_direct_stocks_thresh = prepare_df_for_concat(
        health_educ_direct_stocks_thresh,
        unit="kg",
        stock_type="direct",
        dimension="education",
        stock="nonres_buildings",
        year=2015,
        product="nonres_buildings",
    )
    health_educ_direct_stocks_thresh.sum() / 1e9

    # separate dimenions education and health
    educ_direct_stocks_thresh = health_educ_direct_stocks_thresh[
        health_educ_direct_stocks_thresh.index.get_level_values(2) == "education"
    ]
    health_direct_stocks_thresh = health_educ_direct_stocks_thresh[
        health_educ_direct_stocks_thresh.index.get_level_values(2) == "Health care"
    ]
    health_direct_stocks_thresh.rename({"Health care": "health"}, inplace=True)

    """##############################################
            EDUCATION & HEALTHCARE: IN-DIRECT STOCKS
       ##############################################"""
    # use shelter_indstocks_dict from dimension 'housing' to derive indirect stock intensities
    # modify material demand dataframe (from direct stocks) and calc. respective indirect stock requirements

    # HEALTH - DLS provided
    # prepare dataframe for indirect material demand of health
    health_material_demand_prov = health_educ_direct_stocks_prov_piv[
        health_educ_direct_stocks_prov_piv.index.get_level_values(1) == "Health care"
    ]
    health_material_demand_prov.index = (
        health_material_demand_prov.index.get_level_values(0)
    )
    # health_material_demand_prov.insert(0,'year',2015)
    # health_material_demand_prov = health_material_demand_prov.set_index('year',append=True)
    health_material_demand_prov.sum(axis=0)

    # regionalize material demand to match to regional ecoinvent processes
    # Define a list of materials to process and path for regional ecoinvent process lists per region
    materials = ["concrete", "wood"]
    path_ecoinvent_housing_reg = input_paths_dict.get("ecoinvent_housing_reg_mapping_path")
    # HEALTH - DLS provided
    health_material_demand_prov_reg = regionalize_material_demand_ecoinvent(
        materials, health_material_demand_prov, path_ecoinvent_housing_reg
    )
    health_material_demand_prov = health_material_demand_prov_reg.copy()
    health_material_demand_prov.insert(0, "year", 2015)
    health_material_demand_prov = health_material_demand_prov.set_index(
        "year", append=True
    )
    health_material_demand_prov.sum(axis=0)

    # mult. housing_material_demand (tons) with ind. stock intens. by material (kg/kg); result = tons
    health_indirect_stocks_prov_dict = create_nested_dict(
        shelter_indstocks_dict, health_material_demand_prov
    )
    health_indirect_stocks_prov = expand_nested_dict_to_df(
        health_indirect_stocks_prov_dict
    )
    health_indirect_stocks_prov.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    health_indirect_stocks_prov = prepare_df_for_concat(
        health_indirect_stocks_prov,
        unit="kg",
        stock_type="indirect",
        dimension="health_i",
        stock="health_related",
        year=2015,
        product="health_related",
    )
    health_indirect_stocks_prov = (
        health_indirect_stocks_prov.reset_index()
        .replace(
            {
                "wood_RER": "wood",
                "wood_RoW": "wood",
                "concrete_AT": "concrete",
                "concrete_BR": "concrete",
                "concrete_ZA": "concrete",
                "concrete_RoW": "concrete",
                "glass_RER": "glass",
                "glass_RoW": "glass",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )
    health_indirect_stocks_prov.sum() / 1e9

    # HEALTH - DLS threshold
    # prepare dataframe for indirect material demand of health
    health_material_demand_thresh = health_educ_direct_stocks_thresh_piv[
        health_educ_direct_stocks_thresh_piv.index.get_level_values(1) == "Health care"
    ]
    health_material_demand_thresh.index = (
        health_material_demand_thresh.index.get_level_values(0)
    )
    health_material_demand_thresh.sum(axis=0)

    # regionalize material demand to match to regional ecoinvent processes
    health_material_demand_thresh_reg = regionalize_material_demand_ecoinvent(
        materials, health_material_demand_thresh, path_ecoinvent_housing_reg
    )
    health_material_demand_thresh = health_material_demand_thresh_reg.copy()
    health_material_demand_thresh.insert(0, "year", 2015)
    health_material_demand_thresh = health_material_demand_thresh.set_index(
        "year", append=True
    )
    health_material_demand_thresh.sum(axis=0)

    # mult. health_material_demand (tons) with ind. stock intens. by material (kg/kg), result = tons
    health_indirect_stocks_thresh_dict = create_nested_dict(
        shelter_indstocks_dict, health_material_demand_thresh
    )
    health_indirect_stocks_thresh = expand_nested_dict_to_df(
        health_indirect_stocks_thresh_dict
    )
    health_indirect_stocks_thresh.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    health_indirect_stocks_thresh = prepare_df_for_concat(
        health_indirect_stocks_thresh,
        unit="kg",
        stock_type="indirect",
        dimension="health_i",
        stock="health_related",
        year=2015,
        product="health_related",
    )
    health_indirect_stocks_thresh = (
        health_indirect_stocks_thresh.reset_index()
        .replace(
            {
                "wood_RER": "wood",
                "wood_RoW": "wood",
                "concrete_AT": "concrete",
                "concrete_BR": "concrete",
                "concrete_ZA": "concrete",
                "concrete_RoW": "concrete",
                "glass_RER": "glass",
                "glass_RoW": "glass",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )
    health_indirect_stocks_thresh.sum() / 1e9

    # EDUCATION - DLS provided
    # prepare dataframe for indirect material demand of health
    educ_material_demand_prov = health_educ_direct_stocks_prov_piv[
        health_educ_direct_stocks_prov_piv.index.get_level_values(1) == "education"
    ]
    educ_material_demand_prov.index = educ_material_demand_prov.index.get_level_values(
        0
    )
    educ_material_demand_prov.sum(axis=0)

    # regionalize material demand to match to regional ecoinvent processes
    educ_material_demand_prov_reg = regionalize_material_demand_ecoinvent(
        materials, educ_material_demand_prov, path_ecoinvent_housing_reg
    )
    educ_material_demand_prov = educ_material_demand_prov_reg.copy()
    educ_material_demand_prov.insert(0, "year", 2015)
    educ_material_demand_prov = educ_material_demand_prov.set_index("year", append=True)

    # mult. housing_material_demand (tons) with ind. stock intens. by material (kg/kg); result = tons
    educ_indirect_stocks_prov_dict = create_nested_dict(
        shelter_indstocks_dict, educ_material_demand_prov
    )
    educ_indirect_stocks_prov = expand_nested_dict_to_df(educ_indirect_stocks_prov_dict)
    educ_indirect_stocks_prov.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    educ_indirect_stocks_prov = prepare_df_for_concat(
        educ_indirect_stocks_prov,
        unit="kg",
        stock_type="indirect",
        dimension="education_i",
        stock="education_related",
        year=2015,
        product="education_related",
    )
    educ_indirect_stocks_prov = (
        educ_indirect_stocks_prov.reset_index()
        .replace(
            {
                "wood_RER": "wood",
                "wood_RoW": "wood",
                "concrete_AT": "concrete",
                "concrete_BR": "concrete",
                "concrete_ZA": "concrete",
                "concrete_RoW": "concrete",
                "glass_RER": "glass",
                "glass_RoW": "glass",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )
    educ_indirect_stocks_prov.sum() / 1e9

    # EDUCATION - DLS threshold
    # prepare dataframe for indirect material demand of educ
    educ_material_demand_thresh = health_educ_direct_stocks_thresh_piv[
        health_educ_direct_stocks_thresh_piv.index.get_level_values(1) == "education"
    ]
    educ_material_demand_thresh.index = (
        educ_material_demand_thresh.index.get_level_values(0)
    )
    educ_material_demand_thresh.sum(axis=0)

    # regionalize material demand to match to regional ecoinvent processes
    educ_material_demand_thresh_reg = regionalize_material_demand_ecoinvent(
        materials, educ_material_demand_thresh, path_ecoinvent_housing_reg
    )
    educ_material_demand_thresh = educ_material_demand_thresh_reg.copy()
    educ_material_demand_thresh.insert(0, "year", 2015)
    educ_material_demand_thresh = educ_material_demand_thresh.set_index(
        "year", append=True
    )

    # mult. educ_material_demand (tons) with ind. stock intens. by material (kg/kg), result = tons
    educ_indirect_stocks_thresh_dict = create_nested_dict(
        shelter_indstocks_dict, educ_material_demand_thresh
    )
    educ_indirect_stocks_thresh = expand_nested_dict_to_df(
        educ_indirect_stocks_thresh_dict
    )
    educ_indirect_stocks_thresh.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    educ_indirect_stocks_thresh = prepare_df_for_concat(
        educ_indirect_stocks_thresh,
        unit="kg",
        stock_type="indirect",
        dimension="education_i",
        stock="education_related",
        year=2015,
        product="education_related",
    )
    educ_indirect_stocks_thresh = (
        educ_indirect_stocks_thresh.reset_index()
        .replace(
            {
                "wood_RER": "wood",
                "wood_RoW": "wood",
                "concrete_AT": "concrete",
                "concrete_BR": "concrete",
                "concrete_ZA": "concrete",
                "concrete_RoW": "concrete",
                "glass_RER": "glass",
                "glass_RoW": "glass",
            }
        )
        .groupby(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .sum()
    )

    """##############################################
           HEATING & COOLING: IN-DIRECT STOCKS
      ##############################################"""
    # we assume heatpumps for delivering heat/cooling; we do not consider equipment for hot water and radiators

    # HEATING & COOLING - DLS provided
    DLS_2015_funct_prov_cool = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Cooling CON|total"
    ]
    DLS_2015_funct_prov_heat = DLS_2015_funct_prov.reset_index()[
        DLS_2015_funct_prov.reset_index().variable == "Heating CON|total"
    ]

    # use the maximum value of heating/cooling requirement to determine m2 that need to be conditioned
    DLS_2015_funct_prov_temp = (
        DLS_2015_funct_prov_cool.groupby("region")
        .sum()
        .rename(columns={"value": "heat"})
        .merge(
            DLS_2015_funct_prov_heat.groupby("region")
            .sum()
            .rename(columns={"value": "cool"}),
            on="region",
        )
    )
    DLS_2015_funct_prov_temp["max"] = DLS_2015_funct_prov_temp[["cool", "heat"]].max(
        axis=1
    )
    # calculate total m2 housing
    DLS_2015_funct_prov_hous_scale = calc_functions_scale(
        DLS_2015_funct_prov_hous, MISO2_population_2015_subset
    )
    DLS_2015_funct_prov_condm2 = DLS_2015_funct_prov_temp[["max"]].merge(
        DLS_2015_funct_prov_hous_scale, on="region"
    )
    # calculate m2 that need to be conditioned
    DLS_2015_funct_prov_condm2["value"] = (
        DLS_2015_funct_prov_condm2["max"] * DLS_2015_funct_prov_condm2["functions"]
    )

    ## determine appliance capacity required to condition floorspace
    # assumption: 0.06kW/m2 - based on: https://www.thermondo.de/info/rat/waermepumpe/leistung-und-groesse-einer-waermepumpe-berechnen/
    cond_intes_m2 = 0.06
    DLS_2015_funct_prov_condm2["pump"] = (
        DLS_2015_funct_prov_condm2["value"] * cond_intes_m2
    )
    # format dataframe to calc. respective indirect stock requirements
    cond_material_demand_prov = pd.DataFrame(DLS_2015_funct_prov_condm2["pump"]).rename(
        columns={"pump": "m_heat_pump_30kW_GLO"}
    )
    cond_material_demand_prov.insert(0, "year", 2015)
    cond_material_demand_prov = cond_material_demand_prov.set_index("year", append=True)
    cond_material_demand_prov.sum(axis=0)

    # multiply cond_material_demand (kW required) with indirect stock intens. by material (kg/kg); result = tons
    # divide  cond_material_demand_prov by factor 30, as MI in shelter_indstocks_dict are for a 30kW heatpump
    cond_indirect_stocks_prov_dict = create_nested_dict(
        shelter_indstocks_dict, cond_material_demand_prov / 30
    )
    cond_indirect_stocks_prov = expand_nested_dict_to_df(cond_indirect_stocks_prov_dict)
    cond_indirect_stocks_prov.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    cond_indirect_stocks_prov = prepare_df_for_concat(
        cond_indirect_stocks_prov,
        unit="kg",
        stock_type="indirect",
        dimension="cond_i",
        stock="cond_related",
        year=2015,
        product="cond_related",
    )

    # HEATING & COOLING - DLS threshold
    DLS_2015_thresh_cool = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Cooling CON|total"
    ]
    DLS_2015_thresh_heat = DLS_2015_thresh.reset_index()[
        DLS_2015_thresh.reset_index().variable == "Heating CON|total"
    ]

    # use the maximum value of heating/cooling requirement to determine m2 that need to be conditioned
    DLS_2015_thresh_temp = (
        DLS_2015_thresh_cool.groupby("region")
        .sum()
        .rename(columns={"value": "heat"})
        .merge(
            DLS_2015_thresh_heat.groupby("region")
            .sum()
            .rename(columns={"value": "cool"}),
            on="region",
        )
    )
    DLS_2015_thresh_temp["max"] = DLS_2015_thresh_temp[["cool", "heat"]].max(axis=1)
    DLS_2015_thresh_hous_scale = calc_functions_scale(
        DLS_2015_thresh_hous, MISO2_population_2015_subset
    )
    DLS_2015_thresh_condm2 = (
        DLS_2015_thresh_temp[["max"]]
        .merge(DLS_2015_thresh_hous_scale, on="region")
        .rename(columns={"value": "functions"})
    )
    DLS_2015_thresh_condm2["value"] = (
        DLS_2015_thresh_condm2["max"] * DLS_2015_thresh_condm2["functions"]
    )
    # m2 that need to be conditioned

    ## determine appliances required to condition m2
    DLS_2015_thresh_condm2["pump"] = DLS_2015_thresh_condm2["value"] * cond_intes_m2
    # modify pump demand dataframe and calc. respective indirect stock requirements
    cond_material_demand_thresh = pd.DataFrame(DLS_2015_thresh_condm2["pump"]).rename(
        columns={"pump": "m_heat_pump_30kW_GLO"}
    )
    cond_material_demand_thresh.insert(0, "year", 2015)
    cond_material_demand_thresh = cond_material_demand_thresh.set_index(
        "year", append=True
    )
    cond_material_demand_thresh.sum(axis=0)

    # mult. cond_material_demand (kw) with ind. stock intens. by material (kg/kg); result = tons
    ## need to divide  cond_material_demand_prov by factor 30, as MI in shelter_indstocks_dict fpr 30kW heatpump
    cond_indirect_stocks_thresh_dict = create_nested_dict(
        shelter_indstocks_dict, cond_material_demand_thresh / 30
    )
    cond_indirect_stocks_thresh = expand_nested_dict_to_df(
        cond_indirect_stocks_thresh_dict
    )
    cond_indirect_stocks_thresh.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    cond_indirect_stocks_thresh = prepare_df_for_concat(
        cond_indirect_stocks_thresh,
        unit="kg",
        stock_type="indirect",
        dimension="cond_i",
        stock="cond_related",
        year=2015,
        product="cond_related",
    )

    """##############################################
            LAMPS: IN-DIRECT STOCKS (including direct stocks)
       ##############################################"""

    # calculate amount of lamps required per building floorspace from Veléz-Henao & Pauliuk (2023
    # housing: 1 lamp / 15 m2
    lamps_housing_prov = DLS_2015_funct_prov_hous_scale / 15
    lamps_housing_thresh = DLS_2015_thresh_hous_scale / 15
    # education:  4 lamps / 2.48 m2
    lamps_educ_prov = DLS_funct_prov_educ_m2 / (
        4 / (m2_pupil_education * educ_shares * sensitivity_factor_health_education)
    )
    lamps_educ_thresh = DLS_thresh_educ_m2 / (
        4 / (m2_pupil_education * educ_shares * sensitivity_factor_health_education)
    )
    # health:  4 lamps / 1.6 m2
    lamps_health_prov = DLS_funct_prov_health_m2 / (
        4 / (m2_cap_health * sensitivity_factor_health_education)
    )
    lamps_health_thresh = DLS_thresh_health_m2 / (
        4 / (m2_cap_health * sensitivity_factor_health_education)
    )

    # LAMPS for housing - DLS provided
    # prepare dataframe for calc indirect material demand of lamps
    lamps_housing_prov = (
        lamps_housing_prov.reset_index()
        .rename(columns={"functions": "m_lamp_GLO"})[["region", "m_lamp_GLO"]]
        .set_index("region")
    )
    lamps_housing_prov.insert(0, "year", 2015)
    lamps_housing_prov = lamps_housing_prov.set_index("year", append=True)
    lamps_housing_prov.sum(axis=0)
    # multiply with (in)direct stock intensities
    lamps_housing_indirect_stocks_prov_dict_scale = create_nested_dict(
        shelter_indstocks_dict, lamps_housing_prov
    )
    lamps_housing_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        lamps_housing_indirect_stocks_prov_dict_scale
    )
    lamps_housing_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    lamps_housing_indirect_stocks_prov = prepare_df_for_concat(
        lamps_housing_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="housing_i",
        stock="lamps_related",
    )
    lamps_housing_indirect_stocks_prov.sum() / 1e9

    # LAMPS for housing - DLS threshold
    # prepare dataframe for calc indirect material demand of lamps
    lamps_housing_thresh = (
        lamps_housing_thresh.reset_index()
        .rename(columns={"functions": "m_lamp_GLO"})[["region", "m_lamp_GLO"]]
        .set_index("region")
    )
    lamps_housing_thresh.insert(0, "year", 2015)
    lamps_housing_thresh = lamps_housing_thresh.set_index("year", append=True)
    lamps_housing_thresh.sum(axis=0)
    # multiply with (in)direct stock intensities
    lamps_housing_indirect_stocks_thresh_dict_scale = create_nested_dict(
        shelter_indstocks_dict, lamps_housing_thresh
    )
    lamps_housing_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        lamps_housing_indirect_stocks_thresh_dict_scale
    )
    lamps_housing_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    lamps_housing_indirect_stocks_thresh = prepare_df_for_concat(
        lamps_housing_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="housing_i",
        stock="lamps_related",
    )
    lamps_housing_indirect_stocks_thresh.sum() / 1e9

    # LAMPS for education - DLS provided
    # prepare dataframe for calc indirect material demand of lamps
    lamps_educ_prov = (
        lamps_educ_prov.reset_index()
        .rename(columns={"value": "m_lamp_GLO"})[["region", "m_lamp_GLO"]]
        .set_index("region")
    )
    lamps_educ_prov.insert(0, "year", 2015)
    lamps_educ_prov = lamps_educ_prov.set_index("year", append=True)
    lamps_educ_prov.sum(axis=0)
    # multiply with (in)direct stock intensities
    lamps_educ_indirect_stocks_prov_dict_scale = create_nested_dict(
        shelter_indstocks_dict, lamps_educ_prov
    )
    lamps_educ_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        lamps_educ_indirect_stocks_prov_dict_scale
    )
    lamps_educ_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    lamps_educ_indirect_stocks_prov = prepare_df_for_concat(
        lamps_educ_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="education_i",
        stock="lamps_related",
    )
    lamps_educ_indirect_stocks_prov.sum() / 1e9

    # LAMPS for education - DLS threshold
    # prepare dataframe for calc indirect material demand of lamps
    lamps_educ_thresh = (
        lamps_educ_thresh.reset_index()
        .rename(columns={"value": "m_lamp_GLO"})[["region", "m_lamp_GLO"]]
        .set_index("region")
    )
    lamps_educ_thresh.insert(0, "year", 2015)
    lamps_educ_thresh = lamps_educ_thresh.set_index("year", append=True)
    lamps_educ_thresh.sum(axis=0)
    # multiply with (in)direct stock intensities
    lamps_educ_indirect_stocks_thresh_dict_scale = create_nested_dict(
        shelter_indstocks_dict, lamps_educ_thresh
    )
    lamps_educ_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        lamps_educ_indirect_stocks_thresh_dict_scale
    )
    lamps_educ_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    lamps_educ_indirect_stocks_thresh = prepare_df_for_concat(
        lamps_educ_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="education_i",
        stock="lamps_related",
    )
    lamps_educ_indirect_stocks_thresh.sum() / 1e9

    # LAMPS for healthcare - DLS provided
    # prepare dataframe for calc indirect material demand of lamps
    lamps_health_prov = (
        lamps_health_prov.reset_index()
        .rename(columns={"value": "m_lamp_GLO"})[["region", "m_lamp_GLO"]]
        .set_index("region")
    )
    lamps_health_prov.insert(0, "year", 2015)
    lamps_health_prov = lamps_health_prov.set_index("year", append=True)
    lamps_health_prov.sum(axis=0)
    # multiply with (in)direct stock intensities
    lamps_health_indirect_stocks_prov_dict_scale = create_nested_dict(
        shelter_indstocks_dict, lamps_health_prov
    )
    lamps_health_indirect_stocks_prov_df_scale = expand_nested_dict_to_df(
        lamps_health_indirect_stocks_prov_dict_scale
    )
    lamps_health_indirect_stocks_prov_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    lamps_health_indirect_stocks_prov = prepare_df_for_concat(
        lamps_health_indirect_stocks_prov_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="health_i",
        stock="lamps_related",
    )
    lamps_health_indirect_stocks_prov.sum() / 1e9

    # LAMPS for healthcare - DLS threshold
    # prepare dataframe for calc indirect material demand of lamps
    lamps_health_thresh = (
        lamps_health_thresh.reset_index()
        .rename(columns={"value": "m_lamp_GLO"})[["region", "m_lamp_GLO"]]
        .set_index("region")
    )
    lamps_health_thresh.insert(0, "year", 2015)
    lamps_health_thresh = lamps_health_thresh.set_index("year", append=True)
    lamps_health_thresh.sum(axis=0)
    # multiply with (in)direct stock intensities
    lamps_health_indirect_stocks_thresh_dict_scale = create_nested_dict(
        shelter_indstocks_dict, lamps_health_thresh
    )
    lamps_health_indirect_stocks_thresh_df_scale = expand_nested_dict_to_df(
        lamps_health_indirect_stocks_thresh_dict_scale
    )
    lamps_health_indirect_stocks_thresh_df_scale.index.names = [
        "region,year",
        "product",
        "stock",
        "material",
    ]
    lamps_health_indirect_stocks_thresh = prepare_df_for_concat(
        lamps_health_indirect_stocks_thresh_df_scale,
        unit="kg",
        stock_type="indirect",
        dimension="health_i",
        stock="lamps_related",
    )
    lamps_health_indirect_stocks_thresh.sum() / 1e9

    """##############################################
                ENERGY SYSTEM: IN-DIRECT STOCKS
       ##############################################"""

    # read decent living energy (DLE) data for DLS provided and DLS threshold from Kikstra et al. (2025)
    energy_prov_cap = pd.read_csv(DLE_provided).rename(
        columns={"iso": "region"}
    )
    energy_thresh_cap = pd.read_csv(DLE_threshold).rename(
        columns={"iso": "region"}
    )

    # rename regions according to MISO2 country labels and drop all which are not in country correspondence list
    for i in [energy_prov_cap, energy_thresh_cap]:
        for key, value in country_correspondence_dict.items():
            i["region"].replace({key: value}, inplace=True)
        i = i[i.region.isin(country_correspondence.MISO2.to_list())]
        i.set_index(["region"], inplace=True)
    energy_prov_cap = energy_prov_cap[
        energy_prov_cap.region.isin(country_correspondence.MISO2)
    ]
    energy_thresh_cap = energy_thresh_cap[
        energy_thresh_cap.region.isin(country_correspondence.MISO2)
    ]
    energy_thresh_cap.drop("thres.energy", axis=1, inplace=True)

    # convert MJ to kWh (0.2777778 kWh/MJ)
    energy_prov_cap.replace({"MJ/cap/year": "kWh/cap/year"}, inplace=True)
    energy_prov_cap = (
        energy_prov_cap.set_index(["region", "variable", "elec", "unit.energy"])
        * 0.2777778
    )
    energy_thresh_cap.replace({"MJ/cap/year": "kWh/cap/year"}, inplace=True)
    energy_thresh_cap = (
        energy_thresh_cap.set_index(["region", "variable", "elec", "unit.energy"])
        * 0.2777778
    )

    # substitute DLS indicator dimension details with higher level categories
    subst_dim = {
        "Appliance|clean_cooking_fuel": "nutrition_i",
        "Appliance|television": "communication_i",
        "Appliance|mobile_telephone": "communication_i",
        "Appliance|refrigerator": "nutrition_i",
        "Water": "water_i",
        "Sanitation": "sanitation_i",
        "Nutrition": "nutrition_i",
        "Clothing|clothing": "clothing_i",
        "Clothing|footwear": "clothing_i",
        "Health care": "health_i",
        "Education|primary": "education_i",
        "Education|lower_secondary": "education_i",
        "Housing|total": "housing_i",
        "Roads": "transport_i",
        "Hot Water OP|total": "housing_i",
        "Transport": "transport_i",
        "Heating": "housing_i",
        "Cooling": "housing_i",
    }
    energy_prov_cap.reset_index(inplace=True)
    energy_prov_cap = energy_prov_cap.replace(subst_dim)
    energy_prov_cap = (
        energy_prov_cap.set_index(["region", "variable", "elec", "unit.energy"])
        .groupby(["region", "variable", "elec", "unit.energy"])
        .sum()
    )
    energy_thresh_cap.reset_index(inplace=True)
    energy_thresh_cap = energy_thresh_cap.replace(subst_dim)
    energy_thresh_cap = (
        energy_thresh_cap.set_index(["region", "variable", "elec", "unit.energy"])
        .groupby(["region", "variable", "elec", "unit.energy"])
        .sum()
    )

    # energy use per capita in categories electricity and non-electricity (=fuel)
    electricity_prov_cap = energy_prov_cap[
        energy_prov_cap.index.get_level_values(2) == "elec"
    ]
    electricity_thresh_cap = energy_thresh_cap[
        energy_thresh_cap.index.get_level_values(2) == "elec"
    ]
    fuel_prov_cap = energy_prov_cap[
        energy_prov_cap.index.get_level_values(2) == "non.elec"
    ]
    fuel_thresh_cap = energy_thresh_cap[
        energy_thresh_cap.index.get_level_values(2) == "non.elec"
    ]

    # ELECTRICITY
    # for electricity by dimension: multiply by indirect stock intensity per country (dep. on electricity mix)

    # Read all MI sheets into a dictionary of dataframes
    # manually renamed the lavel Cote d'Ivoire to Cote dIvoire in the input data
    electricity_indstocks_path = input_paths_dict.get("electricity_indstocks_path")
    electricity_indstocks_dict = read_excel_into_dict(electricity_indstocks_path)
    del electricity_indstocks_dict["summary"]
    for key in electricity_indstocks_dict:
        electricity_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)
        electricity_indstocks_dict.get(key).columns = electricity_indstocks_dict.get(
            key
        ).columns.astype(int)

    # countries not in dictionary
    countries_not_in = energy_thresh_cap.index.get_level_values("region").unique()[
        ~energy_thresh_cap.index.get_level_values("region")
        .unique()
        .isin(electricity_indstocks_dict.keys())
    ]
    # for countries not in dictionary, assign surrogate country for which ecoinvent process of electricity production used
    assign_dict = {
        "Afghanistan": "Iran",
        "Burundi": "Kenya",
        "Burkina Faso": "Ghana",
        "Bahamas": "Cuba",
        "Belize": "Guatemala",
        "Brunei": "Indonesia",
        "Bhutan": "Bangladesh",
        "Central African Republic": "Cameroon",
        "Comoros": "Mozambique",
        "Cape Verde": "Senegal",
        "Djibouti": "Ethiopia",
        "Fiji": "Indonesia",
        "Guinea": "Cote dIvoire",
        "Guadeloupe": "France",
        "The Gambia": "Senegal",
        "Guinea Bissau": "Cote dIvoire",
        "Equatorial Guinea": "Cameroon",
        "Guyana": "Brazil, Northern grid",
        "Hong Kong SAR": "China",
        "Laos": "Vietnam",
        "Liberia": "Cote dIvoire",
        "Lesotho": "South Africa",
        "Moldova": "Ukraine",
        "Madagascar": "Kenya",
        "Maldives": "Sri Lanka",
        "Mali": "Algeria",
        "Mauritania": "Morocco",
        "Martinique": "France",
        "Malawi": "Zambia",
        "Papua New Guinea": "Indonesia",
        "Reunion": "France",
        "Rwanda": "Kenya",
        "Solomon Islands": "Indonesia",
        "Sierra Leone": "Cote dIvoire",
        "Somalia": "Ethiopia",
        "Suriname": "Brazil, Northern grid",
        "Swaziland": "South Africa",
        "Syria": "Iraq",
        "Chad": "Sudan",
        "Timor-Leste": "Indonesia",
        "Tanzania": "Kenya",
        "Uganda": "Kenya",
    }
    # for countries with different spelling, match both data source spellings + for countries with regional grid defferentiation, choose one mix
    rename_dict = {
        "Bolivia": "Bolivia, Plurinational State of",
        "Brazil": "Brazil, Mid-western grid",
        "Canada": "Canada, British Columbia",
        "China": "State Grid Corporation of China",
        "Congo, DR": "Congo, Democratic Republic of t",
        "Czech Republic": "Czechia",
        "India": "India, North-eastern grid",
        "Iran": "Iran (Islamic Republic of)",
        "South Korea": "Korea, Republic of",
        "North Korea": "Korea, Democratic People's Repu",
        "Serbia (incl. Kosovo)": "Serbia",
        "United States of America": "United States of America, Alask",
        "Vietnam": "Viet Nam",
    }

    for key, value in rename_dict.items():
        electricity_indstocks_dict[key] = electricity_indstocks_dict.pop(value)
    for key, value in assign_dict.items():
        electricity_indstocks_dict[key] = electricity_indstocks_dict[value]
    # assemble dictionary for indirect stocks of electricity system
    electricity_indstocks_dict = {
        key: value
        for key, value in electricity_indstocks_dict.items()
        if key in country_correspondence.MISO2.to_list()
    }

    # DLS ELECTRICITY PROVIDED
    # calculate indirect stocks
    electricity_indstocks_prov_cap_dict = create_nested_dict_energy(
        electricity_indstocks_dict, electricity_prov_cap
    )
    electricity_indstocks_prov_cap_df = expand_nested_dict_to_df(
        electricity_indstocks_prov_cap_dict
    )
    electricity_indstocks_prov_cap_df.index.names = ["tuple", "stock", "material"]
    electricity_indstocks_prov_cap_df.rename(columns={0: "value"}, inplace=True)
    electricity_indstocks_prov_cap_df = electricity_indstocks_prov_cap_df[
        electricity_indstocks_prov_cap_df.value != 0
    ]
    # expand tupled index
    tupled_index = electricity_indstocks_prov_cap_df.index.get_level_values(0)
    split_tuples = [list(t) for t in tupled_index]
    electricity_indstocks_prov_cap_df[
        ["region", "dimension", "energy_carrier", "unit", "energy_scope"]
    ] = split_tuples
    electricity_indstocks_prov_cap_df["unit"] = "kg/cap"
    electricity_indstocks_prov_cap_df.insert(
        1,
        "product",
        electricity_indstocks_prov_cap_df["energy_carrier"]
        + "_"
        + electricity_indstocks_prov_cap_df["energy_scope"],
    )
    electricity_indstocks_prov_cap_df.insert(1, "year", "2015")
    electricity_indstocks_prov_cap_df.insert(1, "stock_type", "indirect")
    # substitute dimension details with higher level labels
    subst_dim = {
        "Appliance|clean_cooking_fuel": "nutrition_i",
        "Appliance|television": "communication_i",
        "Appliance|mobile_telephone": "communication_i",
        "Appliance|refrigerator": "nutrition_i",
        "Water": "water_i",
        "Sanitation": "sanitation_i",
        "Nutrition": "nutrition_i",
        "Clothing|clothing": "clothing_i",
        "Clothing|footwear": "clothing_i",
        "Health care": "health_i",
        "Education|primary": "education_i",
        "Education|lower_secondary": "education_i",
        "Housing|total": "housing_i",
        "Roads": "transport_i",
        "Hot Water OP|total": "housing_i",
        "Transport": "transport_i",
        "Heating": "housing_i",
        "Cooling": "housing_i",
    }
    electricity_indstocks_prov_cap_df = electricity_indstocks_prov_cap_df.replace(
        subst_dim
    )
    electricity_indstocks_prov_cap_df = (
        electricity_indstocks_prov_cap_df.reset_index()
        .set_index(
            [
                "region",
                "year",
                "stock_type",
                "dimension",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .drop(columns=["tuple", "energy_carrier", "energy_scope"])
    )
    electricity_indstocks_prov_cap_df = electricity_indstocks_prov_cap_df.groupby(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ]
    ).sum()

    # scale up from per capita to national level values
    electricity_indstocks_prov_df_scale = electricity_indstocks_prov_cap_df.copy().join(
        MISO2_population_2015_subset.rename(columns={"2015": "pop"}), on="region"
    )
    electricity_indstocks_prov_df_scale["value_scale"] = (
        electricity_indstocks_prov_df_scale["value"]
        * electricity_indstocks_prov_df_scale["pop"]
    )
    electricity_indstocks_prov_df_scale = electricity_indstocks_prov_df_scale.rename(
        {"kg/cap": "tons"}
    )
    electricity_indstocks_prov_df_scale = (
        electricity_indstocks_prov_df_scale.reset_index()
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .drop(columns=["value", "pop"])
    )
    electricity_indstocks_prov_df_scale.rename(
        columns={"value_scale": "value"}, inplace=True
    )
    electricity_indstocks_prov_df_scale.sum() / 1e9

    # DLS ELECTRICITY THRESHOLD
    # calculate indirect stocks
    electricity_indstocks_thresh_cap_dict = create_nested_dict_energy(
        electricity_indstocks_dict, electricity_thresh_cap
    )
    electricity_indstocks_thresh_cap_df = expand_nested_dict_to_df(
        electricity_indstocks_thresh_cap_dict
    )
    electricity_indstocks_thresh_cap_df.index.names = ["tuple", "stock", "material"]
    electricity_indstocks_thresh_cap_df.rename(columns={0: "value"}, inplace=True)
    electricity_indstocks_thresh_cap_df = electricity_indstocks_thresh_cap_df[
        electricity_indstocks_thresh_cap_df.value != 0
    ]
    # expand tupled index
    tupled_index = electricity_indstocks_thresh_cap_df.index.get_level_values(0)
    split_tuples = [list(t) for t in tupled_index]
    electricity_indstocks_thresh_cap_df[
        ["region", "dimension", "energy_carrier", "unit", "energy_scope"]
    ] = split_tuples
    electricity_indstocks_thresh_cap_df["unit"] = "kg/cap"
    electricity_indstocks_thresh_cap_df.insert(
        1,
        "product",
        electricity_indstocks_thresh_cap_df["energy_carrier"]
        + "_"
        + electricity_indstocks_thresh_cap_df["energy_scope"],
    )
    electricity_indstocks_thresh_cap_df.insert(1, "year", "2015")
    electricity_indstocks_thresh_cap_df.insert(1, "stock_type", "indirect")
    # substitute dimension details with higher level labels
    subst_dim = {
        "Appliance|clean_cooking_fuel": "nutrition_i",
        "Appliance|television": "communication_i",
        "Appliance|mobile_telephone": "communication_i",
        "Appliance|refrigerator": "nutrition_i",
        "Water": "water_i",
        "Sanitation": "sanitation_i",
        "Nutrition": "nutrition_i",
        "Clothing|clothing": "clothing_i",
        "Clothing|footwear": "clothing_i",
        "Health care": "health_i",
        "Education|primary": "education_i",
        "Education|lower_secondary": "education_i",
        "Housing|total": "housing_i",
        "Roads": "transport_i",
        "Hot Water OP|total": "housing_i",
        "Transport": "transport_i",
        "Heating": "housing_i",
        "Cooling": "housing_i",
    }
    electricity_indstocks_thresh_cap_df = electricity_indstocks_thresh_cap_df.replace(
        subst_dim
    )
    electricity_indstocks_thresh_cap_df = (
        electricity_indstocks_thresh_cap_df.reset_index()
        .set_index(
            [
                "region",
                "year",
                "stock_type",
                "dimension",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .drop(columns=["tuple", "energy_carrier", "energy_scope"])
    )
    electricity_indstocks_thresh_cap_df = electricity_indstocks_thresh_cap_df.groupby(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ]
    ).sum()

    # scale up from capita to national level values
    electricity_indstocks_thresh_df_scale = (
        electricity_indstocks_thresh_cap_df.copy().join(
            MISO2_population_2015_subset.rename(columns={"2015": "pop"}), on="region"
        )
    )
    electricity_indstocks_thresh_df_scale["value_scale"] = (
        electricity_indstocks_thresh_df_scale["value"]
        * electricity_indstocks_thresh_df_scale["pop"]
    )
    electricity_indstocks_thresh_df_scale = (
        electricity_indstocks_thresh_df_scale.rename({"kg/cap": "tons"})
    )
    electricity_indstocks_thresh_df_scale = (
        electricity_indstocks_thresh_df_scale.reset_index()
        .set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ]
        )
        .drop(columns=["value", "pop"])
    )
    electricity_indstocks_thresh_df_scale.rename(
        columns={"value_scale": "value"}, inplace=True
    )
    electricity_indstocks_thresh_df_scale.sum() / 1e9

    # FUELS = non-electricity
    # split non-electricity energy use by fuel type
    fuel_shares_path = input_paths_dict.get("fuel_shares_path")
    fuel_shares = pd.read_csv(fuel_shares_path)
    fuel_share_reg_corresp_path = input_paths_dict.get("fuel_shares_reg_mapping_path")
    fuel_shares_reg_corresp = pd.read_excel(fuel_share_reg_corresp_path)
    fuel_shares_reg_corresp_dict = dict(
        zip(fuel_shares_reg_corresp.iloc[:, 0], fuel_shares_reg_corresp.iloc[:, 1])
    )
    fuel_shares_2015 = fuel_shares[fuel_shares.Year == 2015]
    fuel_shares_2015_prep = (
        fuel_shares_2015.copy()
        .rename(
            columns={
                "Entity": "fuel_regions",
                "Oil (% equivalent primary energy)": "oil",
                "Coal (% equivalent primary energy)": "coal",
                "Gas (% equivalent primary energy)": "gas",
            }
        )[["fuel_regions", "oil", "coal", "gas"]]
        .set_index("fuel_regions")
    )
    fuel_shares_2015_prep = fuel_shares_2015_prep.div(
        fuel_shares_2015_prep.sum(axis=1), axis=0
    )
    # 92 countries/regions

    # map fuel shares data for countries/regions to DLS indicator countries
    fuel_prov_cap["fuel_regions"] = fuel_prov_cap.index.get_level_values(0).map(
        fuel_shares_reg_corresp_dict
    )
    fuel_prov_cap = fuel_prov_cap.join(fuel_shares_2015_prep, on=["fuel_regions"])
    fuel_thresh_cap["fuel_regions"] = fuel_thresh_cap.index.get_level_values(0).map(
        fuel_shares_reg_corresp_dict
    )
    fuel_thresh_cap = fuel_thresh_cap.join(fuel_shares_2015_prep, on=["fuel_regions"])

    # split into operational and construction fuel use to ease calculation process
    fuel_prov_cap_op = fuel_prov_cap[["current.dle.op", "oil", "coal", "gas"]]
    fuel_prov_cap_con = fuel_prov_cap[["current.dle.conrep", "oil", "coal", "gas"]]
    fuel_thresh_cap_op = fuel_thresh_cap[["thres.energy.op", "oil", "coal", "gas"]]
    fuel_thresh_cap_con = fuel_thresh_cap[["thres.energy.conrep", "oil", "coal", "gas"]]

    # assign fuel use per fuel type (column names according to ecoinvent process for indirect stock intensity)
    # name oil and gas according to ecoinvent process label in fuels_indstocks_dict (below)
    fuel_prov_cap_op["m_petroleum_GLO"] = (
        fuel_prov_cap_op["current.dle.op"] * fuel_prov_cap_op["oil"]
    )
    fuel_prov_cap_op["coal"] = (
        fuel_prov_cap_op["current.dle.op"] * fuel_prov_cap_op["coal"]
    )
    fuel_prov_cap_op["m_gas_GLO"] = (
        fuel_prov_cap_op["current.dle.op"] * fuel_prov_cap_op["gas"]
    )
    fuel_prov_cap_op = fuel_prov_cap_op[["m_petroleum_GLO", "coal", "m_gas_GLO"]]

    fuel_prov_cap_con["m_petroleum_GLO"] = (
        fuel_prov_cap_con["current.dle.conrep"] * fuel_prov_cap_con["oil"]
    )
    fuel_prov_cap_con["coal"] = (
        fuel_prov_cap_con["current.dle.conrep"] * fuel_prov_cap_con["coal"]
    )
    fuel_prov_cap_con["m_gas_GLO"] = (
        fuel_prov_cap_con["current.dle.conrep"] * fuel_prov_cap_con["gas"]
    )
    fuel_prov_cap_con = fuel_prov_cap_con[["m_petroleum_GLO", "coal", "m_gas_GLO"]]

    fuel_thresh_cap_op["m_petroleum_GLO"] = (
        fuel_thresh_cap_op["thres.energy.op"] * fuel_thresh_cap_op["oil"]
    )
    fuel_thresh_cap_op["coal"] = (
        fuel_thresh_cap_op["thres.energy.op"] * fuel_thresh_cap_op["coal"]
    )
    fuel_thresh_cap_op["m_gas_GLO"] = (
        fuel_thresh_cap_op["thres.energy.op"] * fuel_thresh_cap_op["gas"]
    )
    fuel_thresh_cap_op = fuel_thresh_cap_op[["m_petroleum_GLO", "coal", "m_gas_GLO"]]

    fuel_thresh_cap_con["m_petroleum_GLO"] = (
        fuel_thresh_cap_con["thres.energy.conrep"] * fuel_thresh_cap_con["oil"]
    )
    fuel_thresh_cap_con["coal"] = (
        fuel_thresh_cap_con["thres.energy.conrep"] * fuel_thresh_cap_con["coal"]
    )
    fuel_thresh_cap_con["m_gas_GLO"] = (
        fuel_thresh_cap_con["thres.energy.conrep"] * fuel_thresh_cap_con["gas"]
    )
    fuel_thresh_cap_con = fuel_thresh_cap_con[["m_petroleum_GLO", "coal", "m_gas_GLO"]]

    # regionalize material for coal (oil, gas do not have regional representation in ecoinvent) demand to match to regional ecoinvent processes
    # Define a list of materials to process and path for regional ecoinvent process lists per region
    coal_uses = ["coal"]
    path_ecoinvent_coal_reg = input_paths_dict.get("ecoinvent_coal_reg_mapping_path")
    fuel_prov_cap_op_reg = regionalize_material_demand_ecoinvent_coal(
        coal_uses, fuel_prov_cap_op, path_ecoinvent_coal_reg
    )
    fuel_prov_cap_con_reg = regionalize_material_demand_ecoinvent_coal(
        coal_uses, fuel_prov_cap_con, path_ecoinvent_coal_reg
    )
    fuel_thresh_cap_op_reg = regionalize_material_demand_ecoinvent_coal(
        coal_uses, fuel_thresh_cap_op, path_ecoinvent_coal_reg
    )
    fuel_thresh_cap_con_reg = regionalize_material_demand_ecoinvent_coal(
        coal_uses, fuel_thresh_cap_con, path_ecoinvent_coal_reg
    )

    ## multiply by indirect stock intensity per fuel type and adjust index for concat with other dimensions and products
    # FUELS indirect stock dictionary
    # I manually renamed the lavel Cote d'Ivoire to Cote dIvoire in the input data
    fuels_indstocks_path = input_paths_dict.get("fuels_indstocks_path")
    fuels_indstocks_dict = read_excel_into_dict(fuels_indstocks_path)
    del fuels_indstocks_dict["summary"]
    for key in fuels_indstocks_dict:
        fuels_indstocks_dict.get(key).set_index("ISIC_stocks", inplace=True)
        fuels_indstocks_dict.get(key).columns = fuels_indstocks_dict.get(
            key
        ).columns.astype(int)

    # Adjust units for the specific materials
    for key in list(fuels_indstocks_dict.keys()):
        # convert MJ to kWh (0.2777778 kWh/MJ)
        # ecoinvent description gas: 39 MJ/m3, pretroleum: 43.2 MJ/kg, hard coal: 27.91 MJ/kg
        if key.startswith("m_gas"):
            fuels_indstocks_dict[key] = fuels_indstocks_dict[key] / 39 * 0.2777778
        elif key.startswith("m_petroleum"):
            fuels_indstocks_dict[key] = fuels_indstocks_dict[key] / 43.2 * 0.2777778
        elif key.startswith("market for hard coal_"):
            fuels_indstocks_dict[key] = fuels_indstocks_dict[key] / 27.91 * 0.2777778

    # calculate indirect stocks by multiplying fuel use (kwh) with material stock intensity
    # FUELS THRESHOLD - operational
    fuel_op_indstocks_thresh_cap_dict = create_nested_dict_fuel(
        fuels_indstocks_dict, fuel_thresh_cap_op_reg
    )
    fuel_op_indstocks_thresh_cap_df = expand_nested_dict_to_df(
        fuel_op_indstocks_thresh_cap_dict
    )
    # if no non-zero entry in df, insert a tiny non-zero value to ensure that script works for assumption of only renewable energy
    if fuel_op_indstocks_thresh_cap_df.sum().sum() == 0:
        fuel_op_indstocks_thresh_cap_df.iloc[0, 0] = 0.1
    fuel_op_indstocks_thresh_cap_df = adjust_index_fuel(
        fuel_op_indstocks_thresh_cap_df,
        ["tuple", "product", "stock", "material"],
        0,
        ["region", "dimension", "fuel", "unit"],
        "op",
    )

    # FUELS THRESHOLD - construction (industrial)
    fuel_con_indstocks_thresh_cap_dict = create_nested_dict_fuel(
        fuels_indstocks_dict, fuel_thresh_cap_con_reg
    )
    fuel_con_indstocks_thresh_cap_df = expand_nested_dict_to_df(
        fuel_con_indstocks_thresh_cap_dict
    )
    # if no non-zero entry in df, insert a tiny non-zero value to ensure that script works for assumption of only renewable energy
    if fuel_con_indstocks_thresh_cap_df.sum().sum() == 0:
        fuel_con_indstocks_thresh_cap_df.iloc[0, 0] = 0.1
    fuel_con_indstocks_thresh_cap_df = adjust_index_fuel(
        fuel_con_indstocks_thresh_cap_df,
        ["tuple", "product", "stock", "material"],
        0,
        ["region", "dimension", "fuel", "unit"],
        "con",
    )

    # FUELS PROVIDED - operational
    fuel_op_indstocks_prov_cap_dict = create_nested_dict_fuel(
        fuels_indstocks_dict, fuel_prov_cap_op_reg
    )
    fuel_op_indstocks_prov_cap_df = expand_nested_dict_to_df(
        fuel_op_indstocks_prov_cap_dict
    )
    # if no non-zero entry in df, insert a tiny non-zero value to ensure that script works for assumption of only renewable energy
    if fuel_op_indstocks_prov_cap_df.sum().sum() == 0:
        fuel_op_indstocks_prov_cap_df.iloc[0, 0] = 0.1
    fuel_op_indstocks_prov_cap_df = adjust_index_fuel(
        fuel_op_indstocks_prov_cap_df,
        ["tuple", "product", "stock", "material"],
        0,
        ["region", "dimension", "fuel", "unit"],
        "op",
    )

    # FUELS PROVIDED - construction (industrial)
    fuel_con_indstocks_prov_cap_dict = create_nested_dict_fuel(
        fuels_indstocks_dict, fuel_prov_cap_con_reg
    )
    fuel_con_indstocks_prov_cap_df = expand_nested_dict_to_df(
        fuel_con_indstocks_prov_cap_dict
    )
    # if no non-zero entry in df, insert a tiny non-zero value to ensure that script works for assumption of only renewable energy
    if fuel_con_indstocks_prov_cap_df.sum().sum() == 0:
        fuel_con_indstocks_prov_cap_df.iloc[0, 0] = 0.1
    fuel_con_indstocks_prov_cap_df = adjust_index_fuel(
        fuel_con_indstocks_prov_cap_df,
        ["tuple", "product", "stock", "material"],
        0,
        ["region", "dimension", "fuel", "unit"],
        "con",
    )

    # concat and scale to country-level
    fuel_indstocks_thresh_cap = pd.concat(
        [fuel_op_indstocks_thresh_cap_df, fuel_con_indstocks_thresh_cap_df]
    )
    fuel_indstocks_thresh_scale = scale_kg_cap(
        fuel_indstocks_thresh_cap, MISO2_population_2015_subset
    )
    fuel_indstocks_thresh_scale.rename(columns={"value_scale": "value"}, inplace=True)
    fuel_indstocks_thresh_scale.sum() / 1e9  # Gt
    fuel_indstocks_prov_cap = pd.concat(
        [fuel_op_indstocks_prov_cap_df, fuel_con_indstocks_prov_cap_df]
    )
    fuel_indstocks_prov_scale = scale_kg_cap(
        fuel_indstocks_prov_cap, MISO2_population_2015_subset
    )
    fuel_indstocks_prov_scale.rename(columns={"value_scale": "value"}, inplace=True)
    fuel_indstocks_prov_scale.sum() / 1e9  # Gt

    """#######################################################
            SUMMARY OF DIRECT & INDIRECT STOCKS BY DLS DIMENSION
       #######################################################"""

    # check that all countries covered (and not more) - should be 175 countries
    phone_direct_stocks_prov.index.get_level_values(0).unique()  # 175
    stove_direct_stocks_prov.index.get_level_values(0).unique()  # 175
    fridge_direct_stocks_prov.index.get_level_values(0).unique()  # 175
    housing_direct_stocks_prov.index.get_level_values(0).unique()  # 175
    mobility_direct_stocks_prov.index.get_level_values(0).unique()  # 175
    health_educ_direct_stocks_prov.index.get_level_values(0).unique()  # 175
    clothing_direct_stocks.index.get_level_values(0).unique()  # 175

    phone_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    stove_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    fridge_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    housing_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    mobility_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    health_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    educ_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    clothing_indirect_stocks.index.get_level_values(0).unique()  # 175
    sanitation_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    water_indirect_stocks_prov.index.get_level_values(0).unique()  # 175
    electricity_indstocks_prov_df_scale.index.get_level_values(0).unique()  # 175
    fuel_indstocks_prov_scale.index.get_level_values(0).unique()

    phone_direct_stocks_thresh.index.get_level_values(0).unique()  # 175
    stove_direct_stocks_thresh.index.get_level_values(0).unique()  # 175
    fridge_direct_stocks_thresh.index.get_level_values(0).unique()  # 175
    housing_direct_stocks_thresh.index.get_level_values(0).unique()  # 175
    mobility_direct_stocks_thresh.index.get_level_values(0).unique()  # 175
    health_educ_direct_stocks_thresh.index.get_level_values(0).unique()  # 175
    clothing_direct_stocks.index.get_level_values(0).unique()  # 175

    phone_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    stove_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    fridge_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    housing_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    mobility_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    health_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    educ_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    clothing_indirect_stocks.index.get_level_values(0).unique()  # 175
    sanitation_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    water_indirect_stocks_thresh.index.get_level_values(0).unique()  # 175
    electricity_indstocks_thresh_df_scale.index.get_level_values(0).unique()  # 175
    fuel_indstocks_thresh_scale.index.get_level_values(0).unique()  ##175

    DLS_direct_stocks_prov = pd.concat(
        [
            phone_direct_stocks_prov,
            stove_direct_stocks_prov,
            fridge_direct_stocks_prov,
            housing_direct_stocks_prov,
            mobility_direct_stocks_prov,
            clothing_direct_stocks,
            educ_direct_stocks_prov,
            health_direct_stocks_prov,
        ]
    )

    DLS_indirect_stocks_prov = pd.concat(
        [
            food_indirect_stocks_prov_2015,
            stove_indirect_stocks_prov_DCfree,
            fridge_indirect_stocks_prov_DCfree,
            phone_indirect_stocks_prov_DCfree,
            TV_indirect_stocks_prov,
            housing_indirect_stocks_prov,
            mobility_indirect_stocks_prov_DCfree,
            clothing_indirect_stocks,
            health_indirect_stocks_prov,
            educ_indirect_stocks_prov,
            sanitation_indirect_stocks_prov,
            water_indirect_stocks_prov,
            cond_indirect_stocks_prov,
            lamps_housing_indirect_stocks_prov,
            lamps_educ_indirect_stocks_prov,
            lamps_health_indirect_stocks_prov,
            electricity_indstocks_prov_df_scale,
            fuel_indstocks_prov_scale,
        ]
    )

    DLS_stocks_prov = pd.concat([DLS_direct_stocks_prov, DLS_indirect_stocks_prov])

    DLS_direct_stocks_thresh = pd.concat(
        [
            phone_direct_stocks_thresh,
            stove_direct_stocks_thresh,
            fridge_direct_stocks_thresh,
            housing_direct_stocks_thresh,
            mobility_direct_stocks_thresh,
            clothing_direct_stocks,
            educ_direct_stocks_thresh,
            health_direct_stocks_thresh,
        ]
    )

    DLS_indirect_stocks_thresh = pd.concat(
        [
            food_indirect_stocks_thresh_2015,
            stove_indirect_stocks_thresh_DCfree,
            fridge_indirect_stocks_thresh_DCfree,
            phone_indirect_stocks_thresh_DCfree,
            TV_indirect_stocks_thresh,
            housing_indirect_stocks_thresh,
            mobility_indirect_stocks_thresh_DCfree,
            clothing_indirect_stocks,
            health_indirect_stocks_thresh,
            educ_indirect_stocks_thresh,
            sanitation_indirect_stocks_thresh,
            water_indirect_stocks_thresh,
            cond_indirect_stocks_thresh,
            lamps_housing_indirect_stocks_thresh,
            lamps_educ_indirect_stocks_thresh,
            lamps_health_indirect_stocks_thresh,
            electricity_indstocks_thresh_df_scale,
            fuel_indstocks_thresh_scale,
        ]
    )

    DLS_stocks_thresh = pd.concat(
        [DLS_direct_stocks_thresh, DLS_indirect_stocks_thresh]
    )

    # drop zero rows
    DLS_stocks_prov = DLS_stocks_prov[DLS_stocks_prov.value != 0]
    DLS_stocks_thresh = DLS_stocks_thresh[DLS_stocks_thresh.value != 0]

    # save
    current_date = datetime.now().strftime("%Y-%m-%d")
    provided_path = f"{output_paths_dict.get('provided')}.pkl"
    threshold_path = f"{output_paths_dict.get('threshold')}.pkl"

    DLS_stocks_prov.to_pickle(provided_path)
    DLS_stocks_thresh.to_pickle(threshold_path)
