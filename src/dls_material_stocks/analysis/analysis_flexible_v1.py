# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from ..load.DLS_load_data import (
    load_country_correspondence_dict,
    load_population_2015,
    load_ew_MFA_data,
    load_DLS_2015,
    load_population,
)

from ..analysis.DLS_functions_v3 import (
    plot_stacked_bars_sub_geo,
    plot_country_distribution,
    plot_country_distribution_one,
    expand_nested_dict_to_df,
    plot_geo_multiplot,
    plot_country_distribution_one_withTotal,
    plot_bars_horiz_gap_headroom_two_subplots_doubleGlob_mod,
    write_dict_to_excel,
    extrapolate_NAS,
    plot_timing_close_DLS_gaps_noYlim,
    plot_timing_close_DLS_gaps,
    share_plot,
    plot_stacked_bars_sub_geo_converge,
    plot_country_distribution_one_converge_small,
    share_plot_converge,
    plot_bars_Vert_gap_headroom_two_subplots_doubleGlob_mod_eff,
    plot_timing_close_DLS_gaps_converged,
    plot_country_distribution_one_converge_small_current,
    plot_stacked_bars_sub_geo_converge_onlyMap,
    plot_bars_Vert_gap_headroom_two_subplots_doubleGlob_mod_eff_ii,
    plot_bars_Vert_gap_headroom_two_subplots_doubleGlob_mod_eff_i,
)

from ..harmonization.harmonization import (
    STOCKS_OF_MATERIALS,
    DLS_STOCKS,
    HARMONIZE_MATERIALS_DICT,
    HARMONIZE_STOCKS_DICT,
    AGGR_DIM,
    AGGR_DIRECT_INDIRECT_GF,
    AGGR_PROD,
)

"""
@author: jstreeck

"""

"""############################################################

       Description:
         
        This script pulls together all results and creates the
        main manuscript figures and results data.
        
   ############################################################"""


def run_analysis(
    scenario_name,
    DLS_stocks_prov,
    DLS_stocks_thresh,
    activate_converge,
    DLS_stocks_thresh_converge,
    input_paths_dict,
    output_path,
    MISO2_MFA_data_path,
    save_results=None,
    conv_gap_mode=False
):
    pd.options.mode.copy_on_write = True

    DLS_data_path = input_paths_dict.get("DLS_data_path")

    # cover sheet for results file
    cover = pd.read_excel(input_paths_dict.get("cover"), index_col=0)
    cover = cover.rename(columns={"Unnamed: 1": "", "Unnamed: 2": "Description"})

    """ ##################
        #1 LOAD BASE DATA
        ##################"""

    # country corespondence (Wiedernhofer et al., 2024) & DLS indicators (Kikstra et al., 2025)
    country_correspondence, country_correspondence_dict = (
        load_country_correspondence_dict(input_paths_dict.get("country_correspondence"))
    )

    # load population data (Wiedernhofer et al., 2024) with Sudan & South Sudan aggregated to match to sytem boundaries of DLS indicators (Kikstra et al., 2025)
    MISO2_population_2015, MISO2_population_2015_subset = load_population_2015(
        input_paths_dict["MISO2_population"],
        sheet_name="values",
        country_correspondence=country_correspondence,
    )
    MISO2_population = load_population(
        input_paths_dict["MISO2_population"], sheet_name="values"
    )

    # load  material stock and flow data (Wiedernhofer et al., 2024) with Sudan & South Sudan aggregated to match to sytem boundaries of DLS indicators (Kikstra et al., 2025)
    MISO2_stocks_GAS_2015 = load_ew_MFA_data(
        filename=MISO2_MFA_data_path,
        country_correspondence=country_correspondence,
    )
    MISO2_stocks_GAS_2015.reset_index(inplace=True)
    MISO2_stocks_2015 = MISO2_stocks_GAS_2015[
        MISO2_stocks_GAS_2015["name"] == "S10_stock_enduse"
    ][["region", "name", "material", "sector", "2015"]]

    # load DLS data (Kikstra et al., 2025)
    DLS_2015_funct_prov, DLS_2015_thresh = load_DLS_2015(
        filename=DLS_data_path,
        country_correspondence=country_correspondence,
        country_correspondence_dict=country_correspondence_dict,
    )

    """############################################################################## 
    #2 ASSEMBLE BOTTOM-UP RESULTS FROM COMBINING DLS INDICATORS + MATERIAL INTENSITIES 
             & HARMONIZE DEFINITIONS BETWEEN DIRECT/INDIRECT STOCKS
             ###############################################################"""

    ## for our main scenario, we combine:
    #   (1) estimate of currently existing DLS stocks based on current technologies and practices
    #   (2) estimate of DLS stock thresholds based on current or converged practices
    #  --> the DLS stock gap is thus the gap between currently existing DLS stocks and DLS stocks thresholds

    ## FORMAT & PARTIALLY HARMONIZE SYSTEM SCOPE & LABELS IN BOTTOM-UP AND TOP-DOWN RESULTS
    # remove materials as stocks in their own right

    DLS_stocks_prov_noMats = DLS_stocks_prov[
        ~DLS_stocks_prov.index.get_level_values(5).isin(STOCKS_OF_MATERIALS)
    ]
    DLS_stocks_thresh_noMats = DLS_stocks_thresh[
        ~DLS_stocks_thresh.index.get_level_values(5).isin(STOCKS_OF_MATERIALS)
    ]
    if activate_converge is not None:
        DLS_stocks_thresh_converge_noMats = DLS_stocks_thresh_converge[
            ~DLS_stocks_thresh_converge.index.get_level_values(5).isin(
                STOCKS_OF_MATERIALS
            )
        ]

    DLS_stocks_prov_noMats.reset_index(inplace=True)
    DLS_stocks_thresh_noMats.reset_index(inplace=True)
    if activate_converge is not None:
        DLS_stocks_thresh_converge_noMats.reset_index(inplace=True)

    # remove materials which are not related to stocks (fertilizer, pesticides, soap, basic chemicals, chemical products n.e.c., man-made fibres)
    DLS_stocks_prov_noMats = DLS_stocks_prov_noMats[
        ~(DLS_stocks_prov_noMats.material.isin(DLS_STOCKS))
    ]
    DLS_stocks_thresh_noMats = DLS_stocks_thresh_noMats[
        ~(DLS_stocks_thresh_noMats.material.isin(DLS_STOCKS))
    ]
    if activate_converge is not None:
        DLS_stocks_thresh_converge_noMats = DLS_stocks_thresh_converge_noMats[
            ~(DLS_stocks_thresh_converge_noMats.material.isin(DLS_STOCKS))
        ]

    # drop materials which cannot be identified according to type
    DLS_stocks_prov_noMats = DLS_stocks_prov_noMats[
        ~DLS_stocks_prov_noMats.material.isin(["other_materials"])
    ]
    DLS_stocks_thresh_noMats = DLS_stocks_thresh_noMats[
        ~DLS_stocks_thresh_noMats.material.isin(["other_materials"])
    ]
    if activate_converge is not None:
        DLS_stocks_thresh_converge_noMats = DLS_stocks_thresh_converge_noMats[
            ~DLS_stocks_thresh_converge_noMats.material.isin(["other_materials"])
        ]

    ## drop aggregates in road and civil engineering foundations (only keeping only those from buildings) from MISO2_stocks_2015 and MISO2_stocks_GAS_2015 because too uncertain
    MISO2_stocks_2015 = MISO2_stocks_2015[
        ~(
            (MISO2_stocks_2015.material.isin(["aggregates"]))
            & (
                MISO2_stocks_2015.sector.isin(
                    ["Roads", "Civil_engineering_except_roads"]
                )
            )
        )
    ]
    MISO2_stocks_GAS_2015 = MISO2_stocks_GAS_2015[
        ~(
            (MISO2_stocks_GAS_2015.material.isin(["aggregates"]))
            & (
                MISO2_stocks_GAS_2015.sector.isin(
                    ["Roads", "Civil_engineering_except_roads"]
                )
            )
        )
    ]

    # rename materials and stocks from ecoinvent material labels to labels in manuscript

    ecoinvent_sector_labels = pd.read_excel(
        input_paths_dict.get("Velez_ecoinvent_target_sectors")
    )
    ecoinvent_materials_dict = ecoinvent_sector_labels[
        ["ISIC_materials_code", "ISIC_materials_label_2"]
    ].dropna()
    ecoinvent_materials_dict["ISIC_materials_code"] = (
        ecoinvent_materials_dict["ISIC_materials_code"].astype(int).astype(str)
    )
    ecoinvent_materials_dict = dict(
        zip(
            ecoinvent_materials_dict["ISIC_materials_code"],
            ecoinvent_materials_dict["ISIC_materials_label_2"],
        )
    )
    ecoinvent_stocks_dict = ecoinvent_sector_labels[
        ["ISIC_stocks_code", "ISIC_stocks_label_1"]
    ].dropna()
    ecoinvent_stocks_dict["ISIC_stocks_code"] = ecoinvent_stocks_dict[
        "ISIC_stocks_code"
    ].astype(int)
    ecoinvent_stocks_dict = dict(
        zip(
            ecoinvent_stocks_dict["ISIC_stocks_code"],
            ecoinvent_stocks_dict["ISIC_stocks_label_1"],
        )
    )

    # harmonize materials and stock types in indirect stocks
    # check that all values are strings for 'material'
    DLS_stocks_prov_noMats["material"] = DLS_stocks_prov_noMats["material"].astype(str)
    DLS_stocks_prov_noMats["material"] = DLS_stocks_prov_noMats["material"].replace(
        ecoinvent_materials_dict
    )

    DLS_stocks_prov_noMats["stock"] = DLS_stocks_prov_noMats["stock"].replace(
        ecoinvent_stocks_dict
    )
    DLS_stocks_prov_noMats["material"] = DLS_stocks_prov_noMats["material"].replace(
        HARMONIZE_MATERIALS_DICT
    )
    DLS_stocks_prov_noMats["stock"] = DLS_stocks_prov_noMats["stock"].replace(
        HARMONIZE_STOCKS_DICT
    )

    DLS_stocks_thresh_noMats["material"] = DLS_stocks_thresh_noMats["material"].astype(
        str
    )

    DLS_stocks_thresh_noMats["material"] = DLS_stocks_thresh_noMats["material"].replace(
        ecoinvent_materials_dict
    )
    DLS_stocks_thresh_noMats["stock"] = DLS_stocks_thresh_noMats["stock"].replace(
        ecoinvent_stocks_dict
    )
    DLS_stocks_thresh_noMats["material"] = DLS_stocks_thresh_noMats["material"].replace(
        HARMONIZE_MATERIALS_DICT
    )
    DLS_stocks_thresh_noMats["stock"] = DLS_stocks_thresh_noMats["stock"].replace(
        HARMONIZE_STOCKS_DICT
    )

    if activate_converge is not None:
        DLS_stocks_thresh_converge_noMats["material"] = (
            DLS_stocks_thresh_converge_noMats["material"].astype(str)
        )
        DLS_stocks_thresh_converge_noMats["material"] = (
            DLS_stocks_thresh_converge_noMats["material"].replace(
                ecoinvent_materials_dict
            )
        )
        DLS_stocks_thresh_converge_noMats["stock"] = DLS_stocks_thresh_converge_noMats[
            "stock"
        ].replace(ecoinvent_stocks_dict)
        DLS_stocks_thresh_converge_noMats["material"] = (
            DLS_stocks_thresh_converge_noMats["material"].replace(
                HARMONIZE_MATERIALS_DICT
            )
        )
        DLS_stocks_thresh_converge_noMats["stock"] = DLS_stocks_thresh_converge_noMats[
            "stock"
        ].replace(HARMONIZE_STOCKS_DICT)

    # set index
    DLS_stocks_prov_noMats.set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ],
        inplace=True,
    )
    DLS_stocks_thresh_noMats.set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ],
        inplace=True,
    )

    # if also considering DLS stock thresholds for converged practices, also set index for the respective data frame
    if activate_converge is not None:
        DLS_stocks_thresh_converge_noMats.set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ],
            inplace=True,
        )

    """ #####################################################################
         #2A PREPARE BOTTOM-UP DATA ON DLS MATERIAL STOCKS FOR COUNTRY-LEVEL
         ####################################################################"""

    # prepare population data to calculate PER CAPITA stocks
    MISO2_2015_pop_country_subset = MISO2_population_2015.reset_index()[
        MISO2_population_2015.reset_index()["region"].isin(
            DLS_stocks_prov_noMats.reset_index()["region"].unique()
        )
    ].set_index("region")

    # calculate per capita stocks and format dataframes for DLS stocks provided and to reach DLS threshold
    DLS_stocks_prov_noMats_country_cap = DLS_stocks_prov_noMats.reset_index().merge(
        MISO2_2015_pop_country_subset.reset_index(), how="left", on="region"
    )
    DLS_stocks_prov_noMats_country_cap["value_cap"] = (
        DLS_stocks_prov_noMats_country_cap["value"]
        / DLS_stocks_prov_noMats_country_cap["2015"]
    )
    DLS_stocks_prov_noMats_country_cap["unit"] = "kg/cap"
    DLS_stocks_prov_noMats_country_cap.set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ],
        inplace=True,
    )
    DLS_stocks_prov_noMats_country_cap = DLS_stocks_prov_noMats_country_cap["value_cap"]

    DLS_stocks_thresh_noMats_country_cap = DLS_stocks_thresh_noMats.reset_index().merge(
        MISO2_2015_pop_country_subset.reset_index(), how="left", on="region"
    )
    DLS_stocks_thresh_noMats_country_cap["value_cap"] = (
        DLS_stocks_thresh_noMats_country_cap["value"]
        / DLS_stocks_thresh_noMats_country_cap["2015"]
    )
    DLS_stocks_thresh_noMats_country_cap["unit"] = "kg/cap"
    DLS_stocks_thresh_noMats_country_cap.set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ],
        inplace=True,
    )
    DLS_stocks_thresh_noMats_country_cap = DLS_stocks_thresh_noMats_country_cap[
        "value_cap"
    ]

    if activate_converge != None:
        DLS_stocks_thresh_converge_noMats_country_cap = (
            DLS_stocks_thresh_converge_noMats.reset_index().merge(
                MISO2_2015_pop_country_subset.reset_index(), how="left", on="region"
            )
        )
        DLS_stocks_thresh_converge_noMats_country_cap["value_cap"] = (
            DLS_stocks_thresh_converge_noMats_country_cap["value"]
            / DLS_stocks_thresh_converge_noMats_country_cap["2015"]
        )
        DLS_stocks_thresh_converge_noMats_country_cap["unit"] = "kg/cap"
        DLS_stocks_thresh_converge_noMats_country_cap.set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ],
            inplace=True,
        )
        DLS_stocks_thresh_converge_noMats_country_cap = (
            DLS_stocks_thresh_converge_noMats_country_cap["value_cap"]
        )

    """ #####################################################################
         #2B PREPARE BOTTOM-UP DATA ON DLS MATERIAL STOCKS FOR WORLD REGIONS
         ####################################################################"""

    # read world region country correspondence
    R11_country_correspondence = pd.read_excel(
        input_paths_dict.get("country_correspondence")
    )

    R11_country_correspondence_dict = dict(
        zip(R11_country_correspondence.MISO2, R11_country_correspondence["R11"])
    )

    # DLS stocks in world regions AT SCALE
    DLS_stocks_prov_noMats_R11 = DLS_stocks_prov_noMats.copy().reset_index()
    DLS_stocks_prov_noMats_R11["region"] = DLS_stocks_prov_noMats_R11["region"].replace(
        R11_country_correspondence_dict
    )
    DLS_stocks_prov_noMats_R11["value"] = DLS_stocks_prov_noMats_R11["value"].astype(
        float
    )
    DLS_stocks_prov_noMats_R11 = DLS_stocks_prov_noMats_R11.groupby(
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

    DLS_stocks_thresh_noMats_R11 = DLS_stocks_thresh_noMats.copy().reset_index()
    DLS_stocks_thresh_noMats_R11.region = DLS_stocks_thresh_noMats_R11.region.replace(
        R11_country_correspondence_dict
    )
    DLS_stocks_thresh_noMats_R11["value"] = DLS_stocks_thresh_noMats_R11[
        "value"
    ].astype(float)
    DLS_stocks_thresh_noMats_R11 = DLS_stocks_thresh_noMats_R11.groupby(
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

    if activate_converge != None:
        DLS_stocks_thresh_converge_noMats_R11 = (
            DLS_stocks_thresh_converge_noMats.copy().reset_index()
        )
        DLS_stocks_thresh_converge_noMats_R11.region = (
            DLS_stocks_thresh_converge_noMats_R11.region.replace(
                R11_country_correspondence_dict
            )
        )
        DLS_stocks_thresh_converge_noMats_R11["value"] = (
            DLS_stocks_thresh_converge_noMats_R11["value"].astype(float)
        )
        DLS_stocks_thresh_converge_noMats_R11 = (
            DLS_stocks_thresh_converge_noMats_R11.groupby(
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
        )

    # DLS stocks AT SCALE per region
    # region (R), dimension (D), product(P) material (M)
    DLS_stocks_thresh_noMats_R11_scale_RDPM = DLS_stocks_thresh_noMats_R11.groupby(
        ["region", "dimension", "stock", "material"]
    ).sum()
    DLS_stocks_thresh_noMats_R11_scale_RDPM_piv = (
        DLS_stocks_thresh_noMats_R11_scale_RDPM.reset_index().pivot(
            index=["region", "dimension", "stock"], columns="material", values="value"
        )
    )
    DLS_stocks_prov_noMats_R11_scale_RDPM = DLS_stocks_prov_noMats_R11.groupby(
        ["region", "dimension", "stock", "material"]
    ).sum()
    DLS_stocks_prov_noMats_R11_scale_RDPM_piv = (
        DLS_stocks_prov_noMats_R11_scale_RDPM.reset_index().pivot(
            index=["region", "dimension", "stock"], columns="material", values="value"
        )
    )

    # region (R), dimension (D), material (M)
    DLS_stocks_thresh_noMats_R11_scale_RDM = DLS_stocks_thresh_noMats_R11.groupby(
        ["region", "dimension", "material"]
    ).sum()
    DLS_stocks_thresh_noMats_R11_scale_RDM_piv = (
        DLS_stocks_thresh_noMats_R11_scale_RDM.reset_index().pivot(
            index=["region", "dimension"], columns="material", values="value"
        )
    )
    DLS_stocks_prov_noMats_R11_scale_RDM = DLS_stocks_prov_noMats_R11.groupby(
        ["region", "dimension", "material"]
    ).sum()
    DLS_stocks_prov_noMats_R11_scale_RDM_piv = (
        DLS_stocks_prov_noMats_R11_scale_RDM.reset_index().pivot(
            index=["region", "dimension"], columns="material", values="value"
        )
    )
    # region (R), product(P), material (M)
    DLS_stocks_prov_noMats_R11_scale_RPM = DLS_stocks_prov_noMats_R11.groupby(
        ["region", "stock", "material"]
    ).sum()
    DLS_stocks_prov_noMats_R11_scale_RPM_piv = (
        DLS_stocks_prov_noMats_R11_scale_RPM.reset_index().pivot(
            index=["region", "stock"], columns="material", values="value"
        )
    )
    DLS_stocks_thresh_noMats_R11_scale_RPM = DLS_stocks_thresh_noMats_R11.groupby(
        ["region", "stock", "material"]
    ).sum()
    DLS_stocks_thresh_noMats_R11_scale_RPM_piv = (
        DLS_stocks_thresh_noMats_R11_scale_RPM.reset_index().pivot(
            index=["region", "stock"], columns="material", values="value"
        )
    )
    # DLS stock gap
    DLS_stocks_gap_noMats_R11_scale_RDM_piv = (
        DLS_stocks_thresh_noMats_R11_scale_RDM_piv
        - DLS_stocks_prov_noMats_R11_scale_RDM_piv
    )
    DLS_stocks_gap_noMats_R11_scale_RPM_piv = (
        DLS_stocks_thresh_noMats_R11_scale_RPM_piv
        - DLS_stocks_prov_noMats_R11_scale_RPM_piv
    )

    # DLS stocks in world regions PER CAPITA (unit: tons/cap)
    # prepare population data to calculate per capita stocks
    MISO2_2015_pop_country_subset = MISO2_population_2015.reset_index()[
        MISO2_population_2015.reset_index()["region"].isin(
            DLS_stocks_prov_noMats.reset_index()["region"].unique()
        )
    ].set_index("region")
    MISO2_2015_pop_R11 = MISO2_2015_pop_country_subset.copy()
    MISO2_2015_pop_R11 = (
        MISO2_2015_pop_R11.rename(R11_country_correspondence_dict)
        .groupby("region")
        .sum()
    )

    # DLS provided
    DLS_stocks_prov_noMats_R11_cap = DLS_stocks_prov_noMats_R11.reset_index().merge(
        MISO2_2015_pop_R11.reset_index(), how="left", on="region"
    )
    DLS_stocks_prov_noMats_R11_cap["value_cap"] = (
        DLS_stocks_prov_noMats_R11_cap["value"] / DLS_stocks_prov_noMats_R11_cap["2015"]
    )
    DLS_stocks_prov_noMats_R11_cap["unit"] = "kg/cap"
    DLS_stocks_prov_noMats_R11_cap.set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ],
        inplace=True,
    )
    DLS_stocks_prov_noMats_R11_cap = DLS_stocks_prov_noMats_R11_cap["value_cap"]

    # DLS threshold
    DLS_stocks_thresh_noMats_R11_cap = DLS_stocks_thresh_noMats_R11.reset_index().merge(
        MISO2_2015_pop_R11.reset_index(), how="left", on="region"
    )
    DLS_stocks_thresh_noMats_R11_cap["value_cap"] = (
        DLS_stocks_thresh_noMats_R11_cap["value"]
        / DLS_stocks_thresh_noMats_R11_cap["2015"]
    )
    DLS_stocks_thresh_noMats_R11_cap["unit"] = "kg/cap"
    DLS_stocks_thresh_noMats_R11_cap.set_index(
        [
            "region",
            "year",
            "dimension",
            "stock_type",
            "product",
            "stock",
            "material",
            "unit",
        ],
        inplace=True,
    )
    DLS_stocks_thresh_noMats_R11_cap = DLS_stocks_thresh_noMats_R11_cap["value_cap"]

    if activate_converge is not None:
        # DLS threshold converge
        DLS_stocks_thresh_converge_noMats_R11_cap = (
            DLS_stocks_thresh_converge_noMats_R11.reset_index().merge(
                MISO2_2015_pop_R11.reset_index(), how="left", on="region"
            )
        )
        DLS_stocks_thresh_converge_noMats_R11_cap["value_cap"] = (
            DLS_stocks_thresh_converge_noMats_R11_cap["value"]
            / DLS_stocks_thresh_converge_noMats_R11_cap["2015"]
        )
        DLS_stocks_thresh_converge_noMats_R11_cap["unit"] = "kg/cap"
        DLS_stocks_thresh_converge_noMats_R11_cap.set_index(
            [
                "region",
                "year",
                "dimension",
                "stock_type",
                "product",
                "stock",
                "material",
                "unit",
            ],
            inplace=True,
        )
        DLS_stocks_thresh_converge_noMats_R11_cap = (
            DLS_stocks_thresh_converge_noMats_R11_cap["value_cap"]
        )

    # R11 - aggregate dataframe dimensions but keep details for explorative plots in excel
    DLS_stocks_prov_R11_RDTPM = DLS_stocks_prov_noMats_R11_cap.groupby(
        ["region", "dimension", "stock_type", "product", "stock", "material"]
    ).sum()
    DLS_stocks_thresh_R11_cap_RDTPM = DLS_stocks_thresh_noMats_R11_cap.groupby(
        ["region", "dimension", "stock_type", "product", "stock", "material"]
    ).sum()
    DLS_stocks_prov_R11_DTPM_piv = DLS_stocks_prov_R11_RDTPM.reset_index().pivot(
        index=["region", "dimension", "stock_type", "product", "stock"],
        columns="material",
        values="value_cap",
    )
    DLS_stocks_thresh_R11_DTPM_piv = (
        DLS_stocks_thresh_R11_cap_RDTPM.reset_index().pivot(
            index=["region", "dimension", "stock_type", "product", "stock"],
            columns="material",
            values="value_cap",
        )
    )

    """ ##################################################################################################################
         #3 PREPARE MISO2 TOP-DOWN DATA ON ECONOMY-WIDE MATERIAL STOCKS AND NET ADDITIONS TO STOCKS (NAS) FOR R11 WORLD REGIONS
         ##################################################################################################################"""

    ### MISO2 STOCKS by country to R11
    MISO2_stocks_2015_total = MISO2_stocks_2015.groupby(
        ["region", "sector", "material"]
    ).sum()["2015"]
    MISO2_stocks_2015_total_subset = MISO2_stocks_2015_total.reset_index()[
        MISO2_stocks_2015_total.reset_index()["region"].isin(
            DLS_stocks_prov_noMats.reset_index()["region"].unique()
        )
    ]
    MISO2_stocks_2015_total_subset_R11 = MISO2_stocks_2015_total_subset.copy()
    MISO2_stocks_2015_total_subset_R11.region = (
        MISO2_stocks_2015_total_subset_R11.region.replace(
            R11_country_correspondence_dict
        )
    )
    MISO2_stocks_2015_total_subset_R11 = MISO2_stocks_2015_total_subset_R11.groupby(
        ["region", "sector", "material"]
    ).sum()

    # calc per capita values (unit: tons/capita)
    MISO2_stocks_2015_R11_cap = MISO2_stocks_2015_total_subset_R11.reset_index().merge(
        MISO2_2015_pop_R11.reset_index(), how="left", on="region"
    )
    MISO2_stocks_2015_R11_cap["value_cap"] = (
        MISO2_stocks_2015_R11_cap["2015_x"] / MISO2_stocks_2015_R11_cap["2015_y"]
    )
    MISO2_stocks_2015_R11_cap["unit"] = "kg/cap"
    MISO2_stocks_2015_R11_cap.set_index(["region", "sector", "material"], inplace=True)
    MISO2_stocks_2015_R11_cap = MISO2_stocks_2015_R11_cap["value_cap"]

    MISO2_stocks_2015_R11_cap_piv = MISO2_stocks_2015_R11_cap.reset_index().pivot(
        index=["region", "sector"], columns="material", values="value_cap"
    )

    ### MISO2 Net additions to stocks NAS (GAS - EOL)
    MISO2_GAS = (
        MISO2_stocks_GAS_2015[MISO2_stocks_GAS_2015.name == "F_9_10_GAS_enduse"]
        .set_index(["region", "name", "material", "sector"])
        .groupby(["region", "material", "sector"])
        .sum()
    )
    
    # pre aggregate these

    MISO2_EoL = (
        MISO2_stocks_GAS_2015[
            MISO2_stocks_GAS_2015.name == "F_10_11_supply_EoL_waste_enduse"
        ]
        .set_index(["region", "name", "material", "sector"])
        .groupby(["region", "material", "sector"])
        .sum()
    )
    MISO2_NAS = MISO2_GAS - MISO2_EoL

    """ ####################################################################
                #4 HARMONIZE DEFINITIONS BETWEEN BOTTOM-UP & TOP-DOWN
         #####################################################################"""

    # harmonize sector names (product groups)
    R11_harmonize_stocks = {
        "buildings": "nonres_buildings",
        "Civil_engineering_except_roads": "other_construction",
        "Computers_and_precision_instruments": "machinery",
        "Electrical_equipment": "machinery",
        "Food_packaging": "other",
        "Furniture_and_other_manufactured_goods_nec": "machinery",
        "Machinery_and_equipment": "machinery",
        "Motor_vehicles_trailers_and_semi-trailers": "motor_vehicles",
        "Non_residential_buildings": "nonres_buildings",
        "Other_transport_equipment": "other_transport",
        "Printed_matter_and_recorded_media": "other",
        "Products_nec": "other",
        "Residential_buildings": "res_buildings",
        "Roads": "road_rail",
        "Textiles": "textiles",
    }

    ## 1) MISO2 STOCKS
    ## harmonize MISO2 per capita STOCKS regarding sectors and four material groups
    MISO2_R11_cap_S = (
        MISO2_stocks_2015_R11_cap_piv.rename(R11_harmonize_stocks)
        .groupby(["region", "sector"])
        .sum()
    )
    # MISO2 at scale
    MISO2_stocks_2015_R11_scale = (
        MISO2_stocks_2015_total_subset_R11.rename(R11_harmonize_stocks)
        .groupby(["region", "sector", "material"])
        .sum()
    )
    MISO2_stocks_2015_R11_scale_piv = MISO2_stocks_2015_R11_scale.reset_index().pivot(
        index=["region", "sector"], columns="material", values="2015"
    )
    
    MISO2_R11_cap_S = MISO2_R11_cap_S[["biomass", "fossils", "metals", "minerals"]]

    MISO2_stocks_2015_R11_scale_piv = MISO2_stocks_2015_R11_scale_piv[
        ["biomass", "fossils", "metals", "minerals"]
    ]

    # MISO2 per capita stocks total per region for 2015
    MISO2_stocks_2015_grouped = MISO2_stocks_2015.set_index(
        ["region", "name", "material", "sector"]
    ).groupby(["region", "material"])
    intermed_list = []
    for group_key, group_data in MISO2_stocks_2015_grouped:
        df = (
            MISO2_stocks_2015_grouped.get_group(group_key)
            / MISO2_population_2015_subset[
                MISO2_population_2015_subset.index == group_key[0]
            ]
        )
        intermed_list.append(df)
    MISO2_stocks_2015_cap = pd.concat(intermed_list)
    MISO2_stocks_2015_cap_total = MISO2_stocks_2015_cap.groupby(["region"]).sum()
    MISO2_stocks_2015_cap_total.rename(columns={"2015": "value"}, inplace=True)

    ## 2) MISO2 NAS (net additions to stocks)
    # MISO2 NAS at scale harmonize materials to 4 groups
    MISO2_NAS_subset = MISO2_NAS.reset_index()[
        MISO2_NAS.reset_index()["region"].isin(
            DLS_stocks_prov_noMats.reset_index()["region"].unique()
        )
    ]
    MISO2_NAS_subset_melted = MISO2_NAS_subset.reset_index(drop=True).melt(
        id_vars=["region", "sector", "material"], var_name="year", value_name="value"
    )
    MISO2_NAS_subset_piv = MISO2_NAS_subset_melted.pivot_table(
        index=["region", "sector", "year"], columns="material", values="value"
    )

    # here, aggr are included
    MISO2_NAS_subset_piv["minerals"] = MISO2_NAS_subset_piv[
        ["aggregates", "minerals"]
    ].sum(axis=1)

    MISO2_NAS_subset_piv = MISO2_NAS_subset_piv[
        ["biomass", "fossils", "metals", "minerals"]
    ]

    MISO2_NAS_subset_piv_materials = MISO2_NAS_subset_piv.groupby(
        ["region", "year"]
    ).sum()

    # MISO2 NAS per capita
    MISO2_population_melted = MISO2_population.reset_index().melt(
        id_vars=["region"], var_name="year", value_name="value"
    )
    MISO2_NAS_subset_piv_materials_cap = MISO2_NAS_subset_piv_materials.merge(
        MISO2_population_melted, how="left", on=["region", "year"]
    )
    pop_case = MISO2_NAS_subset_piv_materials_cap["value"] * 1000
    (
        MISO2_NAS_subset_piv_materials_cap["biomass_cap"],
        MISO2_NAS_subset_piv_materials_cap["fossils_cap"],
        MISO2_NAS_subset_piv_materials_cap["metals_cap"],
        MISO2_NAS_subset_piv_materials_cap["minerals_cap"],
    ) = (
        MISO2_NAS_subset_piv_materials_cap["biomass"] / pop_case,
        MISO2_NAS_subset_piv_materials_cap["fossils"] / pop_case,
        MISO2_NAS_subset_piv_materials_cap["metals"] / pop_case,
        MISO2_NAS_subset_piv_materials_cap["minerals"] / pop_case,
    )
    # tons/capita
    MISO2_NAS_subset_piv_materials_cap = (
        MISO2_NAS_subset_piv_materials_cap[
            [
                "region",
                "year",
                "biomass_cap",
                "fossils_cap",
                "metals_cap",
                "minerals_cap",
            ]
        ].set_index(["region", "year"])
        * 1000
    )

    ## calculate GLOBAL aggregate NAS per capita, 10^3 tons/cap
    MISO2_NAS_global = MISO2_NAS_subset_piv_materials.groupby("year").sum()
    MISO2_NAS_global_cap = MISO2_NAS_global.merge(
        MISO2_population_melted.set_index(["region", "year"]).groupby("year").sum(),
        how="left",
        on=["year"],
    )
    pop_use_globe = MISO2_NAS_global_cap["value"] * 1000
    (
        MISO2_NAS_global_cap["biomass_cap"],
        MISO2_NAS_global_cap["fossils_cap"],
        MISO2_NAS_global_cap["metals_cap"],
        MISO2_NAS_global_cap["minerals_cap"],
    ) = (
        MISO2_NAS_global_cap["biomass"] / pop_use_globe,
        MISO2_NAS_global_cap["fossils"] / pop_use_globe,
        MISO2_NAS_global_cap["metals"] / pop_use_globe,
        MISO2_NAS_global_cap["minerals"] / pop_use_globe,
    )
    MISO2_NAS_global_cap = MISO2_NAS_global_cap[
        ["biomass_cap", "fossils_cap", "metals_cap", "minerals_cap"]
    ]
    # tons/cap
    MISO2_NAS_global_cap = MISO2_NAS_global_cap * 1000

    ## calculate R11 aggregate NAS per capita, 10^3 tons/cap
    MISO2_NAS_subset_R11 = MISO2_NAS_subset.copy().reset_index(drop=True)
    MISO2_NAS_subset_R11.region = MISO2_NAS_subset_R11.region.replace(
        R11_country_correspondence_dict
    )
    MISO2_NAS_subset_R11 = MISO2_NAS_subset_R11.groupby(
        ["region", "sector", "material"]
    ).sum()
    MISO2_NAS_subset_R11 = (
        MISO2_NAS_subset_R11.rename(R11_harmonize_stocks)
        .groupby(["region", "sector", "material"])
        .sum()
    )
    MISO2_NAS_subset_R11_piv = MISO2_NAS_subset_R11.reset_index().pivot(
        index=["region", "sector"], columns="material", values=["2015"]
    )
    # Reshape with melt
    MISO2_NAS_subset_R11_melted = MISO2_NAS_subset_R11.reset_index().melt(
        id_vars=["region", "sector", "material"], var_name="year", value_name="value"
    )
    # Pivot with year as an additional index
    MISO2_NAS_subset_R11_piv = MISO2_NAS_subset_R11_melted.pivot_table(
        index=["region", "sector", "year"], columns="material", values="value"
    )

    MISO2_NAS_subset_R11_piv["minerals"] = MISO2_NAS_subset_R11_piv[
        ["minerals", "aggregates"]
    ].sum(axis=1)
    MISO2_NAS_subset_R11_piv = MISO2_NAS_subset_R11_piv[
        ["biomass", "fossils", "metals", "minerals"]
    ]

    MISO2_NAS_subset_R11_piv_materials = MISO2_NAS_subset_R11_piv.groupby(
        ["region", "year"]
    ).sum()

    MISO2_NAS_subset_R11_piv_materials_2015 = MISO2_NAS_subset_R11_piv_materials[
        MISO2_NAS_subset_R11_piv_materials.index.get_level_values(1) == "2015"
    ]
    MISO2_NAS_subset_R11_piv_materials_2015.sum().sum() / 1e6

    #
    MISO2_population_melted_R11 = (
        MISO2_population_melted.replace(R11_country_correspondence_dict)
        .groupby(["region", "year"])
        .sum()
        .loc[R11_country_correspondence.R11.unique()]
    )
    MISO2_population_melted_R11_2015 = MISO2_population_melted_R11[
        MISO2_population_melted_R11.index.get_level_values(1) == "2015"
    ]

    MISO2_NAS_subset_R11_piv_materials_cap = (
        MISO2_NAS_subset_R11_piv_materials.reset_index().merge(
            MISO2_population_melted_R11.reset_index(), how="left", on=["region", "year"]
        )
    )
    pop_use_R11 = MISO2_NAS_subset_R11_piv_materials_cap["value"] * 1000
    (
        MISO2_NAS_subset_R11_piv_materials_cap["biomass_cap"],
        MISO2_NAS_subset_R11_piv_materials_cap["fossils_cap"],
        MISO2_NAS_subset_R11_piv_materials_cap["metals_cap"],
        MISO2_NAS_subset_R11_piv_materials_cap["minerals_cap"],
    ) = (
        MISO2_NAS_subset_R11_piv_materials_cap["biomass"] / pop_use_R11,
        MISO2_NAS_subset_R11_piv_materials_cap["fossils"] / pop_use_R11,
        MISO2_NAS_subset_R11_piv_materials_cap["metals"] / pop_use_R11,
        MISO2_NAS_subset_R11_piv_materials_cap["minerals"] / pop_use_R11,
    )
    # tons/cap (before 10^3 tons/cap)
    MISO2_NAS_subset_R11_piv_materials_cap = (
        MISO2_NAS_subset_R11_piv_materials_cap[
            [
                "region",
                "year",
                "biomass_cap",
                "fossils_cap",
                "metals_cap",
                "minerals_cap",
            ]
        ].set_index(["region", "year"])
        * 1000
    )

    """ ####################################################################
                              #5 PREPARE DATA FOR PLOTS
         #####################################################################"""

    # harmonize DLS dimension labels
    aggr_direct_indirect = {
        "clothing_i": "clothing",
        "communication_i": "communication",
        "educ_i": "education",
        "education_i": "education",
        "health_i": "health",
        "hh_appliance_i": "hh_appliance",
        "housing_i": "housing",
        "transport_i": "transport",
    }
    aggr_dim = {
        "communication": "Socialization_cur",
        "clothing": "Shelter_cur",
        "cond_i": "Shelter_cur",
        "education": "Socialization_cur",
        "health": "Health_cur",
        "hh_appliance": "Nutrition_cur",
        "housing": "Shelter_cur",
        "nutrition_i": "Nutrition_cur",
        "sanitation_i": "Health_cur",
        "transport": "Mobility_cur",
        "water_i": "Health_cur",
    }
    DLS_stocks_gap_noMats_R11_scale_RDM_piv = (
        DLS_stocks_gap_noMats_R11_scale_RDM_piv.reset_index()
        .replace(aggr_direct_indirect)
        .groupby(["region", "dimension"])
        .sum()
    )
    DLS_stocks_gap_noMats_R11_scale_RDM_piv = (
        DLS_stocks_gap_noMats_R11_scale_RDM_piv.reset_index()
        .replace(aggr_dim)
        .groupby(["region", "dimension"])
        .sum()
    )

    #! set negatives in gap zero (which occur in case that the provided DLS are higher than the ones required in converging practices scenario)
    # - however, these won't enter the calculations as reducing the stock gap because it's unlikely that old DLS stocks are abandoned in the short/medium term
    # - the gap is thus only the positive values
    
    if not conv_gap_mode:
        DLS_stocks_gap_noMats_R11_scale_RDM_piv = DLS_stocks_gap_noMats_R11_scale_RDM_piv[
            DLS_stocks_gap_noMats_R11_scale_RDM_piv > 0
        ]

    # harmonize product labels
    aggr_prod = {
        "lamps": "other",
        "motor_vehicles": "transport_machinery",
        "other_transport": "transport_machinery",
        "textiles": "other",
    }

    MISO2_stocks_2015_R11_scale_piv = (
        MISO2_stocks_2015_R11_scale_piv.reset_index()
        .replace(aggr_prod)
        .groupby(["region", "sector"])
        .sum()
    )

    # format DLS stocks prov by product
    DLS_stocks_prov_noMats_R11_scale_RPM_piv.index.rename(
        {"stock": "sector"}, inplace=True
    )
    DLS_stocks_prov_noMats_R11_scale_RPM_piv = (
        DLS_stocks_prov_noMats_R11_scale_RPM_piv.rename(R11_harmonize_stocks)
        .reset_index()
        .groupby(["region", "sector"])
        .sum()
    )
    DLS_stocks_prov_noMats_R11_scale_RPM_piv = (
        DLS_stocks_prov_noMats_R11_scale_RPM_piv.rename(aggr_prod)
        .reset_index()
        .groupby(["region", "sector"])
        .sum()
    )

    """ ######################################################################################################
            #6 CALCULATE BEYOND-DLS STOCKS AND ASSEMBLE GLOBAL AND REGIONAL DFs with DLS & BEYOND DLS STOCKS
         #####################################################################################################"""

    # beyond-DLS stock by product (*1000 conversion ktons to tons)
    Beyond_DLS_stocks_R11_piv = (
        MISO2_stocks_2015_R11_scale_piv * 1000
        - DLS_stocks_prov_noMats_R11_scale_RPM_piv
    )



    ### 1) REGIONAL

    # assemble data for REGIONAL bar plot at scale per material
    # stocks for DLS provided, beyond-DLS, gap - targeted, gap - regional ratio, gap - global trickle down
    DLS_stocks_prov_noMats_regio_scale = DLS_stocks_prov_noMats_R11.groupby(
        ["region", "stock", "material", "unit"]
    ).sum()
    Beyond_DLS_stocks_regio_scale = Beyond_DLS_stocks_R11_piv.groupby(
        ["region", "sector"]
    ).sum()

    DLS_stocks_gap_noMats_regio_scale = DLS_stocks_gap_noMats_R11_scale_RDM_piv.groupby(
        ["region", "dimension"]
    ).sum()

    # per material
    DLS_stocks_prov_noMats_regio_scale_materials = (
        DLS_stocks_prov_noMats_regio_scale.groupby(["region", "material"]).sum()
    )
    Beyond_DLS_stocks_regio_scale_materials = (
        pd.melt(
            Beyond_DLS_stocks_R11_piv.reset_index(),
            id_vars=["region", "sector"],
            value_vars=["biomass", "fossils", "metals", "minerals"],
            var_name="material",
            value_name="value",
        )
        .set_index(["region", "sector", "material"])
        .groupby(["region", "material"])
        .sum()
    )

    ##! beyond DLS stocks negative for some materials because top-down estimate of total stocks larger than bottom-up estimate of existing DLS stocks
    # negatives in terms of count and mass
    negative_count_bDLS_RM = (Beyond_DLS_stocks_regio_scale_materials < 0).sum().sum()
    total_count_bDLS_RM = Beyond_DLS_stocks_regio_scale_materials.size
    negative_mass_bDLS_RM = (
        (
            Beyond_DLS_stocks_regio_scale_materials[
                Beyond_DLS_stocks_regio_scale_materials < 0
            ]
        )
        .sum()
        .sum()
    )
    positive_mass_bDLS_RM = (
        (
            Beyond_DLS_stocks_regio_scale_materials[
                Beyond_DLS_stocks_regio_scale_materials > 0
            ]
        )
        .sum()
        .sum()
    )
    negative_on_positive_mass_bDLS_RM = abs(
        negative_mass_bDLS_RM / positive_mass_bDLS_RM * 100
    )
    # we set negative values to zero here, implying that the regional ratio of existing DLS and beyond DLS stocks is 1 and thus no beyond-DLS stock is built in 'regional ratios' scenario
    Beyond_DLS_stocks_regio_scale_materials[
        Beyond_DLS_stocks_regio_scale_materials < 0
    ] = 0

    # calculate additional beyond-DLS stocks that would be added if regional beyond-DLS ratios are built in the future
    DLS_stocks_gap_noMats_regio_scale_materials = (
        pd.melt(
            DLS_stocks_gap_noMats_regio_scale.reset_index(),
            id_vars=["region", "dimension"],
            value_vars=["biomass", "fossils", "metals", "minerals"],
            var_name="material",
            value_name="value",
        )
        .set_index(["region", "dimension", "material"])
        .groupby(["region", "material"])
        .sum()
    )
    regio_ratio_DLS_beyond_DLS = (
        Beyond_DLS_stocks_regio_scale_materials
        / DLS_stocks_prov_noMats_regio_scale_materials
    )

    if not conv_gap_mode:
        add_regio_gap_beyond_DLS_per_DLS = (
            DLS_stocks_gap_noMats_regio_scale_materials * regio_ratio_DLS_beyond_DLS
        )
    else:
        add_regio_gap_beyond_DLS_per_DLS = (
            DLS_stocks_gap_noMats_regio_scale_materials.clip(0) * regio_ratio_DLS_beyond_DLS)

    # calculate additional beyond-DLS stocks that would be added if regional beyond-DLS ratios are built in the future
    # any regions that have a sum larger than the hard-coded threshold in 'per_capita_trickle_down_regio' already have existing total stocks higher than the threshold
    per_capita_trickle_down_regio = pd.DataFrame(
        [4.28, 1.16, 7.2, 164.73] * 11,
        index=add_regio_gap_beyond_DLS_per_DLS.index,
        columns=add_regio_gap_beyond_DLS_per_DLS.columns,
    )

    if not conv_gap_mode:
        add_gap_trickle_down_regio = (
            (
                MISO2_2015_pop_R11.rename(columns={"2015": "value"})
                * 1000
                * per_capita_trickle_down_regio
            )
            - DLS_stocks_prov_noMats_regio_scale_materials
            - Beyond_DLS_stocks_regio_scale_materials
            - DLS_stocks_gap_noMats_regio_scale_materials
            - add_regio_gap_beyond_DLS_per_DLS
        )
    else:
        add_gap_trickle_down_regio = (
            (
                MISO2_2015_pop_R11.rename(columns={"2015": "value"})
                * 1000
                * per_capita_trickle_down_regio
            )
            - DLS_stocks_prov_noMats_regio_scale_materials
            - Beyond_DLS_stocks_regio_scale_materials
            - DLS_stocks_gap_noMats_regio_scale_materials.clip(0)
            - add_regio_gap_beyond_DLS_per_DLS
        )


    add_gap_trickle_down_regio.sum()
    if conv_gap_mode:
        add_gap_trickle_down_regio = add_gap_trickle_down_regio.clip(0)


    # assemble all stock elements for Figure 3
    regio_stock_dict = {
        "DLS_prov": DLS_stocks_prov_noMats_regio_scale_materials,
        "beyond_DLS": Beyond_DLS_stocks_regio_scale_materials,
        "gap_targeted": DLS_stocks_gap_noMats_regio_scale_materials,
        "gap_regional": add_regio_gap_beyond_DLS_per_DLS,
        "gap_trickle": add_gap_trickle_down_regio,
    }
    # Convert to a single DataFrame
    regio_stocks = pd.concat(regio_stock_dict, axis=1)
    regio_stocks.columns = regio_stocks.columns.droplevel(1)
    #! for some regions, cross-sectional trickle down stocks are smaller than already existing stocks leading to negative 'gap trickle' --> set zero
    if not conv_gap_mode:
        regio_stocks[regio_stocks < 0] = 0

    # calculate global stock items per capita (t/cap)
    regio_stocks_cap = regio_stocks.reset_index().merge(
        MISO2_2015_pop_R11.reset_index(), how="left", on="region"
    )
    (
        regio_stocks_cap["DLS_prov_cap"],
        regio_stocks_cap["beyond_DLS_cap"],
        regio_stocks_cap["gap_targeted_cap"],
        regio_stocks_cap["gap_regional_cap"],
        regio_stocks_cap["gap_trickle_cap"],
    ) = (
        regio_stocks_cap["DLS_prov"] / (regio_stocks_cap["2015"] * 1000),
        regio_stocks_cap["beyond_DLS"] / (regio_stocks_cap["2015"] * 1000),
        regio_stocks_cap["gap_targeted"] / (regio_stocks_cap["2015"] * 1000),
        regio_stocks_cap["gap_regional"] / (regio_stocks_cap["2015"] * 1000),
        regio_stocks_cap["gap_trickle"] / (regio_stocks_cap["2015"] * 1000),
    )
    regio_stocks_cap = regio_stocks_cap[
        [
            "region",
            "material",
            "DLS_prov_cap",
            "beyond_DLS_cap",
            "gap_targeted_cap",
            "gap_regional_cap",
            "gap_trickle_cap",
        ]
    ].set_index(["region", "material"])

    # check
    regio_stocks_cap.sum(axis=1)

    # aggregate to globe
    global_stocks = regio_stocks.groupby("material").sum()
    global_stocks_cap = global_stocks / (
        MISO2_2015_pop_country_subset.sum().sum() * 1000
    )
    global_stocks_cap = global_stocks_cap.rename(
        columns={
            "DLS_prov": "DLS_prov_cap",
            "beyond_DLS": "beyond_DLS_cap",
            "gap_targeted": "gap_targeted_cap",
            "gap_regional": "gap_regional_cap",
            "gap_trickle": "gap_trickle_cap",
        }
    )
    ## global_stocks will be different from global_stocks_prod (below) because of setting negative beyond DLS stocks to 0 - these negatives are more when including the dimensions products



    ### 2) REGIONAL - INCLUDING PRODUCTS DIMENSION

    # assemble data for REGIONAL bar plot at scale per material, product group

    # aggregate dimensions to groups
    aggr_dim = {
        "lamps": "other",
        "motor_vehicles": "transport_machinery",
        "other_transport": "transport_machinery",
        "textiles": "other",
    }
    DLS_stocks_gap_noMats_R11_scale_RPM_piv_harm = (
        DLS_stocks_gap_noMats_R11_scale_RPM_piv.reset_index()
        .replace(aggr_dim)
        .replace(R11_harmonize_stocks)
        .groupby(["region", "stock"])
        .sum()
    )
    DLS_stocks_gap_noMats_R11_scale_RPM_piv_harm.index.names = ["region", "sector"]

    # assemble all stock elements for Figure 3
    regio_stock_prod_dict = {
        "DLS_prov": DLS_stocks_prov_noMats_R11_scale_RPM_piv,
        "beyond_DLS": Beyond_DLS_stocks_R11_piv,
        "gap_targeted": DLS_stocks_gap_noMats_R11_scale_RPM_piv_harm,
    }

    # # Convert to a single DataFrame
    regio_stock_prod = pd.concat(
        regio_stock_prod_dict, axis=1, names=["region", "sector"]
    )

    #! for some regions, cross-sectional trickle down stocks are smaller than already existing stocks leading to negative 'gap trickle' --> set zero
    # negatives in terms of count and mass
    # region - material - product
    negative_count_bDLS_RMP = (Beyond_DLS_stocks_R11_piv < 0).sum().sum()
    total_count_bDLS_RMP = Beyond_DLS_stocks_R11_piv.size
    negative_mass_bDLS_RMP = (
        (Beyond_DLS_stocks_R11_piv[Beyond_DLS_stocks_R11_piv < 0]).sum().sum()
    )
    positive_mass_bDLS_RMP = (
        (Beyond_DLS_stocks_R11_piv[Beyond_DLS_stocks_R11_piv > 0]).sum().sum()
    )
    negative_on_positive_mass_bDLS_RMP = abs(
        negative_mass_bDLS_RMP / positive_mass_bDLS_RMP * 100
    )
    # region - product
    negative_count_bDLS_RP = (Beyond_DLS_stocks_R11_piv.sum(axis=1) < 0).sum().sum()
    total_count_bDLS_RP = Beyond_DLS_stocks_R11_piv.sum(axis=1).size
    negative_mass_bDLS_RP = (
        (
            Beyond_DLS_stocks_R11_piv.sum(axis=1)[
                Beyond_DLS_stocks_R11_piv.sum(axis=1) < 0
            ]
        )
        .sum()
        .sum()
    )
    positive_mass_bDLS_RP = (
        (
            Beyond_DLS_stocks_R11_piv.sum(axis=1)[
                Beyond_DLS_stocks_R11_piv.sum(axis=1) > 0
            ]
        )
        .sum()
        .sum()
    )
    negative_on_positive_mass_bDLS_RP = abs(
        negative_mass_bDLS_RP / positive_mass_bDLS_RP * 100
    )
    
       
    # we set negative values to zero here, removing negatives due to mismatches of bottom-up estimate of DLS stocks and top-down estimate of economy-wide stocks
    # if adding that to the stock gap again, existing economy-wide stocks might be higher due to removed negatives (slight deviation)
    regio_stock_prod[regio_stock_prod < 0] = 0
    
    # aggregate to globe
    global_stock_prod = regio_stock_prod.groupby(["sector"]).sum()

    # calculate global stock items per capita (t/cap)
    regio_stock_prod_cap = regio_stock_prod.copy()
    population_dict = MISO2_2015_pop_R11["2015"].to_dict()
    regio_stock_prod_cap["population"] = (
        regio_stock_prod_cap.reset_index()["region"].map(population_dict).to_list()
    )
    for col in regio_stock_prod_cap.columns:
        regio_stock_prod_cap[(col[0] + "_cap", col[1])] = regio_stock_prod_cap[col] / (
            regio_stock_prod_cap["population"] * 1000
        )
    regio_stock_prod_cap = regio_stock_prod_cap.iloc[:, 13:25]

    # check if correct against regio_stocks_cap
    

    
    # import numpy as np
    # rssum = regio_stock_prod_cap.sum().sum()
    # rescsum = regio_stocks_cap.sum().sum()
    #print(f"Regio stock prod cap sum: {rssum} vs {rescsum}")
    #assert np.isclose(rssum, rescsum) is True



    # 3) REGIONAL - INCLUDING PRODUCTS & DIMENSION

    ### assemble data for REGIONAL bar plot at scale per material, product group and dimension
    aggr_direct_indirect = {
        "clothing_i": "clothing",
        "communication_i": "communication",
        "educ_i": "education",
        "education_i": "education",
        "health_i": "health",
        "hh_appliance_i": "hh_appliance",
        "housing_i": "housing",
        "transport_i": "transport",
        "nutrition_i": "nutrition",
        "sanitation_i": "sanitation",
        "water_i": "water",
    }
    # harmonize dimensions, products
    DLS_stocks_prov_noMats_R11_scale_RDPM_piv_harm = (
        DLS_stocks_prov_noMats_R11_scale_RDPM_piv.reset_index()
        .replace(aggr_dim)
        .replace(R11_harmonize_stocks)
        .replace(aggr_direct_indirect)
        .set_index(["region", "dimension", "stock"])
        .groupby(["region", "dimension", "stock"])
        .sum()
    )
    DLS_stocks_thresh_noMats_R11_scale_RDPM_piv_harm = (
        DLS_stocks_thresh_noMats_R11_scale_RDPM_piv.reset_index()
        .replace(aggr_dim)
        .replace(R11_harmonize_stocks)
        .replace(aggr_direct_indirect)
        .set_index(["region", "dimension", "stock"])
        .groupby(["region", "dimension", "stock"])
        .sum()
    )

    # calc DLS stock gap including all data dimensions
    DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm = (
        DLS_stocks_thresh_noMats_R11_scale_RDPM_piv_harm
        - DLS_stocks_prov_noMats_R11_scale_RDPM_piv_harm
    )

    ###! set negatives in gap zero (which occur in case that the provided DLS are higher than the ones required in converging practices scenario)
    # - however, these won't enter the calculations as reducing the stock gap because it's unlikely that old DLS stocks are abandoned in the short/medium term
    # - the gap is thus only the positive values
    # DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm = DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm[DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm>0]    
    if not conv_gap_mode:
        DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm = (
        DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm.clip(0)
    )

    # assemble in one dataframe
    regio_stock_prod_dim_dict = {
        "DLS_prov": DLS_stocks_prov_noMats_R11_scale_RDPM_piv_harm,
        "gap_targeted": DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm,
    }
    regio_stock_prod_dim = pd.concat(regio_stock_prod_dim_dict, axis=1)
    regio_stock_prod_dim.index.names = ["region", "dimension", "sector"]

    # aggregate dimensions for data SI of material stock gap
    # material stock gap per material, product, dimension (global and regional, absolute)
    aggr_dim = {
        "communication": "Communication",
        "clothing": "Shelter",
        "cond_i": "Shelter",
        "education": "Education",
        "health": "Health",
        "hh_appliance": "Nutrition",
        "housing": "Shelter",
        "nutrition": "Nutrition",
        "sanitation": "Sanitation",
        "transport": "Mobility",
        "water": "Water",
    }
    DLS_stocks_gap_noMats_R11_scale_RDPM = (
        DLS_stocks_gap_noMats_R11_scale_RDPM_piv_harm.reset_index().melt(
            id_vars=["region", "dimension", "stock"], var_name="material"
        )
    )
    DLS_stocks_gap_noMats_R11_scale_RDPM = DLS_stocks_gap_noMats_R11_scale_RDPM.replace(
        aggr_dim
    )
    DLS_stocks_gap_noMats_R11_scale_RDPM.set_index(
        ["region", "dimension", "stock", "material"], inplace=True
    )

    # check that equivalent to regio_stock_prod.groupby('sector').sum() (as we cannot calculate beyond DLS stocks per DLS dimension and thus not take this one as base df)
    # regio_stock_prod_dim.groupby('sector').sum()

    # calculate global stock items per capita (t/cap)
    regio_stock_prod_dim_cap = regio_stock_prod_dim.copy()
    regio_stock_prod_dim_cap["population"] = (
        regio_stock_prod_dim_cap.reset_index()["region"].map(population_dict).to_list()
    )
    for col in regio_stock_prod_dim_cap.columns:
        regio_stock_prod_dim_cap[(col[0] + "_cap", col[1])] = regio_stock_prod_dim_cap[
            col
        ] / (regio_stock_prod_dim_cap["population"] * 1000)
    # only keep columns with per capita values
    # TODO are we sure these are correct columns?
    regio_stock_prod_dim_cap = regio_stock_prod_dim_cap.iloc[:, 9:17]

    # check if correct against regio_stocks_cap
    # regio_stock_prod_dim_cap.groupby('region').sum()

    # check that regional, global and diff dimensional resolutions come to same result
    # global_stocks and regio_Stocks
    # global and regional correspondence per resolution pairs

    #print(f"GS vs RS: {global_stocks} vs {regio_stocks.groupby('material').sum()}")
    #print(f"GSPD vs RGSPD: {global_stock_prod} vs {regio_stock_prod.groupby('sector').sum()}")
    #assert np.isclose(global_stocks, regio_stocks.groupby("material").sum()) is True, "global and regional results do not match"
    #assert np.isclose(global_stock_prod, regio_stock_prod.groupby("sector").sum()) is True, "global and regional results do not match"
    # over resolution pairs
    # these match
    global_stocks_prod_dim = regio_stock_prod_dim.groupby(["dimension", "sector"]).sum()

    aggr_dim = {
        "communication": "Socialization",
        "clothing": "Shelter",
        "cond_i": "Shelter",
        "education": "Socialization",
        "health": "Health",
        "hh_appliance": "Nutrition",
        "housing": "Shelter",
        "nutrition": "Nutrition",
        "sanitation": "Health",
        "transport": "Mobility",
        "water": "Health",
    }
    global_stocks_prod_dim = global_stocks_prod_dim.rename(
        index=aggr_dim, level="dimension"
    )
    global_stocks_prod_dim = global_stocks_prod_dim.groupby(
        ["dimension", "sector"]
    ).sum()
    global_stocks_prod_dim_thresh = pd.DataFrame(global_stocks_prod_dim.sum(axis=1))
    global_stocks_prod_dim_thresh.columns = ["DLSstock_threshold"]
    global_stocks_prod_dim_thresh_cap = global_stocks_prod_dim_thresh / (
        MISO2_2015_pop_country_subset.sum().sum() * 1000
    )
    global_stocks_dim_thresh_cap = global_stocks_prod_dim_thresh_cap.groupby(
        "dimension"
    ).sum()
    global_stocks_dim_thresh_cap.columns = ["DLSstock_threshold"]


    """ ############################################################
             #7 PREPARE DATA: CONSTRUCTION SPEED TO CLOSE DLS GAPS
         ############################################################"""

    ### EXTRAPOLATE NEW CONSTRUCTION SPEED (NAS) FOR SCENARIOS TO FILL DLS GAP

    #### NAS assembly for plot
    MISO2_NAS_global_cap_4concat = MISO2_NAS_global_cap.copy()
    MISO2_NAS_global_cap_4concat.insert(0, "region", "Global")
    MISO2_NAS_global_cap_4concat = MISO2_NAS_global_cap_4concat.reset_index().set_index(
        ["region", "year"]
    )

    # concat GLOBAL and R11 NAS per capita
    NAS_all = pd.concat(
        [MISO2_NAS_subset_R11_piv_materials_cap, MISO2_NAS_global_cap_4concat]
    )

    # years to extrapolate
    additional_years = list(range(2017, 2051))

    # for China, reasoning of Wiedenhofer et al. (2021) taken up: -1% per annum on NAS per capita
    # # Apply the extrapolation function; this is per capita
    NAS_all_histprosp = extrapolate_NAS(
        NAS_all[NAS_all.index.get_level_values(1).astype(int) > 2004], additional_years
    )
    NAS_all_histprosp = pd.concat(
        [
            NAS_all_histprosp,
            NAS_all[
                NAS_all.index.get_level_values(1)
                .astype(int)
                .isin(list(range(1970, 2005)))
            ],
        ]
    )
    NAS_all_histprosp.reset_index(inplace=True)
    NAS_all_histprosp.year = NAS_all_histprosp.year.astype(int)
    NAS_all_histprosp.set_index(["region", "year"], inplace=True)
    NAS_all_histprosp.sort_index(inplace=True)

    ### rename historical and extrapolated prospective NAS
    # NAS_all_histprosp in per capita values
    NAS_all_histprosp = NAS_all_histprosp.rename(
        columns={
            "biomass_cap": "biomass",
            "fossils_cap": "fossils",
            "metals_cap": "metals",
            "minerals_cap": "minerals",
        }
    )
    NAS_all_histprosp["total"] = (
        NAS_all_histprosp["biomass"]
        + NAS_all_histprosp["fossils"]
        + NAS_all_histprosp["metals"]
        + NAS_all_histprosp["minerals"]
    )
    # NAS_all_histprosp['total'].reset_index().set_index('region').T.plot(legend=False)
    NAS_all_hist = NAS_all_histprosp[
        NAS_all_histprosp.index.get_level_values(1).astype(int) < 2017
    ]

    ### 7-1 FOR WORLD REGIONAL SPEED

    #### CALCULATE DLS STOCK TRAJECTORY AT ASSUMED CONSTRUCTION SPEED (NAS)
    # prepare regional initital DLS stock levels in 2015 and requirements to close DLS gaps
    regio_DLSstocks_cap_curr = pd.DataFrame(
        regio_stocks_cap[["DLS_prov_cap"]].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    regio_stocks_cap_curr = pd.DataFrame(
        regio_stocks_cap[["DLS_prov_cap", "beyond_DLS_cap"]].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    regio_stocks_cap_targeted = pd.DataFrame(
        regio_stocks_cap[["DLS_prov_cap", "gap_targeted_cap"]].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    regio_stocks_cap_regspill = pd.DataFrame(
        regio_stocks_cap[
            ["DLS_prov_cap", "beyond_DLS_cap", "gap_targeted_cap", "gap_regional_cap"]
        ].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    regio_stocks_cap_globspill = pd.DataFrame(
        regio_stocks_cap[
            [
                "DLS_prov_cap",
                "beyond_DLS_cap",
                "gap_targeted_cap",
                "gap_regional_cap",
                "gap_trickle_cap",
            ]
        ].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)

    # prepare global initital DLS stock levels in 2015 and requirements to close DLS gaps
    global_stocks_cap_index = global_stocks_cap.copy()
    global_stocks_cap_index.insert(0, "region", "Global")
    global_stocks_cap_index.reset_index(inplace=True)
    global_stocks_cap_index.set_index(["region", "material"], inplace=True)
    global_DLSstocks_cap_curr = pd.DataFrame(
        global_stocks_cap_index[["DLS_prov_cap"]].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    global_stocks_cap_curr = pd.DataFrame(
        global_stocks_cap_index[["DLS_prov_cap", "beyond_DLS_cap"]].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    global_stocks_cap_targeted = pd.DataFrame(
        global_stocks_cap_index[["DLS_prov_cap", "gap_targeted_cap"]].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    global_stocks_cap_regspill = pd.DataFrame(
        global_stocks_cap_index[
            ["DLS_prov_cap", "beyond_DLS_cap", "gap_targeted_cap", "gap_regional_cap"]
        ].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)
    global_stocks_cap_globspill = pd.DataFrame(
        global_stocks_cap_index[
            [
                "DLS_prov_cap",
                "beyond_DLS_cap",
                "gap_targeted_cap",
                "gap_regional_cap",
                "gap_trickle_cap",
            ]
        ].sum(axis=1)
    ).pivot_table(index=["region"], columns="material", values=0)

    # add global to regional
    regio_DLSstocks_cap_curr = regio_DLSstocks_cap_curr._append(
        global_DLSstocks_cap_curr
    )
    regio_stocks_cap_curr = regio_stocks_cap_curr._append(global_stocks_cap_curr)
    regio_stocks_cap_targeted = regio_stocks_cap_targeted._append(
        global_stocks_cap_targeted
    )
    regio_stocks_cap_regspill = regio_stocks_cap_regspill._append(
        global_stocks_cap_regspill
    )
    regio_stocks_cap_globspill = regio_stocks_cap_globspill._append(
        global_stocks_cap_globspill
    )

    # format dataframes above and add value for sum over all materials
    for df in [
        regio_DLSstocks_cap_curr,
        regio_stocks_cap_curr,
        regio_stocks_cap_targeted,
        regio_stocks_cap_regspill,
        regio_stocks_cap_globspill,
    ]:
        df.insert(0, "year", "2015")
        df.reset_index(inplace=True)
        df.set_index(["region", "year"], inplace=True)
        df["total"] = df["biomass"] + df["fossils"] + df["metals"] + df["minerals"]

    ## assemble propsective DLS stock timeseries with extrapolated NAS, starting from 2015 DLS stock
    # prospective NAS per capita
    NAS_all_prosp = NAS_all_histprosp[
        NAS_all_histprosp.index.get_level_values(1).astype(int) > 2015
    ].copy()
    #! set negative NAS to 0 (assuming that no negative NAS exist)
    NAS_all_prosp[NAS_all_prosp < 0] = 0

    # initital DLS + beyond DLS stocks in 2015 + NAS trajectory
    NAS_stock_prosp = pd.concat(
        [
            regio_stocks_cap_curr,
            NAS_all_prosp.rename(
                columns={
                    "biomass_cap": "biomass",
                    "fossils_cap": "fossils",
                    "metals_cap": "metals",
                    "minerals_cap": "minerals",
                }
            ),
        ]
    )
    NAS_stock_prosp["total"] = (
        NAS_stock_prosp["biomass"]
        + NAS_stock_prosp["fossils"]
        + NAS_stock_prosp["metals"]
        + NAS_stock_prosp["minerals"]
    )

    # initital DLS stocks in 2015 + NAS trajectory
    NAS_DLS_stock_prosp = pd.concat(
        [
            regio_DLSstocks_cap_curr,
            NAS_all_prosp.rename(
                columns={
                    "biomass_cap": "biomass",
                    "fossils_cap": "fossils",
                    "metals_cap": "metals",
                    "minerals_cap": "minerals",
                }
            ),
        ]
    )
    NAS_DLS_stock_prosp["total"] = (
        NAS_DLS_stock_prosp["biomass"]
        + NAS_DLS_stock_prosp["fossils"]
        + NAS_DLS_stock_prosp["metals"]
        + NAS_DLS_stock_prosp["minerals"]
    )

    ### 7-2 FOR GLOBAL SPEED

    # (set as per capita global speed for each world region)
    # initital DLS + beyond-DLS stocks in 2015 + NAS trajectory at global speed
    NAS_stock_prosp_GlobSpeed = NAS_stock_prosp[
        NAS_stock_prosp.index.get_level_values(0) == "Global"
    ]
    NAS_stock_prosp_GlobSpeed.index = [
        NAS_stock_prosp_GlobSpeed.index.get_level_values(0),
        NAS_stock_prosp_GlobSpeed.index.get_level_values(1).astype(int),
    ]
    NAS_stock_prosp_allGlobSpeed = NAS_stock_prosp.copy()
    NAS_stock_prosp_allGlobSpeed.index = [
        NAS_stock_prosp_allGlobSpeed.index.get_level_values(0),
        NAS_stock_prosp_allGlobSpeed.index.get_level_values(1).astype(int),
    ]
    NAS_stock_prosp_allGlobSpeed = NAS_stock_prosp_allGlobSpeed[
        NAS_stock_prosp_allGlobSpeed.index.get_level_values(0) != "Global"
    ]

    # Identify the years greater than 2015 in the first DataFrame
    mask = NAS_stock_prosp_allGlobSpeed.index.get_level_values(1) > 2015

    # Update the values in "NAS_stock_prosp_allGlobSpeed" for years > 2015 with values from df2
    for region in NAS_stock_prosp_allGlobSpeed.index.get_level_values(0).unique():
        mask_allGlobSpeed = (
            NAS_stock_prosp_allGlobSpeed.index.get_level_values("region") == region
        ) & (NAS_stock_prosp_allGlobSpeed.index.get_level_values("year") > 2015)
        NAS_stock_prosp_allGlobSpeed.loc[
            mask_allGlobSpeed, ["biomass", "fossils", "metals", "minerals", "total"]
        ] = NAS_stock_prosp_GlobSpeed.loc[
            NAS_stock_prosp_GlobSpeed.index.get_level_values("year") > 2015,
            ["biomass", "fossils", "metals", "minerals", "total"],
        ].values

    # initital DLS stocks in 2015 + NAS trajectory at global speed
    NAS_DLS_stock_prosp_GlobSpeed = NAS_DLS_stock_prosp[
        NAS_DLS_stock_prosp.index.get_level_values(0) == "Global"
    ]
    NAS_DLS_stock_prosp_GlobSpeed.index = [
        NAS_DLS_stock_prosp_GlobSpeed.index.get_level_values(0),
        NAS_DLS_stock_prosp_GlobSpeed.index.get_level_values(1).astype(int),
    ]
    NAS_DLS_stock_prosp_allGlobSpeed = NAS_DLS_stock_prosp.copy()
    NAS_DLS_stock_prosp_allGlobSpeed.index = [
        NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values(0),
        NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values(1).astype(int),
    ]
    NAS_DLS_stock_prosp_allGlobSpeed = NAS_DLS_stock_prosp_allGlobSpeed[
        NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values(0) != "Global"
    ]

    # Identify the years greater than 2015 in the first DataFrame
    mask = NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values(1) > 2015

    # Update the values in "NAS_DLS_stock_prosp_allGlobSpeed" for years > 2015 with values from df2
    for region in NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values(0).unique():
        mask_allGlobSpeed = (
            NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values("region") == region
        ) & (NAS_DLS_stock_prosp_allGlobSpeed.index.get_level_values("year") > 2015)
        NAS_DLS_stock_prosp_allGlobSpeed.loc[
            mask_allGlobSpeed, ["biomass", "fossils", "metals", "minerals", "total"]
        ] = NAS_DLS_stock_prosp_GlobSpeed.loc[
            NAS_DLS_stock_prosp_GlobSpeed.index.get_level_values("year") > 2015,
            ["biomass", "fossils", "metals", "minerals", "total"],
        ].values

    """ ####################################################################################
             #7A CALCULATE DLS REACHED - CONTINUANCE OF WORLD -- REGIONAL -- CONSTRUCTION SPEED
         ###################################################################################"""

    ## #7A-1 CALCULATE DLS REACHED IN WORLD-REGIONS - IF ALL CONSTRUCTION FOCUSED ON DLS ONLY (TARGETED)

    df = NAS_DLS_stock_prosp.copy()
    df = df.sort_index()
    # calculate cumulative sum for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum = df.groupby(level="region").cumsum()
    # concat with stocks required for targeted closure of DLS gap
    df_cumsum_rel_target = (
        df_cumsum.reset_index()
        .merge(regio_stocks_cap_targeted, on="region")
        .set_index(["region", "year"])
    )
    # calculate relative achievement of DLS for all per material
    df_cumsum_rel_target["biomass_target"] = (
        df_cumsum_rel_target["biomass_x"] / df_cumsum_rel_target["biomass_y"]
    )
    df_cumsum_rel_target["fossils_target"] = (
        df_cumsum_rel_target["fossils_x"] / df_cumsum_rel_target["fossils_y"]
    )
    df_cumsum_rel_target["metals_target"] = (
        df_cumsum_rel_target["metals_x"] / df_cumsum_rel_target["metals_y"]
    )
    df_cumsum_rel_target["minerals_target"] = (
        df_cumsum_rel_target["minerals_x"] / df_cumsum_rel_target["minerals_y"]
    )
    df_cumsum_rel_target["total_target"] = (
        df_cumsum_rel_target["total_x"] / df_cumsum_rel_target["total_y"]
    )
    df_cumsum_rel_target = df_cumsum_rel_target[
        [
            "total_target",
            "biomass_target",
            "fossils_target",
            "metals_target",
            "minerals_target",
        ]
    ]

    ## #7A-2 CALCULATE DLS REACHED FOR THE GLOBE - IF ALL CONSTRUCTION FOCUSED ON DLS ONLY (TARGETED)

    # calculate cumulative sum for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum_rel_target_toGlobe = NAS_DLS_stock_prosp.groupby(level="region").cumsum()
    # merge with required stocks to fill DLS gap
    df_cumsum_rel_target_toGlobe = (
        df_cumsum_rel_target_toGlobe.reset_index()
        .merge(regio_stocks_cap_targeted, on="region")
        .set_index(["region", "year"])
    )
    ## calculate relative target reached (per capita)
    df_cumsum_rel_target_toGlobe["biomass_target"] = (
        df_cumsum_rel_target_toGlobe["biomass_x"]
        / df_cumsum_rel_target_toGlobe["biomass_y"]
    )
    df_cumsum_rel_target_toGlobe["fossils_target"] = (
        df_cumsum_rel_target_toGlobe["fossils_x"]
        / df_cumsum_rel_target_toGlobe["fossils_y"]
    )
    df_cumsum_rel_target_toGlobe["metals_target"] = (
        df_cumsum_rel_target_toGlobe["metals_x"]
        / df_cumsum_rel_target_toGlobe["metals_y"]
    )
    df_cumsum_rel_target_toGlobe["minerals_target"] = (
        df_cumsum_rel_target_toGlobe["minerals_x"]
        / df_cumsum_rel_target_toGlobe["minerals_y"]
    )
    df_cumsum_rel_target_toGlobe["total_target"] = (
        df_cumsum_rel_target_toGlobe["total_x"]
        / df_cumsum_rel_target_toGlobe["total_y"]
    )
    df_cumsum_rel_target_toGlobe = df_cumsum_rel_target_toGlobe[
        [
            "total_target",
            "biomass_target",
            "fossils_target",
            "metals_target",
            "minerals_target",
        ]
    ]
    # bound DLS reached at 100% (1)
    df_cumsum_rel_target_toGlobe[df_cumsum_rel_target_toGlobe > 1] = 1

    ### we have two different elements here:
    # global average gap with global average speed assumed that DLS stock thresholds are globally equal (DLS_trajectory_targeted_globalAV)
    # regional gaps with sum of regional speeds takes into account the different needs and speeds of the regions (DLS_trajectory_targeted_regional_specifics below)
    # in the base year (2015) this leads to a different figure of DLS reached! (which makes sense as the one is average, the other includes distribution)
    # the 'correct' one is including distribution (dala2 below, which depicts global DLS gap closure at regional speeds, considering regional DLS stock threshold needs)

    # calc 'global' population for country set
    MISO2_2015_pop_R11_glob = MISO2_2015_pop_R11.copy()
    MISO2_2015_pop_R11_glob.loc["Global"] = MISO2_2015_pop_R11_glob.sum()
    # aggregate to globe by weighting with population
    df_cumsum_rel_target_Regional = df_cumsum_rel_target_toGlobe.copy()
    df_cumsum_rel_target_toGlobe = (
        df_cumsum_rel_target_toGlobe.reset_index()
        .merge(MISO2_2015_pop_R11_glob, on=["region"])
        .set_index(["region", "year"])
    )
    df_cumsum_rel_target_toGlobe = (
        df_cumsum_rel_target_toGlobe[
            [
                "total_target",
                "biomass_target",
                "fossils_target",
                "metals_target",
                "minerals_target",
            ]
        ]
        * df_cumsum_rel_target_toGlobe[["2015"]].to_numpy()
    )

    # results with one global aggregate average (all DLS thresholds in regional are the same = globalAV) and one taking regional DLS tresholds into account (.._regional_specifics)
    DLS_trajectory_targeted_regional_specifics = (
        df_cumsum_rel_target_toGlobe[
            df_cumsum_rel_target_toGlobe.index.get_level_values(0) != "Global"
        ]
        .groupby("year")
        .sum()
        / MISO2_2015_pop_R11.sum().sum()
    )
    DLS_trajectory_targeted_globalAV = (
        df_cumsum_rel_target_toGlobe[
            df_cumsum_rel_target_toGlobe.index.get_level_values(0) == "Global"
        ]
        / MISO2_2015_pop_R11.sum().sum()
    )

    ## #7A-3 CALCULATE DLS REACHED FOR THE GLOBE - IF SOME CONSTRUCTION FOCUSED ON BEYOND-DLS TOO (regional ratios of beyond-DLS to DLS stocks in 2015)

    # calculate cumulative sum of DLS and beyond DLS stocks + new construction for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum_rel_regio_toGlobe = NAS_stock_prosp.groupby(level="region").cumsum()
    # merge with required stocks to fill DLS gap (including beyond-DLS stocks according to regional ratios of beyond-DLS to DLS stocks in 2015)
    df_cumsum_rel_regio_toGlobe = (
        df_cumsum_rel_regio_toGlobe.reset_index()
        .merge(regio_stocks_cap_regspill, on="region")
        .set_index(["region", "year"])
    )
    ## calculate relative target reached (per capita)
    df_cumsum_rel_regio_toGlobe["biomass_regio"] = (
        df_cumsum_rel_regio_toGlobe["biomass_x"]
        / df_cumsum_rel_regio_toGlobe["biomass_y"]
    )
    df_cumsum_rel_regio_toGlobe["fossils_regio"] = (
        df_cumsum_rel_regio_toGlobe["fossils_x"]
        / df_cumsum_rel_regio_toGlobe["fossils_y"]
    )
    df_cumsum_rel_regio_toGlobe["metals_regio"] = (
        df_cumsum_rel_regio_toGlobe["metals_x"]
        / df_cumsum_rel_regio_toGlobe["metals_y"]
    )
    df_cumsum_rel_regio_toGlobe["minerals_regio"] = (
        df_cumsum_rel_regio_toGlobe["minerals_x"]
        / df_cumsum_rel_regio_toGlobe["minerals_y"]
    )
    df_cumsum_rel_regio_toGlobe["total_regio"] = (
        df_cumsum_rel_regio_toGlobe["total_x"] / df_cumsum_rel_regio_toGlobe["total_y"]
    )
    df_cumsum_rel_regio_toGlobe = df_cumsum_rel_regio_toGlobe[
        [
            "total_regio",
            "biomass_regio",
            "fossils_regio",
            "metals_regio",
            "minerals_regio",
        ]
    ]
    # bound DLS reached at 100% (1)
    df_cumsum_rel_regio_toGlobe[df_cumsum_rel_regio_toGlobe > 1] = 1
    # aggregate to globe by weighting with population
    df_cumsum_rel_regio_toGlobe = (
        df_cumsum_rel_regio_toGlobe.reset_index()
        .merge(MISO2_2015_pop_R11, on=["region"])
        .set_index(["region", "year"])
    )
    df_cumsum_rel_regio_toGlobe_weighted = (
        df_cumsum_rel_regio_toGlobe[
            [
                "total_regio",
                "biomass_regio",
                "fossils_regio",
                "metals_regio",
                "minerals_regio",
            ]
        ]
        * df_cumsum_rel_regio_toGlobe[["2015"]].to_numpy()
    )
    # results with regional DLS tresholds (.._regional_specifics)
    DLS_trajectory_somebDLS_regional_specifics = (
        df_cumsum_rel_regio_toGlobe_weighted.groupby("year").sum()
        / MISO2_2015_pop_R11.sum().sum()
    )

    ## #7A-4 CALCULATE DLS REACHED FOR THE GLOBE - IF MUCH CONSTRUCTION FOCUSED ON BEYOND-DLS TOO (global trickle-down beyond-DLS stocks in 2015)
    # calculate cumulative sum for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum_rel_glob_toGlobe = NAS_stock_prosp.groupby(level="region").cumsum()

    # calculate the stock intensity of achievieng 1% more DLS reached on basis on 2015 value for %DLS reached
    # this is required to calculate %DLS improvement (as there is no basis in 2015 stocks representing the global trickle down which would allow for setting
    # prospective new construction (NAS) into relation to %DLS imrpovement)
    DLS_bDLS_stock_2015 = df_cumsum_rel_glob_toGlobe[
        df_cumsum_rel_glob_toGlobe.index.get_level_values(1) == "2015"
    ]
    DLS_bDLS_stock_2015.index = [
        DLS_bDLS_stock_2015.index.get_level_values(0),
        DLS_bDLS_stock_2015.index.get_level_values(1).astype(int),
    ]
    DLS_bDLS_stock_2015 = DLS_bDLS_stock_2015.drop("Global")
    stock_target_2015 = regio_stocks_cap_globspill.drop("Global")
    stock_target_2015.index = [
        stock_target_2015.index.get_level_values(0),
        stock_target_2015.index.get_level_values(1).astype(int),
    ]

    DLS_reached_2015 = (
        df_cumsum_rel_target[df_cumsum_rel_target.index.get_level_values(1) == "2015"]
        .drop("Global")
        .rename(
            columns={
                "total_target": "total",
                "biomass_target": "biomass",
                "fossils_target": "fossils",
                "metals_target": "metals",
                "minerals_target": "minerals",
            }
        )
    )
    if conv_gap_mode:
        DLS_reached_2015 = DLS_reached_2015.clip(0,1)


    DLS_reached_2015.index = [
        DLS_reached_2015.index.get_level_values(0),
        DLS_reached_2015.index.get_level_values(1).astype(int),
    ]

    if not conv_gap_mode:
        intens_stock_percDLS = (stock_target_2015 - DLS_bDLS_stock_2015) / (1 - DLS_reached_2015)
    else:
        intens_stock_percDLS = (stock_target_2015 - DLS_bDLS_stock_2015).clip(0) / (
        1 - DLS_reached_2015).replace(0, np.nan)

    # rename prospective DLS stock data
    rename_df_cumsum_rel_glob_toGlobe = {
        "total": "total_DLS_bDLS_stock",
        "biomass": "biomass_DLS_bDLS_stock",
        "fossils": "fossils_DLS_bDLS_stock",
        "metals": "metals_DLS_bDLS_stock",
        "minerals": "minerals_DLS_bDLS_stock",
    }
    df_cumsum_rel_glob_toGlobe = df_cumsum_rel_glob_toGlobe.rename(
        columns=rename_df_cumsum_rel_glob_toGlobe
    )
    # add columns on stock intensity of achievieng 1% more DLS reached on basis on 2015 value %DLS reached
    rename_intens_stock_percDLS = {
        "total": "total_intens_per%DLS",
        "biomass": "biomass_intens_per%DLS",
        "fossils": "fossils_intens_per%DLS",
        "metals": "metals_intens_per%DLS",
        "minerals": "minerals_intens_per%DLS",
    }
    df_cumsum_rel_glob_toGlobe = (
        df_cumsum_rel_glob_toGlobe.reset_index()
        .merge(
            intens_stock_percDLS.rename(columns=rename_intens_stock_percDLS),
            on="region",
            how="left",
        )
        .set_index(["region", "year"])
    )
    # add columns for starting DLS stock level in 2015
    rename_DLS_bDLS_stock_2015 = {
        "total": "total_DLS_bDLS_stock_2015",
        "biomass": "biomass_DLS_bDLS_stock_2015",
        "fossils": "fossils_DLS_bDLS_stock_2015",
        "metals": "metals_DLS_bDLS_stock_2015",
        "minerals": "minerals_DLS_bDLS_stock_2015",
    }
    df_cumsum_rel_glob_toGlobe = (
        df_cumsum_rel_glob_toGlobe.reset_index()
        .merge(
            DLS_bDLS_stock_2015.rename(columns=rename_DLS_bDLS_stock_2015),
            on="region",
            how="left",
        )
        .set_index(["region", "year"])
    )

    # calculate gap closure in DLS% = [DLS_bDLS_stock(year t) - DLS_bDLS_stock(year 2015)] / stock intensity of %DLS change = change in % DLS reached by new construction
    df_cumsum_rel_glob_toGlobe["total_%closed"] = (
        df_cumsum_rel_glob_toGlobe["total_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe["total_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe["total_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe["biomass_%closed"] = (
        df_cumsum_rel_glob_toGlobe["biomass_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe["biomass_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe["biomass_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe["metals_%closed"] = (
        df_cumsum_rel_glob_toGlobe["metals_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe["metals_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe["metals_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe["fossils_%closed"] = (
        df_cumsum_rel_glob_toGlobe["fossils_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe["fossils_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe["fossils_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe["minerals_%closed"] = (
        df_cumsum_rel_glob_toGlobe["minerals_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe["minerals_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe["minerals_intens_per%DLS"]

    if conv_gap_mode:
        df_cumsum_rel_glob_toGlobe = df_cumsum_rel_glob_toGlobe.replace(np.nan, 0)


    # calculate total DLS% by adding start value of %DLS reached from 2015
    rename_DLS_reached_2015 = {
        "total": "total_reached_2015",
        "biomass": "biomass_reached_2015",
        "fossils": "fossils_reached_2015",
        "metals": "metals_reached_2015",
        "minerals": "minerals_reached_2015",
    }
    df_cumsum_rel_glob_toGlobe = (
        df_cumsum_rel_glob_toGlobe.reset_index()
        .merge(
            DLS_reached_2015.rename(columns=rename_DLS_reached_2015),
            on="region",
            how="left",
        )
        .set_index(["region", "year"])
    )
    df_cumsum_rel_glob_toGlobe["total_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe["total_reached_2015"]
        + df_cumsum_rel_glob_toGlobe["total_%closed"]
    )
    df_cumsum_rel_glob_toGlobe["biomass_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe["biomass_reached_2015"]
        + df_cumsum_rel_glob_toGlobe["biomass_%closed"]
    )
    df_cumsum_rel_glob_toGlobe["metals_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe["metals_reached_2015"]
        + df_cumsum_rel_glob_toGlobe["metals_%closed"]
    )
    df_cumsum_rel_glob_toGlobe["fossils_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe["fossils_reached_2015"]
        + df_cumsum_rel_glob_toGlobe["fossils_%closed"]
    )
    df_cumsum_rel_glob_toGlobe["minerals_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe["minerals_reached_2015"]
        + df_cumsum_rel_glob_toGlobe["minerals_%closed"]
    )

    # only keep goal columns for %DLS_reached and set DLS limit to 1 (100% reached)
    df_cumsum_rel_glob_toGlobe = df_cumsum_rel_glob_toGlobe[
        [
            "total_%reached_globGap",
            "biomass_%reached_globGap",
            "metals_%reached_globGap",
            "fossils_%reached_globGap",
            "minerals_%reached_globGap",
        ]
    ]
    df_cumsum_rel_glob_toGlobe[df_cumsum_rel_glob_toGlobe > 1] = 1

    # calculate weighted global reaching of DLS targets, based on regional trajectories
    df_cumsum_rel_glob_toGlobe = (
        df_cumsum_rel_glob_toGlobe.reset_index()
        .merge(MISO2_2015_pop_R11, on=["region"])
        .set_index(["region", "year"])
    )
    df_cumsum_rel_glob_toGlobe_weighted = (
        df_cumsum_rel_glob_toGlobe * df_cumsum_rel_glob_toGlobe[["2015"]].to_numpy()
    )
    df_cumsum_rel_glob_toGlobe_weighted_rel = (
        df_cumsum_rel_glob_toGlobe_weighted.groupby("year").sum()
        / MISO2_2015_pop_R11.sum().sum()
    )
    DLS_trajectory_manybDLS_regional_specifics = (
        df_cumsum_rel_glob_toGlobe_weighted_rel.drop("2015", axis=1)
    )

    ## #7A-5 ASSEMBLE GLOBAL TRAJECTORIES @ REGIONAL SPEED

    df_global_regional_speed = DLS_trajectory_targeted_regional_specifics.merge(
        DLS_trajectory_somebDLS_regional_specifics, on=["year"]
    ).merge(DLS_trajectory_manybDLS_regional_specifics, on=["year"])
    df_global_regional_speed.index = df_global_regional_speed.index.astype(int)
    df_global_regional_speed = df_global_regional_speed.sort_index()

    """ ####################################################################################
             #7B CALCULATE DLS REACHED - CONTINUANCE OF WORLD -- GLOBAL -- CONSTRUCTION SPEED
         ###################################################################################"""
    #### repeat to calculate scenarios from above for global speed

    # (starting from 7B-2 for correspondence to 7A steps)

    ## #7B-2 CALCULATE DLS REACHED FOR THE GLOBE - IF ALL CONSTRUCTION FOCUSED ON DLS ONLY (TARGETED)

    # calculate cumulative sum for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum_rel_target_toGlobe_allGlobSpeed = (
        NAS_DLS_stock_prosp_allGlobSpeed.groupby(level="region").cumsum()
    )
    # merge with required stocks to fill DLS gap (including beyond-DLS stocks according to regional ratios of beyond-DLS to DLS stocks in 2015)
    df_cumsum_rel_target_toGlobe_allGlobSpeed = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed.reset_index()
        .merge(regio_stocks_cap_targeted, on="region")
        .set_index(["region", "year"])
    )

    ## calculate relative target reached (per capita)
    df_cumsum_rel_target_toGlobe_allGlobSpeed["biomass_target"] = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed["biomass_x"]
        / df_cumsum_rel_target_toGlobe_allGlobSpeed["biomass_y"]
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed["fossils_target"] = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed["fossils_x"]
        / df_cumsum_rel_target_toGlobe_allGlobSpeed["fossils_y"]
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed["metals_target"] = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed["metals_x"]
        / df_cumsum_rel_target_toGlobe_allGlobSpeed["metals_y"]
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed["minerals_target"] = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed["minerals_x"]
        / df_cumsum_rel_target_toGlobe_allGlobSpeed["minerals_y"]
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed["total_target"] = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed["total_x"]
        / df_cumsum_rel_target_toGlobe_allGlobSpeed["total_y"]
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed[
            [
                "total_target",
                "biomass_target",
                "fossils_target",
                "metals_target",
                "minerals_target",
            ]
        ]
    )

    # bound DLS reached at 100% (1)
    df_cumsum_rel_target_toGlobe_allGlobSpeed[
        df_cumsum_rel_target_toGlobe_allGlobSpeed > 1
    ] = 1

    # aggregate to globe by weighting with population
    MISO2_2015_pop_R11_glob = MISO2_2015_pop_R11.copy()
    MISO2_2015_pop_R11_glob.loc["Global"] = MISO2_2015_pop_R11_glob.sum()
    df_cumsum_rel_target_Regional_allGlobSpeed = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed.copy()
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed.reset_index()
        .merge(MISO2_2015_pop_R11_glob, on=["region"])
        .set_index(["region", "year"])
    )
    df_cumsum_rel_target_toGlobe_allGlobSpeed = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed[
            [
                "total_target",
                "biomass_target",
                "fossils_target",
                "metals_target",
                "minerals_target",
            ]
        ]
        * df_cumsum_rel_target_toGlobe_allGlobSpeed[["2015"]].to_numpy()
    )
    DLS_trajectory_targeted_regional_specifics_allGlobSpeed = (
        df_cumsum_rel_target_toGlobe_allGlobSpeed[
            df_cumsum_rel_target_toGlobe_allGlobSpeed.index.get_level_values(0)
            != "Global"
        ]
        .groupby("year")
        .sum()
        / MISO2_2015_pop_R11.sum().sum()
    )

    ## #7B-3 CALCULATE DLS REACHED FOR THE GLOBE - IF SOME CONSTRUCTION FOCUSED ON BEYOND-DLS TOO (regional ratios of beyond-DLS to DLS stocks in 2015)
    # calculate cumulative sum for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum_rel_regio_toGlobe_allGlobSpeed = NAS_stock_prosp_allGlobSpeed.groupby(
        level="region"
    ).cumsum()
    # merge with required stocks to fill DLS gap (including beyond-DLS stocks according to regional ratios of beyond-DLS to DLS stocks in 2015)
    df_cumsum_rel_regio_toGlobe_allGlobSpeed = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed.reset_index()
        .merge(regio_stocks_cap_regspill, on="region")
        .set_index(["region", "year"])
    )

    ## calculate relative target reached (per capita)
    df_cumsum_rel_regio_toGlobe_allGlobSpeed["biomass_regio"] = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed["biomass_x"]
        / df_cumsum_rel_regio_toGlobe_allGlobSpeed["biomass_y"]
    )
    df_cumsum_rel_regio_toGlobe_allGlobSpeed["fossils_regio"] = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed["fossils_x"]
        / df_cumsum_rel_regio_toGlobe_allGlobSpeed["fossils_y"]
    )
    df_cumsum_rel_regio_toGlobe_allGlobSpeed["metals_regio"] = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed["metals_x"]
        / df_cumsum_rel_regio_toGlobe_allGlobSpeed["metals_y"]
    )
    df_cumsum_rel_regio_toGlobe_allGlobSpeed["minerals_regio"] = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed["minerals_x"]
        / df_cumsum_rel_regio_toGlobe_allGlobSpeed["minerals_y"]
    )
    df_cumsum_rel_regio_toGlobe_allGlobSpeed["total_regio"] = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed["total_x"]
        / df_cumsum_rel_regio_toGlobe_allGlobSpeed["total_y"]
    )
    df_cumsum_rel_regio_toGlobe_allGlobSpeed = df_cumsum_rel_regio_toGlobe_allGlobSpeed[
        [
            "total_regio",
            "biomass_regio",
            "fossils_regio",
            "metals_regio",
            "minerals_regio",
        ]
    ]

    # bound DLS reached at 100% (1)
    df_cumsum_rel_regio_toGlobe_allGlobSpeed[
        df_cumsum_rel_regio_toGlobe_allGlobSpeed > 1
    ] = 1

    # aggregate to globe by weighting with population
    df_cumsum_rel_regio_toGlobe_allGlobSpeed = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed.reset_index()
        .merge(MISO2_2015_pop_R11, on=["region"])
        .set_index(["region", "year"])
    )
    df_cumsum_rel_regio_toGlobe_allGlobSpeed_weighted = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed[
            [
                "total_regio",
                "biomass_regio",
                "fossils_regio",
                "metals_regio",
                "minerals_regio",
            ]
        ]
        * df_cumsum_rel_regio_toGlobe_allGlobSpeed[["2015"]].to_numpy()
    )
    DLS_trajectory_somebDLS_regional_specifics_allGlobSpeed = (
        df_cumsum_rel_regio_toGlobe_allGlobSpeed_weighted.groupby("year").sum()
        / MISO2_2015_pop_R11.sum().sum()
    )

    ## #7B-4 CALCULATE DLS REACHED FOR THE GLOBE - IF MUCH CONSTRUCTION FOCUSED ON BEYOND-DLS TOO (global trickle-down beyond-DLS stocks in 2015)
    # calculate cumulative sum for each year within each region (adding each year's NAS to intital stock in 2015)
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = NAS_stock_prosp_allGlobSpeed.groupby(
        level="region"
    ).cumsum()

    # calculate the stock intensity of achievieng 1% more DLS reached on basis on 2015 value for %DLS reached
    # this is required to calculate %DLS improvement (as their is no basis in 2015 stocks representing the global trickle down which would allow for setting
    # prospective new construction (NAS) into relation to %DLS imrpovement)
    DLS_bDLS_stock_2015 = df_cumsum_rel_glob_toGlobe_allGlobSpeed[
        df_cumsum_rel_glob_toGlobe_allGlobSpeed.index.get_level_values(1) == 2015
    ]
    DLS_bDLS_stock_2015.index = [
        DLS_bDLS_stock_2015.index.get_level_values(0),
        DLS_bDLS_stock_2015.index.get_level_values(1).astype(int),
    ]
    stock_target_2015 = regio_stocks_cap_globspill.drop("Global")
    stock_target_2015.index = [
        stock_target_2015.index.get_level_values(0),
        stock_target_2015.index.get_level_values(1).astype(int),
    ]
    DLS_reached_2015 = (
        df_cumsum_rel_target[df_cumsum_rel_target.index.get_level_values(1) == "2015"]
        .drop("Global")
        .rename(
            columns={
                "total_target": "total",
                "biomass_target": "biomass",
                "fossils_target": "fossils",
                "metals_target": "metals",
                "minerals_target": "minerals",
            }
        )
    )

    if conv_gap_mode:
        DLS_reached_2015 = DLS_reached_2015.clip(0,1)

    DLS_reached_2015.index = [
        DLS_reached_2015.index.get_level_values(0),
        DLS_reached_2015.index.get_level_values(1).astype(int),
    ]
    
    if not conv_gap_mode:
        intens_stock_percDLS = (stock_target_2015 - DLS_bDLS_stock_2015) / (
            1 - DLS_reached_2015)
    else:
        intens_stock_percDLS = (stock_target_2015 - DLS_bDLS_stock_2015).clip(0) / (
            1 - DLS_reached_2015).replace(0, np.nan)

    # rename prospective DLS stock data
    rename_df_cumsum_rel_glob_toGlobe_allGlobSpeed = {
        "total": "total_DLS_bDLS_stock",
        "biomass": "biomass_DLS_bDLS_stock",
        "fossils": "fossils_DLS_bDLS_stock",
        "metals": "metals_DLS_bDLS_stock",
        "minerals": "minerals_DLS_bDLS_stock",
    }
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed.rename(
            columns=rename_df_cumsum_rel_glob_toGlobe_allGlobSpeed
        )
    )
    # add columns on stock intensity of achievieng 1% more DLS reached on basis on 2015 value %DLS reached
    rename_intens_stock_percDLS = {
        "total": "total_intens_per%DLS",
        "biomass": "biomass_intens_per%DLS",
        "fossils": "fossils_intens_per%DLS",
        "metals": "metals_intens_per%DLS",
        "minerals": "minerals_intens_per%DLS",
    }
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed.reset_index()
        .merge(
            intens_stock_percDLS.rename(columns=rename_intens_stock_percDLS),
            on="region",
            how="left",
        )
        .set_index(["region", "year"])
    )
    # add columns for starting DLS stock level in 2015
    rename_DLS_bDLS_stock_2015 = {
        "total": "total_DLS_bDLS_stock_2015",
        "biomass": "biomass_DLS_bDLS_stock_2015",
        "fossils": "fossils_DLS_bDLS_stock_2015",
        "metals": "metals_DLS_bDLS_stock_2015",
        "minerals": "minerals_DLS_bDLS_stock_2015",
    }
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed.reset_index()
        .merge(
            DLS_bDLS_stock_2015.rename(columns=rename_DLS_bDLS_stock_2015),
            on="region",
            how="left",
        )
        .set_index(["region", "year"])
    )

    # calculate gap closure in DLS% = [DLS_bDLS_stock(year t) - DLS_bDLS_stock(year 2015)] / stock intensity of %DLS change = change in % DLS reached by new construction
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_%closed"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_%closed"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_%closed"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_%closed"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_intens_per%DLS"]
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_%closed"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_DLS_bDLS_stock"]
        - df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_DLS_bDLS_stock_2015"]
    ) / df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_intens_per%DLS"]
    
    if conv_gap_mode:
        df_cumsum_rel_glob_toGlobe_allGlobSpeed = (
            df_cumsum_rel_glob_toGlobe_allGlobSpeed.replace(np.nan, 0)
        )
    # calculate total DLS% by adding start value of %DLS reached from 2015
    rename_DLS_reached_2015 = {
        "total": "total_reached_2015",
        "biomass": "biomass_reached_2015",
        "fossils": "fossils_reached_2015",
        "metals": "metals_reached_2015",
        "minerals": "minerals_reached_2015",
    }
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed.reset_index()
        .merge(
            DLS_reached_2015.rename(columns=rename_DLS_reached_2015),
            on="region",
            how="left",
        )
        .set_index(["region", "year"])
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_reached_2015"]
        + df_cumsum_rel_glob_toGlobe_allGlobSpeed["total_%closed"]
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_reached_2015"]
        + df_cumsum_rel_glob_toGlobe_allGlobSpeed["biomass_%closed"]
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_reached_2015"]
        + df_cumsum_rel_glob_toGlobe_allGlobSpeed["metals_%closed"]
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_reached_2015"]
        + df_cumsum_rel_glob_toGlobe_allGlobSpeed["fossils_%closed"]
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_%reached_globGap"] = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_reached_2015"]
        + df_cumsum_rel_glob_toGlobe_allGlobSpeed["minerals_%closed"]
    )

    # only keep goal columns for %reached and set limit to 1 (100% reached)
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = df_cumsum_rel_glob_toGlobe_allGlobSpeed[
        [
            "total_%reached_globGap",
            "biomass_%reached_globGap",
            "metals_%reached_globGap",
            "fossils_%reached_globGap",
            "minerals_%reached_globGap",
        ]
    ]
    df_cumsum_rel_glob_toGlobe_allGlobSpeed[
        df_cumsum_rel_glob_toGlobe_allGlobSpeed > 1
    ] = 1

    # calculate weighted global reaching of DLS targets, based on regional trajectories
    df_cumsum_rel_glob_toGlobe_allGlobSpeed = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed.reset_index()
        .merge(MISO2_2015_pop_R11, on=["region"])
        .set_index(["region", "year"])
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed_weighted = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed
        * df_cumsum_rel_glob_toGlobe_allGlobSpeed[["2015"]].to_numpy()
    )
    df_cumsum_rel_glob_toGlobe_allGlobSpeed_weighted_rel = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed_weighted.groupby("year").sum()
        / MISO2_2015_pop_R11.sum().sum()
    )
    DLS_trajectory_manybDLS_regional_specifics_allGlobSpeed = (
        df_cumsum_rel_glob_toGlobe_allGlobSpeed_weighted_rel.drop("2015", axis=1)
    )

    ## #7B-5 ASSEMBLE GLOBAL TRAJECTORIES @ GLOBAL SPEED

    df_global_regional_speed_allGlobSpeed = (
        DLS_trajectory_targeted_regional_specifics_allGlobSpeed.merge(
            DLS_trajectory_somebDLS_regional_specifics_allGlobSpeed, on=["year"]
        ).merge(DLS_trajectory_manybDLS_regional_specifics_allGlobSpeed, on=["year"])
    )
    df_global_regional_speed_allGlobSpeed.index = (
        df_global_regional_speed_allGlobSpeed.index.astype(int)
    )
    df_global_regional_speed_allGlobSpeed = (
        df_global_regional_speed_allGlobSpeed.sort_index()
    )

    """ ####################################################################################
                     #7C CALCULATE DLS REACHED - SUMMARIZE ALL SCENARIOS
         ###################################################################################"""

    ### concat trajectories @ REGIONAL speed & GLOBAL speed
    scenario_summary = df_global_regional_speed_allGlobSpeed.merge(
        df_global_regional_speed, on=["year"]
    )
    scenario_summary.rename(
        columns={
            "total_target_x": "total_DLS_globSpeed",
            "biomass_target_x": "biomass_DLS_globSpeed",
            "fossils_target_x": "fossils_DLS_globSpeed",
            "metals_target_x": "metals_DLS_globSpeed",
            "minerals_target_x": "minerals_DLS_globSpeed",
            "total_regio_x": "total_bDLSregio_globSpeed",
            "biomass_regio_x": "biomass_bDLSregio_globSpeed",
            "fossils_regio_x": "fossils_bDLSregio_globSpeed",
            "metals_regio_x": "metals_bDLSregio_globSpeed",
            "minerals_regio_x": "minerals_bDLSregio_globSpeed",
            "total_%reached_globGap_x": "total_bDLSglob_globSpeed",
            "biomass_%reached_globGap_x": "biomass_bDLSglob_globSpeed",
            "fossils_%reached_globGap_x": "fossils_bDLSglob_globSpeed",
            "metals_%reached_globGap_x": "metals_bDLSglob_globSpeed",
            "minerals_%reached_globGap_x": "minerals_bDLSglob_globSpeed",
            "total_target_y": "total_DLS_regSpeed",
            "biomass_target_y": "biomass_DLS_regSpeed",
            "fossils_target_y": "fossils_DLS_regSpeed",
            "metals_target_y": "metals_DLS_regSpeed",
            "minerals_target_y": "minerals_DLS_regSpeed",
            "total_regio_y": "total_bDLSregio_regSpeed",
            "biomass_regio_y": "biomass_bDLSregio_regSpeed",
            "fossils_regio_y": "fossils_bDLSregio_regSpeed",
            "metals_regio_y": "metals_bDLSregio_regSpeed",
            "minerals_regio_y": "minerals_bDLSregio_regSpeed",
            "total_%reached_globGap_y": "total_bDLSglob_regSpeed",
            "biomass_%reached_globGap_y": "biomass_bDLSglob_regSpeed",
            "fossils_%reached_globGap_y": "fossils_bDLSglob_regSpeed",
            "metals_%reached_globGap_y": "metals_bDLSglob_regSpeed",
            "minerals_%reached_globGap_y": "minerals_bDLSglob_regSpeed",
        },
        inplace=True,
    )

    """  ############################################################
                                    
                                      8 
                                    PLOTS
                                    
         ############################################################"""

    """  ##############
    
          FIGURE 2 (A) 
          
         ###############"""

    # map values by country of current_DLS_stocks / thresh_DLS_stocks (0-1) to worldmap
    # assumes zipfile from naturalearthdata was downloaded to current directory
    import geopandas as gpd

    world_raw = gpd.read_file(input_paths_dict.get("admin_boundaries"))
    world_df = world_raw[["POP_EST", "CONTINENT", "NAME", "ISO_A3_EH", "geometry"]]
    world_df.columns = world_df.columns.str.lower()
    country_correspondence_appl = country_correspondence.rename(
        columns={"ISO_code3": "iso_a3_eh"}
    )
    world_df = world_df.merge(country_correspondence_appl, on="iso_a3_eh")
    world_df.rename(columns={"MISO2": "region"}, inplace=True)

    df1 = (
        regio_stock_prod_dim_cap.groupby(["region", "dimension"]).sum()["DLS_prov_cap"]
        * 1000
    ).reset_index()
    df2 = (
        regio_stock_prod_dim_cap.groupby(["region", "dimension"]).sum()[
            "gap_targeted_cap"
        ]
        * 1000
    ).reset_index()
    DLS_stocks_prov_noMats_country_cap_sum = (
        DLS_stocks_prov_noMats_country_cap.groupby("region").sum() / 1000
    )  # t/cap resulting
    DLS_stocks_thresh_noMats_country_cap_sum = (
        DLS_stocks_thresh_noMats_country_cap.groupby("region").sum() / 1000
    )  # t/cap resulting
    DLS_stocks_fulfilled = (
        DLS_stocks_prov_noMats_country_cap_sum
        / DLS_stocks_thresh_noMats_country_cap_sum
    )
    world_df = world_df.merge(DLS_stocks_fulfilled, on="region")
    if activate_converge != None:
        df_converge = pd.DataFrame(
            DLS_stocks_thresh_converge_noMats_R11_cap.groupby(
                ["region", "dimension"]
            ).sum()
        )

    # aggregate DLS stock dimensions to groups for bar plot
    aggr_dim = {
        "communication": "Socialization_cur",
        "clothing": "Shelter_cur",
        "cond_i": "Shelter_cur",
        "education": "Socialization_cur",
        "health": "Health_cur",
        "hh_appliance": "Nutrition_cur",
        "housing": "Shelter_cur",
        "nutrition": "Nutrition_cur",
        "sanitation": "Health_cur",
        "transport": "Mobility_cur",
        "water": "Health_cur",
    }
    df1 = (
        df1.reset_index(drop=True)
        .replace(aggr_dim)
        .groupby(["region", "dimension"])
        .sum()
    )
    df2 = (
        df2.reset_index(drop=True)
        .replace(aggr_dim)
        .groupby(["region", "dimension"])
        .sum()
    )
    aggr_direct_indirect_gf = {
        "clothing_i": "clothing",
        "communication_i": "communication",
        "educ_i": "education",
        "education_i": "education",
        "health_i": "health",
        "hh_appliance_i": "hh_appliance",
        "housing_i": "housing",
        "transport_i": "transport",
        "nutrition_i": "nutrition",
        "sanitation_i": "sanitation",
        "water_i": "water",
    }
    if activate_converge != None:
        df_converge = (
            df_converge.reset_index()
            .replace(aggr_direct_indirect_gf)
            .replace(aggr_dim)
            .groupby(["region", "dimension"])
            .sum()
        )

    # set title
    title = (
        "(a) Material stocks providing existing DLS-levels & to close DLS gaps (2015)"
    )

    ## call plot
    # without additional second dataset for DLS stock thresholds
    regional_DLSstocks_toSI = plot_stacked_bars_sub_geo(
        df1, df2, "dimension", title, world_df
    )
    # with converged practices (if set)
    if activate_converge != None:
        regional_DLSstocks_toSI = plot_stacked_bars_sub_geo_converge(
            df1, df2, df_converge, "dimension", title, world_df
        )
        # only map
        plot_stacked_bars_sub_geo_converge_onlyMap(
            df1, df2, df_converge, "dimension", title, world_df
        )

    # prep plot data for SI
    regional_DLSstocks_toSI = (
        regional_DLSstocks_toSI.rename(
            columns={"df1_stock": "existing_DLSstock", "df2_stock": "DLSstock_gap"}
        )
        .reset_index()
        .replace(
            {
                "Health_cur": "health provided",
                "Mobility_cur": "mobility provided",
                "Nutrition_cur": "nutrition provided",
                "Shelter_cur": "shelter provided",
                "Socialization_cur": "socializ provided",
            }
        )
        .set_index(["region", "dimension"])
    )

    """  ##############
          
          FIGURE 2 (A) 
        
      PER MATERIAL GROUP TO SI
    
         ###############"""

    ## DLS stocks per material group
    DLS_stocks_prov_matGroup = (
        DLS_stocks_prov_noMats_country_cap.groupby(["region", "material"]).sum() / 1000
    )  # t/cap resulting
    DLS_stocks_thresh_matGroup = (
        DLS_stocks_thresh_noMats_country_cap.groupby(["region", "material"]).sum()
        / 1000
    )  # t/cap resulting
    DLS_stocks_fulfilled_matGroup = (
        DLS_stocks_prov_matGroup / DLS_stocks_thresh_matGroup
    )
    DLS_stocks_fulfilled_matGroup = (
        DLS_stocks_fulfilled_matGroup.reset_index().pivot_table(
            index=["region"], columns="material", values="value_cap"
        )
    )

    # create world map
    # assumes zipfile from naturalearthdata was downloaded to current directory
    world_raw = gpd.read_file(input_paths_dict.get("admin_boundaries"))
    world_df = world_raw[["POP_EST", "CONTINENT", "NAME", "ISO_A3", "geometry"]]
    world_df.columns = world_df.columns.str.lower()
    country_correspondence_appl = country_correspondence.rename(
        columns={"ISO_code3": "iso_a3"}
    )
    world_df = world_df.merge(country_correspondence_appl, on="iso_a3")
    world_df.rename(columns={"MISO2": "region"}, inplace=True)
    world_df = world_df.merge(DLS_stocks_fulfilled_matGroup, on="region")
    world_df = world_df.merge(DLS_stocks_fulfilled, on="region").rename(
        columns={"value_cap": "total"}
    )
    # Example of how to call the function
    columns = ["total", "biomass", "fossils", "metals", "minerals"]  # Columns to plot
    titles = [
        "Total",
        "Biomass",
        "Fossils-based",
        "Metals",
        "Non-metallic minerals",
    ]  # Corresponding subplot titles

    # call plots
    plot_geo_multiplot(world_df, columns, titles)

    world_df_toSI = (
        world_df[["region", "total", "biomass", "fossils", "metals", "minerals"]]
        .rename(
            columns={
                "total": "%DLS_matstock_total",
                "biomass": "%DLS_matstock_biomass",
                "fossils": "%DLS_matstock_fossils",
                "metals": "%DLS_matstock_metals",
                "minerals": "%DLS_matstock_minerals",
            }
        )
        .sort_values(by="region")
        .set_index(["region"])
    )
    # Calculate the minimum and maximum for each row
    world_df_toSI["min_value"] = world_df_toSI.min(axis=1)
    world_df_toSI["max_value"] = world_df_toSI.max(axis=1)
    # Calculate the deviation in percentage points
    world_df_toSI["maximum deviation %points"] = (
        world_df_toSI["max_value"] - world_df_toSI["min_value"]
    )
    # Drop the min_value and max_value columns if not needed
    world_df_toSI = world_df_toSI.drop(columns=["min_value", "max_value"])
    world_df_toSI["maximum deviation %points"].median()
    world_df_toSI["maximum deviation %points"].mean()

    """  ##############
    
          FIGURE 2 (B) 
          
         ###############"""

    ### assemble data for distribution of per capita total economy wide stocks, DLS current stocks, DLS threshold stocks
    x1 = DLS_stocks_prov_noMats_country_cap_sum
    x = MISO2_stocks_2015_cap_total.merge(
        DLS_stocks_prov_noMats_country_cap_sum, on="region"
    )
    x = x.merge(DLS_stocks_thresh_noMats_country_cap_sum, on="region")
    x.rename(
        columns={
            "value": "economy-wide (top-down)",
            "value_cap_x": "DLS_current (bottom-up)",
            "value_cap_y": "DLS_thresh (bottom-up)",
        },
        inplace=True,
    )
    distribution_df = x.copy()
    distribution_df = distribution_df.reset_index().merge(
        R11_country_correspondence[["MISO2", "R11"]]
        .reset_index()
        .rename(columns={"MISO2": "region"}),
        how="left",
        on="region",
    )
    distribution_df = distribution_df.merge(
        MISO2_2015_pop_R11.reset_index().rename(
            columns={"region": "R11", "2015": "population_R11"}
        ),
        how="left",
        on="R11",
    )
    distribution_df = distribution_df.merge(
        MISO2_population_2015.reset_index().rename(
            columns={"2015": "population_countries"}
        ),
        how="left",
        on="region",
    )
    distribution_df = distribution_df[
        [
            "region",
            "economy-wide (top-down)",
            "DLS_current (bottom-up)",
            "DLS_thresh (bottom-up)",
            "R11",
            "population_R11",
            "population_countries",
        ]
    ].rename(
        columns={
            "economy-wide (top-down)": "(b) Total material stocks (economy-wide)",
            "DLS_current (bottom-up)": "(b) DLS material stocks, existing",
            "DLS_thresh (bottom-up)": "(c) DLS material stocks, threshold",
        }
    )

    # if additional second set of stock thresholds (converged practices) specified,
    # assemble data for distribution of per capita total economy wide stocks, DLS current stocks, DLS threshold stocks
    if activate_converge is not None:
        distribution_conv_df = DLS_stocks_thresh_converge_noMats_country_cap.copy()
        distribution_conv_df = distribution_conv_df.groupby("region").sum()
        distribution_conv_df = distribution_conv_df.reset_index().merge(
            R11_country_correspondence[["MISO2", "R11"]]
            .reset_index()
            .rename(columns={"MISO2": "region"}),
            how="left",
            on="region",
        )
        distribution_conv_df = distribution_conv_df.merge(
            MISO2_2015_pop_R11.reset_index().rename(
                columns={"region": "R11", "2015": "population_R11"}
            ),
            how="left",
            on="R11",
        )
        distribution_conv_df = distribution_conv_df.merge(
            MISO2_population_2015.reset_index().rename(
                columns={"2015": "population_countries"}
            ),
            how="left",
            on="region",
        )

    # Plot threshold mean and distribution by R11

    subpanels = [
        "(b) Total material stocks (economy-wide)",
        "(b) DLS material stocks, existing",
        "(c) DLS material stocks, threshold",
    ]

    # calculate stats and save to dataframe for SI
    stats = plot_country_distribution(distribution_df, subpanels)
    Fig2_stats_toSI = expand_nested_dict_to_df(stats)
    Fig2_stats_toSI = Fig2_stats_toSI.rename(columns={0: "value"})
    Fig2_stats_toSI.index.names = ["variable", "region", "stat_measure"]

    # additional figure with one axis
    subpanels = [
        "(b) DLS material stocks, existing",
        "(c) DLS material stocks, threshold",
    ]
    title = "(b) Country variation of material stocks providing existing DLS-levels & to close DLS gaps"
    plot_country_distribution_one(
        distribution_df.set_index("region").iloc[:, 1:].reset_index(), subpanels, title
    )

    # additional figure with one axis with additional second set of stock thresholds (converged practices) specified
    if activate_converge is not None:
        title = "(b) Region & country variation of existing DLS material stocks & gaps"
        plot_country_distribution_one_converge_small(
            distribution_df.set_index("region").iloc[:, 1:].reset_index(),
            distribution_conv_df,
            subpanels,
            title,
        )
        # only with current practices
        plot_country_distribution_one_converge_small_current(
            distribution_df.set_index("region").iloc[:, 1:].reset_index(),
            distribution_conv_df,
            subpanels,
            title,
        )
        # check statistics
        # distribution_conv_df.groupby('R11')['value_cap'].agg(['mean','min', 'max', 'median']).sort_values(by='mean', ascending=False)

        """  ##############
            
                FIGURE 2 (C) 
        
        WITH EXISTING BEYOND_DLS STOCKS
        
            ###############"""

    ## add economy-wide stocks to bar plot Fig 2B
    # format dataframe

    # TODO We indented this, where does this go?

        distr_df_wconverge_save = (
            distribution_df[
                [
                    "region",
                    "(b) Total material stocks (economy-wide)",
                    "(c) DLS material stocks, threshold",
                    "population_R11",
                    "population_countries",
                ]
            ]
            .copy()
            .rename(
                columns={
                    "(c) DLS material stocks, threshold": "DLS material stocks, threshold (default)"
                }
            )
        )

    if activate_converge is not None:
        distr_df_wconverge_save = distr_df_wconverge_save.merge(
            distribution_conv_df.iloc[:, :2]
            .set_index("region")
            .rename(columns={"value_cap": "DLS material stocks, threshold (converge)"})
            .astype(float)
            / 1e3,
            on="region",
        )
    # choose panels to include
        subpanels = [
            "(b) DLS material stocks, existing",
            "(c) DLS material stocks, threshold",
            "(b) Total material stocks (economy-wide)",
        ]
        # plot
        plot_country_distribution_one_withTotal(
            distribution_df.set_index("region").reset_index(),
            subpanels,
            distribution_conv_df,
        )

    """  ##############
    
          FIGURE 3 (A-B)
          
         ###############"""

    # plot existing DLS and beyond-DLS stocks, as well as stock additions required in scenarios i-iii

    ### calculate required stock additions for case with converged practices

    # DLS provided by product and region
    df1 = (
        regio_stock_prod_cap.groupby(["region", "sector"]).sum()["DLS_prov_cap"] * 1000
    ).reset_index()
    # beyond DLS stocks by product and region
    df2 = (
        regio_stock_prod_cap.groupby(["region", "sector"]).sum()["beyond_DLS_cap"]
        * 1000
    ).reset_index()
    # DLS treshold
    df3 = (
        (
            regio_stock_prod_cap.groupby(["region", "sector"]).sum()["DLS_prov_cap"]
            * 1000
        )
        + (
            regio_stock_prod_cap.groupby(["region", "sector"]).sum()["gap_targeted_cap"]
            * 1000
        )
    ).reset_index()
    df1 = df1.reset_index(drop=True).groupby(["region", "sector"]).sum()
    df2 = df2.reset_index(drop=True).groupby(["region", "sector"]).sum()

    # per capita beyond_DLS stocks that would be built according to DLS_prov : beyond_DLS stock ratio per region
    # DLS_stock_gap = df3.set_index(['region','sector']).groupby('region').sum().sum(axis=1) -  df1.groupby('region').sum().sum(axis=1)

    # calculate required stock additions (NAS) for scenario-ii according to regio_stock_prod
    regio_ratio_DLS_beyond_DLS_PROD = (
        df2.groupby("region").sum() / df1.groupby("region").sum()
    )
    regio_stock_prod_cap_gap = (
        regio_stock_prod_cap.groupby(["region", "sector"]).sum()["gap_targeted_cap"]
        * 1000
    )
    df_add_beyond_stocks_regional = (
        regio_stock_prod_cap_gap.groupby("region").sum()
        * regio_ratio_DLS_beyond_DLS_PROD
    )

    # calculate required stock additions (NAS) for scenario-iii
    # calculate thresholds to be reached for scenario-iii
    per_capita_trickle_down_regio_PROD = pd.DataFrame(
        [[4.28, 1.16, 7.2, 164.73]] * 11,
        index=df_add_beyond_stocks_regional.index,
        columns=df_add_beyond_stocks_regional.columns,
    )
    per_capita_trickle_down_regio_PROD.sum(axis=1)
    # calculate difference to be filled; any regions that have stocks larger than the hard-coded thresholds above in 'per_capita_trickle_down_regio' already have existing total stocks higher than the threshold
    df_add_beyond_stocks_global = (
        (1000 * per_capita_trickle_down_regio_PROD)
        - df1.groupby("region").sum()
        - df2.groupby("region").sum()
        - regio_stock_prod_cap_gap.groupby("region").sum()
        - df_add_beyond_stocks_regional
    )
    df_add_beyond_stocks_global[df_add_beyond_stocks_global < 0] = 0

    # sum four materials to stock total for plot
    df_add_beyond_stocks_regional = df_add_beyond_stocks_regional.sum(axis=1)
    df_add_beyond_stocks_global = df_add_beyond_stocks_global.sum(axis=1)

    # set labels
    # title = "Material stocks: DLS_current (bottom bar1), beyond-DLS (top bar2)"
    line_y_values = [177.33999999999997]  # Replace with actual y-values
    line_labels = ["scenario-iii: economy-wide stock requirement (global)"]

    ### calculate required stock additions for scenario-i with converged practices (if case is present)
    #   to add to case with current national/regional practices

    if activate_converge is not None:
        # aggregate do common DLS dimensions and products

        DLS_stock_thresh_eff_glob_dim_prod = (
            DLS_stocks_thresh_converge_noMats_R11.reset_index()
            .replace(AGGR_DIRECT_INDIRECT_GF)
            .replace(AGGR_DIM)
            .replace(HARMONIZE_STOCKS_DICT)
            .replace(AGGR_PROD)
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
            .groupby(["region", "dimension", "stock", "material"])
            .sum()
        )

        #DLS_stock_thresh_eff_glob_dim_prod.dropna(how="all", axis=0)

        DLS_stock_thresh_eff_glob_dim_prod.index.names = [
            "region",
            "dimension",
            "sector",
            "material",
        ]
        DLS_stock_prov_RDPM = pd.melt(
            regio_stock_prod_dim["DLS_prov"]
            .reset_index()
            .replace(AGGR_DIRECT_INDIRECT_GF)
            .replace(AGGR_DIM)
            .replace(HARMONIZE_STOCKS_DICT)
            .replace(AGGR_PROD)
            .set_index(["region", "dimension", "sector"])
            .groupby(["region", "dimension", "sector"])
            .sum()
            .reset_index(),
            id_vars=["region", "dimension", "sector"],
            var_name="material",
            value_name="value",
        ).set_index(["region", "dimension", "sector", "material"])

        DLS_stock_gap_eff_glob_dim_prod = (
            DLS_stock_thresh_eff_glob_dim_prod - DLS_stock_prov_RDPM
        )

        # set negatives to zero (negatives mean that existing DLS stocks (current practices) higher than thresholds (converged practices)); set zero as these negatives do not count for DLS gap (materials only needed where positive values)
        DLS_stock_gap_eff_glob_dim_prod = DLS_stock_gap_eff_glob_dim_prod[
            DLS_stock_gap_eff_glob_dim_prod > 0
        ]

        # TODO This is not setting anything to zero, its subsetting df

        # calculate DLS stock gap (scenario-i) under converged practices (global total and regional per capita)
        DLS_stock_gap_eff_glob = (
            DLS_stock_gap_eff_glob_dim_prod.groupby("region").sum().sum()
        )
        DLS_stock_gap_eff_cap = (
            DLS_stock_gap_eff_glob_dim_prod.groupby("region").sum()
            / MISO2_population_melted_R11[
                MISO2_population_melted_R11.index.get_level_values(1) == "2015"
            ]
            .groupby("region")
            .sum()
        )

    # plot full plot (horizontal) if no converged practices case
    plot_bars_horiz_gap_headroom_two_subplots_doubleGlob_mod(
        df1,
        df2,
        df3,
        df_add_beyond_stocks_regional,
        df_add_beyond_stocks_global,
        global_stocks,
        global_stock_prod,
        "sector",
        line_y_values,
        line_labels,
    )

    # plots with converged practices case
    if activate_converge != None:
        # plot full plot
        plot_bars_Vert_gap_headroom_two_subplots_doubleGlob_mod_eff(
            df1,
            df2,
            df3,
            df_add_beyond_stocks_regional,
            df_add_beyond_stocks_global,
            global_stocks,
            global_stock_prod,
            "sector",
            line_y_values,
            line_labels,
            DLS_stock_gap_eff_glob,
            DLS_stock_gap_eff_cap,
        )
        # plot only scenarios i-ii
        plot_bars_Vert_gap_headroom_two_subplots_doubleGlob_mod_eff_ii(
            df1,
            df2,
            df3,
            df_add_beyond_stocks_regional,
            df_add_beyond_stocks_global,
            global_stocks,
            global_stock_prod,
            "sector",
            line_y_values,
            line_labels,
            DLS_stock_gap_eff_glob,
            DLS_stock_gap_eff_cap,
        )
        # plot only scenarios i
        plot_bars_Vert_gap_headroom_two_subplots_doubleGlob_mod_eff_i(
            df1,
            df2,
            df3,
            df_add_beyond_stocks_regional,
            df_add_beyond_stocks_global,
            global_stocks,
            global_stock_prod,
            "sector",
            line_y_values,
            line_labels,
            DLS_stock_gap_eff_glob,
            DLS_stock_gap_eff_cap,
        )

    """  ##############
    
            FIGURE 4
          
         ###############"""

    ###  plot proportions of DLS stock gap & beyond-DLS stocks by region and product group

    # calculate shares and format
    regio_stock_prod_cap_4shares = regio_stock_prod_cap.copy()
    cols_to_sum = ["gap_targeted_cap"]
    regio_stock_prod_cap_4shares["sum_DLS_and_gap"] = regio_stock_prod_cap[
        cols_to_sum
    ].sum(axis=1)
    regio_stock_prod_cap_4shares["sum_bDLS"] = regio_stock_prod_cap[
        "beyond_DLS_cap"
    ].sum(axis=1)
    regio_stock_prod_cap_4shares["share_DLS"] = regio_stock_prod_cap_4shares[
        "sum_DLS_and_gap"
    ] / (
        regio_stock_prod_cap_4shares["sum_DLS_and_gap"]
        + regio_stock_prod_cap_4shares["sum_bDLS"]
    )
    regio_stock_prod_cap_4shares["share_bDLS"] = regio_stock_prod_cap_4shares[
        "sum_bDLS"
    ] / (
        regio_stock_prod_cap_4shares["sum_DLS_and_gap"]
        + regio_stock_prod_cap_4shares["sum_bDLS"]
    )
    regio_stock_prod_cap_shares = regio_stock_prod_cap_4shares[
        ["share_DLS", "share_bDLS"]
    ]
    regio_stock_prod_cap_shares.reset_index(inplace=True)
    regio_stock_prod_cap_shares.columns = ["region", "sector", "DLS gap", "bDLS"]
    regio_stock_prod_cap_shares.set_index(["region", "sector"], inplace=True)

    # plot
    regions = [
        "AFR",
        "CPA",
        "EEU",
        "FSU",
        "LAM",
        "MEA",
        "PAS",
        "SAS",
        "NAM",
        "WEU",
        "PAO",
    ]
    share_plot(regio_stock_prod_cap_shares, regions)

    # add values for converged scenario if available
    if activate_converge != None:
        # calculate shares
        DLS_stock_gap_eff_glob_dim_prod_cap = (
            DLS_stock_gap_eff_glob_dim_prod.groupby(["region", "sector"]).sum()
            / MISO2_population_melted_R11[
                MISO2_population_melted_R11.index.get_level_values(1) == "2015"
            ]
            .groupby("region")
            .sum()
            / 1e3
        )
        shares_converge = DLS_stock_gap_eff_glob_dim_prod_cap.join(
            regio_stock_prod_cap_4shares["sum_bDLS"]
        )
        shares_converge.columns = ["gap_converge", "sum_bDLS"]
        shares_converge["share_DLS"] = shares_converge["gap_converge"] / (
            shares_converge["gap_converge"] + shares_converge["sum_bDLS"]
        )
        shares_converge["share_bDLS"] = shares_converge["sum_bDLS"] / (
            shares_converge["gap_converge"] + shares_converge["sum_bDLS"]
        )
        shares_converge = shares_converge[["share_DLS", "share_bDLS"]]
        shares_converge.reset_index(inplace=True)
        shares_converge.columns = ["region", "sector", "DLS gap", "bDLS"]
        shares_converge.set_index(["region", "sector"], inplace=True)

        # plot for all regions
        regions = [
            "AFR",
            "CPA",
            "EEU",
            "FSU",
            "LAM",
            "MEA",
            "PAS",
            "SAS",
            "NAM",
            "WEU",
            "PAO",
        ]
        share_plot_converge(regio_stock_prod_cap_shares, shares_converge, regions)

    """  ##############
    
            FIGURE 5
            
         ###############"""

    # plot historical and extrapolated NAs and how fast the can close DLS stock gaps
    # plot
    plot_timing_close_DLS_gaps(
        NAS_all_histprosp, NAS_all_hist, NAS_all_prosp, scenario_summary, "total"
    )

    # if converged scenario present, get data from a prior run (with script analysis_flexible_convGaps - required because calculations follow different logic for converged case)
    try:
        scenario_summary_converged = pd.read_excel(
            os.path.join(
                output_path,
                "results_data_supplement_gapConverged.xlsx",
            ),
            sheet_name="Fig4c_closeGap_Globscenario",
            index_col=[0],
            header=0,
        ).drop(columns=["unit"])
        # scenario_summary_converged = pd.read_excel(os.path.join(main_path, "output/results_data_supplement_gapConverged_2025-08-05.xlsx"),sheet_name="Fig5c_closeGap_Globscenario", index_col=[0], header=0).drop(columns=['unit'])
        plot_timing_close_DLS_gaps_converged(
            NAS_all_histprosp,
            NAS_all_hist,
            NAS_all_prosp,
            scenario_summary,
            scenario_summary_converged,
            "total",
        )
    except Exception as e:
        # If an error occurs, just print the error and continue
        print(f"An error occurred: {e}")

    """  ##############
    
          FIGURE 5 C 
          BY FOUR MATERIAL 
          GROUPS for supplementary 
          info
          
         ###############"""

    # DLS stock gaps closure by material group

    plot_timing_close_DLS_gaps_noYlim(
        NAS_all_histprosp, NAS_all_hist, NAS_all_prosp, scenario_summary, "biomass"
    )
    plot_timing_close_DLS_gaps_noYlim(
        NAS_all_histprosp, NAS_all_hist, NAS_all_prosp, scenario_summary, "fossils"
    )
    plot_timing_close_DLS_gaps_noYlim(
        NAS_all_histprosp, NAS_all_hist, NAS_all_prosp, scenario_summary, "metals"
    )
    plot_timing_close_DLS_gaps_noYlim(
        NAS_all_histprosp, NAS_all_hist, NAS_all_prosp, scenario_summary, "minerals"
    )

    """  ##############
    
          FIGURE 5 C 
          BY WORLD REGIONS 
          for supplementary 
          info
          
         ###############"""

    # DLS stock gaps closure by world-region

    def plot_scenarios(df, label, columns):
        df[columns].plot(
            kind="line",
            color=["g", "orange", "r", "g", "orange", "r"],
            style=[
                "-",
                "-",
                "-",
                "--",
                "--",
                "--",
            ],
            title=label + "\n DLS achieved over time (several scenarios)",
        )
        plt.ylabel("DLS achieved")
        # Manually create the legend
        from matplotlib.lines import Line2D

        custom_lines = [
            Line2D([0], [0], color="g", lw=2, label="scenario-i"),
            Line2D([0], [0], color="orange", lw=2, label="scenario-ii"),
            Line2D([0], [0], color="r", lw=2, label="scenario-iii"),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="-",
                markersize=10,
                label="global speed of new construction",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                lw=2,
                label="regional speed of new construction",
            ),
        ]
        # Add the custom legend to the plot
        plt.legend(handles=custom_lines, loc="best")
        # plt.legend(loc='best')
        plt.show()
        plt.close("all")

    ### repeat plot for regional figures
    # targeted
    df1 = df_cumsum_rel_target_Regional
    df2 = df_cumsum_rel_target_Regional_allGlobSpeed.add_suffix("_globspeed")
    # regional gap
    df3 = df_cumsum_rel_regio_toGlobe.drop(columns=["2015"])
    df4 = df_cumsum_rel_regio_toGlobe_allGlobSpeed.drop(columns=["2015"]).add_suffix(
        "_globspeed"
    )
    # global gap
    df5 = df_cumsum_rel_glob_toGlobe.drop(columns=["2015"])
    df6 = df_cumsum_rel_glob_toGlobe_allGlobSpeed.drop(columns=["2015"]).add_suffix(
        "_globspeed"
    )

    # targeted
    df1 = df_cumsum_rel_target_Regional.drop("Global")
    df2 = df_cumsum_rel_target_Regional_allGlobSpeed.add_suffix("_globspeed")
    # regional gap
    df3 = df_cumsum_rel_regio_toGlobe
    df4 = df_cumsum_rel_regio_toGlobe_allGlobSpeed.add_suffix("_globspeed")
    # global gap
    df5 = df_cumsum_rel_glob_toGlobe
    df6 = df_cumsum_rel_glob_toGlobe_allGlobSpeed.add_suffix("_globspeed")

    # harmonize indices as int as somehow is mixed
    df1.reset_index(inplace=True)
    df1.year = df1.year.astype(int)
    df1 = df1.set_index(["region", "year"])

    scenario_summary_regional = df1
    for df in [df2, df3, df4, df5, df6]:
        df.reset_index(inplace=True)
        df.year = df.year.astype(int)
        df = df.set_index(["region", "year"])
        scenario_summary_regional = scenario_summary_regional.merge(
            df, on=["region", "year"], how="inner"
        )

    for region in scenario_summary_regional.index.get_level_values(0).unique():
        col_id = "total_"
        columns = [col for col in scenario_summary_regional.columns if col_id in col]
        df = scenario_summary_regional[
            scenario_summary_regional.index.get_level_values(0) == region
        ]
        sorted_columns = [
            col_id + "target_globspeed",
            col_id + "regio_globspeed",
            col_id + "%reached_globGap_globspeed",
            col_id + "target",
            col_id + "regio",
            col_id + "%reached_globGap",
        ]
        df_sorted = df[sorted_columns]
        df_sorted.index = df.index.droplevel(0)
        region_full_names = {
            "AFR": "Sub-Saharan Africa",
            "CPA": "Centrally planned Asia",
            "EEU": "Eastern Europe",
            "FSU": "Former Soviet Union",
            "LAM": "Latin America",
            "MEA": "N. Africa & Middle East",
            "NAM": "North America",
            "PAO": "Japan, Australia, New Zealand",
            "PAS": "Pacific Asia",
            "SAS": "South Asia",
            "WEU": "Western Europe",
        }
        plot_scenarios(df_sorted, region_full_names.get(region), sorted_columns)

    """ ###############################################
                           
                            9 
            FORMAT NUMERICAL RESULTS FOR DATA SI
            
        ###############################################"""

    ### prepare dataframes to output for data SI

    # global average DLS stock threshold
    global_av_DLS_stock_thresh = (
        global_stocks_cap[["DLS_prov_cap", "gap_targeted_cap"]].sum().sum()
    )
    # global ratio of DLS stock on total historical stock
    share_DLS_glob = (
        global_stocks.sum(axis=0)["DLS_prov"]
        / global_stocks.sum(axis=0)[["DLS_prov", "beyond_DLS"]].sum()
    )
    # regional ratio of DLS stock on total historical stock
    share_DLS_reg = regio_stocks.groupby("region").sum()[
        "DLS_prov"
    ] / regio_stocks.groupby("region").sum()[["DLS_prov", "beyond_DLS"]].sum(axis=1)
    # global product share of beyond_DLS stocks
    share_beyondDLS_prod_glob = (
        regio_stock_prod.groupby("sector").sum()["beyond_DLS"].sum(axis=1)
        / regio_stock_prod.groupby("sector").sum()["beyond_DLS"].sum(axis=1).sum()
    )
    # regional product share of beyond_DLS stocks
    share_beyondDLS_prod_reg = regio_stock_prod["beyond_DLS"].sum(
        axis=1
    ) / regio_stock_prod.groupby("region").sum()["beyond_DLS"].sum(axis=1)
    # global share of stock additions required for scenario-iii on existing economy-wide stock
    share_gap_untGlo_on_totalStock_glob = (
        global_stocks.sum(axis=0)[["gap_targeted", "gap_regional", "gap_trickle"]]
        / global_stocks.sum(axis=0)[["DLS_prov", "beyond_DLS"]].sum()
    )
    gap_rename = {
        "gap_targeted": "gap_scenario-i",
        "gap_regional": "+gap_scenario-ii",
        "gap_trickle": "+gap_scenario-iii",
        "gap_targeted_cap": "gap_scenario-i_cap",
        "gap_regional_cap": "+gap_scenario-ii_cap",
        "gap_trickle_cap": "+gap_scenario-iii_cap",
    }
    share_gap_untGlo_on_totalStock_glob.rename(index=gap_rename, inplace=True)
    # share of DLS stock gap material groups
    share_gap_materials_glob = (
        global_stocks["gap_targeted"] / global_stocks.sum(axis=0)["gap_targeted"]
    )
    # share of DLS stock gap dimensions - global
    glob_prod_dim_share_gap = (
        regio_stock_prod_dim["gap_targeted"]
        .sum(axis=1)
        .groupby(["dimension", "sector"])
        .sum()
        / regio_stock_prod_dim["gap_targeted"].sum(axis=1).sum()
    )
    # share of DLS stock gap for product groups - global
    share_gap_products_glob = glob_prod_dim_share_gap.groupby("sector").sum()
    # share of DLS stock gap for regional product groups
    regio_prod_share_gap = (
        regio_stock_prod["gap_targeted"].sum(axis=1).groupby(["region", "sector"]).sum()
        / regio_stock_prod["gap_targeted"].sum(axis=1).groupby(["region"]).sum()
    )
    # share of DLS stock gap for regions
    regio_geogr_share_gap = (
        regio_stock_prod["gap_targeted"].sum(axis=1).groupby(["region"]).sum()
        / regio_stock_prod["gap_targeted"].sum().sum()
    )
    # ratio DLS vs. beyond DLS stock
    ratio_DLS_beyondDLS = regio_stock_prod["beyond_DLS"].groupby("region").sum().sum(
        axis=1
    ) / regio_stock_prod["DLS_prov"].groupby("region").sum().sum(axis=1)
    # how much of gap per product group could be closed by redistr. beyond-DLS stocks?
    multiple_beyondDLS_to_gap = regio_stock_prod["beyond_DLS"].sum(
        axis=1
    ) / regio_stock_prod["gap_targeted"].sum(axis=1)
    multiple_beyondDLS_to_gap[multiple_beyondDLS_to_gap > 1] = 1
    share_redistr_beyondDLS_toGap = multiple_beyondDLS_to_gap * regio_prod_share_gap
    share_redistr_beyondDLS_toGap_tot = share_redistr_beyondDLS_toGap.groupby(
        "region"
    ).sum()

    # calculate average global person DLS reached by weighing DLS_reached_perc by pop, summing regions and dividing by pop
    # calculate: DLS reached (in percent)
    DLS_reached_perc = pd.DataFrame(DLS_2015_funct_prov / DLS_2015_thresh)
    DLS_reached_perc.index = DLS_reached_perc.index.droplevel([2])
    DLS_reached_perc_weighted = DLS_reached_perc.reset_index().merge(
        MISO2_population_2015, on="region", how="left"
    )
    DLS_reached_perc_weighted["weighted"] = (
        DLS_reached_perc_weighted["value"] * DLS_reached_perc_weighted["2015"]
    )
    DLS_reached_perc_weighted_glob = (
        DLS_reached_perc_weighted.set_index(["region", "variable"])
        .groupby("variable")
        .sum()
    )
    DLS_reached_perc_weighted_glob = (
        DLS_reached_perc_weighted_glob
        / DLS_reached_perc_weighted[DLS_reached_perc_weighted.variable == "Nutrition"][
            "2015"
        ].sum()
    )
    DLS_reached_perc_weighted_glob["lacking"] = (
        1 - DLS_reached_perc_weighted_glob["weighted"]
    )

    # calculate average global DLS threshold per dimension by weighting by world population, summing regions and dividing by pop
    DLS_2015_thresh_weighted = DLS_2015_thresh.reset_index().merge(
        MISO2_population_2015, on="region", how="left"
    )
    DLS_2015_thresh_weighted["weighted"] = (
        DLS_2015_thresh_weighted["value"] * DLS_2015_thresh_weighted["2015"]
    )
    DLS_2015_thresh_weighted_glob = (
        DLS_2015_thresh_weighted.set_index(["region", "variable", "unit"])
        .groupby(["variable", "unit"])
        .sum()
    )
    # calc global average thresholds per capita
    DLS_2015_thresh_weighted_glob = (
        DLS_2015_thresh_weighted_glob
        / DLS_2015_thresh_weighted[DLS_2015_thresh_weighted.variable == "Nutrition"][
            "2015"
        ].sum()
    )["weighted"]

    # calculate average service levels per region
    DLS_2015_thresh_weighted_reg = DLS_2015_thresh_weighted.copy()
    DLS_2015_thresh_weighted_reg.region = DLS_2015_thresh_weighted_reg.region.replace(
        R11_country_correspondence_dict
    )
    DLS_2015_thresh_weighted_reg = (
        DLS_2015_thresh_weighted_reg.set_index(["region", "variable", "unit"])
        .groupby(["region", "variable", "unit"])
        .sum()
    )
    DLS_2015_thresh_weighted_reg["average_regional_value"] = (
        DLS_2015_thresh_weighted_reg["weighted"] / DLS_2015_thresh_weighted_reg["2015"]
    )
    DLS_2015_thresh_weighted_reg_sortDim = DLS_2015_thresh_weighted_reg.sort_index(
        axis=0, level=1
    )

    # share of beyond-DLS stocks in global North (according to Kikstra et al. 2021 - all except MEA, CPA, SAS, PAS, LAM)
    global_North = ["EEU", "FSU", "NAM", "PAO", "WEU"]
    global_South_noCPA = ["AFR", "LAM", "MEA", "PAS", "SAS"]
    regions_GNS_mapping = {
        "EEU": "Global North",
        "FSU": "Global North",
        "NAM": "Global North",
        "PAO": "Global North",
        "WEU": "Global North",
        "AFR": "Global South",
        "LAM": "Global South",
        "MEA": "Global South",
        "PAS": "Global South",
        "SAS": "Global South",
        "CPA": "CPA",
    }
    global_NS_correspondence = R11_country_correspondence[["MISO2", "R11"]].replace(
        regions_GNS_mapping
    )
    global_North_countries = global_NS_correspondence[
        global_NS_correspondence.R11 == "Global North"
    ]
    global_South_noCPA_countries = global_NS_correspondence[
        global_NS_correspondence.R11 == "Global South"
    ]
    CPA_countries = global_NS_correspondence[global_NS_correspondence.R11 == "CPA"]

    # economy-wide stocks Global North and South
    ew_total_stocks_scale_R11 = (
        MISO2_stocks_2015_R11_scale_piv.groupby("region").sum().sum(axis=1)
    )
    # North
    ew_total_stocks_scale_R11_North_cap = (
        ew_total_stocks_scale_R11[
            ew_total_stocks_scale_R11.index.get_level_values(0).isin(global_North)
        ].sum()
        / MISO2_population_melted_R11_2015[
            MISO2_population_melted_R11_2015.index.get_level_values(0).isin(
                global_North
            )
        ].sum()
    )
    # South no CPA
    ew_total_stocks_scale_R11_South_noCPA_cap = (
        ew_total_stocks_scale_R11[
            ew_total_stocks_scale_R11.index.get_level_values(0).isin(global_South_noCPA)
        ].sum()
        / MISO2_population_melted_R11_2015[
            MISO2_population_melted_R11_2015.index.get_level_values(0).isin(
                global_South_noCPA
            )
        ].sum()
    )
    # CPA
    ew_total_stocks_scale_R11_CPA_cap = (
        ew_total_stocks_scale_R11[
            ew_total_stocks_scale_R11.index.get_level_values(0).isin(["CPA"])
        ].sum()
        / MISO2_population_melted_R11_2015[
            MISO2_population_melted_R11_2015.index.get_level_values(0).isin(["CPA"])
        ].sum()
    )

    # countries in Global North, South and CPA for subsetting
    global_North_countries = global_NS_correspondence[
        global_NS_correspondence.R11 == "Global North"
    ]
    global_South_noCPA_countries = global_NS_correspondence[
        global_NS_correspondence.R11 == "Global South"
    ]
    CPA_countries = global_NS_correspondence[global_NS_correspondence.R11 == "CPA"]

    ## calculate deprivation of headcount global share by dimension for introduction section from Kikstra et al 2021 data
    # calculate global deprivation headcounts by DLS needs-dimension from Kikstra et al. (2021) - mentioned in paper introduction
    Kikstra2021_population = pd.read_excel(
        input_paths_dict["Kikstra_population"],
        index_col=0,
    )
    Kikstra2021_DLS_indicators = pd.read_csv(input_paths_dict.get("DLS_data_path"))

    Kikstra2021_deprivation_headcount = Kikstra2021_DLS_indicators[
        Kikstra2021_DLS_indicators.type == "Deprivation headcount"
    ]

    Kikstra2021_deprivation_headcount = Kikstra2021_deprivation_headcount.merge(
        Kikstra2021_population.reset_index(), how="left", on="iso"
    )
    Kikstra2021_deprivation_headcount["value_weighted_pop"] = (
        Kikstra2021_deprivation_headcount["value"]
        * Kikstra2021_deprivation_headcount["population"]
    )
    Kikstra2021_deprivation_headcount_global_shares = (
        Kikstra2021_deprivation_headcount.set_index(
            [
                "iso",
                "variable",
                "unit",
                "value",
                "type",
                "note",
                "year",
                "version",
                "author",
                "population",
            ]
        )
        .groupby("variable")
        .sum()
        / Kikstra2021_population.sum().sum()
    )

    # CALCULATE GLOBAL AVERAGE PER CAPITA VALUES TO COMPARE WITH VELEZ-HENAO & PAULIUK (2023)
    DLS_stocks_thresh_noMats_R11["value"] = DLS_stocks_thresh_noMats_R11[
        "value"
    ].astype(float)
    DLS_stocks_thresh_noMats_R11 = DLS_stocks_thresh_noMats_R11.groupby(
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
    DLS_stocks_thresh_noMats_R11_DM = DLS_stocks_thresh_noMats_R11.groupby(
        ["dimension", "material"]
    ).sum()
    DLS_stocks_thresh_noMats_R11_DM_piv = (
        DLS_stocks_thresh_noMats_R11_DM.reset_index().pivot(
            index=["dimension"], columns="material", values="value"
        )
    )

    # negative out of total datapoints in beyond-DLS stocks
    text_negatives_bDLS = (
        "region-materials: "
        + str(negative_count_bDLS_RM)
        + " out of "
        + str(total_count_bDLS_RM)
        + " negatives, "
        "region-products: "
        + str(negative_count_bDLS_RP)
        + " out of "
        + str(total_count_bDLS_RP)
        + " negatives, "
        "region-products-materials: "
        + str(negative_count_bDLS_RMP)
        + " out of "
        + str(total_count_bDLS_RMP)
        + " negatives"
    )
    text_negatives_bDLS_mass = (
        "region-materials: negative mass in Gigations is "
        + str(negative_mass_bDLS_RM / 1e9)
        + " out of "
        + str(positive_mass_bDLS_RM / 1e9)
        + " positive mass, "
        "region-products: negative mass in Gigations is "
        + str(negative_mass_bDLS_RP / 1e9)
        + " out of "
        + str(positive_mass_bDLS_RP / 1e9)
        + " positive mas, "
        "region-products-materials negative mass in Gigations is: "
        + str(negative_mass_bDLS_RMP / 1e9)
        + " out of "
        + str(positive_mass_bDLS_RMP / 1e9)
        + " positive mas"
    )

    # format dfs before saving
    distribution_df = distribution_df.set_index(["region", "R11"]).astype(float)
    distribution_df["DLS material stock gap"] = (
        distribution_df["(c) DLS material stocks, threshold"]
        - distribution_df["(b) DLS material stocks, existing"]
    )
    global_stocks.loc["total"] = global_stocks.sum(axis=0)
    # insert units and prepare for output
    regional_DLSstocks_toSI_save = regional_DLSstocks_toSI.copy()
    regional_DLSstocks_toSI_save.insert(0, "unit", "tons/capita")
    regional_DLSstocks_toSI_save.set_index(["unit"], append=True, inplace=True)
    regional_DLSstocks_toSI_save["DLSstock_threshold"] = (
        regional_DLSstocks_toSI_save["existing_DLSstock"]
        + regional_DLSstocks_toSI_save["DLSstock_gap"]
    )
    distribution_df_save = distribution_df.copy()
    distribution_df_save.insert(0, "unit", "tons/capita")
    distribution_df_save.set_index(["unit"], append=True, inplace=True)
    if activate_converge is not None:
        distribution_conv_df_save = distribution_conv_df.copy()
        distribution_conv_df_save = (
            distribution_conv_df_save[["region", "value_cap"]]
            .set_index(["region"])
            .rename(columns={"value_cap": "DLS_treshold_converge"})
            / 1e3
        )
        distribution_df_save = distribution_df_save.merge(
            distribution_conv_df_save, on="region"
        )
    Fig2_stats_toSI_save = Fig2_stats_toSI.copy()
    Fig2_stats_toSI_save.insert(0, "unit", "tons/capita")
    Fig2_stats_toSI_save.set_index(["unit"], append=True, inplace=True)
    regio_stocks_cap_save = regio_stocks_cap.copy()
    regio_stocks_cap_save.insert(0, "unit", "tons/capita")
    regio_stocks_cap_save.set_index(["unit"], append=True, inplace=True)
    regio_stocks_cap_save.rename(columns=gap_rename, inplace=True)
    regio_stock_prod_cap_save = regio_stock_prod_cap.copy()
    regio_stock_prod_cap_save.insert(0, "unit", "tons/capita")
    regio_stock_prod_cap_save.set_index(["unit"], append=True, inplace=True)
    regio_stock_prod_cap_save.rename(columns=gap_rename, inplace=True)
    regio_stock_prod_dim_cap_save = regio_stock_prod_dim_cap.copy()
    regio_stock_prod_dim_cap_save.insert(0, "unit", "tons/capita")
    regio_stock_prod_dim_cap_save.set_index(["unit"], append=True, inplace=True)
    NAS_all_histprosp_save = NAS_all_histprosp.copy()
    NAS_all_histprosp_save.insert(0, "unit", "tons/capita")
    NAS_all_histprosp_save.set_index(["unit"], append=True, inplace=True)
    NAS_all_hist_save = NAS_all_hist.copy()
    NAS_all_hist_save.insert(0, "unit", "tons/capita")
    NAS_all_hist_save.set_index(["unit"], append=True, inplace=True)
    NAS_all_prosp_save = NAS_all_prosp.copy()
    NAS_all_prosp_save.insert(0, "unit", "tons/capita")
    NAS_all_prosp_save.set_index(["unit"], append=True, inplace=True)
    scenario_summary_save = scenario_summary.copy()
    scenario_summary_save.insert(0, "unit", "%DLS_reached")
    scenario_summary_save.set_index(["unit"], append=True, inplace=True)
    scenario_summary_regional_save = scenario_summary_regional.copy()
    scenario_summary_regional_save.insert(0, "unit", "%DLS_reached")
    scenario_summary_regional_save.set_index(["unit"], append=True, inplace=True)
    regio_ratio_DLS_beyond_DLS_save = regio_ratio_DLS_beyond_DLS.copy()
    regio_ratio_DLS_beyond_DLS_save.insert(0, "unit", "tons_bDLS/tons_DLS")
    regio_ratio_DLS_beyond_DLS_save.set_index(["unit"], append=True, inplace=True)
    global_stocks_save = global_stocks.copy()
    global_stocks_save.insert(0, "unit", "Gigatons")
    global_stocks_save.set_index(["unit"], append=True, inplace=True)
    global_stocks_save[
        ["DLS_prov", "beyond_DLS", "gap_targeted", "gap_regional", "gap_trickle"]
    ] = (
        global_stocks_save[
            ["DLS_prov", "beyond_DLS", "gap_targeted", "gap_regional", "gap_trickle"]
        ]
        / 1e9
    )
    
    global_stocks_prod_save = global_stock_prod.copy()
    global_stocks_prod_save = global_stocks_prod_save /1e9
    global_stocks_prod_save.insert(0, "unit", "Gigatons")
    global_stocks_prod_save.set_index(["unit"], append=True, inplace=True)

    
        
    global_stocks_save.rename(columns=gap_rename, inplace=True)
    global_av_DLS_stock_thresh = pd.DataFrame(
        [[global_av_DLS_stock_thresh]], columns=["value"], index=["tons/capita"]
    )
    ew_total_stocks_scale_R11_North_cap = pd.DataFrame(
        [[ew_total_stocks_scale_R11_North_cap[0]]],
        columns=["value"],
        index=["tons/capita"],
    )
    ew_total_stocks_scale_R11_South_noCPA_cap = pd.DataFrame(
        [[ew_total_stocks_scale_R11_South_noCPA_cap[0]]],
        columns=["value"],
        index=["tons/capita"],
    )
    ew_total_stocks_scale_R11_CPA_cap = pd.DataFrame(
        [[ew_total_stocks_scale_R11_CPA_cap[0]]],
        columns=["value"],
        index=["tons/capita"],
    )
    
    DLS_stocks_gap_noMats_R11_scale_RDPM_save = (
        DLS_stocks_gap_noMats_R11_scale_RDPM / 1e9
    )
    
    DLS_stocks_gap_noMats_R11_scale_RDPM_save.insert(0, "unit", "Gigatons")
    DLS_stocks_gap_noMats_R11_scale_RDPM_save.set_index(
        ["unit"], append=True, inplace=True
    )
    global_stocks_dim_thresh_cap.insert(1, "unit", "tons/capita")
    global_stocks_dim_thresh_cap.set_index(["unit"], append=True, inplace=True)
    if activate_converge is not None:
        DLS_stock_gap_eff_glob_dim_prod_save = DLS_stock_gap_eff_glob_dim_prod.copy()
        DLS_stock_gap_eff_glob_dim_prod_save.insert(1, "unit", "tons")
        DLS_stock_gap_eff_glob_dim_prod_save.set_index(
            ["unit"], append=True, inplace=True
        )

        DLS_stock_gap_eff_cap_save = DLS_stock_gap_eff_cap.copy() / 1e3
        DLS_stock_gap_eff_cap_save.insert(1, "unit", "tons/capita")
        DLS_stock_gap_eff_cap_save.set_index(["unit"], append=True, inplace=True)

    """ ###############################################
                           
                            10
                      SAVE TO DATA SI
            
        ###############################################"""

    data_to_SI = {
        "Cover": cover,
        "Fig2a_map_countries": world_df_toSI,
        "Fig2a_map_countries_GN": world_df_toSI[
            world_df_toSI.index.isin(global_North_countries.MISO2)
        ],
        "Fig2a_map_countries_GS_noCPA": world_df_toSI[
            world_df_toSI.index.isin(global_South_noCPA_countries.MISO2)
        ],
        "Fig2a_map_countries_CPA": world_df_toSI[
            world_df_toSI.index.isin(CPA_countries.MISO2)
        ],
        "Fig2a_map_countries_AFR": world_df_toSI[
            world_df_toSI.index.isin(
                R11_country_correspondence[
                    R11_country_correspondence.R11 == "AFR"
                ].MISO2.to_list()
            )
        ],
        "Fig2a_map_reg_stocks_dim": regional_DLSstocks_toSI_save.rename(
            columns={"converge_value_cap": "DLSstock_threshold_converge"}
        ),
        "Fig2a_map_reg_stocks": regional_DLSstocks_toSI_save.groupby(["region", "unit"])
        .sum()
        .rename(columns={"converge_value_cap": "DLSstock_thresh_converge"}),
        "Fig2bc_stock_distr_countr": distribution_df_save,
        "Fig2b_stock_distr_stats": Fig2_stats_toSI_save,
        "DLS_stock_thresh_factDif_regio": regional_DLSstocks_toSI_save.groupby("region")
        .sum()["DLSstock_threshold"]
        .max()
        / regional_DLSstocks_toSI_save.groupby("region")
        .sum()["DLSstock_threshold"]
        .min(),
        "DLS_stock_thresh_factDif_countr": distribution_df_save[
            "(c) DLS material stocks, threshold"
        ].max()
        / distribution_df_save["(c) DLS material stocks, threshold"].min(),
        "global_av_DLSstock_thresh": global_stocks_dim_thresh_cap,
        "glob_av_DLSservice_threshold": DLS_2015_thresh_weighted_glob,
        "Fig3_global_material": global_stocks_save,
        "Fig3_global_product": global_stocks_prod_save,
        "Fig3_regio": regio_stocks_cap_save.groupby(["region", "unit"]).sum(),
        "Fig3_regio_mater": regio_stocks_cap_save,
        "Fig3_regio_mater_prod": regio_stock_prod_cap_save,
        "Fig3_stockGap_RDPM": DLS_stocks_gap_noMats_R11_scale_RDPM_save,
        "Fig4_DLS_bDLS_prop_curr": regio_stock_prod_cap_shares,
        "%_gap_closedby_beyondDLS_prod": multiple_beyondDLS_to_gap,
        "%_gap_closedby_beyondDLS_tot": share_redistr_beyondDLS_toGap_tot,
        "%_DLS_on_totStock_glob": share_DLS_glob,
        "%_DLS_on_totStock_reg": share_DLS_reg,
        "%_beyondDLS_prod_glob": share_beyondDLS_prod_glob,
        "%_beyondDLS_prod_reg": share_beyondDLS_prod_reg,
        "%_gaps_on_existTotStock": share_gap_untGlo_on_totalStock_glob,
        "%_gap_materials_glob": share_gap_materials_glob,
        "%_gap_products_glob": share_gap_products_glob,
        "%_gap_geogr_glob": regio_geogr_share_gap,
        "%_gap_products_DLSdim_glob": glob_prod_dim_share_gap,
        "Ratio_DLS_beyondDLS_mat": regio_ratio_DLS_beyond_DLS_save,
        "Ratio_DLS_beyondDLS_tot": ratio_DLS_beyondDLS,
        "Negatives_bDLS_stocks_count": text_negatives_bDLS,
        "Negatives_bDLS_stocks_mass": text_negatives_bDLS_mass,
        "Fig5a_speed": NAS_all_hist_save,
        "Fig5a_speed_2005_16_GN": NAS_all_hist_save[
            (NAS_all_hist_save.index.get_level_values(0).isin(global_North))
            & (
                NAS_all_hist_save.index.get_level_values(1).isin(
                    list(range(2005, 2017))
                )
            )
        ],
        "Fig5a_speed_2005_16_GS_noCpaMea": NAS_all_hist_save[
            (
                NAS_all_hist_save.index.get_level_values(0).isin(
                    ["AFR", "LAM", "PAS", "SAS"]
                )
            )
            & (
                NAS_all_hist_save.index.get_level_values(1).isin(
                    list(range(2005, 2017))
                )
            )
        ],
        "Fig5a_speed_2005_16_CPA": NAS_all_hist[
            (NAS_all_hist_save.index.get_level_values(0).isin(["CPA"]))
            & (
                NAS_all_hist_save.index.get_level_values(1).isin(
                    list(range(2005, 2017))
                )
            )
        ],
        "Fig5b_prospSpeed": NAS_all_prosp_save,
        "Fig5c_closeGap_Glob_curr": scenario_summary_save,
        "Fig5c_closeGap_Reg_curr": scenario_summary_regional_save,
        "ew_stock_cap_Global_North": ew_total_stocks_scale_R11_North_cap,
        "ew_stock_cap_Global_South_noCPA": ew_total_stocks_scale_R11_South_noCPA_cap,
        "ew_stock_cap_CPA": ew_total_stocks_scale_R11_CPA_cap,
    }

    # add items, if case of converged practices also calculated
    if activate_converge is not None:
        data_to_SI["Fig3_stockGapConv_RDPM"] = (
            DLS_stock_gap_eff_glob_dim_prod_save
        )
        data_to_SI["Fig3b_stockGapConv_reg_cap"] = DLS_stock_gap_eff_cap_save.dropna(
            how="all", axis=0
        )
        data_to_SI["Fig4_DLS_bDLS_prop_converged"] = shares_converge.dropna(
            how="all", axis=0
        )

    # would be better not to compute unneeded results, but this is a quick fix
    if save_results is None:
        data_to_SI = data_to_SI
    else:
        for k in save_results:
           if k not in data_to_SI:
                print("Warning, cannot save result {k} because I do not know what that is")
        new_data_to_SI = {k: v for k, v in data_to_SI.items() if k in save_results}
        data_to_SI = new_data_to_SI

    write_dict_to_excel(data_to_SI, output_path)
