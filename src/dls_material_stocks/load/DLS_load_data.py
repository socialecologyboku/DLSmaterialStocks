import numpy as np
import pandas as pd

"""Data loading functions for DLS material stocks analysis.

This module provides functions to load and format the main data inputs
for the DLS stocks model including:
- Country correspondence mappings
- Population data 
- Economy-wide material flow analysis data
- DLS service level data
- Regional diet composition data

@author: jstreeck
"""


def load_country_correspondence_dict(filename):
    """Load country correspondence across datasources."""
    country_correspondence = pd.read_excel(filename).iloc[
        :, 0:3
    ]
    country_correspondence_dict = dict(
        zip(country_correspondence["ISO_code3"], country_correspondence.MISO2)
    )
    return country_correspondence, country_correspondence_dict


def load_population_2015(filename, sheet_name, country_correspondence):
    """Load population data for 2015, format, aggregate Sudan & South Sudan (so it matches data from Kikstra et al. 2025)."""
    MISO2_population = pd.read_excel(filename, sheet_name=sheet_name, index_col=0
    ).drop(columns="Code ISO3166-1")
    MISO2_population_2015 = pd.DataFrame(MISO2_population[2015])
    MISO2_population_2015.index.names = ["region"]
    MISO2_population_2015.rename(columns={2015: "2015"}, inplace=True)
    MISO2_population_2015.reset_index(inplace=True)
    # rename, as items make problems in country correspondence (need to rename the two countries in input files accordingly)
    MISO2_population_2015["region"] = MISO2_population_2015["region"].replace(
        {"Côte d'Ivoire": "Cote dIvoire", "Réunion": "Reunion"}
    )
    MISO2_population_2015_subset = MISO2_population_2015[
        MISO2_population_2015.region.isin(country_correspondence.MISO2.to_list())
    ]
    MISO2_population_2015.set_index("region", inplace=True)
    MISO2_population_2015.rename({"South Sudan": "Sudan"}, inplace=True)
    MISO2_population_2015 = MISO2_population_2015.groupby("region").sum()
    MISO2_population_2015_subset.set_index("region", inplace=True)
    MISO2_population_2015_subset.rename({"South Sudan": "Sudan"}, inplace=True)
    MISO2_population_2015_subset = MISO2_population_2015_subset.groupby("region").sum()
    return MISO2_population_2015, MISO2_population_2015_subset


def load_population(filename, sheet_name):
    """Load population data for 2015, format, aggregate Sudan & South Sudan."""
    MISO2_population = pd.read_excel(
        filename, sheet_name=sheet_name, index_col=0
    ).drop(columns="Code ISO3166-1")
    MISO2_population.index.names = ["region"]
    MISO2_population.columns = MISO2_population.columns.astype(str)
    MISO2_population.reset_index(inplace=True)
    # rename, as items make problems in country correspondence (need to rename the two countries in input files accordingly)
    MISO2_population["region"] = MISO2_population["region"].replace(
        {"Côte d'Ivoire": "Cote dIvoire", "Réunion": "Reunion", "South Sudan": "Sudan"})
    MISO2_population.set_index("region", inplace=True)
    #MISO2_population.rename({"South Sudan": "Sudan"}, inplace=True)
    MISO2_population = MISO2_population.groupby("region").sum()
    return MISO2_population


def load_ew_MFA_data(filename, country_correspondence):
    """Load MISO2 stocks per country, end-uses, material for year 2015."""
    MISO2_stocks_2015_raw = pd.read_csv(filename)
    # rename, as items make problems in country correspondence (need to rename the two countries in input files accordingly)
    MISO2_stocks_2015_raw["region"] = MISO2_stocks_2015_raw["region"].replace(
        {"Côte d'Ivoire": "Cote dIvoire", "Réunion": "Reunion"})
    # only select regions that are common with Kikstra et al. 2025

    MISO2_stocks_2015 = MISO2_stocks_2015_raw[
        MISO2_stocks_2015_raw.region.isin(country_correspondence.MISO2.to_list())
    ].copy()
    MISO2_stocks_2015.set_index(["region", "name", "material", "sector"], inplace=True)
    MISO2_stocks_2015.rename({"South Sudan": "Sudan"}, inplace=True)
    MISO2_stocks_2015 = MISO2_stocks_2015.groupby(
        ["region", "name", "material", "sector"]
    ).sum()
    return MISO2_stocks_2015


def load_DLS_2015(
    filename, country_correspondence, country_correspondence_dict
):
    """Load DLS service data for 2015."""
    DLS_2015_raw = pd.read_csv(filename).set_index(
        ["iso", "variable", "unit"]
    )
    DLS_2015_raw.index.names = ["region", "variable", "unit"]
    # drop categories that distinguish urban/rural, only keep totals (as no urban/rural distinction for stocks available)
    DLS_2015_raw = DLS_2015_raw.drop(
        index=[
            "Cooling CON|rural",
            "Cooling CON|urban",
            "Heating CON|rural",
            "Heating CON|urban",
            "Housing|rural",
            "Housing|urban",
            "Hot Water OP|rural",
            "Hot Water OP|urban",
            "Roads|roads",
        ],
        level="variable",
    )
    DLS_2015_raw.reset_index(inplace=True)
    # rename regions according to MISO2 country labels and drop all which are not in MISO2 country list
    for key, value in country_correspondence_dict.items():
        DLS_2015_raw["region"] = DLS_2015_raw["region"].replace({key: value})
    DLS_2015_raw = DLS_2015_raw[
        DLS_2015_raw.region.isin(country_correspondence.MISO2.to_list())
    ]
    DLS_2015_raw.set_index(["region", "variable", "unit"], inplace=True)
    # subset SERVICE GAP:
    # the additional service that would need to be provide to lift all those that are currently
    # deprived up to the decent living standard.
    DLS_2015_gap = DLS_2015_raw[DLS_2015_raw.type == "Service gap"]["value"]
    # subset THRESHOLD:
    DLS_2015_thresh = DLS_2015_raw[
        DLS_2015_raw.type == "Decent Living Standard threshold"
    ]["value"]
    # calculate: FUNCTIONS REACHED = threshold - gap
    DLS_2015_funct_prov = DLS_2015_thresh - DLS_2015_gap
    return DLS_2015_funct_prov, DLS_2015_thresh


def load_regional_diets(
    filename, country_correspondence, country_correspondence_dict
):
    """Load and format regional diet data."""
    reg_diets = pd.read_csv(filename)
    reg_diets_act = reg_diets[reg_diets.Year.isin([2015, 2019])]
    reg_diets_act = reg_diets_act[reg_diets_act.Code != np.nan]
    reg_diets_act.reset_index(inplace=True, drop=True)
    reg_diets_act = reg_diets_act.iloc[reg_diets_act.Code.dropna().index, :]
    # rename regions according to MISO2 country labels and drop all which are not in MISO2 country list
    for key, value in country_correspondence_dict.items():
        reg_diets_act["Code"] = reg_diets_act["Code"].replace({key: value})
    reg_diets_act = reg_diets_act[
        reg_diets_act.Code.isin(country_correspondence.MISO2.to_list())
    ]
    # check which countries missing
    # reg_diets_missing = country_correspondence.MISO2[~country_correspondence.MISO2.isin(reg_diets_act.Code.unique().tolist())]
    # Brunei data only to 2009 --> take 2009 diet
    # Eritrea n.a. --> take Ethiopia diet
    # Equatorial Guineau n.a.? --> take cameroon
    # Guadeloupe n.a. --> take domenican rep
    # Martinique n.a.  --> take domenican rep
    # Puerto Rico n.a. --> take domenican rep
    # Reunion  --> take mauritius
    # Singapore n.a. --> take malaysia
    # Somalia n.a. --> take Ethiopia diet
    reg_diets_act = (
        reg_diets_act.iloc[:, 1:]
        .rename(columns={"Code": "region", "Year": "year"})
        .set_index(["region", "year"])
    )
    Brunei = (reg_diets[(reg_diets.Entity == "Brunei") & (reg_diets.Year == 2009)]
              .drop(columns="Code")
              .rename(columns={"Entity": "region", "Year": "year"})
              .set_index(["region", "year"]))
    reg_diets_act.loc[("Brunei", 2015), :] = Brunei.loc[("Brunei", 2009), :]
    reg_diets_act.loc[("Brunei", 2019), :] = Brunei.loc[("Brunei", 2009), :]
    reg_diets_act.loc[("Bahrain", 2015), :] = reg_diets_act.loc[("Bahrain", 2019), :]
    reg_diets_act.loc[("Bhutan", 2015), :] = reg_diets_act.loc[("Bhutan", 2019), :]
    reg_diets_act.loc[("Equatorial Guinea", 2015), :] = reg_diets_act.loc[
        ("Cameroon", 2015), :
    ]
    reg_diets_act.loc[("Equatorial Guinea", 2019), :] = reg_diets_act.loc[
        ("Cameroon", 2019), :
    ]
    reg_diets_act.loc[("Eritrea", 2015), :] = reg_diets_act.loc[("Ethiopia", 2015), :]
    reg_diets_act.loc[("Eritrea", 2019), :] = reg_diets_act.loc[("Ethiopia", 2019), :]
    reg_diets_act.loc[("Guadeloupe", 2015), :] = reg_diets_act.loc[
        ("Dominican Republic", 2015), :
    ]
    reg_diets_act.loc[("Guadeloupe", 2019), :] = reg_diets_act.loc[
        ("Dominican Republic", 2019), :
    ]
    reg_diets_act.loc[("Martinique", 2015), :] = reg_diets_act.loc[
        ("Dominican Republic", 2015), :
    ]
    reg_diets_act.loc[("Martinique", 2019), :] = reg_diets_act.loc[
        ("Dominican Republic", 2019), :
    ]
    reg_diets_act.loc[("Puerto Rico", 2015), :] = reg_diets_act.loc[
        ("Dominican Republic", 2015), :
    ]
    reg_diets_act.loc[("Puerto Rico", 2019), :] = reg_diets_act.loc[
        ("Dominican Republic", 2019), :
    ]
    reg_diets_act.loc[("Qatar", 2015), :] = reg_diets_act.loc[("Qatar", 2019), :]
    reg_diets_act.loc[("Reunion", 2015), :] = reg_diets_act.loc[("Mauritius", 2015), :]
    reg_diets_act.loc[("Reunion", 2019), :] = reg_diets_act.loc[("Mauritius", 2019), :]
    reg_diets_act.loc[("Singapore", 2015), :] = reg_diets_act.loc[("Malaysia", 2015), :]
    reg_diets_act.loc[("Singapore", 2019), :] = reg_diets_act.loc[("Malaysia", 2019), :]
    reg_diets_act.loc[("Somalia", 2015), :] = reg_diets_act.loc[("Ethiopia", 2015), :]
    reg_diets_act.loc[("Somalia", 2019), :] = reg_diets_act.loc[("Ethiopia", 2019), :]
    reg_diets_act = reg_diets_act.replace(np.nan, 0)
    reg_diets_act_shares = reg_diets_act.divide(reg_diets_act.sum(axis=1), axis=0)
    # len(reg_diets_act.index.get_level_values(0).unique())

    # drop columns for which no representative ecoinvent data (eggs, alcohol, misc.) + scale back to 100% over all other entries
    reg_diets_act_shares = reg_diets_act_shares.drop(
        columns=[
            "Miscellaneous group | 00002928 || Food available for consumption | 0664pc || kilocalories per day per capita",
            "Alcoholic Beverages | 00002924 || Food available for consumption | 0664pc || kilocalories per day per capita",
            "Eggs | 00002949 || Food available for consumption | 0664pc || kilocalories per day per capita",
            "Meat, Other | 00002735 || Food available for consumption | 0664pc || kilocalories per day per capita",
        ]
    )
    # check 100%
    reg_diets_act_shares.sort_index(inplace=True)
    # shares_dropped = reg_diets_act_shares.sum(axis=1)
    reg_diets_act_shares_final = reg_diets_act_shares.divide(
        reg_diets_act_shares.sum(axis=1), axis=0
    )
    return reg_diets_act_shares_final
