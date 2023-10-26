# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Solves optimal operation and capacity for a network with the option to
iteratively optimize while updating line reactances.

This script is used for optimizing the electrical network as well as the
sector coupled network.

Description
-----------

Total annual system costs are minimised with PyPSA. The full formulation of the
linear optimal power flow (plus investment planning
is provided in the
`documentation of PyPSA <https://pypsa.readthedocs.io/en/latest/optimal_power_flow.html#linear-optimal-power-flow>`_.

The optimization is based on the :func:`network.optimize` function.
Additionally, some extra constraints specified in :mod:`solve_network` are added.

.. note::

    The rules ``solve_elec_networks`` and ``solve_sector_networks`` run
    the workflow for all scenarios in the configuration file (``scenario:``)
    based on the rule :mod:`solve_network`.
"""
import os
import logging
import re

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from _helpers import (
    configure_logging,
    override_component_attrs,
    update_config_with_sector_opts,
)
from vresutils.benchmark import memory_logger

logger = logging.getLogger(__name__)
pypsa.pf.logger.setLevel(logging.WARNING)


def add_land_use_constraint(n, config):
    if "m" in snakemake.wildcards.clusters:
        _add_land_use_constraint_m(n, config)
    else:
        _add_land_use_constraint(n, config)


def _add_land_use_constraint(n, config):
    # warning: this will miss existing offwind which is not classed AC-DC and has carrier 'offwind'

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        ext_i = (n.generators.carrier == carrier) & ~n.generators.p_nom_extendable
        existing = (
            n.generators.loc[ext_i, "p_nom"]
            .groupby(n.generators.bus.map(n.buses.location))
            .sum()
        )
        existing.index += " " + carrier + "-" + snakemake.wildcards.planning_horizons
        n.generators.loc[existing.index, "p_nom_max"] -= existing

    # check if existing capacities are larger than technical potential
    existing_large = n.generators[
        n.generators["p_nom_min"] > n.generators["p_nom_max"]
    ].index
    if len(existing_large):
        logger.warning(
            f"Existing capacities larger than technical potential for {existing_large},\
                       adjust technical potential to existing capacities"
        )
        n.generators.loc[existing_large, "p_nom_max"] = n.generators.loc[
            existing_large, "p_nom_min"
        ]

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def _add_land_use_constraint_m(n, config):
    # if generators clustering is lower than network clustering, land_use accounting is at generators clusters

    planning_horizons = config["scenario"]["planning_horizons"]
    grouping_years = config["existing_capacities"]["grouping_years"]
    current_horizon = snakemake.wildcards.planning_horizons

    for carrier in ["solar", "onwind", "offwind-ac", "offwind-dc"]:
        existing = n.generators.loc[n.generators.carrier == carrier, "p_nom"]
        ind = list(
            set(
                [
                    i.split(sep=" ")[0] + " " + i.split(sep=" ")[1]
                    for i in existing.index
                ]
            )
        )

        previous_years = [
            str(y)
            for y in planning_horizons + grouping_years
            if y < int(snakemake.wildcards.planning_horizons)
        ]

        for p_year in previous_years:
            ind2 = [
                i for i in ind if i + " " + carrier + "-" + p_year in existing.index
            ]
            sel_current = [i + " " + carrier + "-" + current_horizon for i in ind2]
            sel_p_year = [i + " " + carrier + "-" + p_year for i in ind2]
            n.generators.loc[sel_current, "p_nom_max"] -= existing.loc[
                sel_p_year
            ].rename(lambda x: x[:-4] + current_horizon)

    n.generators.p_nom_max.clip(lower=0, inplace=True)


def add_co2_sequestration_limit(n, limit=200):
    """
    Add a global constraint on the amount of Mt CO2 that can be sequestered.
    """
    n.carriers.loc["co2 stored", "co2_absorptions"] = -1
    n.carriers.co2_absorptions = n.carriers.co2_absorptions.fillna(0)

    limit = limit * 1e6
    for o in opts:
        if "seq" not in o:
            continue
        limit = float(o[o.find("seq") + 3 :]) * 1e6
        break

    n.add(
        "GlobalConstraint",
        "co2_sequestration_limit",
        sense="<=",
        constant=limit,
        type="primary_energy",
        carrier_attribute="co2_absorptions",
    )


def prepare_network(n, solve_opts=None, config=None):
    if "clip_p_max_pu" in solve_opts:
        for df in (
            n.generators_t.p_max_pu,
            n.generators_t.p_min_pu,  # TODO: check if this can be removed
            n.storage_units_t.inflow,
        ):
            df.where(df > solve_opts["clip_p_max_pu"], other=0.0, inplace=True)

    if solve_opts.get("load_shedding"):
        # intersect between macroeconomic and surveybased willingness to pay
        # http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full
        # TODO: retrieve color and nice name from config
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
        buses_i = n.buses.query("carrier == 'AC'").index
        if not np.isscalar(load_shedding):
            # TODO: do not scale via sign attribute (use Eur/MWh instead of Eur/kWh)
            load_shedding = 1e2  # Eur/kWh

        n.madd(
            "Generator",
            buses_i,
            " load",
            bus=n.buses.index,
            carrier="load",
            sign=1e-3,  # Adjust sign to measure p and p_nom in kW instead of MW
            marginal_cost=load_shedding,  # Eur/kWh
            p_nom=1e9,  # kW
        )

    if solve_opts.get("noisy_costs"):
        for t in n.iterate_components():
            # if 'capital_cost' in t.df:
            #    t.df['capital_cost'] += 1e1 + 2.*(np.random.random(len(t.df)) - 0.5)
            if "marginal_cost" in t.df:
                t.df["marginal_cost"] += 1e-2 + 2e-3 * (
                    np.random.random(len(t.df)) - 0.5
                )

        for t in n.iterate_components(["Line", "Link"]):
            t.df["capital_cost"] += (
                1e-1 + 2e-2 * (np.random.random(len(t.df)) - 0.5)
            ) * t.df["length"]

    if solve_opts.get("nhours"):
        nhours = solve_opts["nhours"]
        n.set_snapshots(n.snapshots[:nhours])
        n.snapshot_weightings[:] = 8760.0 / nhours

    if config["foresight"] == "myopic":
        add_land_use_constraint(n, config)

    if n.stores.carrier.eq("co2 stored").any():
        limit = config["sector"].get("co2_sequestration_potential", 200)
        add_co2_sequestration_limit(n, limit=limit)

    return n


def add_CCL_constraints(n, config):
    """
    Add CCL (country & carrier limit) constraint to the network.

    Add minimum and maximum levels of generator nominal capacity per carrier
    for individual countries. Opts and path for agg_p_nom_minmax.csv must be defined
    in config.yaml. Default file is available at data/agg_p_nom_minmax.csv.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-CCL-24H]
    electricity:
        agg_p_nom_limits: data/agg_p_nom_minmax.csv
    """
    agg_p_nom_minmax = pd.read_csv(
        config["electricity"]["agg_p_nom_limits"], index_col=[0, 1]
    )
    logger.info("Adding generation capacity constraints per carrier and country")
    p_nom = n.model["Generator-p_nom"]

    gens = n.generators.query("p_nom_extendable").rename_axis(index="Generator-ext")
    grouper = [gens.bus.map(n.buses.country), gens.carrier]
    grouper = xr.DataArray(pd.MultiIndex.from_arrays(grouper), dims=["Generator-ext"])
    lhs = p_nom.groupby(grouper).sum().rename(bus="country")

    minimum = xr.DataArray(agg_p_nom_minmax["min"].dropna()).rename(dim_0="group")
    index = minimum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) >= minimum.loc[index], name="agg_p_nom_min"
        )

    maximum = xr.DataArray(agg_p_nom_minmax["max"].dropna()).rename(dim_0="group")
    index = maximum.indexes["group"].intersection(lhs.indexes["group"])
    if not index.empty:
        n.model.add_constraints(
            lhs.sel(group=index) <= maximum.loc[index], name="agg_p_nom_max"
        )


def add_EQ_constraints(n, o, scaling=1e-1):
    """
    Add equity constraints to the network.

    Currently this is only implemented for the electricity sector only.

    Opts must be specified in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    o : str

    Example
    -------
    scenario:
        opts: [Co2L-EQ0.7-24H]

    Require each country or node to on average produce a minimal share
    of its total electricity consumption itself. Example: EQ0.7c demands each country
    to produce on average at least 70% of its consumption; EQ0.7 demands
    each node to produce on average at least 70% of its consumption.
    """
    # TODO: Generalize to cover myopic and other sectors?
    float_regex = "[0-9]*\.?[0-9]+"
    level = float(re.findall(float_regex, o)[0])
    if o[-1] == "c":
        ggrouper = n.generators.bus.map(n.buses.country)
        lgrouper = n.loads.bus.map(n.buses.country)
        sgrouper = n.storage_units.bus.map(n.buses.country)
    else:
        ggrouper = n.generators.bus
        lgrouper = n.loads.bus
        sgrouper = n.storage_units.bus
    load = (
        n.snapshot_weightings.generators
        @ n.loads_t.p_set.groupby(lgrouper, axis=1).sum()
    )
    inflow = (
        n.snapshot_weightings.stores
        @ n.storage_units_t.inflow.groupby(sgrouper, axis=1).sum()
    )
    inflow = inflow.reindex(load.index).fillna(0.0)
    rhs = scaling * (level * load - inflow)
    p = n.model["Generator-p"]
    lhs_gen = (
        (p * (n.snapshot_weightings.generators * scaling))
        .groupby(ggrouper.to_xarray())
        .sum()
        .sum("snapshot")
    )
    # TODO: double check that this is really needed, why do have to subtract the spillage
    if not n.storage_units_t.inflow.empty:
        spillage = n.model["StorageUnit-spill"]
        lhs_spill = (
            (spillage * (-n.snapshot_weightings.stores * scaling))
            .groupby(sgrouper.to_xarray())
            .sum()
            .sum("snapshot")
        )
        lhs = lhs_gen + lhs_spill
    else:
        lhs = lhs_gen
    n.model.add_constraints(lhs >= rhs, name="equity_min")


def add_BAU_constraints(n, config):
    """
    Add a per-carrier minimal overall capacity.

    BAU_mincapacities and opts must be adjusted in the config.yaml.

    Parameters
    ----------
    n : pypsa.Network
    config : dict

    Example
    -------
    scenario:
        opts: [Co2L-BAU-24H]
    electricity:
        BAU_mincapacities:
            solar: 0
            onwind: 0
            OCGT: 100000
            offwind-ac: 0
            offwind-dc: 0
    Which sets minimum expansion across all nodes e.g. in Europe to 100GW.
    OCGT bus 1 + OCGT bus 2 + ... > 100000
    """
    mincaps = pd.Series(config["electricity"]["BAU_mincapacities"])
    p_nom = n.model["Generator-p_nom"]
    ext_i = n.generators.query("p_nom_extendable")
    ext_carrier_i = xr.DataArray(ext_i.carrier.rename_axis("Generator-ext"))
    lhs = p_nom.groupby(ext_carrier_i).sum()
    index = mincaps.index.intersection(lhs.indexes["carrier"])
    rhs = mincaps[index].rename_axis("carrier")
    n.model.add_constraints(lhs >= rhs, name="bau_mincaps")


# TODO: think about removing or make per country
def add_SAFE_constraints(n, config):
    """
    Add a capacity reserve margin of a certain fraction above the peak demand.
    Renewable generators and storage do not contribute. Ignores network.

    Parameters
    ----------
        n : pypsa.Network
        config : dict

    Example
    -------
    config.yaml requires to specify opts:

    scenario:
        opts: [Co2L-SAFE-24H]
    electricity:
        SAFE_reservemargin: 0.1
    Which sets a reserve margin of 10% above the peak demand.
    """
    peakdemand = n.loads_t.p_set.sum(axis=1).max()
    margin = 1.0 + config["electricity"]["SAFE_reservemargin"]
    reserve_margin = peakdemand * margin
    # TODO: do not take this from the plotting config!
    conv_techs = config["plotting"]["conv_techs"]
    ext_gens_i = n.generators.query("carrier in @conv_techs & p_nom_extendable").index
    p_nom = n.model["Generator-p_nom"].loc[ext_gens_i]
    lhs = p_nom.sum()
    exist_conv_caps = n.generators.query(
        "~p_nom_extendable & carrier in @conv_techs"
    ).p_nom.sum()
    rhs = reserve_margin - exist_conv_caps
    n.model.add_constraints(lhs >= rhs, name="safe_mintotalcap")


def add_operational_reserve_margin(n, sns, config):
    """
    Build reserve margin constraints based on the formulation given in
    https://genxproject.github.io/GenX/dev/core/#Reserves.

    Parameters
    ----------
        n : pypsa.Network
        sns: pd.DatetimeIndex
        config : dict

    Example:
    --------
    config.yaml requires to specify operational_reserve:
    operational_reserve: # like https://genxproject.github.io/GenX/dev/core/#Reserves
        activate: true
        epsilon_load: 0.02 # percentage of load at each snapshot
        epsilon_vres: 0.02 # percentage of VRES at each snapshot
        contingency: 400000 # MW
    """
    reserve_config = config["electricity"]["operational_reserve"]
    EPSILON_LOAD = reserve_config["epsilon_load"]
    EPSILON_VRES = reserve_config["epsilon_vres"]
    CONTINGENCY = reserve_config["contingency"]

    # Reserve Variables
    n.model.add_variables(
        0, np.inf, coords=[sns, n.generators.index], name="Generator-r"
    )
    reserve = n.model["Generator-r"]
    lhs = reserve.sum("Generator")

    # Share of extendable renewable capacities
    ext_i = n.generators.query("p_nom_extendable").index
    vres_i = n.generators_t.p_max_pu.columns
    if not ext_i.empty and not vres_i.empty:
        capacity_factor = n.generators_t.p_max_pu[vres_i.intersection(ext_i)]
        p_nom_vres = (
            n.model["Generator-p_nom"]
            .loc[vres_i.intersection(ext_i)]
            .rename({"Generator-ext": "Generator"})
        )
        lhs = lhs + (p_nom_vres * (-EPSILON_VRES * capacity_factor)).sum()

    # Total demand per t
    demand = n.loads_t.p_set.sum(axis=1)

    # VRES potential of non extendable generators
    capacity_factor = n.generators_t.p_max_pu[vres_i.difference(ext_i)]
    renewable_capacity = n.generators.p_nom[vres_i.difference(ext_i)]
    potential = (capacity_factor * renewable_capacity).sum(axis=1)

    # Right-hand-side
    rhs = EPSILON_LOAD * demand + EPSILON_VRES * potential + CONTINGENCY

    n.model.add_constraints(lhs >= rhs, name="reserve_margin")

    reserve = n.model["Generator-r"]

    lhs = n.model.constraints["Generator-fix-p-upper"].lhs
    lhs = lhs + reserve.loc[:, lhs.coords["Generator-fix"]].drop("Generator")
    rhs = n.model.constraints["Generator-fix-p-upper"].rhs
    n.model.add_constraints(lhs <= rhs, name="Generator-fix-p-upper-reserve")

    lhs = n.model.constraints["Generator-ext-p-upper"].lhs
    lhs = lhs + reserve.loc[:, lhs.coords["Generator-ext"]].drop("Generator")
    rhs = n.model.constraints["Generator-ext-p-upper"].rhs
    n.model.add_constraints(lhs >= rhs, name="Generator-ext-p-upper-reserve")


def add_battery_constraints(n):
    """
    Add constraint ensuring that charger = discharger, i.e.
    1 * charger_size - efficiency * discharger_size = 0
    """
    if not n.links.p_nom_extendable.any():
        return

    discharger_bool = n.links.index.str.contains("battery discharger")
    charger_bool = n.links.index.str.contains("battery charger")

    dischargers_ext = n.links[discharger_bool].query("p_nom_extendable").index
    chargers_ext = n.links[charger_bool].query("p_nom_extendable").index

    eff = n.links.efficiency[dischargers_ext].values
    lhs = (
        n.model["Link-p_nom"].loc[chargers_ext]
        - n.model["Link-p_nom"].loc[dischargers_ext] * eff
    )

    n.model.add_constraints(lhs == 0, name="Link-charger_ratio")


def add_chp_constraints(n):
    electric = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("electric")
    )
    heat = (
        n.links.index.str.contains("urban central")
        & n.links.index.str.contains("CHP")
        & n.links.index.str.contains("heat")
    )

    electric_ext = n.links[electric].query("p_nom_extendable").index
    heat_ext = n.links[heat].query("p_nom_extendable").index

    electric_fix = n.links[electric].query("~p_nom_extendable").index
    heat_fix = n.links[heat].query("~p_nom_extendable").index

    p = n.model["Link-p"]  # dimension: [time, link]

    # output ratio between heat and electricity and top_iso_fuel_line for extendable
    if not electric_ext.empty:
        p_nom = n.model["Link-p_nom"]

        lhs = (
            p_nom.loc[electric_ext]
            * (n.links.p_nom_ratio * n.links.efficiency)[electric_ext].values
            - p_nom.loc[heat_ext] * n.links.efficiency[heat_ext].values
        )
        n.model.add_constraints(lhs == 0, name="chplink-fix_p_nom_ratio")

        rename = {"Link-ext": "Link"}
        lhs = (
            p.loc[:, electric_ext]
            + p.loc[:, heat_ext]
            - p_nom.rename(rename).loc[electric_ext]
        )
        n.model.add_constraints(lhs <= 0, name="chplink-top_iso_fuel_line_ext")

    # top_iso_fuel_line for fixed
    if not electric_fix.empty:
        lhs = p.loc[:, electric_fix] + p.loc[:, heat_fix]
        rhs = n.links.p_nom[electric_fix]
        n.model.add_constraints(lhs <= rhs, name="chplink-top_iso_fuel_line_fix")

    # back-pressure
    if not electric.empty:
        lhs = (
            p.loc[:, heat] * (n.links.efficiency[heat] * n.links.c_b[electric].values)
            - p.loc[:, electric] * n.links.efficiency[electric]
        )
        n.model.add_constraints(lhs <= rhs, name="chplink-backpressure")


def add_pipe_retrofit_constraint(n):
    """
    Add constraint for retrofitting existing CH4 pipelines to H2 pipelines.
    """
    gas_pipes_i = n.links.query("carrier == 'gas pipeline' and p_nom_extendable").index
    h2_retrofitted_i = n.links.query(
        "carrier == 'H2 pipeline retrofitted' and p_nom_extendable"
    ).index

    if h2_retrofitted_i.empty or gas_pipes_i.empty:
        return

    p_nom = n.model["Link-p_nom"]

    CH4_per_H2 = 1 / n.config["sector"]["H2_retrofit_capacity_per_CH4"]
    lhs = p_nom.loc[gas_pipes_i] + CH4_per_H2 * p_nom.loc[h2_retrofitted_i]
    rhs = n.links.p_nom[gas_pipes_i].rename_axis("Link-ext")

    n.model.add_constraints(lhs == rhs, name="Link-pipe_retrofit")

### ADDED NEW FUNCTION HERE ###

def ev_share_const(const_country, land_transport_electric_share):
    #const_country: Country ISO code, e.g. 'DK'
    #land_transport_electric_share: Share of EV land transport in constrained country

    global_land_transport_electric_share = n.config["sector"]["land_transport_electric_share"][2030]    #NB 2030 is hardcoded
    k_fac = land_transport_electric_share/global_land_transport_electric_share

    #BEV charger constraint
        #Extract BEV charger links for DK nodes
    link_list = n.links.bus0.to_list()
    mask_bus_BEV_charger = pd.Series([bus.startswith(const_country) for bus in link_list])
    mask_carrier_BEV_charger = n.links.carrier=='BEV charger'
    mask_carrier_BEV_charger.index = mask_bus_BEV_charger.index
    mask_DK_BEV_charger = mask_bus_BEV_charger & mask_carrier_BEV_charger
    mask_DK_BEV_charger.index = n.links.index

        #Scale link capacity
    n.links.loc[mask_DK_BEV_charger,'p_nom'] *= k_fac

    #BEV battery capacity constraint
        #Extract BEV battery stores for DK nodes
    store_list = n.stores.bus.to_list()
    mask_bus_BEV_battery = pd.Series([bus.startswith(const_country) for bus in store_list])
    mask_carrier_BEV_battery = n.stores.carrier=='battery storage'
    mask_carrier_BEV_battery.index = mask_bus_BEV_battery.index
    mask_DK_BEV_battery = mask_bus_BEV_battery & mask_carrier_BEV_battery
    mask_DK_BEV_battery.index = n.stores.index

        #Scale store capacity
    n.stores.loc[mask_DK_BEV_battery,'e_nom'] *= k_fac

    #ICE constraint
        #Transport load
    clusters = snakemake.wildcards.clusters
    simpl = snakemake.wildcards.simpl
    simpl_part = f"s{simpl}_" if simpl else "s_"
    file_path = f"resources/transport_demand_{simpl_part}{clusters}.csv"
    transport = pd.read_csv(file_path, index_col=0, parse_dates=True)

        #Ice scale factor
    global_land_transport_ICE_share = 1 - global_land_transport_electric_share
    land_transport_ICE_share = 1 - land_transport_electric_share
    ICE_fac = land_transport_ICE_share/global_land_transport_ICE_share

    if snakemake.config["co2_local_atmosphere"]:

        #Extract oil emission loads specific to target country nodes
        load_list = n.loads.index.to_list()
        mask_loads_country = pd.Series([load.startswith(const_country) for load in load_list])
        mask_loads_country.index = n.loads.index
        mask_loads_carrier = n.loads.carrier=='land transport oil emissions'
        mask_loads_carrier.index = n.loads.index
        mask_loads = mask_loads_country & mask_loads_carrier
        
        #Scale oil emission capacity
        n.loads.loc[mask_loads,'p_set'] *= ICE_fac

    else:

        #Extract nodes and country specific nodes
        bus_list = n.buses.index.to_list()
        mask_nodes = n.buses.country.apply(lambda x: x!='')
        mask_country = pd.Series([bus.startswith(const_country) for bus in bus_list])
        mask_country.index = mask_nodes.index
        mask_nodes_country = mask_nodes & mask_country
        nodes = n.buses.loc[mask_nodes].index
        nodes_country = n.buses.loc[mask_nodes_country].index
        
        #Extract transport loads in network and in target country
        transport_country = transport[nodes_country].sum().sum()
        transport_global = transport[nodes].sum().sum()
        country_share = transport_country/transport_global
        
        #Scale oil emission capacity
        CO2_global = n.loads.loc['land transport oil emissions'].p_set
        n.loads.loc['land transport oil emissions','p_set'] = CO2_global - CO2_global*country_share + CO2_global*country_share*ICE_fac

def remove_myopic_RE_cap():
    # Load generators
    gen = n.generators

    # Carriers to remove
    mask_carrier = gen.index.str.contains('onwind|solar|offwind-ac|offwind-dc', case=False, regex=True)

    # Remove only non-extendable
    mask_ext = gen['p_nom_extendable'] == False

    # Generators to be removed
    gen_RE_myopic = gen[mask_carrier & mask_ext].index

    # Remove selected generators
    n.generators.drop(gen_RE_myopic,inplace = True)
          
def country_generator_const(const_country, const_carrier, gen_limit, type):

    #const_country: Country ISO code of constrained contries in square brackets, e.g. ['DK']
    #const_carrier: Constrained carriers in squared brackets, e.g. ['offwind-ac','offwind-dc']
    #gen_limit: Country-wide generator limit in MW, e.g. 5000
    #type: Constraint type: 'E'=equality, 'LE'=less or equal, 'GE'=greater or equal

    #Load generators
    gen = n.generators
    mask_ext = gen['p_nom_extendable'] == True

    #Extract constrained generators by 2-character prefix and carrier name
    gen_ext = gen[mask_ext] 
    gen_series = gen_ext.index.to_list()
    gen_series_scrub = pd.DataFrame([s[:2] for s in gen_series])
    gen_series_scrub.index = gen_ext.carrier.index
    grouper = pd.concat([gen_ext.carrier,gen_series_scrub],axis=1)
    const_idx = grouper[(grouper.iloc[:, 0].isin(const_carrier)) & (grouper.iloc[:, 1].isin(const_country))].index
    const_gen = gen_ext.loc[const_idx].index.to_list()

    # Brownfield installed generators
    gen_brown = gen[~mask_ext]
    mask_country = gen_brown['bus'].str.contains('|'.join(const_country), case=False, na=False)
    mask_carrier = gen_brown['carrier'].isin(const_carrier)
    intalled_gen = gen_brown.loc[mask_country & mask_carrier,'p_nom'].sum()

    #Nominal capacity in model object
    gen_p = n.model["Generator-p_nom"]

    #Apply constraint
    name_str = "_".join(const_country + const_carrier + ["custom_const"])

    gen_limit_brown = gen_limit - intalled_gen

    if type == 'E':
        n.model.add_constraints(gen_p.loc[const_gen].sum()==gen_limit_brown,name=name_str)
    elif type == 'LE':
        n.model.add_constraints(gen_p.loc[const_gen].sum()<=gen_limit_brown,name=name_str)
    elif type == 'GE':
        n.model.add_constraints(gen_p.loc[const_gen].sum()>=gen_limit_brown,name=name_str)

def country_p2h_const(const_country, link_limit):
    #const_country: Country ISO code/codes as string, e.g. 'DK'
    #link_limit: Country-wide P2H limit [MW], e.g. 5000

    const_country = 'DK'
    
    #Extract P2H links in constrained country
    links = n.links
    mask_ext = links['p_nom_extendable'] == True
    links_ext = links[mask_ext]
    
    mask_p2h = links_ext.index.str.contains('heat pump|resistive heater', case=False, regex=True)
    mask_country = links_ext['bus0'].str.contains(const_country, case=False, regex=True)
    mask_comb = mask_p2h & mask_country
    links_p2h = links_ext.loc[mask_comb].index.to_list()
    
    # Brownfield installed generators
    links_brown = links[~mask_ext]
    mask_p2h = links_brown.index.str.contains('heat pump|resistive heater', case=False, regex=True)
    mask_country = links_brown['bus0'].str.contains(const_country, case=False, regex=True)
    intalled_links = links_brown.loc[mask_country & mask_p2h,'p_nom'].sum()
    
    #Nominal capacity in model object
    link_p = n.model["Link-p_nom"]
    
    link_limit_brown = link_limit - intalled_links
    
    #Add constraint
    name_str = const_country + "_P2H_custom_const"
    n.model.add_constraints(link_p.loc[links_p2h].sum()<=link_limit_brown, name=name_str)

def country_p2x_const(const_country, link_limit, type):
    #const_country: Country ISO code, e.g. 'DK'
    #link_limit: Country-wide H2 electrolysis limit [MW], e.g. 5000
    #type: Constraint type: 'E'=equality, 'LE'=less or equal, 'GE'=greater or equal

    #Load links
    links = n.links
    const_carrier = "H2 Electrolysis"

    #Extract P2X links in constrained country
    mask_p2x = links.index.str.contains('H2 Electrolysis', case=False, regex=True)
    mask_country = links['bus0'].str.contains(const_country, case=False, regex=True)
    mask_comb = mask_p2x & mask_country
    links_p2x = links.loc[mask_comb].index.to_list()
    
    #Nominal capacity in model object
    link_p = n.model["Link-p_nom"]
    
    #Add constraint
    name_str = const_country + "_P2X_custom_const"

    if type == 'E':
        n.model.add_constraints(link_p.loc[links_p2x].sum()==link_limit, name=name_str)
    elif type == 'LE':
        n.model.add_constraints(link_p.loc[links_p2x].sum()<=link_limit,name=name_str)
    elif type == 'GE':
        n.model.add_constraints(link_p.loc[links_p2x].sum()>=link_limit, name=name_str)


### END OF ADDED NEW FUNCTION ###

def extra_functionality(n, snapshots):
    """
    Collects supplementary constraints which will be passed to
    ``pypsa.optimization.optimize``.

    If you want to enforce additional custom constraints, this is a good
    location to add them. The arguments ``opts`` and
    ``snakemake.config`` are expected to be attached to the network.
    """
    opts = n.opts
    config = n.config
    if "BAU" in opts and n.generators.p_nom_extendable.any():
        add_BAU_constraints(n, config)
    if "SAFE" in opts and n.generators.p_nom_extendable.any():
        add_SAFE_constraints(n, config)
    if "CCL" in opts and n.generators.p_nom_extendable.any():
        add_CCL_constraints(n, config)
    reserve = config["electricity"].get("operational_reserve", {})
    if reserve.get("activate"):
        add_operational_reserve_margin(n, snapshots, config)
    for o in opts:
        if "EQ" in o:
            add_EQ_constraints(n, o)
    add_battery_constraints(n)
    add_pipe_retrofit_constraint(n)

    ### ADDED NEW CODE HERE ###

#Add scenario constraints
    remove_myopic_RE_cap() # It is important that this comes before country_generator_const if active!

    ev_share_const('DK', 0.21)

    country_generator_const(['DK'], ['onwind'], 6075, 'LE')

    country_generator_const(['DK'], ['offwind','offwind-ac','offwind-dc'], 8985, 'LE')

    country_generator_const(['DK'], ['solar','solar rooftop'], 13011, 'LE')

    # country_p2h_const('DK', 2775)

    # country_p2x_const('DK', 4000, 'E')

#Add local co2 constraints

    #Wildcards
    simpl = snakemake.wildcards.simpl
    clusters = snakemake.wildcards.clusters
    ll = snakemake.wildcards.ll
    opts = snakemake.wildcards.opts
    sector_opts = snakemake.wildcards.sector_opts
    planning_horizons = snakemake.wildcards.planning_horizons

    file_path = f'resources/co2_budget_{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.csv'
    df_co2 = pd.read_csv(file_path)
    df_co2.columns = ['co2 atmosphere', 'co2 budget']

    #Sort df in DK nodes and non-DK nodes
    dk_nodes =  [node for node in df_co2.loc[:,'co2 atmosphere'] if 'DK' in node]
    dk_df_co2 = df_co2[df_co2['co2 atmosphere'].isin(dk_nodes)]
    other_df_co2 = df_co2[~df_co2['co2 atmosphere'].isin(dk_nodes)]

    #Store nominal capacity in model object
    store_e = n.model["Store-e_nom"]

    #Apply local CO2 constraints nodally for non-DK countries
    logger.info(f"Adding local CO2 constraint for non-DK nodes")
    co2_budget = other_df_co2['co2 budget']
    co2_budget.index = other_df_co2['co2 atmosphere']

    for atmosphere in other_df_co2.loc[:,'co2 atmosphere']:
        co2_cap = co2_budget[atmosphere]
        name_str = atmosphere + " local co2 constraint"
        n.model.add_constraints(store_e[atmosphere] <= co2_cap, name=name_str)

    #Apply local CO2 constraint by country aggregate in DK
    logger.info(f"Adding local CO2 constraint for DK nodes")
    dk_co2_budget = dk_df_co2['co2 budget'].sum()
    name_str = "DK local co2 constraint"
    n.model.add_constraints(store_e.loc[dk_nodes].sum()<=dk_co2_budget, name=name_str)

    ### END OF ADDED CODE ###

def solve_network(n, config, opts="", **kwargs):
    set_of_options = config["solving"]["solver"]["options"]
    solver_options = (
        config["solving"]["solver_options"][set_of_options] if set_of_options else {}
    )
    solver_name = config["solving"]["solver"]["name"]
    cf_solving = config["solving"]["options"]
    track_iterations = cf_solving.get("track_iterations", False)
    min_iterations = cf_solving.get("min_iterations", 4)
    max_iterations = cf_solving.get("max_iterations", 6)

    # add to network for extra_functionality
    n.config = config
    n.opts = opts

    skip_iterations = cf_solving.get("skip_iterations", False)
    if not n.lines.s_nom_extendable.any():
        skip_iterations = True
        logger.info("No expandable lines found. Skipping iterative solving.")

    from pathlib import Path
    import os

    tmpdir = '/scratch/' + os.environ['SLURM_JOB_ID']

    # if tmpdir is not None:
    #     Path(tmpdir).mkdir(parents=True, exist_ok=True)

    if skip_iterations:
        status, condition = n.optimize(
            solver_name=solver_name,
            model_kwargs = {"solver_dir": tmpdir},
            extra_functionality=extra_functionality,
            **solver_options,
            **kwargs,
        )
    else:
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            solver_name=solver_name,
            model_kwargs = {"solver_dir": tmpdir},
            track_iterations=track_iterations,
            min_iterations=min_iterations,
            max_iterations=max_iterations,
            extra_functionality=extra_functionality,
            **solver_options,
            **kwargs,
        )

    if status != "ok":
        logger.warning(
            f"Solving status '{status}' with termination condition '{condition}'"
        )
    if "infeasible" in condition:
        raise RuntimeError("Solving status 'infeasible'")

    return n


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sector_network",
            configfiles="test/config.overnight.yaml",
            simpl="",
            opts="",
            clusters="5",
            ll="v1.5",
            sector_opts="CO2L0-24H-T-H-B-I-A-solar+p3-dist1",
            planning_horizons="2030",
        )
    configure_logging(snakemake)
    if "sector_opts" in snakemake.wildcards.keys():
        update_config_with_sector_opts(
            snakemake.config, snakemake.wildcards.sector_opts
        )

    opts = snakemake.wildcards.opts
    if "sector_opts" in snakemake.wildcards.keys():
        opts += "-" + snakemake.wildcards.sector_opts
    opts = [o for o in opts.split("-") if o != ""]
    solve_opts = snakemake.config["solving"]["options"]

    np.random.seed(solve_opts.get("seed", 123))

    fn = getattr(snakemake.log, "memory", None)
    with memory_logger(filename=fn, interval=30.0) as mem:
        if "overrides" in snakemake.input.keys():
            overrides = override_component_attrs(snakemake.input.overrides)
            n = pypsa.Network(
                snakemake.input.network, override_component_attrs=overrides
            )
        else:
            n = pypsa.Network(snakemake.input.network)

        n = prepare_network(n, solve_opts, config=snakemake.config)

        n = solve_network(
            n, config=snakemake.config, opts=opts, log_fn=snakemake.log.solver
        )

        n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
        n.export_to_netcdf(snakemake.output[0])

        ### OBS ###

        #Export duals
        
        if n.config["enable"]["save_duals"]:

            #Extract wildcards
            simpl = snakemake.wildcards.simpl
            clusters = snakemake.wildcards.clusters
            ll = snakemake.wildcards.ll
            opts = snakemake.wildcards.opts
            sector_opts = snakemake.wildcards.sector_opts
            planning_horizons = snakemake.wildcards.planning_horizons

            #Create directory
            os.makedirs(f'results/{n.config["run"]["name"]}/duals', exist_ok=True)
            
            #Define name of nodes and co2 constraints for non DK nodes
            nodes = pd.Series([node[:5] for node in n.generators.index.to_list()]).unique()
            nodes = pd.Series(nodes[0:-4]).to_list()
            other_nodes = [node for node in nodes if 'DK' not in node]
            co2_constraints = [node + ' co2 atmosphere local co2 constraint' for node in other_nodes]

            #Extract co2 constraint duals
            file_path_co2_duals = f'results/{n.config["run"]["name"]}/duals/co2_duals_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.csv'
            co2_constraint_duals = pd.Series([n.model.dual[node].values for node in co2_constraints], index=co2_constraints)
            dk_co2_constraint_duals = pd.Series( n.model.dual["DK local co2 constraint"].values, index=["DK local co2 constraint"])
            co2_constraint_duals = co2_constraint_duals.append(dk_co2_constraint_duals)
            co2_constraint_duals = co2_constraint_duals.abs().astype(float)
            co2_constraint_duals.to_csv(file_path_co2_duals)

            #Extract custom constraint duals
            file_path_custom_const_duals = f'results/{n.config["run"]["name"]}/duals/custom_const_duals_s{simpl}_{clusters}_l{ll}_{opts}_{sector_opts}_{planning_horizons}.csv'
            const_names = list(n.model.dual.keys())
            custom_const_names = [name for name in const_names if 'custom_const' in name]
            custom_const_duals = pd.Series([n.model.dual[node].values for node in custom_const_names], index=custom_const_names)
            custom_const_duals = custom_const_duals.abs().astype(float)
            custom_const_duals.to_csv(file_path_custom_const_duals)

        ### OBS ###

    logger.info("Maximum memory usage: {}".format(mem.mem_usage))

