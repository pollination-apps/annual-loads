"""The Pollination Annual Loads App."""
import os
import pathlib

from ladybug.futil import write_to_file_by_name
from ladybug.color import Colorset, Color
from ladybug.legend import LegendParameters
from ladybug.monthlychart import MonthlyChart
from ladybug.sql import SQLiteResult
from ladybug.datacollection import MonthlyCollection
from ladybug.header import Header
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.energyintensity import EnergyIntensity
from honeybee.units import conversion_factor_to_meters
from honeybee_energy.result.loadbalance import LoadBalance
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.result.err import Err
from honeybee_energy.run import run_idf
from honeybee_energy.writer import energyplus_idf_version
from honeybee_energy.config import folders as energy_folders

import streamlit as st
from pollination_streamlit_io import special

from handlers import (bootstrap, web, rhino, revit, sketchup, shared)


# Names of EnergyPlus outputs that will be requested and parsed to make graphics
cool_out = 'Zone Ideal Loads Supply Air Total Cooling Energy'
heat_out = 'Zone Ideal Loads Supply Air Total Heating Energy'
light_out = 'Zone Lights Electricity Energy'
el_equip_out = 'Zone Electric Equipment Electricity Energy'
gas_equip_out = 'Zone Gas Equipment NaturalGas Energy'
process1_out = 'Zone Other Equipment Total Heating Energy'
process2_out = 'Zone Other Equipment Lost Heat Energy'
shw_out = 'Water Use Equipment Heating Energy'
gl_el_equip_out = 'Zone Electric Equipment Total Heating Energy'
gl_gas_equip_out = 'Zone Gas Equipment Total Heating Energy'
gl1_shw_out = 'Water Use Equipment Zone Sensible Heat Gain Energy'
gl2_shw_out = 'Water Use Equipment Zone Latent Gain Energy'


st.set_page_config(
    page_title='Annual Loads',
    page_icon='https://github.com/ladybug-tools/artwork/raw/master/icons_components'
    '/honeybee/png/loadbalance.png',
    initial_sidebar_state='collapsed',
)  # type: ignore
st.sidebar.image(
    'https://uploads-ssl.webflow.com/6035339e9bb6445b8e5f77d7/616da00b76225ec0e4d975ba'
    '_pollination_brandmark-p-500.png',
    use_column_width=True
)


def get_inputs(host: str, target_folder: str):
    """Get all of the inputs for the simulation."""
    # get the input model
    if host.lower() == 'web':
        web.get_model(target_folder)
    elif host.lower() == 'rhino':
        rhino.get_model(target_folder)
    elif host.lower() == 'revit':
        revit.get_model(target_folder)
    elif host.lower() == 'sketchup':
        sketchup.get_model(target_folder)

    # add an option to preview the model in 3D
    if st.session_state.hb_model and st.checkbox(label='Preview Model', value=False):
        shared.generate_vtk_model(target_folder, st.session_state.hb_model)

    # get the input EPW and DDY files
    shared.get_weather_file(target_folder)

    # set up inputs for north
    in_north = st.slider(label='North', min_value=0, max_value=360, value=0)
    if in_north != st.session_state.north:
        st.session_state.north = in_north
        st.session_state.sql_path = None

    # get the inputs that only affect the display and do not require re-simulation
    col1, col2, col3 = st.columns(3)
    in_heat_cop = col1.number_input(
        label='Heating COP', min_value=0.0, max_value=6.0, value=1.0, step=0.05)
    if in_heat_cop != st.session_state.heat_cop:
        st.session_state.heat_cop = in_heat_cop
    in_cool_cop = col2.number_input(
        label='Cooling COP', min_value=0.0, max_value=6.0, value=1.0, step=0.05)
    if in_cool_cop != st.session_state.cool_cop:
        st.session_state.cool_cop = in_cool_cop
    in_ip_units = col3.checkbox(
        label='IP Units', value=False, help='Display output units in kBtu/ft2')
    if in_ip_units != st.session_state.ip_units:
        st.session_state.ip_units = in_ip_units


def run_simulation(target_folder: str):
    """Build the IDF file from the Model and run it through EnergyPlus."""
    # gather all of the inputs and ensure there is a model
    hb_model = st.session_state.hb_model
    epw_path = st.session_state.epw_path
    ddy_path = st.session_state.ddy_path
    north = st.session_state.north
    if not hb_model or not epw_path or not ddy_path:
        return

    # simulate the model if the button is pressed
    if st.button('Compute Loads'):
        # check to be sure that the Model has Rooms
        assert len(hb_model.rooms) != 0, \
            'Model has no Rooms and cannot be simulated in EnergyPlus.'

        # create simulation parameters for the coarsest/fastest E+ sim possible
        sim_par = SimulationParameter()
        sim_par.timestep = 1
        sim_par.shadow_calculation.solar_distribution = 'FullExterior'
        sim_par.output.add_zone_energy_use()
        sim_par.output.reporting_frequency = 'Monthly'
        sim_par.output.add_output(gl_el_equip_out)
        sim_par.output.add_output(gl_gas_equip_out)
        sim_par.output.add_output(gl1_shw_out)
        sim_par.output.add_output(gl2_shw_out)
        sim_par.output.add_gains_and_losses('Total')
        sim_par.output.add_surface_energy_flow()
        sim_par.north_angle = float(north)

        # assign design days to the simulation parameters
        sim_par.sizing_parameter.add_from_ddy(ddy_path.as_posix())

        # create the strings for simulation parameters and model
        ver_str = energyplus_idf_version() if energy_folders.energyplus_version \
            is not None else ''
        sim_par_str = sim_par.to_idf()
        model_str = hb_model.to.idf(hb_model, patch_missing_adjacencies=True)
        idf_str = '\n\n'.join([ver_str, sim_par_str, model_str])

        # write the final string into an IDF
        directory = os.path.join(target_folder, 'data', st.session_state.user_id)
        idf = os.path.join(directory, 'in.idf')
        write_to_file_by_name(directory, 'in.idf', idf_str, True)

        # run the IDF through EnergyPlus
        sql, zsz, rdd, html, err = run_idf(idf, epw_path.as_posix(), silent=True)
        if html is None and err is not None:  # something went wrong; parse the errors
            err_obj = Err(err)
            print(err_obj.file_contents)
            for error in err_obj.fatal_errors:
                raise Exception(error)
        if sql is not None and os.path.isfile(sql):
            st.session_state.sql_path = sql


def data_to_load_intensity(data_colls, floor_area, data_type, cop=1, mults=None):
    """Convert data collections from EnergyPlus to a single load intensity collection.

    Args:
        data_colls: A list of monthly data collections for an energy term.
        floor_area: The total floor area of the rooms, used to compute EUI.
        data_type: Text for the data type of the collections (eg. "Cooling").
        cop: Optional number for the COP, which the results will be divided by.
    """
    if len(data_colls) != 0:
        if mults is not None:
            data_colls = [dat * mul for dat, mul in zip(data_colls, mults)]
        total_vals = [sum(month_vals) / floor_area for month_vals in zip(*data_colls)]
        if cop != 1:
            total_vals = [val / cop for val in total_vals]
    else:  # just make a "filler" collection of 0 values
        total_vals = [0] * 12
    meta_dat = {'type': data_type}
    total_head = Header(EnergyIntensity(), 'kWh/m2', AnalysisPeriod(), meta_dat)
    return MonthlyCollection(total_head, total_vals, range(12))


def create_charts(model, sql_path, heat_cop, cool_cop, ip_units):
    """Create the load charts from the results of the simulation."""
    # get the session variables for the results
    if not sql_path:
        return

    # load up the floor area, get the model units, and the room multipliers
    floor_area = model.floor_area
    assert floor_area != 0, 'Model has no floors with which to compute EUI.'
    floor_area = floor_area * conversion_factor_to_meters(model.units) ** 2
    mults = [rm.multiplier for rm in model.rooms]
    mults = None if all(mul == 1 for mul in mults) else mults

    # get data collections for each energy use term
    sql_obj = SQLiteResult(sql_path)
    cool_init = sql_obj.data_collections_by_output_name(cool_out)
    heat_init = sql_obj.data_collections_by_output_name(heat_out)
    light_init = sql_obj.data_collections_by_output_name(light_out)
    elec_equip_init = sql_obj.data_collections_by_output_name(el_equip_out)
    gas_equip_init = sql_obj.data_collections_by_output_name(gas_equip_out)
    process1_init = sql_obj.data_collections_by_output_name(process1_out)
    process2_init = sql_obj.data_collections_by_output_name(process2_out)
    shw_init = sql_obj.data_collections_by_output_name(shw_out)

    # convert the results to EUI and output them
    cooling = data_to_load_intensity(cool_init, floor_area, 'Cooling', cool_cop)
    heating = data_to_load_intensity(heat_init, floor_area, 'Heating', heat_cop)
    lighting = data_to_load_intensity(light_init, floor_area, 'Lighting', 1, mults)
    equip = data_to_load_intensity(
        elec_equip_init, floor_area, 'Electric Equipment', 1, mults)
    load_terms = [cooling, heating, lighting, equip]
    load_colors = [
        Color(4, 25, 145), Color(153, 16, 0), Color(255, 255, 0), Color(255, 121, 0)
    ]

    # add gas equipment if it is there
    if len(gas_equip_init) != 0:
        gas_equip = data_to_load_intensity(
            gas_equip_init, floor_area, 'Gas Equipment', 1, mults)
        load_terms.append(gas_equip)
        load_colors.append(Color(255, 219, 128))
    # add process load if it is there
    process = []
    if len(process1_init) != 0:
        process1 = data_to_load_intensity(process1_init, floor_area, 'Process', 1, mults)
        process2 = data_to_load_intensity(process2_init, floor_area, 'Process', 1, mults)
        process = process1 + process2
        load_terms.append(process)
        load_colors.append(Color(135, 135, 135))
    # add hot water if it is there
    hot_water = []
    if len(shw_init) != 0:
        hot_water = data_to_load_intensity(
            shw_init, floor_area, 'Service Hot Water', 1, mults)
        load_terms.append(hot_water)
        load_colors.append(Color(255, 0, 0))

    # convert the data to the correct units
    display_units = 'kBtu/ft2' if ip_units else 'kWh/m2'
    if load_terms[0].header.unit != display_units:
        for data in load_terms:
            data.convert_to_unit(display_units)

    # report the total load intensity
    total_load = [dat.total for dat in load_terms]
    st.subheader(
        'Total Load Intensity: {} {}'.format(round(sum(total_load), 2), display_units))

    # plot the monthly data collections on a bar chart
    leg_par = LegendParameters(colors=load_colors)
    leg_par.decimal_count = 0
    month_chart = MonthlyChart(load_terms, leg_par, stack=True)
    figure = month_chart.plot(title='Load Intensity')
    st.plotly_chart(figure)

    # create a monthly chart with the load balance
    bal_obj = LoadBalance.from_sql_file(model, sql_path)
    balance = bal_obj.load_balance_terms(True, True)
    if balance[0].header.unit != display_units:
        for data in balance:
            data.convert_to_unit(display_units)
    bal_colors = Colorset()[19]
    leg_par = LegendParameters(colors=bal_colors)
    leg_par.decimal_count = 0
    month_chart = MonthlyChart(balance, leg_par, stack=True)
    figure = month_chart.plot(title='Load Balance')
    st.plotly_chart(figure)


def main(platform):
    # title
    st.header('Annual Loads')

    # load up all of the inputs
    target_folder = pathlib.Path(__file__).parent

    bootstrap.initialize()
    get_inputs(platform, target_folder)

    # preview the model and/or run the simulation
    run_simulation(target_folder)

    # create the resulting charts
    create_charts(
        st.session_state.hb_model, st.session_state.sql_path,
        st.session_state.heat_cop, st.session_state.cool_cop,
        st.session_state.ip_units
    )


if __name__ == '__main__':
    # get the platform from the query uri
    query = st.experimental_get_query_params()
    platform = special.get_host() or 'web'
    main(platform)