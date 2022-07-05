"""The Pollination Annual Loads App."""
import os
import subprocess

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
from honeybee_energy.run import prepare_idf_for_simulation, output_energyplus_files
from honeybee_energy.writer import energyplus_idf_version
from honeybee_energy.config import folders as energy_folders

from pandas import DataFrame
import streamlit as st
from pollination_streamlit_io import get_host
from handlers import initialize, get_inputs


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


def run_idf(idf_file_path, epw_file_path=None, expand_objects=True):
    """Run an IDF file through EnergyPlus on any operating system.

    Args:
        idf_file_path: The full path to an IDF file.
        epw_file_path: The full path to an EPW file. Note that inputting None here
            is only appropriate when the simulation is just for design days and has
            no weather file run period. (Default: None).
        expand_objects: If True, the IDF run will include the expansion of any
            HVAC Template objects in the file before beginning the simulation.
            This is a necessary step whenever there are HVAC Template objects in
            the IDF but it is unnecessary extra time when they are not
            present. (Default: True).
    """
    # check and prepare the input files
    directory = prepare_idf_for_simulation(idf_file_path, epw_file_path)

    # run the simulation
    cmds = [energy_folders.energyplus_exe, '-i', energy_folders.energyplus_idd_path]
    if epw_file_path is not None:
        cmds.append('-w')
        cmds.append(os.path.abspath(epw_file_path))
    if expand_objects:
        cmds.append('-x')
    process = subprocess.Popen(cmds, cwd=directory, stdout=subprocess.PIPE)

    # print the stdout in the app
    stdout_style = '<style> .std {font-size: 1rem ; margin: 0rem ; ' \
        'padding: 0rem ; color: white ; background-color: black ;} </style>'
    st.markdown(stdout_style, unsafe_allow_html=True)
    with st.empty():
        current_stdout = []
        for line in iter(lambda: process.stdout.readline(), b""):
            std_line = line.decode("utf-8")
            current_stdout.append(std_line)
            stdout_lines = ['<p class="std">{}</p>'.format(li) for li in current_stdout]
            st.markdown(''.join(stdout_lines), unsafe_allow_html=True)
            if len(current_stdout) == 6:
                current_stdout.pop(0)
        st.write('')  # clear the EnergyPlus stdout

    # output the simulation files
    return output_energyplus_files(directory)


def data_to_load_intensity(room_dict, data_colls, floor_area, data_type, mults=None):
    """Convert data collections from EnergyPlus to a single load intensity collection.

    Args:
        data_colls: A list of monthly data collections for an energy term.
        floor_area: The total floor area of the rooms, used to compute EUI.
        data_type: Text for the data type of the collections (eg. "Cooling").
        mults: An optional dictionary of Room identifiers and integers for
            the multipliers of the honeybee Rooms.
    """
    if len(data_colls) != 0:
        # first try adding the data to the room dictionary
        rel_key = 'Zone' if 'Zone' in data_colls[0].header.metadata else 'System'
        for dat in data_colls:
            try:
                z_id = dat.header.metadata[rel_key]
                if rel_key == 'Zone':
                    r_prop = room_dict[z_id]
                elif ' IDEAL LOADS AIR SYSTEM' in z_id:  # E+ HVAC Templates
                    r_prop = room_dict[z_id.split(' IDEAL LOADS AIR SYSTEM')[0]]
                elif '..' in z_id:  # convention used for service hot water
                    r_prop = room_dict[z_id.split('..')[-1]]
                else:
                    r_prop = room_dict[z_id]
                r_prop[-1][data_type] = (dat.total * r_prop[2]) / r_prop[1]
            except KeyError:  # no results of this type for the Room
                pass
        # next, build up the monthly collection of total values
        if mults is not None:
            if rel_key == 'Zone':
                rel_mults = [mults[data.header.metadata['Zone']] for data in data_colls]
                data_colls = [dat * mul for dat, mul in zip(data_colls, rel_mults)]
        total_vals = [sum(month_vals) / floor_area for month_vals in zip(*data_colls)]
    else:  # just make a "filler" collection of 0 values
        total_vals = [0] * 12
    meta_dat = {'type': data_type}
    total_head = Header(EnergyIntensity(), 'kWh/m2', AnalysisPeriod(), meta_dat)
    return MonthlyCollection(total_head, total_vals, range(12))


def load_sql_data(sql_path, model):
    """Load and process the SQL data from the simulation and store it in memory.

    Args:
        sql_path: Path to the SQLite file output from an EnergyPlus simulation.
        model: The honeybee model object used to create the SQL results.
    """
    # load up the floor area, get the model units, and the room multipliers
    con_fac = conversion_factor_to_meters(model.units) ** 2
    floor_areas, rd = [], {}
    for room in model.rooms:
        if not room.exclude_floor_area:
            fa = room.floor_area * room.multiplier * con_fac
            floor_areas.append(fa)
            rd[room.identifier.upper()] = [room.display_name, fa, room.multiplier, {}]
    floor_area = sum(floor_areas)
    assert floor_area != 0, 'Model has no floors with which to compute EUI.'
    mults = {rm.identifier.upper(): rm.multiplier for rm in model.rooms}
    mults = None if all(mul == 1 for mul in mults.values()) else mults

    # get data collections for each energy use term
    sql_obj = SQLiteResult(sql_path)
    cool_init = sql_obj.data_collections_by_output_name(cool_out)
    heat_init = sql_obj.data_collections_by_output_name(heat_out)
    light_init = sql_obj.data_collections_by_output_name(light_out)
    elec_eq_init = sql_obj.data_collections_by_output_name(el_equip_out)
    gas_equip_init = sql_obj.data_collections_by_output_name(gas_equip_out)
    process1_init = sql_obj.data_collections_by_output_name(process1_out)
    process2_init = sql_obj.data_collections_by_output_name(process2_out)
    shw_init = sql_obj.data_collections_by_output_name(shw_out)

    # convert the results to a single monthly EUI data collection
    cooling = data_to_load_intensity(rd, cool_init, floor_area, 'Cooling')
    heating = data_to_load_intensity(rd, heat_init, floor_area, 'Heating')
    lighting = data_to_load_intensity(rd, light_init, floor_area, 'Lighting', mults)
    equip = data_to_load_intensity(
        rd, elec_eq_init, floor_area, 'Electric Equipment', mults)
    load_terms = [cooling, heating, lighting, equip]
    load_colors = [
        Color(4, 25, 145), Color(153, 16, 0), Color(255, 255, 0), Color(255, 121, 0)
    ]

    # add gas equipment if it is there
    if len(gas_equip_init) != 0:
        gas_equip = data_to_load_intensity(
            rd, gas_equip_init, floor_area, 'Gas Equipment', mults)
        load_terms.append(gas_equip)
        load_colors.append(Color(255, 219, 128))
    # add process load if it is there
    process = []
    if len(process1_init) != 0:
        process1 = data_to_load_intensity(
            rd, process1_init, floor_area, 'Process', mults)
        process2 = data_to_load_intensity(
            rd, process2_init, floor_area, 'Process', mults)
        process = process1 + process2
        load_terms.append(process)
        load_colors.append(Color(135, 135, 135))
    # add hot water if it is there
    hot_water = []
    if len(shw_init) != 0:
        hot_water = data_to_load_intensity(
            rd, shw_init, floor_area, 'Service Hot Water', mults)
        load_terms.append(hot_water)
        load_colors.append(Color(255, 0, 0))

    # create a monthly load balance
    bal_obj = LoadBalance.from_sql_file(model, sql_path)
    balance = bal_obj.load_balance_terms(True, True)

    # return a dictionary containing all relevant results of the simulation
    return {
        'room_results': rd,
        'floor_area': floor_area,
        'load_terms': load_terms,
        'load_colors': load_colors,
        'balance': balance
    }


def run_simulation(target_folder, user_id, hb_model, epw_path, ddy_path, north):
    """Build the IDF file from a Model and run it through EnergyPlus.

    Args:
        target_folder: Text for the target folder out of which the simulation will run.
        user_id: A unique user ID for the session, which will be used to ensure
            other simulations do not overwrite this one.
        hb_model: A Honeybee Model object to be simulated.
        epw_path: Path to an EPW file to be used in the simulation.
        ddy_path: Path to a DDY file to be used in the simulation.
        north: Integer for the angle from the Y-axis where North is.
    """
    # check to be sure there is a model
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
        directory = os.path.join(target_folder, 'data', user_id)
        idf = os.path.join(directory, 'in.idf')
        write_to_file_by_name(directory, 'in.idf', idf_str, True)

        # run the IDF through EnergyPlus
        sql, zsz, rdd, html, err = run_idf(idf, epw_path.as_posix())
        if html is None and err is not None:  # something went wrong; parse the errors
            err_obj = Err(err)
            print(err_obj.file_contents)
            for error in err_obj.fatal_errors:
                raise Exception(error)
        if sql is not None and os.path.isfile(sql):
            st.session_state.sql_results = load_sql_data(sql, hb_model)


def create_charts(container, sql_results, heat_cop, cool_cop, ip_units, normalize):
    """Create the load charts from the loaded sql results of the simulation.

    Args:
        container: The streamlit container to which the charts will be added.
        sql_results: Dictionary of the EnergyPlus SQL results (or None).
        heat_cop: Number for the heating COP.
        cool_cop: Number for the cooling COP.
        ip_units: Boolean to indicate whether IP units should be used.
        normalize: Boolean to indicate whether data should be normalized by
            the Model floor area.
    """
    # get the session variables for the results
    if not sql_results:
        return
    load_terms = sql_results['load_terms'].copy()
    load_colors = sql_results['load_colors']
    balance = sql_results['balance'].copy()
    room_results = sql_results['room_results']
    floor_area = sql_results['floor_area']

    # convert the data to the correct units system
    display_units = 'kBtu/ft2' if ip_units else 'kWh/m2'
    if load_terms[0].header.unit != display_units:
        for data in load_terms:
            data.convert_to_unit(display_units)
    if balance[0].header.unit != display_units:
        for data in balance:
            data.convert_to_unit(display_units)

    # total the data over the floor area if normalize is false
    if not normalize:
        if ip_units:
            display_units, a_unit, total_area = 'kBtu', 'ft2', floor_area * 10.7639
        else:
            display_units, a_unit, total_area = 'kWh', 'm2', floor_area
        load_terms = [data.aggregate_by_area(total_area, a_unit) for data in load_terms]
        balance = [data.aggregate_by_area(total_area, a_unit) for data in balance]

    # multiply the results by the COP if it is not equal to 1
    if cool_cop != 1:
        load_terms[0] = load_terms[0] / cool_cop
    if heat_cop != 1:
        load_terms[1] = load_terms[1] / heat_cop

    # report the total load and the breakdown into different terms
    total_load = [dat.total for dat in load_terms]
    container.subheader(
        'Total Load: {:,.2f} {}'.format(sum(total_load), display_units))

    # create a table with totaled data
    total_data = {d.header.metadata['type']: [d.total] for d in load_terms}
    tot_df = DataFrame(total_data)
    container.dataframe(tot_df.style.format('{:,.2f}'))

    # plot the monthly data collections on a bar chart
    leg_par = LegendParameters(colors=load_colors)
    leg_par.decimal_count = 0
    month_chart = MonthlyChart(load_terms, leg_par, stack=True)
    figure = month_chart.plot(title='Monthly Load')
    container.plotly_chart(figure)

    # create a monthly chart with the load balance
    bal_colors = Colorset()[19]
    leg_par = LegendParameters(colors=bal_colors)
    leg_par.decimal_count = 0
    month_chart = MonthlyChart(balance, leg_par, stack=True)
    figure = month_chart.plot(title='Monthly Load Balance')
    container.plotly_chart(figure)

    # process all of the detailed room results into a table
    container.write('Room Summary ({})'.format(display_units))
    table_data = {'Room': []}
    load_types = [dat.header.metadata['type'] for dat in load_terms]
    for lt in load_types:
        table_data[lt] = []
    for room_data in room_results.values():
        name, fa, mult, res = room_data
        table_data['Room'].append(name)
        for lt in load_types:
            try:
                val = res[lt] if normalize else res[lt] * fa * mult
                table_data[lt].append(val)
            except KeyError:
                table_data[lt].append(0.0)
    # perform any unit conversions on the table data
    if ip_units:
        conv = 0.316998 if normalize else 3.41214
        for col, dat in table_data.items():
            if col != 'Room':
                table_data[col] = [val * conv for val in table_data[col]]
    if cool_cop != 1:
        table_data['Cooling'] = [val / cool_cop for val in table_data['Cooling']]
    if heat_cop != 1:
        table_data['Heating'] = [val / heat_cop for val in table_data['Heating']]
    # add a column for the total data of each room
    totals = [0] * len(table_data['Cooling'])
    for col, dat in table_data.items():
        if col != 'Room':
            for i, v in enumerate(dat):
                totals[i] += v
    table_data['Total'] = totals
    container.dataframe(table_data)


def main(platform):
    """Perform the main calculation of the App."""
    # title
    st.header('Annual Loads')

    # initialize the app and load up all of the inputs
    initialize()
    get_inputs(platform)
    container = st.container()

    # preview the model and/or run the simulation
    run_simulation(
        st.session_state.target_folder, st.session_state.user_id,
        st.session_state.hb_model,
        st.session_state.epw_path, st.session_state.ddy_path, st.session_state.north
    )

    # create the resulting charts
    create_charts(
        container, st.session_state.sql_results,
        st.session_state.heat_cop, st.session_state.cool_cop,
        st.session_state.ip_units, st.session_state.normalize
    )


if __name__ == '__main__':
    # get the platform from the query uri
    query = st.experimental_get_query_params()
    platform = get_host() or 'web'
    main(platform)
