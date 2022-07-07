"""Functions for running the simulation and performing initial result postprocessing."""
import os
import subprocess

from ladybug.futil import write_to_file_by_name
from ladybug.sql import SQLiteResult
from ladybug.datacollection import MonthlyCollection
from ladybug.header import Header
from ladybug.analysisperiod import AnalysisPeriod
from ladybug.datatype.energyintensity import EnergyIntensity
from ladybug.color import Color
from honeybee.units import conversion_factor_to_meters
from honeybee_energy.result.loadbalance import LoadBalance
from honeybee_energy.simulation.parameter import SimulationParameter
from honeybee_energy.result.err import Err
from honeybee_energy.run import prepare_idf_for_simulation, output_energyplus_files
from honeybee_energy.writer import energyplus_idf_version
from honeybee_energy.config import folders as energy_folders

import streamlit as st


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


def simulate_idf(idf_file_path, epw_file_path=None, expand_objects=True):
    """Run an IDF file through EnergyPlus and report the STDOUT in the app.

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
    if not hb_model or not epw_path or not ddy_path or \
            st.session_state.sql_results is not None:
        return

    # simulate the model if the button is pressed
    button_holder = st.empty()
    if button_holder.button('Compute Loads'):
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
        sql, zsz, rdd, html, err = simulate_idf(idf, epw_path.as_posix())
        if html is None and err is not None:  # something went wrong; parse the errors
            err_obj = Err(err)
            print(err_obj.file_contents)
            for error in err_obj.fatal_errors:
                raise Exception(error)
        if sql is not None and os.path.isfile(sql):
            st.session_state.sql_results = load_sql_data(sql, hb_model)
            button_holder.write('')
