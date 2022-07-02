"""Init all session state variables"""
import os
import json
import uuid
from pathlib import Path

import streamlit as st

from ladybug.epw import EPW
from honeybee.model import Model
from honeybee_vtk.model import Model as VTKModel

from pollination_streamlit_viewer import viewer
from pollination_streamlit_io import get_hbjson


def initialize():
    # user session
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    if 'target_folder' not in st.session_state:
        st.session_state.target_folder = Path(__file__).parent
    # sim session
    if 'hb_model' not in st.session_state:
        st.session_state.hb_model = None
    if 'vtk_path' not in st.session_state:
        st.session_state.vtk_path = None
    if 'epw_path' not in st.session_state:
        st.session_state.epw_path = None
    if 'ddy_path' not in st.session_state:
        st.session_state.ddy_path = None
    if 'north' not in st.session_state:
        st.session_state.north = None
    # output session
    if 'heat_cop' not in st.session_state:
        st.session_state.heat_cop = None
    if 'cool_cop' not in st.session_state:
        st.session_state.cool_cop = None
    if 'ip_units' not in st.session_state:
        st.session_state.ip_units = False
    if 'sql_results' not in st.session_state:
        st.session_state.sql_results = None


def new_weather_file():
    # reset the simulation results and get the file data
    st.session_state.sql_results = None
    epw_file = st.session_state.epw_data
    if epw_file:
        # save EPW in data folder
        epw_path = Path(
            f'./{st.session_state.target_folder}/data/'
            f'{st.session_state.user_id}/{epw_file.name}'
        )
        epw_path.parent.mkdir(parents=True, exist_ok=True)
        epw_path.write_bytes(epw_file.read())
        # create a DDY file from the EPW
        ddy_file = epw_path.as_posix().replace('.epw', '.ddy')
        epw_obj = EPW(epw_path.as_posix())
        epw_obj.to_ddy(ddy_file)
        ddy_path = Path(ddy_file)
        # set the session state variables
        st.session_state.epw_path = epw_path
        st.session_state.ddy_path = ddy_path


def get_weather_file():
    # upload weather file
    st.file_uploader(
        'Upload a weather file (EPW)', type=['epw'],
        on_change=new_weather_file, key='epw_data'
    )


def new_model_web():
    # reset the simulation results and get the file data
    st.session_state.vtk_path = None
    st.session_state.sql_results = None
    # load the model object from the file data
    hbjson_file = st.session_state.hbjson_data
    if hbjson_file:
        data = hbjson_file.read()
        model_data = json.loads(data)
        hb_model = Model.from_dict(model_data)
        st.session_state.hb_model = hb_model


def get_model_web():
    st.file_uploader(
        'Upload a hbjson file', type=['hbjson', 'json'],
        on_change=new_model_web, key='hbjson_data'
    )


def get_model_cad():
    hbjson = get_hbjson('hbjson_data')


def generate_vtk_model(hb_model: Model) -> str:
    if not st.session_state.vtk_path:
        directory = os.path.join(
            st.session_state.target_folder.as_posix(),
            'data', st.session_state.user_id
        )
        if not os.path.isdir(directory):
            os.makedirs(directory)
        hbjson_path = hb_model.to_hbjson(hb_model.identifier, directory)
        vtk_model = VTKModel.from_hbjson(hbjson_path)
        vtk_path = vtk_model.to_vtkjs(folder=directory, name=hb_model.identifier)
        st.session_state.vtk_path = vtk_path
    vtk_path = st.session_state.vtk_path
    viewer(content=Path(vtk_path).read_bytes(), key='vtk_preview_model')
