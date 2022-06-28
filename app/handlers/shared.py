"""A module to collect shared logic."""
import streamlit as st
import os
from pathlib import Path
from honeybee.model import Model
from honeybee_vtk.model import Model as VTKModel
from pollination_streamlit_viewer import viewer
from ladybug.epw import EPW


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


def new_weather_file():
    st.session_state.sql_path = None
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
    epw_file = st.file_uploader(
        'Upload a weather file (EPW)', type=['epw'],
        on_change=new_weather_file, key='epw_data'
    )
    
