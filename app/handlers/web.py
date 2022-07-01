"""A module to collect web setup."""

import json
import streamlit as st
from honeybee.model import Model


def new_model():
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


def get_model():
    st.file_uploader(
        'Upload a hbjson file', type=['hbjson', 'json'],
        on_change=new_model, key='hbjson_data'
    )
