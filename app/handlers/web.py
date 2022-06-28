"""A module to collect web setup."""

import json
import streamlit as st
from honeybee.model import Model


def new_model():
    st.session_state.vtk_path = None
    st.session_state.sql_path = None


def get_model():
    hbjson_file = st.file_uploader(
        'Upload a hbjson file', type=['hbjson', 'json'], on_change=new_model)
    if hbjson_file:
        # save HBJSON in data folder
        data = hbjson_file.read()
        model_data = json.loads(data)
        hb_model = Model.from_dict(model_data)
        st.session_state.hb_model = hb_model
