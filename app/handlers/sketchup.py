"""A module to collect sketchup setup."""
import pathlib
import streamlit as st
from honeybee.model import Model
from pollination_streamlit_io import button


def get_model():
    # save HBJSON in data folder
    st.warning('Sketchup does not support sync for now...')
    data = button.get(
        is_pollination_model=True, key='pollination-model', platform='sketchup')
    if data:
        hb_model = Model.from_dict(data)
        st.session_state.hb_model = hb_model
        st.session_state.vtk_path = None
        st.session_state.sql_path = None
