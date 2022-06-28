"""A module to collect rhino setup."""
import pathlib
import streamlit as st
from honeybee.model import Model
from pollination_streamlit_io import special, button


def get_model(here: pathlib.Path):
    # save HBJSON in data folder
    token = special.sync(key='pollination-sync', delay=50)
    data = button.get(
        is_pollination_model=True, sync_token=token, key='pollination-model')
    if data:
        hb_model = Model.from_dict(data)
        st.session_state.hb_model = hb_model
        st.session_state.vtk_path = None
        st.session_state.sql_path = None
