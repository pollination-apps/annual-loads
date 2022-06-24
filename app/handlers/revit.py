"""A module to collect revit setup."""
import json
import pathlib
import streamlit as st
from honeybee.model import Model
from pollination_streamlit_io import special


# ADN: @Konrad can you check this? :)
def get_model(here: pathlib.Path):
    data = special.get_hbjson(key='my-revit-json')
    if data:
        model_data = json.loads(data)
        hb_model = Model.from_dict(model_data)
        st.session_state.hb_model = hb_model
        st.session_state.sql_path = None
