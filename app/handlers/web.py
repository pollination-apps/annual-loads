"""A module to collect web setup."""

import json
import pathlib
import streamlit as st
from honeybee.model import Model


def new_model():
    st.session_state.sql_path = None
    st.session_state.rebuild_viz = True


def get_model(here: pathlib.Path):
    hbjson_file = st.file_uploader(
        'Upload a hbjson file', type=['hbjson', 'json'], on_change=new_model)
    if hbjson_file:
        # save HBJSON in data folder
        hbjson_path = pathlib.Path(
            f'./{here}/data/{st.session_state.user_id}/{hbjson_file.name}')
        hbjson_path.parent.mkdir(parents=True, exist_ok=True)
        data = hbjson_file.read()
        hbjson_path.write_bytes(data)
        model_data = json.loads(data)
        hb_model = Model.from_dict(model_data)

        # add to session state
        st.session_state.hb_model = hb_model
