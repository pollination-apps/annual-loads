"""Init all session state variables"""
import uuid
import pathlib
import streamlit as st


def initialize():
    # user session
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    if 'target_folder' not in st.session_state:
        st.session_state.target_folder = pathlib.Path(__file__).parent.parent
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
