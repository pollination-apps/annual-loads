"""Init all session state variables"""
import uuid
import streamlit as st


def initialize():
    # user session
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())[:8]
    # sim session
    if 'hb_model' not in st.session_state:
        st.session_state.hb_model = None
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
    if 'rebuild_viz' not in st.session_state:
        st.session_state.rebuild_viz = True
    if 'sql_path' not in st.session_state:
        st.session_state.sql_path = None
