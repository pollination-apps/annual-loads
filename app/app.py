"""The Pollination Annual Loads App."""
import streamlit as st
from pollination_streamlit_io import get_host

from inputs import initialize, get_inputs
from simulation import run_simulation
from outputs import display_results


st.set_page_config(
    page_title='Annual Loads',
    page_icon='https://github.com/ladybug-tools/artwork/raw/master/icons_components'
    '/honeybee/png/loadbalance.png',
    initial_sidebar_state='collapsed',
)  # type: ignore
st.sidebar.image(
    'https://uploads-ssl.webflow.com/6035339e9bb6445b8e5f77d7/616da00b76225ec0e4d975ba'
    '_pollination_brandmark-p-500.png',
    use_column_width=True
)


def main(platform):
    """Perform the main calculation of the App."""
    # title
    st.header('Annual Loads')

    # initialize the app and load up all of the inputs
    initialize()
    get_inputs(platform)
    container = st.container()  # container to eventually hold the results

    # preview the model and/or run the simulation
    run_simulation(
        st.session_state.target_folder, st.session_state.user_id,
        st.session_state.hb_model,
        st.session_state.epw_path, st.session_state.ddy_path, st.session_state.north
    )

    # create the resulting charts
    display_results(
        container, st.session_state.sql_results,
        st.session_state.heat_cop, st.session_state.cool_cop,
        st.session_state.ip_units, st.session_state.normalize
    )


if __name__ == '__main__':
    # get the platform from the query uri
    query = st.experimental_get_query_params()
    platform = get_host() or 'web'
    main(platform)
