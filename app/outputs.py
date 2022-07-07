"""Functions for creating output visualizations and changing the units of the results."""
from ladybug.color import Colorset
from ladybug.legend import LegendParameters
from ladybug.monthlychart import MonthlyChart


def display_results(container, sql_results, heat_cop, cool_cop, ip_units, normalize):
    """Create the charts and metrics from the loaded sql results of the simulation.

    Args:
        container: The streamlit container to which the charts will be added.
        sql_results: Dictionary of the EnergyPlus SQL results (or None).
        heat_cop: Number for the heating COP.
        cool_cop: Number for the cooling COP.
        ip_units: Boolean to indicate whether IP units should be used.
        normalize: Boolean to indicate whether data should be normalized by
            the Model floor area.
    """
    # get the session variables for the results
    if not sql_results:
        return
    load_terms = sql_results['load_terms'].copy()
    load_colors = sql_results['load_colors']
    balance = sql_results['balance'].copy()
    room_results = sql_results['room_results']
    floor_area = sql_results['floor_area']

    # convert the data to the correct units system
    display_units = 'kBtu/ft2' if ip_units else 'kWh/m2'
    if load_terms[0].header.unit != display_units:
        for data in load_terms:
            data.convert_to_unit(display_units)
    if balance[0].header.unit != display_units:
        for data in balance:
            data.convert_to_unit(display_units)

    # total the data over the floor area if normalize is false
    if not normalize:
        if ip_units:
            display_units, a_unit, total_area = 'kBtu', 'ft2', floor_area * 10.7639
        else:
            display_units, a_unit, total_area = 'kWh', 'm2', floor_area
        load_terms = [data.aggregate_by_area(total_area, a_unit) for data in load_terms]
        balance = [data.aggregate_by_area(total_area, a_unit) for data in balance]

    # multiply the results by the COP if it is not equal to 1
    if cool_cop != 1:
        load_terms[0] = load_terms[0] / cool_cop
    if heat_cop != 1:
        load_terms[1] = load_terms[1] / heat_cop

    # report the total load and the breakdown into different terms
    tot_ld = [dat.total for dat in load_terms]
    val = '{:,.1f}'.format(sum(tot_ld)) if normalize else '{:,.0f}'.format(sum(tot_ld))
    container.header('Total Load: {} {}'.format(val, display_units))

    # add metrics for the individual load components
    eui_cols = container.columns(len(load_terms))
    for d, col in zip(load_terms, eui_cols):
        val = '{:.1f}'.format(d.total) if normalize else '{:,.0f}'.format(d.total)
        col.metric(d.header.metadata['type'], val)

    # plot the monthly data collections on a bar chart
    leg_par = LegendParameters(colors=load_colors)
    leg_par.decimal_count = 0
    month_chart = MonthlyChart(load_terms, leg_par, stack=True)
    figure = month_chart.plot(title='Monthly Load')
    container.plotly_chart(figure)

    # create a monthly chart with the load balance
    bal_colors = Colorset()[19]
    leg_par = LegendParameters(colors=bal_colors)
    leg_par.decimal_count = 0
    month_chart = MonthlyChart(balance, leg_par, stack=True)
    figure = month_chart.plot(title='Monthly Load Balance')
    container.plotly_chart(figure)

    # process all of the detailed room results into a table
    container.write('Room Summary ({})'.format(display_units))
    table_data = {'Room': []}
    load_types = [dat.header.metadata['type'] for dat in load_terms]
    for lt in load_types:
        table_data[lt] = []
    for room_data in room_results.values():
        name, fa, mult, res = room_data
        table_data['Room'].append(name)
        for lt in load_types:
            try:
                val = res[lt] if normalize else res[lt] * fa * mult
                table_data[lt].append(val)
            except KeyError:
                table_data[lt].append(0.0)
    # perform any unit conversions on the table data
    if ip_units:
        conv = 0.316998 if normalize else 3.41214
        for col, dat in table_data.items():
            if col != 'Room':
                table_data[col] = [val * conv for val in table_data[col]]
    if cool_cop != 1:
        table_data['Cooling'] = [val / cool_cop for val in table_data['Cooling']]
    if heat_cop != 1:
        table_data['Heating'] = [val / heat_cop for val in table_data['Heating']]
    # add a column for the total data of each room
    totals = [0] * len(table_data['Cooling'])
    for col, dat in table_data.items():
        if col != 'Room':
            for i, v in enumerate(dat):
                totals[i] += v
    table_data['Total'] = totals
    container.dataframe(table_data)
