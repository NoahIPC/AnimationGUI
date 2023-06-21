from dash import dcc, html
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html


import os

import dash_bootstrap_components as dbc
from dash import html

from views.mapPlot import mapPlot


def make_ESPAM_layout():


    # Get all files from GIS folder
    Files = os.listdir('GIS')
    # Remove the extension from the files
    Files = [i.split('.')[0] for i in Files]
    GIS_Files = dcc.Store(id='GIS_Files', data=Files)

    Tabs = []
    for file in Files:
        # Add a tab with a checkbox, and 5 number inputs
        tab = dcc.Tab(label=file, children=[
            dbc.Tooltip("Choose whether this layer is visible or invisible",
                        target={'type':'visible', 'index':file}),
            dbc.RadioItems(id={'type':'visible', 'index':file}, 
                        options=[{'label': 'Visible', 'value': 1}, {'label': 'Invisible', 'value': 0}],
                        value=0, inline=True),
            dbc.Row([
                dbc.Col([
                    dbc.Label('Line Color'),
                    dbc.Tooltip("Select the line color for this layer",
                                target={'type':'line-color', 'index':file}),
                    dbc.Input(id={'type':'line-color', 'index':file}, type='color', value='#000000'),
                ], style={'padding': '10px'}),
                dbc.Col([
                    dbc.Label('Fill Color'),
                    dbc.Tooltip("Select the fill color for this layer",
                                target={'type':'fill-color', 'index':file}),
                    dbc.Input(id={'type':'fill-color', 'index':file}, type='color', value='#ffffff'),
                ], style={'padding': '10px'}),
                dbc.Col([
                    dbc.Label('Line Width'),
                    dbc.Tooltip("Set the line width for this layer",
                                target={'type':'line-width', 'index':file}),
                    dbc.Input(id={'type':'line-width', 'index':file}, type='number', value=1, step=0.1, min=0.1),
                ], style={'padding': '10px'}),
                dbc.Col([
                    dbc.Label('Opacity'),
                    dbc.Tooltip("Set the opacity for this layer",
                                target={'type':'opacity', 'index':file}),
                    dbc.Input(id={'type':'opacity', 'index':file}, type='number', value=0.3, step=0.1, min=0, max=1),
                ], style={'padding': '10px'}),
                dbc.Col([
                    dbc.Label('Draw Order'),
                    dbc.Tooltip("Set the draw order for this layer",
                                target={'type':'draw-order', 'index':file}),
                    dbc.Input(id={'type':'draw-order', 'index':file}, type='number', value=10),
                ], style={'padding': '10px'}),
            ])
        ], style={'float': 'left', 'border': '1px solid #2574B6', 'margin': '5px'})
        Tabs.append(tab)

    # Make a modal with a tab for each file
    modal = html.Div([
        dbc.Modal([
            dbc.ModalHeader("GIS Files"),
            dbc.ModalBody([
                dcc.Tabs(Tabs, id='ESPAM_modal_tabs', vertical=True, parent_style={'float': 'left', 'width': '90%', 'margin': 'auto'}),
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="ESPAM_modal_close", className="me-1", color='primary')
            ),
        ], id='ESPAM_modal', size='xl')
    ])

    GIS_Options = dcc.Store(id='GIS-options', data={})
    Color_Values = dcc.Store(id='color-values', data={'data': [-25, -2, 2, 25]})
    Colors = dcc.Store(id='colors', data={'data':['red', 'white', 'white', 'green']})

    Zoom = dcc.Store(id='zoom', data={'data': [2379000, 2752000, 1191000, 1508000]})
    SettingsValues = dcc.Store(id='settings-values')

    date_format_options = [
        {"label": "MM/DD/YY", "value": "%m/%d/%y"},
        {"label": "DD/MM/YY", "value": "%d/%m/%y"},
        {"label": "MMM-DD-YYYY", "value": "%b-%d-%Y"},
        {"label": "YYYY-MM-DD", "value": "%Y-%m-%d"},
        {"label": "Month Year (Jan 2020)", "value": "%B %Y"},
        {"label": "Year only (2020)", "value": "%Y"},
    ]

    tooltips = {
        'figure-title': 'Enter the title for the figure.',
        'ESPAM_slider': 'Select the model timestep.',
        'date-format': 'Select the format for the date.',
        'ESPAM_slider_label': 'Current selected timestep.',
        'start_date': 'Enter the start date.',
        'date_freq': 'Enter the model timestep frequency in days.',
        'animation-length': 'Enter the desired animation length in seconds.',
        'ESPAM_modal_open': 'Click to edit base GIS layers.',
        'ESPAM_upload': 'Upload your files here.',
        'settings-save': 'Save the current settings.',
        'settings-load': 'Load the saved settings.',
        'generate-animation': 'Generate the animation.',
        'download-animation': 'Download the animation.',
        'figure-height': 'Adjust the height of the figure.',
        'figure-width': 'Adjust the width of the figure.',
        'color-1-position': 'Enter position for color 1.',
        'color-2-position': 'Enter position for color 2.',
        'color-3-position': 'Enter position for color 3.',
        'color-4-position': 'Enter position for color 4.',
        'color-1': 'Select the color for position 1.',
        'color-2': 'Select the color for position 2.',
        'color-3': 'Select the color for position 3.',
        'color-4': 'Select the color for position 4.',
    }

    def create_tooltip(target):
        return dbc.Tooltip(tooltips[target], target=target, placement="right")

    def create_input(id, type, value, label=None):
        input_div = html.Div([
            dbc.Row([
                html.Label(label, id=f'{id}-label') if label else None,
                dbc.Input(id=id, type=type, value=value, className="my-input"),
            ]),
            create_tooltip(id),
        ])
        return input_div

    def create_color_input(id, position, color, tooltip=None):
        color_input_div = html.Div([
            create_input(id=f'{id}-position', type='number', value=position),
            create_input(id=id, type='color', value=color),
        ], id=f"{id}-row")
        return color_input_div

   # Create buttons with tooltips
    def create_button(id, label, disabled=False):
        button_div = html.Div([
            dbc.Button(label, id=id, className="my-button", disabled=disabled),
            create_tooltip(id),
        ])
        return button_div

    # Create sliders with tooltips
    def create_slider(id, label, min, max, step, value, marks, vertical=False, verticalHeight=400, labelID=''):
        slider_div = dbc.Row([
            html.Label(label, id=f'{id}-label', className="slider-label"+labelID),
            dcc.Slider(id=id, min=min, max=max, step=step, value=value, marks=marks, className="my-slider",
                       vertical=vertical, verticalHeight=verticalHeight),
            create_tooltip(id),
        ])
        return slider_div

    # Use the helper functions to create inputs
    figure_title_input = create_input(id='figure-title', type='text', value='ESPAM', label='Figure Title ')
    ESPAM_slider = create_slider(id='ESPAM_slider', min=0, max=100, step=1, value=0, marks={0: '0', 100: '100'}, label='Select Timestep')
    ESPAM_slider_label = html.Label('Timestep: ', id='ESPAM_slider_label', className="my-label")
    date_dropdown_label = html.Label('Date Format: ', id='date-dropdown-label', className="my-label")
    date_format_dropdown = dcc.Dropdown(id='date-format', options=date_format_options, value="%B %Y", className="my-dropdown")
    start_date_input = create_input(id='start_date', type='text', value='2000-01-01', label='Start Date ')
    date_freq_input = create_input(id='date_freq', type='number', value=1, label='Date Frequency ')
    animation_length_input = create_input(id='animation-length', type='number', value=60, label='Length of Animation (s) ')


    color_input_1 = create_color_input(id='color-1', position=-25, color='#ff0000')
    color_input_2 = create_color_input(id='color-2', position=-2, color='#ffffff')
    color_input_3 = create_color_input(id='color-3', position=2, color='#ffffff')
    color_input_4 = create_color_input(id='color-4', position=25, color='#00FF00')




    # Create ESPAM upload with tooltip
    ESPAM_upload = html.Div([
        dcc.Upload(id='ESPAM_upload', children=html.Div(['Drag and Drop or ', html.A('Select Files')]), className="my-upload"),
        create_tooltip('ESPAM_upload'),
    ])

    # Create buttons
    ESPAM_modal_open_button = create_button(id='ESPAM_modal_open', label='Edit Base GIS Layers', disabled=True)
    settings_save_button = dbc.Button('Download Settings', id='settings-save', disabled=True)
    settings_load_button = dbc.Button('Load Settings', id='settings-load', disabled=True)
    generate_animation_button = dbc.Button('Download Animation', id='generate-animation', disabled=True)
    generate_warning_label = html.Label('', id='generate-warning-label', className="my-label")

    # Create sliders
    figure_height_slider = create_slider(id='figure-height', label='Figure Height', 
                                         min=0, max=720, step=50, value=720, 
                                         marks={i: f'{i}' for i in range(0, 721, 200)}.update({720: '720'}),
                                         vertical=True, verticalHeight=720, labelID='-height')
    figure_width_slider = create_slider(id='figure-width', label='Figure Width', 
                                        min=0, max=720, step=50, value=720, 
                                        marks={i: f'{i}' for i in range(0, 721, 200)}.update({720: '720'}),
                                        labelID='-width')

    # Create a DBC card
    buttons_card = dbc.Card([
        dbc.CardHeader("Output Settings"),
        dbc.CardBody([
            dbc.Row([
                figure_title_input,
                dbc.ButtonGroup([
                    settings_save_button,
                    settings_load_button,
                    generate_animation_button,
                    create_tooltip('settings-save'),
                    create_tooltip('settings-load'),
                    create_tooltip('generate-animation'),
                    create_tooltip('download-animation'),
                    dcc.Download(id='download-animation-file'),
                    dcc.Download(id='download-settings-file'),
                ], vertical=True),
            ]),
            generate_warning_label,
        ])
    ])

    # Create a DBC card for date settings
    date_card = dbc.Card([
        dbc.CardHeader("Date Settings"),
        dbc.CardBody([
            dbc.Row([
                ESPAM_slider,
                ESPAM_slider_label,
                start_date_input,
                date_freq_input,
                date_dropdown_label,
                date_format_dropdown,
            ]),
        ]),
    ])


    # Create the layout
    content = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    ESPAM_upload,
                    ESPAM_modal_open_button,
                    animation_length_input,
                    date_card,
                    buttons_card,  
                    modal,
                    GIS_Options,
                    GIS_Files,
                    Color_Values,
                    Colors,
                    Zoom,
                    SettingsValues,
                ], className="my-div"),
            ], width=3),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        figure_height_slider,
                    ], width=1),
                    dbc.Col([
                        dcc.Graph(id='ESPAM_graph', figure=mapPlot(), className='ESPAM-graph'),
                    ], width=7),
                    dbc.Col([
                        color_input_1,
                        color_input_2,
                        color_input_3,
                        color_input_4,
                    ], width=1),
                ]),
                dbc.Row([
                    figure_width_slider,
                ]),
            ], width=9),
        ])
    ])



    return content

