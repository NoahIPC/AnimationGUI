from dash import dcc, html, Input, Output, State, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html

from scipy.ndimage import rotate


import numpy as np
import pandas as pd
import flopy

import os
import json


def mapPlot(WL=[], GIS_Options=None, height=930, width=1870,
            colors=['red', 'white', 'white', 'green'], values=[-25, -2, 2, 25],
            Zoom=[2379000, 2752000, 1191000, 1508000]):

    # Get zoom level of old figure

    if not GIS_Options:
        GIS_Options = {}

    fig = go.Figure()

    if len(GIS_Options)>0:
        for file in GIS_Options:

            shp = GIS_Options[file]

            Params = ['color', 'alpha', 'linewidth', 'shapetype', 'zorder']
            ParamDefault = ['black', 1, 1, 'Polyline', 5]
            ParamVals = {'filename':f'GIS/{shp}.json'}
        
            for param, default in zip(Params, ParamDefault):
                ParamVals[param] = default
                    
                
            with open(f"GIS/{file}.json") as json_file:
                data = json.load(json_file)

            geomType = list(data['features'][0]['geometry'].keys())[0]

            visible = bool(shp['visible'])
            line_color = shp['line-color']
            fill_color = shp['fill-color']
            line_width = shp['line-width']
            opacity = shp['opacity']
            zorder = shp['draw-order']


                           

            if geomType=='rings':
                try:
                    points = np.array(data['features'][0]['geometry']['rings'][1])
                except IndexError:
                    points = np.array(data['features'][0]['geometry']['rings'][0])
                fig.add_trace(go.Scatter(x=points[:,0], y=points[:, 1], fill='toself', 
                                        fillcolor=fill_color, line_color=line_color, opacity=opacity, line_width=line_width,
                                        hoverinfo='skip', visible=visible))
            elif geomType=='paths':
                for feat in data['features']:
                    for points in feat['geometry']['paths']:
                        points = np.array(points)
                        fig.add_trace(go.Scatter(x=points[:,0], y=points[:,1], mode='lines',
                                                line_color=line_color, opacity=opacity, line_width=line_width,
                                                hoverinfo='skip', visible=visible))


    # Remove the x and y ticks
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    # Set the figure size to 18.7 x 9.3 inch figure
    fig.update_layout(
        width=width,
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
    )


    # Set the X and Y axis to the min and max of the model grid
    fig.update_xaxes(range=[Zoom[0], Zoom[1]])
    fig.update_yaxes(range=[Zoom[2], Zoom[3]])

    # Set the aspect ratio to 1
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    # Turn the legend off
    fig.update_layout(showlegend=False)

    # Set the background color to white and remove the grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(plot_bgcolor='white')

    def hex_to_rgb(hexa):
        hexa = hexa.lstrip('#')
        return tuple(int(hexa[i:i+2], 16)  for i in (0, 2, 4))
    # Convert colors to rgb
    colors = ['rgb({}, {}, {})'.format(*hex_to_rgb(i[1:])) if '#' in i else i for i in colors]

    # normalize the colorscale 0 to 1
    colorscale = [[(values[i]-values[0])/(values[-1]-values[0]), colors[i]] for i in range(len(values))]

    if len(WL)>0:

        i0 = 1191000
        j0 = 2379000

        Y = np.linspace(i0, 1.508e6, WL.shape[0])
        X = np.linspace(j0, 2.752e6, WL.shape[1])

        WL = pd.DataFrame(WL)
        WL = WL.fillna(np.nan)
        fig.add_trace(go.Contour(x=X, y=Y, z=WL.values, zmin=min(values), zmax=max(values), 
                                 contours=dict(start=min(values), end=max(values), size=1),
                                 line=dict(width=0.5, color='gray'), colorscale=colorscale))
        fig.update_layout(coloraxis_showscale=False)


    return fig


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
            dbc.RadioItems(id={'type':'visible', 'index':file}, 
                           options=[{'label': 'Visible', 'value': 1}, {'label': 'Invisible', 'value': 0}],
                            value=0, inline=True),
            dbc.Row([
                dbc.Col([
                    dbc.Label('Line Color'),
                    dbc.Input(id={'type':'line-color', 'index':file}, type='color', value='#000000'),
                ]),
                dbc.Col([
                    dbc.Label('Fill Color'),
                    dbc.Input(id={'type':'fill-color', 'index':file}, type='color', value='#ffffff'),
                ]),
                dbc.Col([
                    dbc.Label('Line Width'),
                    dbc.Input(id={'type':'line-width', 'index':file}, type='number', value=1),
                ]),
                dbc.Col([
                    dbc.Label('Opacity'),
                    dbc.Input(id={'type':'opacity', 'index':file}, type='number', value=0.3),
                ]),
                dbc.Col([
                    dbc.Label('Draw Order'),
                    dbc.Input(id={'type':'draw-order', 'index':file}, type='number', value=0),
                ]),
            ])
        ], style={'float': 'left'})
        Tabs.append(tab)

    # Make a modal with a tab for each file
    modal = html.Div([
        dbc.Modal([
            dbc.ModalHeader("GIS Files"),
            dbc.ModalBody([
                dcc.Tabs(Tabs, id='ESPAM_modal_tabs', vertical=True, parent_style={'float': 'left'}),
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
        {"label": "Year only (2020)", "value": "%Y"}
    ]

    # Make content with a graph, a slider, a dropdown, and a file selector
    content = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Input(id='figure-title', type='text', value='ESPAM', className="my-input"),
                    html.Label('Select a Model Timestep'),
                    dcc.Slider(id='ESPAM_slider', className="my-slider"),
                    html.Label('Timestep: ', id='ESPAM_slider_label', className="my-label"),
                    html.Label('Start Date'),
                    dcc.Input(id='start_date', type='text', value='2000-01-01', className="my-input", pattern=r'\d{4}-\d{2}-\d{2}'),
                    html.Label('Model Timestep Frequency (Days)'),
                    dcc.Input(id='date_freq', type='number', value=1, className="my-input"),
                    html.Label('Animation Length (Seconds)', id='animation-length-label', className="my-label"),
                    dcc.Input(id='animation-length', type='number', value=60, className="my-input"),
                    html.Button('Edit Base GIS Layers', id='ESPAM_modal_open', className="my-button"),
                    dcc.Upload(id='ESPAM_upload', children=html.Div(['Drag and Drop or ', html.A('Select Files')])),
                    html.Button('Save Settings', id='settings-save', className="my-button"),
                    dcc.Dropdown(id='date-format', options=date_format_options, value="%m/%d/%y", className="my-dropdown"),
                    modal,
                    GIS_Options,
                    GIS_Files,
                    Color_Values,
                    Colors,
                    Zoom,
                    SettingsValues,
                ], className="my-div"),
            ], width=3, xl=2),
            dbc.Col([
                dbc.Row([
                    dbc.Row([
                        dbc.Col([
                            html.Label('Figure Height', className="height-slider-label"),
                            dcc.Slider(id='figure-height', min=0, max=720, step=50, value=720, marks={i: f'{i}' for i in range(0, 721, 200)}.update({720: '720'}),
                                    className="height-slider", vertical=True, verticalHeight=720),
                        ], width=1),
                        dbc.Col([
                                dcc.Graph(id='ESPAM_graph', figure=mapPlot(), className='ESPAM-graph'),
                        ], width=9),
                        dbc.Col([
                            dbc.Row([
                                dcc.Input(id='color-1-position', type='number', value=-25, className="color-position", debounce=True),
                                dbc.Input(id='color-1', type='color', value='#ff0000', className="color-picker", style={'width': '50px', 'height': '50px'}, debounce=True)
                            ], id="color-row-1"),
                            dbc.Row([
                                dcc.Input(id='color-2-position', type='number', value=-2, className="color-position", debounce=True),
                                dbc.Input(id='color-2', type='color', value='#ffffff', className="color-picker", style={'width': '50px', 'height': '50px'}, debounce=True)
                            ], id="color-row-2"),
                            dbc.Row([
                                dcc.Input(id='color-3-position', type='number', value=2, className="color-position", debounce=True),
                                dbc.Input(id='color-3', type='color', value='#ffffff', className="color-picker", style={'width': '50px', 'height': '50px'}, debounce=True)
                            ], id="color-row-3"),
                            dbc.Row([
                                dcc.Input(id='color-4-position', type='number', value=25, className="color-position", debounce=True),
                                dbc.Input(id='color-4', type='color', value='#00FF00', className="color-picker", style={'width': '50px', 'height': '50px'}, debounce=True)
                            ], id="color-row-4"),
                        ], width=2),
                    ]),
                    dbc.Row([
                        html.Label('Figure Width', className="width-slider-label"),
                        dcc.Slider(id='figure-width', min=0, max=1280, step=50, value=1280, marks={i: f'{i}' for i in range(0, 1281, 200)}.update({1280: '1280'}),
                            className="width-slider", debounce=True),
                    ]),
                ])
            ], width=9, xl=10),
        ])
    ])



    return content



def get_ESPAM_callbacks(app):
        
        @app.callback(
            Output('zoom', 'data'),
            Input('ESPAM_graph', 'relayoutData'),
            State('ESPAM_graph', 'figure'),
        )
        def update_zoom(relayoutData, figure):
            if not relayoutData:
                return no_update
            try:
                return {'data': list(relayoutData.values())}
            except KeyError:
                return {'data': figure['layout']['mapbox']['zoom']}
    
        @app.callback(
            Output('ESPAM_graph', 'figure'),
            Output('WL_Store', 'data'),
            Input('ESPAM_slider', 'value'),
            State('WLs_Store', 'data'),
            Input('GIS-options', 'data'),
            Input('figure-height', 'value'),
            Input('figure-width', 'value'),
            Input('color-values', 'data'),
            Input('colors', 'data'),
            State('zoom', 'data'),
        )
        def update_graph(slider_value, WLs, GIS_Options, height, width, color_values, colors, zoom):
            # Make a simple plot
            WLs = pd.DataFrame.from_dict(WLs)
            WL = WLs.iloc[:, slider_value]
            WL = WL.values.reshape(198, 233)

            WL[(WL>-0.1) & (WL<0.1)] = 0

            return (mapPlot(WL, GIS_Options=GIS_Options, height=height, width=width, 
                            colors=colors['data'], values=color_values['data'], Zoom=zoom['data']),
                   pd.DataFrame(WL).to_dict('records'))
        # @app.callback(
        #     Output('WLs_Store', 'data'),
        #     Input('ESPAM_upload', 'contents'),
        #     State('ESPAM_upload', 'filename'),
        #     State('ESPAM_upload', 'last_modified'),
        # )
        # def update_WLs(contents, filename, last_modified):
            
        #     def rotate_image(image, xy, angle):
        #         im_rot = rotate(image, angle) 
        #         org_center = (np.array(image.shape[:2][::-1])-1)/2.
        #         rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
        #         org = xy-org_center
        #         a = np.deg2rad(angle)
        #         new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
        #                 -org[0]*np.sin(a) + org[1]*np.cos(a) ])
        #         return im_rot, new+rot_center

        #     alpha = 31.4

        #     Active = pd.read_csv('Input/ModelBoundary.csv')

        #     Boundary = np.zeros((104, 209), dtype=float)
        #     for i, j in zip(Active[Active['ACTIVE'] == 1]['ROW_ID']-1, Active[Active['ACTIVE'] == 1]['COL_ID']-1):
        #         Boundary[int(i), int(j)] = 1


        #     BoundaryRot = rotate(Boundary, alpha)

        #     BoundaryRot[BoundaryRot < 0.5] = 0
        #     BoundaryRot = BoundaryRot.astype(bool)
        #     BoundaryRot = np.flip(BoundaryRot, axis=0)

        #     Boundary = Boundary.astype(bool)

        #     HeadFile = 'Input/ActualRecharge.hds'

        #     hdobj = flopy.utils.binaryfile.HeadFile(HeadFile, precision='single')
        #     WLTest = []
        #     for time in hdobj.times:
        #         WL = hdobj.get_data(totim=time)
        #         WL = WL[0,:,:]
        #         WLTest.append(WL)

        #     WLTest = np.dstack(WLTest)

        #     WLRot = []

        #     for i in range(WLTest.shape[2]):

        #         WL = WLTest[:,:,i]

        #         WL[~Boundary] = np.nan

        #         WL, [jt, it] = rotate_image(WL, [0, 0], alpha)
        #         WL = np.flip(WL, axis=0)
        #         WL[~BoundaryRot] = np.nan
        #         # WL = interpolate_missing_pixels(WL, ~BoundaryRot)
        #         WL = WL.ravel()
        #         WLRot.append(WL)

        #     WLRot = np.array(WLRot).T

        #     WLRot = pd.DataFrame(WLRot, columns=hdobj.times)

        #     # Make a store for WLTest
        #     WLs_Store = dcc.Store(id='WLs_Store', data=WLRot.to_dict('dict'))

        #     return WLs_Store

        @app.callback(
            Output('ESPAM_modal', 'is_open'),
            Input('ESPAM_modal_open', 'n_clicks'),
            Input('ESPAM_modal_close', 'n_clicks'),
            State('ESPAM_modal', 'is_open')
        )
        def toggle_modal(n1, n2, is_open):
            if n1 or n2:
                return not is_open
            return is_open
        
        @app.callback(
            Output('GIS-options', 'data'),
            Input('ESPAM_modal_close', 'n_clicks'),
            State({'type':'visible', 'index':ALL}, 'value'),
            State({'type':'line-color', 'index':ALL}, 'value'),
            State({'type':'fill-color', 'index':ALL}, 'value'),
            State({'type':'line-width', 'index':ALL}, 'value'),
            State({'type':'opacity', 'index':ALL}, 'value'),
            State({'type':'draw-order', 'index':ALL}, 'value'),
            State('GIS_Files', 'data'),
        )
        def update_gis_options(n_clicks, visible, line_color, fill_color, line_width, opacity, draw_order, GIS_Files):
            if n_clicks is None:
                raise PreventUpdate
            options = {}
            for i, file in enumerate(GIS_Files):
                options[file] = {
                    'visible': visible[i],
                    'line-color': line_color[i],
                    'fill-color': fill_color[i],
                    'line-width': line_width[i],
                    'opacity': opacity[i],
                    'draw-order': draw_order[i],
                }
            return options

        @app.callback(
                Input('ESPAM_slider', 'value'),
                Output('ESPAM_slider_label', 'children'),
                State('start_date', 'value'),
                State('date_freq', 'value'),
        )
        def update_slider_label(value, start_date, date_freq):
            return (pd.to_datetime(start_date) + pd.Timedelta(days=value*date_freq)).strftime('%Y-%m-%d')


        @app.callback(
            Output('ESPAM_slider', 'max'),
            Output('ESPAM_slider', 'marks'),
            Output('ESPAM_slider', 'value'),
            Input('WLs_Store', 'data'),
            Input('start_date', 'value'),
            Input('date_freq', 'value'),
        )
        def update_slider(WLs, start_date, date_freq):
            WLs = pd.DataFrame.from_dict(WLs)
            max = WLs.shape[1]
            try:
                start_date = pd.to_datetime(start_date)
            except:
                return no_update, no_update, no_update
            marks = {i: str((start_date + pd.Timedelta(days=i*date_freq)).year) for i in np.linspace(0, max, 3)}
            return int(max/2), marks, max

        @app.callback(
            Output('color-row-1', 'style'),
            Output('color-row-2', 'style'),
            Output('color-row-3', 'style'),
            Output('color-row-4', 'style'),
            Output('color-1-position', 'min'),
            Output('color-2-position', 'min'),
            Output('color-3-position', 'min'),
            Output('color-4-position', 'min'),
            Output('color-1-position', 'max'),
            Output('color-2-position', 'max'),
            Output('color-3-position', 'max'),
            Output('color-4-position', 'max'),
            Output('color-values', 'data'),
            Output('colors', 'data'),
            Input('color-1-position', 'value'),
            Input('color-2-position', 'value'),
            Input('color-3-position', 'value'),
            Input('color-4-position', 'value'),
            Input('color-1', 'value'),
            Input('color-2', 'value'),
            Input('color-3', 'value'),
            Input('color-4', 'value'),
        )
        def update_color_rows(color_1_position, color_2_position, color_3_position, color_4_position,
                              color_1, color_2, color_3, color_4):
            positions = np.array([color_1_position, color_2_position, color_3_position, color_4_position])
            positions = (positions - np.min(positions)) / (np.max(positions) - np.min(positions))

            Top = 70
            Height = 500

            positions = Top + positions * Height
            for i in range(3):
                if (positions[i]-positions[i+1])<120:
                    positions[i+1] += 120

            return [{'top':f'{positions[3]}px', 'position':'absolute'},
                    {'top':f'{positions[1]}px', 'position':'absolute'},
                    {'top':f'{positions[2]}px', 'position':'absolute'},
                    {'top':f'{positions[0]}px', 'position':'absolute'},
                    -np.inf, color_1_position, color_2_position, color_3_position,
                    color_2_position, color_3_position, color_4_position, np.inf,
                    {'data':[color_1_position, color_2_position, color_3_position, color_4_position]},
                    {'data':[color_1, color_2, color_3, color_4]},
                    ]



        @app.callback(
            Output('settings-values', 'value'),
            Input('settings-save', 'n_clicks'),
            State('color-values', 'data'),
            State('colors', 'data'),
            State('GIS-options', 'data'),
            State('date-format', 'value'),
            State('date-freq', 'value'),
            State('start-date', 'value'),
            State('figure-height', 'value'),
            State('figure-width', 'value'),
            State('zoom', 'value'),
            State('Project_ID', 'value'),
            State('WLs_Store', 'data'),
            State('animation-length', 'value'),
            State('figure-title', 'value'),
        )
        def save_settings(n_clicks,
                        color_values,
                        colors,
                        GIS_options,
                        figure_height,
                        figure_width,
                        zoom,
                        date_format,
                        date_freq,
                        start_date,
                        Project_ID,
                        WLs,
                        animation_length,
                        figure_title,
                        ):
            if n_clicks is None:
                raise PreventUpdate
            settings = {
                'color_values': color_values,
                'colors': colors,
                'GIS_options': GIS_options,
                'figure_height': figure_height,
                'figure_width': figure_width,
                'zoom': zoom,
                'date_format': date_format,
                'date_freq': date_freq,
                'start_date': start_date,
                'Project_ID': Project_ID,
                'animation_length': animation_length,
                'figure_title': figure_title,
            }


            with open(f'Output/{Project_ID}/settings.json', 'w') as f:
                json.dump(settings, f)

            return settings

