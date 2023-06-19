import dash
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table, ALL
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html, MultiplexerTransform, DashProxy

import numpy as np
import pandas as pd
import flopy

import os
import json

from scipy import interpolate


def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0
):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image



def UpdateDataPlot(DataValues, PltLines, PltTypes, PltData, PltAxes, Date):
    i=0
    for (ax, data, p) in zip(PltAxes, PltData, PltTypes):
        BaseHeight = np.zeros(len(DataValues))
        for vals in PltData[data]:
            mask = DataValues.index[DataValues.index<=Date]
            line = PltLines[i]
            if PltTypes[p]=='Line':
                line.set_xdata(mask)
                line.set_ydata(PltData[data][vals].loc[mask])
            else:
                LineData = PltData[data][vals].copy()
                LineData.loc[DataValues.index[DataValues.index>Date]] = 0
                c = line[0]._facecolor
                z = line[0].zorder
                line.remove()
                line = PltAxes[ax].bar(DataValues.index, LineData+BaseHeight, width=31, color=c, zorder=z)
                BaseHeight += LineData.values
                PltLines[i] = line

            i += 1

def mapPlot(WL=[], GIS_Options=None, height=930, width=1870, oldFig=None):

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
    fig.update_xaxes(range=[2379000, 2752000])
    fig.update_yaxes(range=[1191000, 1508000])

    # Set the aspect ratio to 1
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    # Turn the legend off
    fig.update_layout(showlegend=False)

    # Set the background color to white and remove the grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_layout(plot_bgcolor='white')

    # Add a border to the figure
    fig.update_layout(
        xaxis=dict(
            linecolor='black',
            linewidth=2,
            mirror=True
        ),
        yaxis=dict(
            linecolor='black',
            linewidth=2,
            mirror=True
        )
    )

    colors = ['red', 'white', 'white', 'green']
    values = [-25, -2, 2, 25]
    # normalize the colorscale 0 to 1
    colorscale = [[(values[i]-values[0])/(values[-1]-values[0]), colors[i]] for i in range(len(values))]

    if len(WL)>0:

        rotR = np.deg2rad(31.4)

        i0 = 1191000
        j0 = 2379000

        Y = np.linspace(i0, 1.508e6, WL.shape[0])
        X = np.linspace(j0, 2.752e6, WL.shape[1])

        WL = pd.DataFrame(WL)
        WL = WL.fillna(np.nan)
        fig.add_trace(go.Contour(x=X, y=Y, z=WL.values, zmin=-25, zmax=25, 
                                 contours=dict(start=-25, end=25, size=1),
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

    # Make content with a graph, a slider, a dropdown, and a file selector
    content = html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label('Select a Model Timestep'),
                    dcc.Slider(id='ESPAM_slider', min=0, max=100, step=1, value=75, marks={i: f'{i}' for i in range(0, 101, 25)}, 
                            className="my-slider"),
                    # Set figure size
                    dbc.Row([html.Label('Figure Height'),
                            dcc.Input(id='figure-height', type='number', value=930, min=0, max=10000, step=50, debounce=True),
                            html.Label('Figure Width'),
                            dcc.Input(id='figure-width', type='number', value=1870, min=0, max=10000, step=50, debounce=True)]),
                    html.Button('Edit Base GIS Layers', id='ESPAM_modal_open', className="my-button"),
                    dcc.Upload(id='ESPAM_upload', children=html.Div(['Drag and Drop or ', html.A('Select Files')])),
                    modal,
                    GIS_Options,
                    GIS_Files,
                ], className="my-div"),
            ], width=3, xl=2, style={"border": "2px solid black"}),
            dbc.Col([
                dcc.Graph(id='ESPAM_graph', figure=mapPlot()),
            ], width=9, xl=10, style={"border": "2px solid black"}),
        ])
    ])



    return content



def get_ESPAM_callbacks(app):
    
        @app.callback(
            Output('ESPAM_graph', 'figure'),
            Output('WL_Store', 'data'),
            Input('ESPAM_slider', 'value'),
            State('WLs_Store', 'data'),
            State('GIS-options', 'data'),
            State('figure-height', 'value'),
            State('figure-width', 'value'),
        )
        def update_graph(slider_value, WLs, GIS_Options, height, width):
            # Make a simple plot
            WLs = pd.DataFrame.from_dict(WLs)
            WL = WLs.iloc[:, slider_value]
            WL = WL.values.reshape(198, 233)
            WL[WL<1] = 0

            # WL = interpolate.grid


            return mapPlot(WL, GIS_Options, height, width), pd.DataFrame(WL).to_dict('records')
        

        @app.callback(
            Output('ESPAM_graph', 'figure'),
            Input('GIS-options', 'data'),
            Input('figure-height', 'value'),
            Input('figure-width', 'value'),
            State('WL_Store', 'data'),
        )
        def update_shps(GIS_Options, height, width, WL):
            WL = pd.DataFrame.from_dict(WL)
            return mapPlot(WL, GIS_Options, height, width)
            
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

        