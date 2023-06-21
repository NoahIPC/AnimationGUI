from dash import dcc, html, Input, Output, State, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html

from scipy.ndimage import rotate

from views.ModelAnimation import ModelAnimation

import numpy as np
import pandas as pd
import flopy

import os
import json

import dash_bootstrap_components as dbc
import dash_html_components as html
from dash import dcc



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
