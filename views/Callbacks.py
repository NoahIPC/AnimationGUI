from dash import dcc, html, Input, Output, State, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html
import base64

from scipy.ndimage import rotate
from scipy import interpolate

from views.ModelAnimation import ModelAnimation

import numpy as np
import pandas as pd
import flopy

import os
import json

import dash_bootstrap_components as dbc

from views.mapPlot import mapPlot

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

def rotate_image(image, xy, angle):
    im_rot = rotate(image, angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center

def get_ESPAM_callbacks(app, background_callback_manager):
        
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
            Input('WLs_Store', 'data'),
            Input('GIS-options', 'data'),
            Input('figure-height', 'value'),
            Input('figure-width', 'value'),
            Input('color-values', 'data'),
            Input('colors', 'data'),
            State('zoom', 'data'),
            prevent_initial_call=True
        )
        def update_graph(slider_value, WLs, GIS_Options, height, width, color_values, colors, zoom):
            # Make a simple plot
            WLs = pd.DataFrame.from_dict(WLs)
            try:
                WL = WLs.iloc[:, slider_value]
            except IndexError:
                raise PreventUpdate
            WL = WL.values.reshape(198, 233)

            WL[(WL>-0.1) & (WL<0.1)] = 0

            return (mapPlot(WL, GIS_Options=GIS_Options, height=height, width=width, 
                            colors=colors['data'], values=color_values['data'], Zoom=zoom['data']),
                   pd.DataFrame(WL).to_dict('records'))
        
        @app.callback(
            Output('WLs_Store', 'data'),
            Output('date_freq', 'value'),
            Input('ESPAM_upload', 'contents'),
            State('ESPAM_upload', 'filename'),
            State('ESPAM_upload', 'last_modified'),
            State('Project_ID', 'data'),
            prevent_initial_call=True,
        )
        def update_WLs(contents, filename, last_modified, Project_ID):

            alpha = 31.4

            Active = pd.read_csv('Input/ModelBoundary.csv')

            Boundary = np.zeros((104, 209), dtype=float)
            for i, j in zip(Active[Active['ACTIVE'] == 1]['ROW_ID']-1, Active[Active['ACTIVE'] == 1]['COL_ID']-1):
                Boundary[int(i), int(j)] = 1


            BoundaryRot = rotate(Boundary, alpha)

            BoundaryRot[BoundaryRot < 0.5] = 0
            BoundaryRot = BoundaryRot.astype(bool)
            BoundaryRot = np.flip(BoundaryRot, axis=0)

            Boundary = Boundary.astype(bool)

            print('step 1')

            if filename.endswith('.hds'):


                content_type, content_string = contents.split(',')
                decoded = base64.b64decode(content_string)
                
                # now we can read the file using FloPy
                with open(filename, 'wb') as fp:
                    fp.write(decoded)
                hdobj = flopy.utils.binaryfile.HeadFile(filename, precision='single')
                
                WLTest = []
                for time in hdobj.times:
                    WL = hdobj.get_data(totim=time)
                    WL = WL[0,:,:]
                    WLTest.append(WL)

            WLTest = np.dstack(WLTest)

            # If folder doesn't exist, create it
            if not os.path.exists(os.path.join(os.getcwd(), f'Output/{Project_ID}')):
                os.makedirs(f'Output/{Project_ID}')
                
                np.save(f'Output/{Project_ID}/WLTest.npy', WLTest)
                

            WLRot = []

            for i in range(WLTest.shape[2]):

                WL = WLTest[:,:,i]

                WL[~Boundary] = np.nan

                WL = interpolate_missing_pixels(WL, ~Boundary)

                WL, [jt, it] = rotate_image(WL, [0, 0], alpha)
                WL = np.flip(WL, axis=0)
                WL[~BoundaryRot] = np.nan

                # WL = interpolate_missing_pixels(WL, ~BoundaryRot)
                WL = WL.ravel()
                WLRot.append(WL)

            WLRot = np.array(WLRot).T

            WLRot = pd.DataFrame(WLRot, columns=hdobj.times)

            freq = hdobj.times[1]-hdobj.times[0]

            return WLRot.to_dict('records'), freq

        @app.callback(
            Output('ESPAM_modal', 'is_open'),
            Input('ESPAM_modal_open', 'n_clicks'),
            Input('ESPAM_modal_close', 'n_clicks'),
            State('ESPAM_modal', 'is_open'),
            prevent_initial_call=True
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
            prevent_initial_call=True
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
                State('date-format', 'value'),
                prevent_initial_call=True
        )
        def update_slider_label(value, start_date, date_freq, date_format):
            return 'Date: '+(pd.to_datetime(start_date) + pd.Timedelta(days=value*date_freq)).strftime(date_format)


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
            return max, marks, int(max/2)

        @app.callback(
            # Output('color-row-1', 'style'),
            # Output('color-row-2', 'style'),
            # Output('color-row-3', 'style'),
            # Output('color-row-4', 'style'),
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
            prevent_initial_call=True
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

            # return [{'top':f'{positions[3]}px', 'position':'absolute'},
            #         {'top':f'{positions[1]}px', 'position':'absolute'},
            #         {'top':f'{positions[2]}px', 'position':'absolute'},
            #         {'top':f'{positions[0]}px', 'position':'absolute'},
            return [-np.inf, color_1_position, color_2_position, color_3_position,
                    color_2_position, color_3_position, color_4_position, np.inf,
                    {'data':[color_1_position, color_2_position, color_3_position, color_4_position]},
                    {'data':[color_1, color_2, color_3, color_4]},
                    ]



        @app.callback(
            Output('settings-values', 'data'),
            Input('settings-save', 'n_clicks'),
            State('color-values', 'data'),
            State('colors', 'data'),
            State('GIS-options', 'data'),
            State('date-format', 'value'),
            State('date_freq', 'value'),
            State('start_date', 'value'),
            State('figure-height', 'value'),
            State('figure-width', 'value'),
            State('zoom', 'data'),
            State('Project_ID', 'data'),
            State('WLs_Store', 'data'),
            State('animation-length', 'value'),
            State('figure-title', 'value'),
        )
        def save_settings(
                n_clicks,
                color_values,
                colors,
                GIS_options,
                date_format,
                date_freq,
                start_date,
                figure_height,
                figure_width,
                zoom,
                Project_ID,
                WLs,
                animation_length,
                figure_title):
            if n_clicks is None:
                raise PreventUpdate
            settings = {
                'color_values': color_values['data'],
                'colors': colors['data'],
                'GIS_options': GIS_options,
                'figure_height': figure_height,
                'figure_width': figure_width,
                'zoom': zoom['data'],
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
        
        @app.callback(
            Input('WLs_Store', 'data'),
            Output('ESPAM_modal_open', 'disabled'),
            Output('settings-save', 'disabled'),
            Output('settings-load', 'disabled'),
            Output('generate-animation', 'disabled'),
            prevent_initial_call=True
        )
        def enable_buttons(WLs):
            if WLs is None:
                raise PreventUpdate
            return False, False, False, False

        @app.callback(
            Input('generate-warning-label', 'children'),
            State('settings-values', 'data'),
            State('Project_ID', 'data'),
            Output("download-animation-file", "data"),
            prevent_initial_call=True,
            background=True,
            manager=background_callback_manager,
        )
        def generate_animation(n_clicks, settings, Project_ID):
            
            ModelAnimation(settings)

            return dcc.send_file(f'Output/{Project_ID}/Animation.mp4')

        @app.callback(
            Output("download-settings-file", "data"),
            State('Project_ID', 'data'),
            Input("settings-save", "n_clicks"),
            prevent_initial_call=True,
        )
        def func(Project_ID, n_clicks):
            if n_clicks is None:
                raise PreventUpdate
            
            return dcc.send_file(f'Output/{Project_ID}/settings.json')
        
        @app.callback(
            Output('generate-warning-label', 'children'),
            Output('generate-warning-label', 'style'),
            Input('generate-animation', 'n_clicks'),
            prevent_initial_call=True
        )
        def generate_animation(n_clicks):
            if n_clicks is None:
                raise PreventUpdate
            
            return 'Generating animation. This may take a while.', {'color':'red'}

        # @app.callback(
        #     Input('settings-load', 'n_clicks'),
        #     Output('settings-values', 'value'),
        #     Output('Project_ID', 'value'),
        #     Output('WLs_Store', 'data'),
        #     Output('color-values', 'data'),
        #     Output('colors', 'data'),
        #     Output('GIS-options', 'data'),
        #     Output('date-format', 'value'),
        #     Output('date_freq', 'value'),
        #     Output('start_date', 'value'),
        #     Output('figure-height', 'value'),
        #     Output('figure-width', 'value'),
        #     Output('zoom', 'value'),
        #     Output('animation-length', 'value'),
        #     Output('figure-title', 'value'),
        # )
        # def load_settings(n_clicks):
        #     if n_clicks is None:
        #         raise PreventUpdate
        #     Project_ID = 'test'
        #     with open(f'Output/{Project_ID}/settings.json', 'r') as f:
        #         settings = json.load(f)

        #     WL = np.load(f'Output/{Project_ID}/WLTest.npy', allow_pickle=True)


        #     return (settings, Project_ID, WL, settings['color_values'], settings['colors'], 
        #             settings['GIS_options'], settings['date_format'], settings['date_freq'], 
        #             settings['start_date'], settings['figure_height'], settings['figure_width'], 
        #             settings['zoom'], settings['animation_length'], settings['figure_title'])
