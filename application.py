from dash import dcc, html
from dash_extensions.enrich import DashProxy, MultiplexerTransform, dcc, html

import dash_bootstrap_components as dbc
from datetime import datetime

import pandas as pd
import numpy as np
import json

import os

import flopy

# from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import rotate
from scipy import interpolate

from views.Layout import make_ESPAM_layout
from views.Callbacks import get_ESPAM_callbacks

ESPAM_content = make_ESPAM_layout()

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


# Make a store for WLTest
# WLs_Store = dcc.Store(id='WLs_Store', data=WLRot.to_dict('dict'))

# Make a store for WLTest
WLs_Store = dcc.Store(id='WLs_Store', data={})
WL_Store = dcc.Store(id='WL_Store')
SHP_Store = dcc.Store(id='SHP_Store', data=[])

# Project_ID = dcc.Store(id='Project_ID', data=datetime.now().strftime('%Y%m%d%H%M%S%f'))
Project_ID = '20230620'

    
Project_ID = dcc.Store(id='Project_ID', data=Project_ID)


print('step 2')


app = DashProxy(transforms=[MultiplexerTransform()],
                title="ESPAM",
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                )


app.layout = html.Div(
    [
        html.H1("ESPAM"),
        ESPAM_content,
        WLs_Store,
        WL_Store,
        SHP_Store,
        Project_ID,
    ]
)



get_ESPAM_callbacks(app)

print('step 3')

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)