from dash import dcc, html
from dash_extensions.enrich import DashProxy, MultiplexerTransform, dcc, html

import dash_bootstrap_components as dbc

import os

from views.Layout import make_ESPAM_layout
from views.Callbacks import get_ESPAM_callbacks

ESPAM_content = make_ESPAM_layout()

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


# Make a store for WLTest
WLs_Store = dcc.Store(id='WLs_Store', data={})
WL_Store = dcc.Store(id='WL_Store')
SHP_Store = dcc.Store(id='SHP_Store', data=[])


Project_ID = '20230620'

    
Project_ID = dcc.Store(id='Project_ID', data=Project_ID)



app = DashProxy(transforms=[MultiplexerTransform()],
                title="ESPAM",
                external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
                )


app.layout = html.Div(
    [
        html.H1("Animation Generation Tool"),
        ESPAM_content,
        WLs_Store,
        WL_Store,
        SHP_Store,
        Project_ID,
    ]
)

from dash import Dash, DiskcacheManager, CeleryManager, Input, Output, html, callback

if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)

else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)

get_ESPAM_callbacks(app, background_callback_manager)

if __name__ == "__main__":
    app.run_server(debug=True, port=8050)