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

from views.ESPAM import get_ESPAM_callbacks, make_ESPAM_layout

ESPAM_content = make_ESPAM_layout()

from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')



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

HeadFile = 'Input/ActualRecharge.hds'

hdobj = flopy.utils.binaryfile.HeadFile(HeadFile, precision='single')
WLTest = []
for time in hdobj.times:
    WL = hdobj.get_data(totim=time)
    WL = WL[0,:,:]
    WLTest.append(WL)

WLTest = np.dstack(WLTest)

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

WLRot.to_csv('Input/WL_Rot.csv', index=False)

WLRot = pd.read_csv('Input/WL_Rot.csv')


# Make a store for WLTest
WLs_Store = dcc.Store(id='WLs_Store', data=WLRot.to_dict('dict'))

# Make a store for WLTest
# WLs_Store = dcc.Store(id='WLs_Store', data={})
WL_Store = dcc.Store(id='WL_Store')
SHP_Store = dcc.Store(id='SHP_Store', data=[])

# Project_ID = dcc.Store(id='Project_ID', data=datetime.now().strftime('%Y%m%d%H%M%S%f'))
Project_ID = dcc.Store(id='Project_ID', data='20230620')

# If folder doesn't exist, create it
if not os.path.exists(f'./data/{Project_ID}'):
    os.makedirs(f'Output/{Project_ID}')
    
    np.save(f'Output/{Project_ID}/WLTest.npy', WLTest)
    

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
    app.run_server(debug=True)