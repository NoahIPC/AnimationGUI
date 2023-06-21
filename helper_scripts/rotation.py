#%%
import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

from scipy.ndimage import rotate
from scipy import interpolate

import flopy

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

Active = pd.read_csv('../Input/ModelBoundary.csv')

Boundary = np.zeros((104, 209), dtype=float)
for i, j in zip(Active[Active['ACTIVE'] == 1]['ROW_ID']-1, Active[Active['ACTIVE'] == 1]['COL_ID']-1):
    Boundary[int(i), int(j)] = 1

p = 130
q = 50

alpha = 31.4  # rotation angle

BoundaryRot = rotate(Boundary, alpha)

BoundaryRot[BoundaryRot < 0.1] = 0
BoundaryRot[np.isnan(BoundaryRot)] = 0
BoundaryRot = BoundaryRot.astype(bool)
BoundaryRot = np.flip(BoundaryRot, axis=0)

Boundary = Boundary.astype(bool)

norm = matplotlib.colors.Normalize(-25, 25)
colors = [[norm(-25.0), "red"],
          [norm(-2), "white"],
          [norm(2), "white"],
          [norm(25), "green"]]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)


HeadFile = '../Input/ActualRecharge.hds'

hdobj = flopy.utils.binaryfile.HeadFile(HeadFile, precision='single')
WLTest = []
for time in hdobj.times:
    WL = hdobj.get_data(totim=time)
    WL = WL[0,:,:]
    WLTest.append(WL)

WLTest = np.dstack(WLTest)

#%%

#Setup model grid coordinate field
i0 = 1332998.93
j0 = 2378350.35
rot = 31.4
sc = 1609.344


WL = WLTest[:, :, 75]
WL[~Boundary] = np.nan

WL = interpolate_missing_pixels(WL, ~Boundary)

def rotate_image(image, xy, angle):
    im_rot = rotate(image, angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center

WL, [jt, it] = rotate_image(WL, [0, 0], rot)
WL = np.flip(WL, axis=0)
WL[~BoundaryRot] = np.nan

#%%
rotR = np.deg2rad(31.4)

i0 = 1191000
j0 = 2379000

Y = np.linspace(i0, 1.508e6, len(WL[:, 0]))
X = np.linspace(j0, 2.752e6, len(WL[0, :]))

plt.figure(figsize=(10, 10))
c = plt.contourf(X, Y, WL, vmin=-2, vmax=2, cmap=cmap)
plt.colorbar(c)

plt.scatter([j0], [i0], color='black', marker='x', s=100)

import json

with open(f'../GIS/Boundary.json') as json_file:
    data = json.load(json_file)

geomType = list(data['features'][0]['geometry'].keys())[0]

for feat in data['features']:
    for points in feat['geometry'][geomType]:
        points = np.array(points)
        plt.plot(points[:,0],points[:,1])

# %%
import dash
from dash import Dash, dcc, html, Input, Output, State, no_update, dash_table
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html, MultiplexerTransform, DashProxy


fig = go.Figure()

SHPs = ['Boundary']
for i, shp in enumerate(SHPs):

    Params = ['color', 'alpha', 'linewidth', 'shapetype', 'zorder']
    ParamDefault = ['black', 1, 1, 'Polyline', 5]
    ParamVals = {'filename':f'../GIS/{shp}.json'}
    
    for param, default in zip(Params, ParamDefault):
        ParamVals[param] = default
            
        
    with open(f'../GIS/{shp}.json') as json_file:
        data = json.load(json_file)

    geomType = list(data['features'][0]['geometry'].keys())[0]
    shapetype = 'Polyline'
    if shapetype=='Polygon':
        points = np.array(data['features'][0]['geometry']['rings'][1])
        fig.add_trace(go.Scatter(x=points[:,0], y=points[:, 1], fill='toself', 
                                    fillcolor='blue', line_color='black', opacity=0.2, line_width=1,
                                    hoverinfo='skip'))
    else:
        for feat in data['features']:
            for points in feat['geometry'][geomType]:
                points = np.array(points)
                fig.add_trace(go.Scatter(x=points[:,0], y=points[:,1], 
                                            line_color='black', opacity=0.5, line_width=1,
                                            hoverinfo='skip'))


fig.add_trace(go.Contour(x=X, y=Y, z=WL, colorscale='jet', zmin=0, zmax=10, 
                            contours=dict(start=-25, end=250, size=0.5, coloring='lines'),
                            line=dict(width=1, color='black')))
# %%

WL_flat = WL.flatten()

WL = WL_flat.reshape(198, 233)
# %%
