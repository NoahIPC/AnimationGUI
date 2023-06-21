#%%

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
from scipy.ndimage.filters import gaussian_filter
import matplotlib.animation as animation
import matplotlib.colors
import flopy
import configparser
import os


#Needs to be formatted JSON can use ArcMap to convert shapefile
# def ShapefilePlot(filename, ax, line_color, fill_color, line_width, opacity, zorder):

#%%
shp = 'AmericanFalls'

filename = f'../GIS/{shp}.json'
ax=WLAx
line_color=ConfigFile['GIS_options'][shp]['line-color']
fill_color=ConfigFile['GIS_options'][shp]['fill-color']
line_width=ConfigFile['GIS_options'][shp]['line-width']
opacity=ConfigFile['GIS_options'][shp]['opacity']
zorder=ConfigFile['GIS_options'][shp]['draw-order']


with open(filename) as json_file:
    data = json.load(json_file)

for feat in data['features']:
    geom = feat['geometry']
    geomType = list(geom.keys())[0]
    if geomType=='rings':
        for polygon in geom[geomType]:
            points = np.array(polygon)
            plt.fill(points[:,0], points[:,1], facecolor=fill_color, edgecolor=line_color, 
                    alpha=opacity, linewidth=line_width, zorder=zorder)
    elif geomType=='paths':
            for points in geom[geomType]:
                points = np.array(points)
                plt.plot(points[:,0], points[:,1], c=line_color, alpha=opacity, linewidth=line_width, 
                        zorder=zorder)

#%%

with open('../Output/20230620/settings.json') as json_file:
    ConfigFile = json.load(json_file)

#Read section files
Output = f'../Output/{ConfigFile["Project_ID"]}/Animation.mp4'
WL_Data = f'../Output/{ConfigFile["Project_ID"]}/WLTest.npy'
GIS_Path = ConfigFile["GIS_options"]

print(WL_Data)

WLTest = np.load(WL_Data)

StartDate = ConfigFile["start_date"]
Dates = pd.date_range(start=StartDate, periods=WLTest.shape[2], freq='D')



Width = ConfigFile['figure_width']
Height = ConfigFile['figure_height']
px = 1/plt.rcParams['figure.dpi']
#Setup the plot formating
fig, WLAx = plt.subplots(figsize=(Width*px, Height*px))

Title = ConfigFile['figure_title']
WLAx.set_title(Title, fontsize=24)


#Setup model grid coordinate field
X = np.zeros((104,209))
Y = np.zeros((104,209))
i0 = 1332998.93
j0 = 2378350.35
rot = 31.4*np.pi/180
sc = 1609.344

for i in range(104):
    for j in range(209):
        X[i,j] = j0+i*sc*np.sin(rot)+j*sc*np.cos(rot)
        Y[i,j] = i0-i*sc*np.cos(rot)+j*sc*np.sin(rot)


Active = pd.read_csv('../Input/ModelBoundary.csv')

Boundary = np.zeros((104, 209), dtype=float)
for i, j in zip(Active[Active['ACTIVE'] == 1]['ROW_ID']-1, Active[Active['ACTIVE'] == 1]['COL_ID']-1):
    Boundary[int(i), int(j)] = 1
Boundary = Boundary.astype(bool)

SHPs = ConfigFile['GIS_options']

for i, shp in enumerate(SHPs):

    if not ConfigFile['GIS_options'][shp]['visible']:
        continue  
        
    ShapefilePlot(f'../GIS/{shp}.json',
                    ax=WLAx,
                    line_color=ConfigFile['GIS_options'][shp]['line-color'],
                    fill_color=ConfigFile['GIS_options'][shp]['fill-color'],
                    line_width=ConfigFile['GIS_options'][shp]['line-width'],
                    opacity=ConfigFile['GIS_options'][shp]['opacity'],
                    zorder=ConfigFile['GIS_options'][shp]['draw-order']
                    )


Lims = ConfigFile['zoom']
WLAx.set_yticklabels([])
WLAx.set_xticklabels([])
WLAx.axis('off')

WLAx.set_xlim([Lims[0], Lims[1]])
WLAx.set_ylim([Lims[2], Lims[3]])

#Initialize contours
Ti = WLTest[:,:,50]
Ti = gaussian_filter(Ti, 1)
Ti[~Boundary] = np.nan

#Set color scale
ScalePoints = ConfigFile['color_values']
ScalePoints = [float(v) for v in ScalePoints]
ScaleColors = ConfigFile['colors']
DateFormat = ConfigFile['date_format']

norm = matplotlib.colors.Normalize(min(ScalePoints), max(ScalePoints))
colors=[]
for col, pnt in zip(ScaleColors, ScalePoints):
    colors.append([norm(pnt), col])

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)

contf=WLAx.contourf(X, Y, Ti, np.linspace(min(ScalePoints), max(ScalePoints), 50),extend='both',cmap=cmap)
fig.colorbar(contf, ax=WLAx, ticks=np.linspace(min(ScalePoints), max(ScalePoints), 4))

# %%
