

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation
import matplotlib.colors
import flopy
import configparser
import os


#Needs to be formatted JSON can use ArcMap to convert shapefile
def ShapefilePlot(filename, ax, line_color, fill_color, line_width, opacity, zorder):
    with open(filename) as json_file:
        data = json.load(json_file)

    for feat in data['features']:
        geom = feat['geometry']
        geomType = list(geom.keys())[0]
        if geomType=='rings':
            for polygon in geom[geomType]:
                points = np.array(polygon)
                ax.fill(points[:,0], points[:,1], facecolor=fill_color, edgecolor=line_color, 
                        alpha=opacity, linewidth=line_width, zorder=zorder)
        elif geomType=='paths':
                for points in geom[geomType]:
                    points = np.array(points)
                    ax.plot(points[:,0], points[:,1], c=line_color, alpha=opacity, linewidth=line_width, 
                            zorder=zorder)



def ModelAnimation(ConfigFile):
  

    #Read section files
    Output = f'Output/Animation.mp4'
    WL_Data = f'Output/WLTest.npy'

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


    Active = pd.read_csv('Input/ModelBoundary.csv')

    Boundary = np.zeros((104, 209), dtype=float)
    for i, j in zip(Active[Active['ACTIVE'] == 1]['ROW_ID']-1, Active[Active['ACTIVE'] == 1]['COL_ID']-1):
        Boundary[int(i), int(j)] = 1
    Boundary = Boundary.astype(bool)

    SHPs = ConfigFile['GIS_options']

    for i, shp in enumerate(SHPs):

        if not ConfigFile['GIS_options'][shp]['visible']:
            continue  
            
        ShapefilePlot(f'GIS/{shp}.json',
                      ax=WLAx,
                      line_color=ConfigFile['GIS_options'][shp]['line-color'],
                      fill_color=ConfigFile['GIS_options'][shp]['fill-color'],
                      line_width=ConfigFile['GIS_options'][shp]['line-width'],
                      opacity=ConfigFile['GIS_options'][shp]['opacity'],
                      zorder=10
                      )
        
        # ConfigFile['GIS_options'][shp]['draw-order']


    Lims = ConfigFile['zoom']
    WLAx.set_yticklabels([])
    WLAx.set_xticklabels([])
    WLAx.axis('off')

    WLAx.set_xlim([Lims[0], Lims[1]])
    WLAx.set_ylim([Lims[2], Lims[3]])

    #Initialize contours
    Ti = WLTest[:,:,0]
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

    contf=WLAx.contourf(X, Y, np.zeros((104,209)), np.linspace(min(ScalePoints), max(ScalePoints), 50),extend='both',cmap=cmap)
    fig.colorbar(contf, ax=WLAx, ticks=np.linspace(min(ScalePoints), max(ScalePoints), 4))


    def update(i):
            global cont
            global contf
            global dateText
            date = Dates[i]

            print(i)

            #Filter model data for cleaner visualization
            Ti = WLTest[:,:,i]
            Ti[~Boundary] = 0
            Ti = gaussian_filter(Ti, 1)
            Ti[~Boundary] = np.nan



            #Remove old data and plot new data on top
            if 'cont' in globals():
                for coll in cont.collections:
                    coll.remove()
            if 'contf' in globals():
                for coll in contf.collections:
                    coll.remove()
            if 'dateText' in globals():
                dateText.remove()


            contf = WLAx.contourf(X, Y, Ti, np.linspace(min(ScalePoints), max(ScalePoints), 50), extend='both', cmap=cmap)
            cont = WLAx.contour(X, Y, Ti, np.linspace(min(ScalePoints), max(ScalePoints), 50), extend='both', colors='gray', linewidths=0.5)

            dateText = WLAx.text(2400000, 1460000, date.strftime(DateFormat),fontsize=18,
                       bbox=dict(boxstyle="round",
                       fc=(1., 1, 1),
                       ec=(0, 0, 0),
                       ))




    #Run and save animation
    plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg-2021-07-06-git-758e2da289-full_build/bin/ffmpeg.exe'

    n = WLTest.shape[2]

    ani = animation.FuncAnimation(fig, update, frames=n, repeat=False,
                                  interval=int(ConfigFile['animation_length']*1000/n)*60)
    ani.save(Output, writer=animation.FFMpegWriter(fps=10), dpi=300)

if __name__ == '__main__':
    with open('Output/settings.json') as json_file:
        ConfigFile = json.load(json_file)
    ModelAnimation(ConfigFile)