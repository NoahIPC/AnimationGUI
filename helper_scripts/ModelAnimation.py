

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
def ShapefilePlot(filename, ax, color, alpha=1, linewidth=1, shapetype='Polyline', zorder=5):
    with open(filename) as json_file:
        data = json.load(json_file)

    geomType = list(data['features'][0]['geometry'].keys())[0]

    if shapetype=='Polygon':
        points = np.array(data['features'][0]['geometry']['rings'][1])
        ax.fill(points[:,0], points[:, 1], facecolor=color, edgecolor='black', alpha=alpha,linewidth=linewidth, zorder=zorder)
    else:
        for feat in data['features']:
            for points in feat['geometry'][geomType]:
                points = np.array(points)
                ax.plot(points[:,0],points[:,1], c=color, alpha=alpha,linewidth=linewidth, zorder=zorder)




def SectionRead(config, Section, Value, ErrorMessage):
    try:
        val = config.get(Section, Value)
        if not val or val=='""':
            return False
        else:
            return val
    except:
        if ErrorMessage:
            print(ErrorMessage)
            return False
        else:
            return False


def ModelAnimation(ConfigFile):

  

    os.chdir(os.path.dirname(ConfigFile))
    config = configparser.ConfigParser()
    config.read(ConfigFile)

    Sections = config.sections()


    #Read section files
    Output = SectionRead(config, 'Animation', 'Output', 'Missing Output Directory')
    WL_Data = SectionRead(config, 'Animation', 'HeadFile', 'Missing Head File')
    GIS_Path = SectionRead(config, 'Graphs', 'GIS_Files', False)

    print(WL_Data)

    hdobj = flopy.utils.binaryfile.HeadFile(WL_Data, precision='single')
    WLTest = []
    for time in hdobj.times:
        WL = hdobj.get_data(totim=time)
        WL = WL[0,:,:]
        WLTest.append(WL)

    WLTest = np.dstack(WLTest)


    Width = SectionRead(config, 'Animation', 'Width', False)
    Height = SectionRead(config, 'Animation', 'Height', False)
    px = 1/plt.rcParams['figure.dpi']
    #Setup the plot formating
    fig, WLAx = plt.subplots(figsize=(Width*px, Height*px))
    
    Title = SectionRead(config, 'Animation', 'Title', False)
    WLAx.set_title(Title)


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


    Boundary = pd.read_csv(SectionRead(config, 'Graphs', 'Active', 'Missing Active Cells Data')).values
    Boundary = Boundary.astype(bool)

    SHPs = eval(SectionRead(config, 'GIS', 'Files', False))

    for i, shp in enumerate(SHPs):

        Params = ['color', 'alpha', 'linewidth', 'shapetype', 'zorder']
        ParamDefault = ['black', 1, 1, 'Polyline', 5]
        ParamVals = {'filename':f'{GIS_Path}/{shp}.json', 'ax':WLAx}
       
        for param, default in zip(Params, ParamDefault):
            val = SectionRead(config, f'GIS_{i+1}', param, False)
            if val:
                try:
                    ParamVals[param] = float(val)
                except ValueError:
                    ParamVals[param] = val
            else:
                ParamVals[param] = default
                
            
        ShapefilePlot(**ParamVals)


    Lims = eval(SectionRead(config, 'Animation', 'ZoomBox', False))
    WLAx.set_yticklabels([])
    WLAx.set_xticklabels([])
    WLAx.set_xlim([Lims[0], Lims[2]])
    WLAx.set_ylim([Lims[1], Lims[3]])

    #Initialize contours
    Ti = WLTest[:,:,0]
    Ti[~Boundary] = 0
    Ti = gaussian_filter(Ti, 1)
    Ti[Boundary] = np.nan

    #Set color scale
    ScalePoints = eval(SectionRead(config, 'Animation', 'Scale_Points', False))
    ScalePoints = [float(v) for v in ScalePoints]
    ScaleColors = eval(SectionRead(config, 'Animation', 'Scale_Colors', False))
    DateFormat = SectionRead(config, 'Animation', 'Date_Format', False)

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
            global Sent
            date = DataValues.index[i]

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


            contf = WLAx.contourf(X, Y, Ti, np.linspace(min(ScalePoints), max(ScalePoints), 50),extend='both',cmap=cmap)

            dateText = WLAx.text(2400000, 1460000, date.strftime(DateFormat),fontsize=36,
                       bbox=dict(boxstyle="round",
                       fc=(1., 1, 1),
                       ec=(0, 0, 0),
                       ))




    #Run and save animation
    plt.rcParams['animation.ffmpeg_path'] = SectionRead(config, 'Graphs', 'ffmpeg', 'Missing ffmpeg Directory')

    ani = animation.FuncAnimation(fig, update, frames=range(WLTest.shape[2]),repeat=False)
    print(Output)
    ani.save(f'{os.path.dirname(ConfigFile)}/{Output}.mp4', writer=animation.FFMpegWriter())


if __name__ == '__main__':
    ModelAnimation('C:/Users/Nstewart-MAddox/Documents/GitHub/ESPAM2/ModelRuns/MP31_v22/Test.ini')