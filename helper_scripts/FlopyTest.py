import dash
import dash_core_components as dcc
from dash import html
import flopy.utils.binaryfile as bf
from dash.dependencies import Input, Output, State
from flask import Flask
import base64
import io

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Only allow uploading one file
        multiple=False
    ),
    html.Div(id='output-data-upload'),
])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(content, name, date):
    if content is not None:
        # the content is in base64 string format, so we decode it
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        
        # now we can read the file using FloPy
        with open(name, 'wb') as fp:
            fp.write(decoded)
        headobj = bf.HeadFile(name)
        data = headobj.get_data(totim=1.0)  # get data for a specific timestep

        # you can then use this data as you wish
        # for example, you can just return the shape of the data
        return 'File {} successfully uploaded! Data shape: {}'.format(name, data.shape)

if __name__ == '__main__':
    app.run_server(debug=False)
