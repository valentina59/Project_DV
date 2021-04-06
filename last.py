import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dash.dependencies import Input, Output
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import plotly as py
import plotly.graph_objs as go
#py.offline.init_notebook_mode(connected = True)
plt.rcParams['figure.dpi'] = 140
from io import BytesIO
from wordcloud import WordCloud
import base64
from plotly.subplots import make_subplots


df = pd.read_csv('netflix_df.csv')
genres = pd.read_csv('genres.csv')
data = pd.read_csv('df_realise.csv')
df_0 = pd.read_csv('netflix_titles.csv')

df = df[df['release_year'] > 2007]
data = data[data['release_year'] > 2007]


filtered_genres = df.set_index('title').listed_in.str.split(', ', expand=True).stack().reset_index(level=1, drop=True)
df['listed_in'] = df['listed_in'].str.replace('[^\w\d\s]', '').str.strip()
#############################################################################################################


### filters Graph dataset
ddt = df[['title', 'type', 'country', 'date_added', 'release_year', 'director', 'cast', 'duration', 'listed_in']]
ddt = ddt.assign(var1=ddt['country'].str.split(',')).explode('var1').reset_index(drop=True)
ddt['var1'] = ddt['var1'].str.replace('[^\w\d\s]', '').str.strip()
ddt = ddt.drop(columns='country').rename(columns={'var1':'country'})
ddt = ddt[ddt['country'].str.contains('[a-z]')][ddt['country']!='other']

ddt = ddt.assign(var1=ddt['cast'].str.split(',')).explode('var1').reset_index(drop=True)
ddt['var1'] = ddt['var1'].str.replace('[^\w\d\s]', '').str.strip()
ddt = ddt.drop(columns='cast').rename(columns={'var1':'cast'})
ddt = ddt[ddt['cast'].str.contains('[a-z]')].drop_duplicates()

ddt = ddt.assign(var1=ddt['listed_in'].str.split(',')).explode('var1').reset_index(drop=True)
ddt['var1'] = ddt['var1'].str.replace('[^\w\d\s]', '').str.strip()
ddt = ddt.drop(columns='listed_in').rename(columns={'var1':'genres'})
ddt = ddt[ddt['genres'].str.contains('[a-z]')].drop_duplicates()


### Types Graph dataset
df_realise = df[['type', 'country', 'date_added', 'release_year']]
df_realise['count'] = 1
df_realise = df_realise.assign(var1=df_realise['country'].str.split(',')).explode('var1').reset_index(drop=True)
df_realise['var1'] = df_realise['var1'].str.replace('[^\w\d\s]', '').str.strip()
df_realise = df_realise.drop(columns='country').rename(columns={'var1': 'country'})
df_realise = df_realise[df_realise['country'].str.contains('[a-z]')][df_realise['country'] != 'other']
###

### Genres Graph dataset
sum_column = genres.sum(axis=0)
genres = sum_column.to_frame()
genres.reset_index(level=0, inplace=True)
genres = genres.rename(columns={"index": "Genres", 0: "Titles"})
genres = genres.sort_values(by=['Titles'], ascending=False)


############################WordCloud######################

# Custom colour map based on Netflix palette
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#221f1f', '#b20710'])

# Requirements for the dash core components

descriptions = list(df['description'])
text = (' ').join(descriptions)

wc = WordCloud(background_color='rgb(230, 230, 230)', width=480, height=360, colormap=cmap, mode="RGBA").generate(text)

wc_img = wc.to_image()
with BytesIO() as buffer:
    wc_img.save(buffer, 'png')
    img2 = base64.b64encode(buffer.getvalue()).decode()


# The app itself
types_options = [
    {'label': 'Movie', 'value': 'Movie'},
    {'label': 'TV Show', 'value': 'TV Show'}
]

all_title_options = {
    'Movie': ddt['title'][ddt['type'] == 'Movie'].unique(),
    'TV Show': ddt['title'][ddt['type'] == 'TV Show'].unique()}

title_options = [{'label': k, 'value': k} for k in all_title_options.keys()]

################




country_options = [
    dict(label='Country ' + country, value=country)
    for country in df_realise['country'].unique()]

genre_options = [
    dict(label='Genre ' + genre, value=genre)
    for genre in genres['Genres'].unique()]

year_slider = dcc.Slider(
        id='year_slider',
        min=2008,
        max=2021,
        marks={i: '{}'.format(i) for i in range(2008, 2022)},
        value=2021,
        step=1)

country_drop = dcc.Dropdown(
        id='country_drop',
        options=country_options,
        value=['Spain', 'Portugal','Canada'],
        multi=True)

genre_drop = dcc.Dropdown(
        id='genre_drop',
        options=genre_options,
        value=['Thrillers', 'Comedies'],
        multi=True)

type_radio = dcc.RadioItems(
        id='types-dropdown',
        options=title_options,
        value='Movie'
        )

title_dropdown = dcc.Dropdown(id='title-dropdown')

app = dash.Dash(__name__)

app.layout = html.Div([
    # title
    html.Div([
        html.Img(src='assets/netflix_image1.png')

    ], className='banner'),
    html.Br(),
    html.Br(),
    html.Br(),
    # Filters
    html.Div([
        html.Div([
            html.Label('Filter by Year'),
            year_slider,
            html.Br(),
            html.Label('Filter by Country'),
            country_drop,
            html.Br(),
            html.Label('Filter by Genre'),
            genre_drop,
            html.Br(),
            html.Label('Filter by Type'),
            type_radio

        ], style={'width': '40%'}, className='box'),

        html.Div([
            html.Label('Filter by Title'),
            title_dropdown,

            html.Label(id='year'),
            html.Label( id='duration'),
            html.Label(id='genre'),
            html.Label(id='director'),
            html.Label(id='cast')

        ], style={'width': '60%'}, className='box'),

    ], style={'display': 'flex'}),

    html.Br(),
    html.Br(),
    html.Br(),

    # Map
    html.Div([

        html.Div([
            html.Div([
                dcc.Graph(id='graph_example')
            ],style={'margin': '2px'}),

            html.Div([
                html.H2('Keywords in the Description of Movies and TV Shows'),
                html.Img(src="data:image/png;base64," + img2, style={'width': '80%',
                                                                     'margin-left': '50px', 'margin-top': '5px'})
            ]),
        ], style={'width': '35%'}),


        html.Div([
            dcc.Graph(id='choropleth')
        ], style={'width': '65%'})
    ], style={'display': 'flex'}, className='box'),

    html.Br(),
    html.Br(),

    html.Div([
        html.Div([
            dcc.Graph(id='graph_example1')
        ], style={'width': '34%'}, className='box1'),
        html.Div([
            dcc.Graph(id='graph_example2')
        ], style={'width': '30%'}, className='box2'),
        html.Div([
            dcc.Graph(id='graph_example4')
        ], style={'width': '27%'}, className='box3')
    ], style={'display': 'flex', 'margin-left': '20px', 'margin-top': '-25px', 'margin-bottom': '20px'}),

    html.Div([
        dcc.Graph(id='graph_example3')
    ], className='box4'),

    html.P([html.Strong('Authors:'),
            html.Br(),
            'Ana Sofia Silva nº m20200220',
            html.Br(),
            'José Francisco Alves nº m20200653',
            html.Br(),
            'Miguel Nunes nº m20200615',
            html.Br(),
            'Valentyna Rusinova nº m20200591'], className='names1')


])



@app.callback(
    Output('title-dropdown', 'options'),
    [Input('types-dropdown', 'value')]
)
def set_title_options(types):
    return [{'label': i, 'value': i} for i in all_title_options[types]]


@app.callback(
    Output('title-dropdown', 'value'),
    [Input('title-dropdown', 'options')]
)
def set_title_value(available_options):
    return available_options[0]['value']



@app.callback(
    [Output('graph_example', 'figure'), Output('graph_example1', 'figure'), Output('graph_example2', 'figure'),
     Output("choropleth", "figure"), Output('graph_example3', 'figure'), Output('graph_example4', 'figure')],
    [Input('year_slider', 'value'), Input('types-dropdown', 'value'), Input("country_drop", "value"),
     Input("genre_drop", "value")]
)
def update_graph(year, types, countries, genre):
    df_year = df[(df['release_year'] == year)]
    ##################################################Movie Graph#######################################################

    df1 = df_year.groupby(['release_year', 'type'])[['type']].count().rename(columns={'type': 'value'})
    df1["sum"] = df1.groupby(["release_year"])["value"].transform(sum)
    df1['ratio'] = ((df1['value']/df1['sum'])).round(2)
    df1 = df1.reset_index()

    nights = []

    Movie = dict(type='bar',
                 x=df1[df1['type']=='Movie']['ratio'],
                 y=df1[df1['type']=='Movie']['release_year'], text=df1['type'],
                 name='Movie', texttemplate="%{value}% Movie", textposition='inside', textfont_size=20,
                 marker_color='#b20710', insidetextanchor='middle', orientation='h')

    TV_Show = dict(type='bar',
                   x=df1[df1['type']=='TV Show']['ratio'],
                   y=df1[df1['type']=='TV Show']['release_year'], text=df1['type'], marker_color='#221f1f',
                   name='Show', texttemplate="%{value}% TV Show", textposition='inside', textfont_size=20,
                   insidetextanchor='middle', orientation='h')

    nights.append(Movie)
    nights.append(TV_Show)


    ex1_layout = dict(title=dict(text='<b>Movie or TV Show<b>', x=0.5, y=0.8,
                                 font=dict(family="Arial", size=16, color='black')
                                 ),
                      yaxis=dict(type='category', showgrid=False, showline=False, domain=[0.15, 1]),
                      xaxis=dict(title='Ratio %', showgrid=False, showline=False, showticklabels=True, zeroline=False),
                      font=dict(family="Arial", size=12, color='black'),
                      barnorm='percent', barmode='stack',
                      paper_bgcolor='#b3b3b3',
                      plot_bgcolor='#b3b3b3',
                      height=245, margin=dict(t=80, b=20))

    fig = go.Figure(data=nights, layout=ex1_layout)

    #################################################Types Graph########################################################

    df_type = df_realise[df_realise["type"] == types]

    df_type = df_type.loc[df_type[
        'country'].isin(countries)].groupby('country')['release_year', 'date_added'].mean().round().reset_index()

    ordered_df_rev = df_type.sort_values(by='release_year', ascending=True)

    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(x=ordered_df_rev['release_year'].values, y=ordered_df_rev['country'],
                              name='release_year', mode='markers',
                              marker=dict(color='#b20710', line=dict(color='#b20710'))
                              ))

    fig1.add_trace(go.Scatter(x=ordered_df_rev['date_added'], y=ordered_df_rev['country'],
                              name='date_added', mode='markers',
                              marker=dict(color='#221f1f', line=dict(color='#221f1f'))
                              ))


    def customLegend(fig, nameSwap):
        for i, dat in enumerate(fig.data):
            for elem in dat:
                if elem == 'name':
                        fig.data[i].name = nameSwap[fig.data[i].name]
        return (fig)

    fig1 = customLegend(fig=fig1, nameSwap={'release_year': "Release Year",
                                            'date_added': "Date Added to Netflix"})


    for i in range(ordered_df_rev.shape[0]):
        fig1.add_shape(
            type='line',
            x0=ordered_df_rev['release_year'].iloc[i], y0=ordered_df_rev['country'].iloc[i],
            x1=ordered_df_rev['date_added'].iloc[i], y1=ordered_df_rev['country'].iloc[i],
            line_color="grey"
        )

    fig1.update_traces(mode='markers', marker_line_width=1.6, marker_size=10)
    fig1.update_layout(title=dict(text=f'<b>Average Gap Between Release and Netflix Added {types}s<b>',x=0.5, y=0.95,
                                  font=dict(family="Arial", size=16, color='black')
                                  ),
                       font=dict(family="Arial", size=12, color='black'),
                       xaxis=dict(title="Year",showgrid=False,showline=False, showticklabels=True),
                       yaxis=dict(title="", showgrid=False, showline=False, showticklabels=True, zeroline=False),
                       paper_bgcolor='#b3b3b3',
                       plot_bgcolor='#b3b3b3'
                       )

    fig1.update_layout(xaxis=dict(tickmode="linear"))
    ################################################Country Graph#######################################################
    # Lets retrieve just the first country
    df['first_country'] = df['country'].apply(lambda x: x.split(",")[0])
    df['first_country'] = df['first_country'].str.replace('[^\w\d\s]', '').str.strip()

    df_country_order = df.loc[df['first_country'].isin(countries)]
    country_order = df_country_order['first_country'].value_counts().index
    data_q2q3 = df[['type', 'first_country']].groupby('first_country')['type'].value_counts().unstack().loc[
        country_order]
    data_q2q3['sum'] = data_q2q3.sum(axis=1)
    data_q2q3_ratio = (data_q2q3.T / data_q2q3['sum']).T[['Movie', 'TV Show']].sort_values(by='Movie', ascending=False)[
                      ::-1]


    fig2 = make_subplots(
        rows=1, cols=1,
    )

    fig2.add_trace(go.Bar(
        x=data_q2q3_ratio['Movie'],
        y=data_q2q3_ratio.index,
        text=data_q2q3_ratio['Movie'],
        texttemplate="%{text:%}",
        textposition='inside',
        textfont_size=10,
        marker=dict(color='#b20710', line=dict(color='#f5f5f1', width=1)),
        name='Movie',
        orientation='h'
    ), 1, 1)

    fig2.add_trace(go.Bar(
        x=data_q2q3_ratio['TV Show'],
        y=data_q2q3_ratio.index,
        text=data_q2q3_ratio['TV Show'],
        texttemplate="%{text:%}",
        textposition='inside',
        textfont_size=10,
        marker=dict(color='#221f1f', line=dict(color='#f5f5f1', width=1)),
        name='TV Show',
        orientation='h'
    ), 1, 1)

    fig2.update_layout(
        title=dict(text='<b>Countries Movie & TV Show Distribution<b>', x=0.5, y=0.95,
                   font=dict(family="Arial", size=16, color='black')
                   ),
        xaxis=dict(title='Ratio', showgrid=False, showline=False, showticklabels=True, zeroline=False,
                   #domain=[0.15, 1],
        ),
        yaxis=dict( type='category', showgrid=False, showline=False, showticklabels=True),
        font=dict(family="Arial", size=12, color='black'),
        barmode='stack', barnorm='percent',
        paper_bgcolor='#b3b3b3',
        plot_bgcolor='#b3b3b3',
        #margin=dict(l=120, r=10, t=140, b=80),
    )

    ####################################################Map Graph#######################################################
    df_countries = data.loc[data['date_added'] <= year]
    df_countries = df_countries.groupby(['country'])[['country', 'count']].sum().sort_values(by='count', ascending=False
                                                                                             ).reset_index()
    z = (df_countries['count'])
    vmin= 0
    vmax=2800
    data_choropleth = dict(type='choropleth',
                           locations=df_countries['country'],
                           locationmode='country names',
                           z=z,
                           zmin=vmin,
                           zmax=vmax,
                           text=df_countries['country'],
                           colorscale=['#fdcfce','#FB9996', '#ff4d4d', '#ff0000',
                                                  '#b30000', '#660000','#221f1f'],


                           )

    layout_choropleth = dict(geo=dict(scope='world',bgcolor= '#b3b3b3',
                                      projection=dict(type='equirectangular'),
                                      # showland=True,   # default = True
                                      landcolor='#b3b3b3',
                                      lakecolor='#b3b3b3',
                                      showocean=True,  # default = False
                                      oceancolor='#b3b3b3'
                                      #landcolor='white',
                                      #lakecolor='grey',
                                      #showocean=True,  # default = False
                                      #oceancolor='azure'
                                      ),

                             title=dict(text=f'<b>World Netflix Released Content (Total until {year})<b>',
                                        x=.5  # Title relative position according to the xaxis, range (0,1)
                                        )
                             )
    map = go.Figure(data=data_choropleth, layout=layout_choropleth)
    map.update_layout(
        title=dict(x=0.5, y=0.93,font=dict(family="Arial", size=25, color='black')
                   ),
        xaxis=dict(showgrid=False, showline=False, showticklabels=True, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, showticklabels=True),
        font=dict(family="Arial", size=12, color='black'),
        paper_bgcolor='#b3b3b3',
        plot_bgcolor='#b3b3b3',
        height=760
    )


    #################################################Genere Graph#######################################################
    genres1 = genres.loc[genres['Genres'].isin(genre)]

    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=genres1['Titles'],
            y=genres1['Genres'][:10],
            orientation='h',
            marker=dict(color='#b20710', line=dict(color='#f5f5f1', width=1))
        )
    )
    fig3.update_layout(
        title=dict(text='<b>Genres on Netflix<b>', x=0.5, y=0.90,
                   font=dict(family="Arial", size=20, color='black')
                   ),
        xaxis=dict(title='Quantity of Movies and TV Show', showgrid=False, showline=False, showticklabels=True, zeroline=False),
        yaxis=dict(autorange="reversed", showgrid=False, showline=False),
        font=dict(family="Arial", size=12, color='black'),
        paper_bgcolor='#b3b3b3',
        plot_bgcolor='#b3b3b3'
    )

    ##################################################Month Graph#######################################################
    df_1 = df_0[~df_0['date_added'].isna()]
    df_1['year_added'] = df_1['date_added'].str.split(', ').str[-1].astype(int)

    df_1 = df_1.loc[df_1['year_added'] <= year]

    df_1["date_added"] = pd.to_datetime(df_1['date_added'])
    df_1['month_name_added'] = df_1['date_added'].dt.month_name()
    df_netflix_date = df_1.sort_values(by="date_added")
    data_sub2 = df_netflix_date.groupby('type')['month_name_added'].value_counts().unstack().fillna(0).loc[
        ['TV Show', 'Movie']].cumsum(axis=0).T

    data_sub2['Value'] = data_sub2['Movie'] + data_sub2['TV Show']
    data_sub2 = data_sub2.reset_index()
    data_sub2 = data_sub2.sort_values(by="month_name_added")

    data_sub2['colors'] = 0
    data_sub2 = data_sub2.sort_values(by='Value', ascending=False)

    fig4 = go.Figure(data=[go.Pie(labels=data_sub2['month_name_added'],
                                  values=data_sub2['Value'],
                                  textinfo='label,value',
                                  pull=[0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                           ])

    fig4.update(layout_title_text=f'<b>Best Month For New Content (until {year})<b>',
                layout_showlegend=False)

    fig4.update_traces(marker=dict(
        colors=['#f5f5f1', '#b20710', '#b20710', '#b20710', '#b20710', '#b20710', '#b20710', '#b20710', '#b20710',
                '#b20710', '#b20710', '#b20710'],
        line=dict(color='#000000', width=2)), textfont_size=10)

    fig4.update_layout(
        title=dict(x=0.5, y=0.95,font=dict(family="Arial", size=16, color='black')
                   ),
        xaxis=dict(showgrid=False, showline=False, showticklabels=True, zeroline=False),
        yaxis=dict(showgrid=False, showline=False, showticklabels=True),
        font=dict(family="Arial", size=12, color='black'),
        paper_bgcolor='#b3b3b3',
        plot_bgcolor='#b3b3b3'
    )

    return fig, fig1, fig2, map, fig3, fig4





@app.callback(
    [
        Output("year", "children"),
        Output("duration", "children"),
        Output("cast", "children"),
        Output("director", "children"),
        Output("genre", "children")
    ],
    [
        Input("title-dropdown", "value"),
    ]
)
def indicator(title):

    ################ data #################
    df_loc = ddt[ddt['title'] == title]

    release_year = df_loc['release_year'].unique()

    duration = df_loc['duration'].drop_duplicates()

    cast = df_loc['cast'].drop_duplicates()
    cast = (', '.join(cast))

    director = df_loc['director'].drop_duplicates()
    director = (', '.join(director))

    genre = df_loc['genres'].drop_duplicates().tolist()
    genre = ', '.join(genre)

############################################

    year = html.P(children=[

        html.Strong('Year: '),
        html.Span(str(release_year).replace('[', '').replace(']', ''))

    ])

    duration = html.P(children=[

        html.Strong('Duration: '),
        html.Span(duration)

    ])

    cast = html.P(children=[

        html.Strong('Cast: '),
        html.Span(str(cast))

    ])


    director = html.P(children=[

        html.Strong('Director: '),
        html.Span(str(director))

    ])


    Genre = html.P(children=[

        html.Strong('Genre: '),
        html.Span(str(genre).replace('  ', ', '))

    ])

    return year, duration, cast, director, Genre



if __name__ == '__main__':
    app.run_server(debug=True)