import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import folium
import plotly.express as px
from plotnine import *
import json

df = pd.read_csv('watches.csv')
st.set_page_config(
    page_title="Comparador de Relojes Rolex",
    layout="wide",
)

st.title("Comparador de Relojes Rolex ⌚")

#########################
## Menu lateral
#########################

sidebar_prueba2 = st.sidebar.header("Bienvenido a nuestra web")
with st.sidebar:
    selected = option_menu("Menu", ["Modelos", 'Países', 'Screener'], 
        icons=['Currency dollar', 'Globe Americas', 'Search'], menu_icon="cast", default_index=1)
    selected

sidebar_prueba1 = st.sidebar.write("About us:")
sidebar_prueba2 = st.sidebar.write("Este comparador de Rolex permite visualizar datos sobre los diferentes modelos de rolex del mercado. Permite analizar los relojes segun su precio o el pais donde se encuentran. Por último, tenemos un screener de busqueda de relojes que cumplan con los requisitos que exijamos") 

    
if selected == "Modelos":
    
    # Histograma de precios
    
    selected_model = st.selectbox('Selecciona un modelo', df['models'].unique())
    model_df = df[df['models'] == selected_model]
    st.write('Has seleccionado:', selected_model)
    st.subheader("Histograma de precios")
    hist_values = model_df['prices'].tolist()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    sns.distplot(hist_values, bins=100, kde=False, color='blue', hist_kws={'alpha': 0.6})
    st.pyplot(fig)
    
    #MAPA
    
    st.subheader(f"Paises en los que puedes encontrar el modelo {selected_model}")
    
    with st.container():
        with open('countries_geo.json', 'r', encoding = 'utf-8') as file:
            geo_world = json.load(file)
            
            map_world = folium.Map()
            folium.GeoJson(geo_world).add_to(map_world)
            
    def color_countries(model):

        for feature in geo_world['features']:
            country = feature['properties']['name']

            if country in model_df['name'].tolist():
                country_data = model_df[model_df['name'] == country]
                min_price = country_data['prices'].min()
                max_price = country_data['prices'].max()
                avg_price = round(country_data['prices'].mean(),)
                counts = country_data['models'].count()
                
                folium.GeoJson(feature, style_function=lambda feature: {
                    'fillColor': 'red',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7,
                }).add_to(map_world).add_child(
                    folium.Popup(
                        
                        f"{selected_model}<br>"
                        f"Min: {min_price} €<br>"
                        f"Max: {max_price} €<br>"
                        f"Medio: {avg_price} €",
                        max_width=100
                    )
                )
    
    color_countries(selected_model)
    st.write(map_world)
    
    ## Scatter plot
    st.subheader(f"Distribución del precio del modelo {selected_model} segun el pais")
    sns.swarmplot(x="name", y="prices", data=model_df)
    plt.xticks(rotation=90)
    st.pyplot()
    
if selected == "Países":
    
    # Bar Plot
    st.subheader("Distribución de Relojes Vendidos por País y Modelo")
    selected_country = st.selectbox("Selecciona un país", df['name'].unique())
    df_country = df[df['name'] == selected_country]
    st.write("Número de modelos de relojes vendidos en", selected_country, ":", df_country.shape[0])
    df_grouped = df_country.groupby(['models'])['prices'].agg(['count', 'mean']).reset_index()
    fig = px.bar(df_grouped, x='models', y='count', hover_data=['count', 'mean'])
    
    fig.update_layout(xaxis_title = "Modelo", yaxis_title = "Total")
    st.plotly_chart(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    #Grafico horizotal
    st.subheader(f"Top 5 modelos mas vendidos de {selected_country}")
    modelos_agrupados = df_country.groupby("models").count()["prices"]
    modelos_ordenados = modelos_agrupados.sort_values(ascending=False)
    modelos_top_5 = modelos_ordenados[:5]
    plt.barh(modelos_top_5.index, modelos_top_5.values)
    plt.xlabel("Cantidad de ventas")
    plt.ylabel("Modelo de reloj")
    st.pyplot()
        
    #Grafico stacked
    st.subheader("Precio medio de los modelos más vendidos en los países seleccionados")
    paises_seleccionados = st.multiselect("Selecciona los países que deseas incluir:", df["name"].unique(), key="paises_seleccionados")
    st.write("Compara los modelos mas populares de cada pais y su precio medio")
    df_selected = df[df["name"].isin(paises_seleccionados)]
    df_grouped = df_selected.groupby(["name", "models"]).mean().reset_index()
    df_grouped = df_grouped.sort_values(by=["name", "prices"], ascending=[True, False])
    fig = px.bar(df_grouped, x="name", y="prices", color="models", barmode="stack")
    fig.update_layout(yaxis=dict(visible=False))
    plt.xlabel("precio")
    plt.ylabel("Paises")
    st.plotly_chart(fig)

    ###MAPA DE CALOR
    st.subheader("Mapa de calor de los modelos mas comprados en cada pais")
    df_grouped = df.groupby(['models', 'name']).size().reset_index(name='counts')
    fig = px.imshow(df_grouped.pivot(index='models', columns='name', values='counts'), color_continuous_scale = "viridis")
    fig.update_coloraxes(cmin=0, cmax=300)
    fig.update_layout(xaxis_title = "Pais", yaxis_title = "Modelo")
    st.write(fig)


if selected == 'Screener':
    
    st.subheader("Encuentra relojes al precio que buscas")

    countries = df['name'].unique()
    min_price = df['prices'].min()
    max_price = df['prices'].max()
    selected_price_range = st.slider("Rango de precios:", min_price, max_price, (min_price, max_price))
    selected_country = st.selectbox("Selecciona un país:", df['name'].unique())    
    filtered_df = df[(df['name'] == selected_country) & (df['prices'] >= selected_price_range[0]) & (df['prices'] <= selected_price_range[1])]
    
    sort_order = st.radio("Ordenar por:", ('Ascendente', 'Descendente'))
    if sort_order == 'Ascendente':
        filtered_df = filtered_df.sort_values('prices')
    else:
        filtered_df = filtered_df.sort_values('prices', ascending=False)
    
    st.dataframe(filtered_df)
    
    
    
    
    
    #########                                                       ########
    ######### Marcar la opcion de reloj mas caro o reloj mas barato ########    
    ########################################################################
