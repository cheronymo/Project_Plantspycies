import streamlit as st
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.markdown(''' ##  Introduction ''')


liste_choix = ["Deep Learning", "Sécurité Alimentaire", "Plantes et virus"]
choix = st.sidebar.radio(" ", options=liste_choix)

if choix == liste_choix[0]:

    st.markdown(''' ### Deep learning 
                    ''')


    #st.markdown(''' La technologie de la reconnaissance d'images est apparue dans les années 1940,
    #mais elle était limitée par l'environnement technique et les équipements matériels de l'époque,
    #de sorte qu'elle n'a pas connu un long développement. Jusqu'aux années 1990,
    #les réseaux de neurones artificiels ont été combinés avec des machines à vecteurs de support pour
    #soutenir le développement des techniques de reconnaissance d'images, ce qui a permis une large utilisation,
    #par exemple, dans la reconnaissance des plaques d'immatriculation, la reconnaissance des visages,
    #la détection des objets, etc. Récemment, la révolution industrielle a fait appel à la vision par
    #ordinateur pour son travail. Les industries de l'automatisation, la robotique,
    #le domaine médical et les secteurs de la surveillance font un usage intensif de
    #l'apprentissage profond. ''')

    image = Image.open('images/deep_learning.jpg')
    st.image(image)

if choix == liste_choix[1]:
    st.markdown(''' ### Securité alimentaire 
                                                    ''')
    #st.markdown(''' Récemment, l'agriculture et la sécurité alimentaire sont devenues un nouveau
    #domaine d'application de la reconnaissance d'images par apprentissage profond. En effet, les
    #technologies modernes ont donné à la société humaine la capacité de produire suffisamment de
    #nourriture pour répondre à la demande de plus de 7 milliards de personnes. Cependant, la sécurité
    #alimentaire reste menacée par un certain nombre de facteurs, notamment le changement climatique
    #, le déclin des pollinisateurs, les maladies des plantes, et autres.
    #Les maladies des plantes constituent non seulement une menace pour la sécurité alimentaire
    #à l'échelle mondiale, mais peuvent également avoir des conséquences désastreuses pour les
    #petits exploitants agricoles dont les moyens de subsistance dépendent de cultures saines.  ''')


    #plot

    age = st.slider('Date', 1000, 2020, (1000, 1700))
    pop_mondial = pd.read_csv('images/population_mondiale.csv', sep=";")

    import plotly.express as px
    fig = px.line(pop_mondial, x="Date", y="Population Mondiale",
                  height = 500,
                  width = 1000,
                  range_x=[age[0],age[1]])
    st.write(fig)


if choix == liste_choix[2]:
    st.markdown(''' ### Des plantes et des virus 
                                                        ''')

    st.markdown(''' #### Des feuilles uniques 
                                                        ''')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        image = Image.open('images/Saine.JPG')
        st.image(image, caption="Pomme", width=150)
        image = Image.open('images/Orange.JPG')
        st.image(image, caption="Orange", width=150)
        image = Image.open('images/Squach.JPG')
        st.image(image, caption="Courge", width=150)
        image = Image.open('images/Poivron.JPG')
        st.image(image, caption="Poivron", width=150)

    with col2:
        image = Image.open('images/Blueberry.JPG')
        st.image(image, caption="Myrtille", width=150)
        image = Image.open('images/Peach.JPG')
        st.image(image, caption="Pêche", width=150)
        image = Image.open('images/Strawberry.JPG')
        st.image(image, caption="Fraise", width=150)
        image = Image.open('images/Framboise.JPG')
        st.image(image, caption="Framboise", width=150)
    with col3:
        image = Image.open('images/Cherry.JPG')
        st.image(image, caption="Cerise", width=150)
        image = Image.open('images/Tomato.JPG')
        st.image(image, caption="Tomate", width=150)
        image = Image.open('images/Potato.JPG')
        st.image(image, caption="Patate", width=150)
    with col4:
        image = Image.open('images/Grape.JPG')
        st.image(image, caption="Raisin", width=150)
        image = Image.open('images/Soybean.JPG')
        st.image(image, caption="Soja", width=150)
        image = Image.open('images/Corn.JPG')
        st.image(image, caption="Maïs", width=150)
    st.markdown(''' #### Des virus uniques 
                                                        ''')
    option = st.selectbox(
            'Les différente maladies du pommier : ',
            ('Saine', 'Tavelure du pommier',  'Rouille du pommier ', 'Black rot'))
    image = Image.open('images/'+option+'.JPG')
    st.image(image, caption=option)

