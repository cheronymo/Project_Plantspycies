import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import time
from tensorflow.keras.models import Sequential  # linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # convolution and pooling
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3       #model


st.set_page_config(layout="wide")

#path = '/Users/gio/Desktop/DataScientest/Project/'
#path = 'images/'

font_bullet = 19
font_bullet_2 = 18
font_archi = 22
color_models = 'Darkgreen'
color_intro = 'Darkgreen'
font_intros = 28
font_intro = 28
spaces =  4
intro_col1_w = 5
intro_col2_w = 80
n_class=38
img_height = 256
img_width = 256

label_overview ='Architecture'
label_VGG16 = 'VGG16'
label_ResNet50 = 'ResNet50'
label_InceptionV3 = 'InceptionV3'
label_three_models = 'Trois Modèles pré-entraînés'

## Sidebar
pages = [label_overview, label_three_models ,label_VGG16, label_ResNet50, label_InceptionV3]
page = st.sidebar.radio("Menu", options = pages)


    
     


if page == label_three_models:
    #st.markdown('# ' + str(page))
    ### Overview ###


    
    bullet = f" &nbsp; VGG16 &nbsp; &nbsp; &nbsp; &nbsp; ----> &nbsp; &nbsp; Freezed, Unfreezed 4 layers  "
    text = f'<p style="font-family:sans-serif; color:  {color_intro} ; font-size: {font_intro}px;">{bullet}<p>'
    st.write(text, unsafe_allow_html=True)

    col1, col2 = st.columns([intro_col1_w,intro_col2_w])
    with col2:
        image_vgg = Image.open('images/VGG16.png')
        st.image(image_vgg, width = 500)

    st.markdown('#') 
    bullet = f" &nbsp; ResNet50 &nbsp; &nbsp;&nbsp; ----> &nbsp; &nbsp; Freezed, Unfreezed 10 layers  "
    text = f'<p style="font-family:sans-serif; color:  {color_intro} ; font-size: {font_intro}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)

    col1, col2 = st.columns([intro_col1_w,intro_col2_w])
    with col2:
        image_vgg = Image.open('images/ResNet50.png')
        st.image(image_vgg, width = 700)
   
    st.markdown('#') 
    bullet = f" &nbsp; InceptionV3 &nbsp; ----> &nbsp; &nbsp; Freezed, Unfreezed 12 et 23 layer"
    text = f'<p style="font-family:sans-serif; color:  {color_intro} ; font-size: {font_intro}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)

    col1, col2 = st.columns([intro_col1_w,intro_col2_w])
    with col2:
        image_vgg = Image.open('images/inceptionV3.png')
        st.image(image_vgg, width = 500)


        
#########################
########## VGG ##########
#########################
if page == label_VGG16:
    st.markdown('# ' + str(page))
    image_vgg = Image.open('images/VGG16.png')
    st.image(image_vgg, width = 1000)
    #st.write("#")
    st.write("")



    bullet = f"-) L'idée clé de VGG (2013), est l'utilisation de plusieurs filtres de convolution très petits, \
            plutôt que des filtres plus grands comme ceux utilisés à l'époque"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)
    st.write("")

    bullet = f"-) VGG utilise une pile de couches de convolution avec des filtres 3 X 3, \
            qui est la plus petite taille pour capturer la notion de gauche/droite, haut/bas, centre"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)
    st.write("")

    bullet = f"-) L'utilisation de filtres 3X3 n’augmente pas la complexité du modèle et l'utilisation de plus couches d'activation non linéaires augmente la capacité à converger plus rapidement"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)
    st.write("")

    bullet = f"-) L'utilisation systématique de convolutions 3 x 3 dans l'ensemble du réseau rende VGG16 très simple, \
                élégant et facile à utiliser"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)
    st.write("")

    bullet = f"-) VGG a obtenu la première et la deuxième place au ImageNet Challenge 2014, \
                            respectivement dans les catégories localisation et classification"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)


    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')


    base_model = VGG16(weights='imagenet', include_top=False,
                    classes= n_class, input_shape = (img_height,img_width,3))# Freeze the layers of base_model
    st.header("VGG16 Model Summary")
    base_model.summary(print_fn=lambda x: st.text(x))




#########################
######## ResNet #########
#########################
if page == label_ResNet50: 
    st.markdown('# ' + str(page))
    test_size_img = 1200
    image_vgg = Image.open('images/ResNet50.png')
    st.image(image_vgg, width = test_size_img)
    
    st.write("")

    bullet = f"-) L'idée clé de ResNet (2015) est l'introduction des “Shortcut Connections” et  du  “Residual Learning” :"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)


    # bullet = f" &nbsp;  &nbsp;  -  &nbsp; Un Shortcut Connection est une connexion directe qui permet de sauter certaines couches du modèle"
    # text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    # st.write(text, unsafe_allow_html=True)

    # bullet = f" &nbsp;  &nbsp;  -  &nbsp; Le Residual Learning se base sur l'idée que, étant donné une entrée x, si un bloc de couches peut approximer H(x) alors peut aussi approximer H(x) - x"
    # text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    # st.write(text, unsafe_allow_html=True)

    # bullet = f" &nbsp;  &nbsp;  -  &nbsp; Le Residual Learning est appliqué avec des Shortcut Connection d'un bloc au suivant"
    # text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    # st.write(text, unsafe_allow_html=True)
    

    col1, col2, col3 = st.columns([2,1,80])


    with col2:

        bullet = f"-"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)
    
        bullet = f"-"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

        bullet = f""
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        #st.write(text, unsafe_allow_html=True)
        

        bullet = f""
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        #st.write(text, unsafe_allow_html=True)

        bullet = f"-"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

    with col3:

        bullet = f" Un Shortcut Connection est une connexion directe qui permet de sauter certaines couches du modèle"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

        bullet = f" Le Residual Learning se base sur l'idée que, étant donné une entrée x, si un bloc de couches peut approximer H(x) alors peut aussi approximer H(x) - x"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

        bullet = f" Le Residual Learning est appliqué avec des Shortcut Connection d'un bloc au suivant"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)


    col1, col2 = st.columns([7,2])

    with col1:    

        st.write("")      
        bullet = f"-) Pour améliorer  le temps d'apprentissage ResNet est structuré en blocs dits “bottleneck” building block "
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)
        st.write("")

        bullet = f"-) ResNet est devenu le lauréat de l'ILSVRC 2015 en classification, détection et localisation d'images, ainsi que le lauréat de MS COCO 2015 en détection et segmentation"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)
    with col2:        
        image= Image.open('images/bottleneck.png')
        st.image(image, width = 220)

    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')

    base_model =ResNet50 (weights='imagenet', include_top=False,
                        classes= n_class, input_shape = (img_height,img_width,3))

    st.header("ResNet50 Model Summary")
    base_model.summary(print_fn=lambda x: st.text(x))

    
#########################
####### Inception #######
#########################

if page == label_InceptionV3 :
    st.markdown('# ' + str(page))
    image_vgg = Image.open('images/inceptionV3.png')
    st.image(image_vgg, width = 900)

    bullet = f"-) L'idée clé d'Inception est la recherche d'une architecture plus efficace en termes de calcul, \
            à la fois en termes de nombre de paramètres générés par le réseau et de coût induit (mémoire et autres ressources)"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)

    bullet = f"-) InceptionV3 (2015) s'attache principalement à consommer moins de puissance de calcul en modifiant les architectures Inception précédentes"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)

    bullet = f"-) La structure d'InceptionV3 est basée sur les principes suivants :"
    text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet}px;">{bullet}</p>'
    st.write(text, unsafe_allow_html=True)

    
    col1, col2, col3 = st.columns([2,1,80])



    with col3:

        bullet = f" - &nbsp Factorized Convolutions : technique qui, en réduisant le nombre de paramètres, permet d'améliorer l'efficience du réseau"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

        bullet = f" - &nbsp Smaller convolutions : remplacement des grandes convolutions par des convolutions plus petites pour permettre un apprentissage plus rapide"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

        #bullet = f" - &nbsp Asymmetric convolutions : reduction du nombre de paramètres en remplaçant les blocs de convolution plus grands par des plus petits"
        #text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        #st.write(text, unsafe_allow_html=True)

        bullet = f" - &nbsp Auxiliary classifier : petit classificateur CNN qui agit comme un régulateur et facilite la convergence"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

        bullet = f" - &nbsp Grid size reduction : ici implémenté avec une technique spécifique plus efficace que les opérations de pooling habituelles"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_bullet_2}px;">{bullet}</p>'
        st.write(text, unsafe_allow_html=True)

    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')
    st.markdown('#')

    base_model =InceptionV3 (weights='imagenet', include_top=False,
                        classes= n_class, input_shape = (img_height,img_width,3))

    st.header("InceptionV3 Model Summary")
    base_model.summary(print_fn=lambda x: st.text(x))
    


#########################
####### Overview# #######
#########################    
if page == label_overview :
    #st.markdown('# ' + str(page))
    #st.header(str(page))
    ### Summary ###
    col1, col2 = st.columns([5,6])
    with col2:
        base_model = VGG16(weights='imagenet', include_top=False,
                    classes= n_class, input_shape = (img_height,img_width,3))# Freeze the layers of base_model
        for layer in base_model.layers: 
            layer.trainable = False

        model_VGG16 = Sequential()

        model_VGG16.add(base_model)

        model_VGG16.add(GlobalAveragePooling2D()) 
        model_VGG16.add(Dense(1024,activation='relu'))
        model_VGG16.add(Dropout(rate=0.2))
        model_VGG16.add(Dense(512, activation='relu'))
        model_VGG16.add(Dropout(rate=0.2))
        model_VGG16.add(Dense(n_class, activation='softmax'))

        #st.header("Model Summary (Freezed)")
        msg= f"Model Summary (Freezed)"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_archi}px;"><b><u>{msg}</p>'
        st.write(text, unsafe_allow_html=True)
        model_VGG16.summary(print_fn=lambda x: st.text(x))

    with col1:
        #st.header("Instance the model")
        msg = "Instance the model"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_archi}px;"><b><u>{msg}</p>'
        st.write(text, unsafe_allow_html=True)
 
        code_str = '''
n_class=38
img_height = 256
img_width = 256

base_model = VGG16(weights='imagenet', include_top=False,
            classes= n_class, input_shape = (img_height,img_width,3))

for layer in base_model.layers: 
    layer.trainable = False

model = Sequential()

model.add(base_model)

model.add(GlobalAveragePooling2D()) 
model.add(Dense(1024,activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(n_class, activation='softmax'))
        '''
        st.code(code_str)

        
        #st.header("Unfreezing")
        msg = "Unfreezing"
        text = f'<p style="font-family:sans-serif; color: {color_models} ; font-size: {font_archi}px;"><b><u>{msg}</p>'
        st.write(text, unsafe_allow_html=True)

        code_str = '''
for layer in model.layers:
    if "Functional" == layer.__class__.__name__:  
        for _layer in layer.layers[-4:]:
              _layer.trainable = True
        '''
        st.code(code_str)
           

