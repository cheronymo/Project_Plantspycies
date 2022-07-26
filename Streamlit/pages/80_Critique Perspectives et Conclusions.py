import streamlit as st
from PIL import Image
from tensorflow import keras  # interface for TensorFlo
from tensorflow.keras.models import Sequential  # linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # convolution and pooling
from tensorflow.keras.layers import Dense, Flatten, Dropout

from tensorflow.python.keras.models import load_model  # to load a saved model

st.markdown('''  ##  Conclusion ''')

liste_choix = ["Augmenter base de données", "Génération d'images", "Batch size ", "Modèle avec 2 sorties",
               "Interprétabilité"]
choix = st.sidebar.radio(" ", options=liste_choix)

if choix == liste_choix[1]:
    st.markdown(''' ### Génération d'images ''')

    option = st.selectbox(
        '',
        ('Tavelure du pommier', 'Saine', 'Rouille du pommier ', 'Black rot'))

    col1, col2 = st.columns(2)

    with col1:
        from keras.preprocessing import image
        from keras.preprocessing.image import ImageDataGenerator
        import numpy as np
        import matplotlib.pyplot as plt

        image_path = 'images/' + option + '.JPG'

        # Loads image in from the set image path
        img = keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
        img_tensor = keras.preprocessing.image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        # Allows us to properly visualize our image by rescaling values in array
        img_tensor /= 255.
        # Plots image
        fig, ax = plt.subplots()
        im = ax.imshow(img_tensor[0])
        im = ax.axis('off')
        st.pyplot(fig)

        st.markdown(''' Originale ''')

    with col2:
        # Loads in image path
        img = keras.preprocessing.image.load_img(image_path, target_size=(500, 500))
        img_tensor = keras.preprocessing.image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        # Uses ImageDataGenerator to flip the images
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                     rotation_range=40,
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     fill_mode='nearest'
                                     )
        # Creates our batch of one image
        pic = datagen.flow(img_tensor, batch_size=1)

        fig, ax = plt.subplots()
        batch = pic.next()
        image_ = batch[0].astype('uint8')
        im = ax.imshow(image_)
        im = ax.axis('off')
        st.pyplot(fig)

        st.markdown(''' Générée ''')

if choix == liste_choix[3]:
    st.markdown(''' ### Modèle avec 2 sorties séparées ''')

    st.markdown(''' L'espèce de plantes : 14 classes ''')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        image = Image.open('images/Saine.JPG')
        st.image(image, caption="Pomme", width=200)
        image = Image.open('images/Orange.JPG')
        st.image(image, caption="Orange", width=200)
        image = Image.open('images/Squach.JPG')
        st.image(image, caption="Courge", width=200)
        image = Image.open('images/Poivron.JPG')
        st.image(image, caption="Poivron", width=200)

    with col2:
        image = Image.open('images/Blueberry.JPG')
        st.image(image, caption="Myrtille", width=200)
        image = Image.open('images/Peach.JPG')
        st.image(image, caption="Pêche", width=200)
        image = Image.open('images/Strawberry.JPG')
        st.image(image, caption="Fraise", width=200)
        image = Image.open('images/Framboise.JPG')
        st.image(image, caption="Framboise", width=200)
    with col3:
        image = Image.open('images/Cherry.JPG')
        st.image(image, caption="Cerise", width=200)
        image = Image.open('images/Tomato.JPG')
        st.image(image, caption="Tomate", width=200)
        image = Image.open('images/Potato.JPG')
        st.image(image, caption="Patate", width=200)
    with col4:
        image = Image.open('images/Grape.JPG')
        st.image(image, caption="Raisin", width=200)
        image = Image.open('images/Soybean.JPG')
        st.image(image, caption="Soja", width=200)
        image = Image.open('images/Corn.JPG')
        st.image(image, caption="Maïs", width=200)

    st.markdown(''' Maladie: Oui / Non  ''')

    col1, col2 = st.columns(2)
    with col1:
        image = Image.open('images/Saine.JPG')
        st.image(image, caption="Saine", width=200)
    with col2:
        image = Image.open('images/Tavelure du pommier.JPG')
        st.image(image, caption="Malade", width=200)

if choix == liste_choix[0]:
    st.markdown(''' ### Compilation différentes bases données ''')

    st.markdown(''' #### Autre bases de données ''')

    image = Image.open('images/Table synthèse databases.png')
    st.image(image)

    st.markdown(''' #### Photos avec fond naturel ''')
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open('images/generateur_1.JPG')
        st.image(image)

    with col2:
        image = Image.open('images/Feuille_pommier_nature.jpeg')
        st.image(image)

if choix == liste_choix[2]:
    st.markdown(''' ### Batch size  ''')

    image = Image.open('images/Tallornot.png')
    st.image(image, width=400)
        
if choix == liste_choix[4]:
    st.markdown(''' ### Interprétabilité des résultats (GradCam) ''')
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        image = Image.open('images/grad_cam_1.png')
        st.image(image, width=200, caption='Dernière couche de convolution')

    with col2:
        image = Image.open('images/fleche.jpeg')
        st.image(image, width=100)

    with col3:
        image = Image.open('images/grad_cam_2.jpeg')
        st.image(image, width=250, caption='Image interpretable')

    st.markdown(''' ### Grad cam sur notre modèle''')
    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        image = Image.open('images/inter_1.png')
        st.image(image, width=250, caption="Feuille d'origine")

    with col2:
        image = Image.open('images/fleche.jpeg')
        st.image(image, width=100)

    with col3:
        image = Image.open('images/inter_2.jpeg')
        st.image(image, width=250, caption='Feuille interpretée')



