
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf # deep neural networks
from tensorflow import keras # interface for TensorFlow
from tensorflow.keras import layers # tensor-in tensor-out computation functions
import sklearn
from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import PIL
from PIL import Image
import os
import random 

from tensorflow.keras.layers.experimental.preprocessing import Rescaling 







from tensorflow.keras.applications.inception_v3 import InceptionV3       #model

#from skimage.transform import resize


liste_choix = ["Prediction", "Drag and Drop"]
choix = st.sidebar.radio(" ", options=liste_choix)

if choix == liste_choix[0]:

    #df_classes
    df_classes = pd.read_excel('classes.xlsx', sheet_name = 'Sheet1', index_col = 0)

    num_images= 12

    def show_class(images, class_path, col1,col2,col3, width= 130, num_images_col1 = 3, num_images_col2 = 7):
        for n, image_name in enumerate(images):
            if n <= num_images_col1:
                with col1:
                    img = plt.imread(class_path+image_name)
                    img_to_show =img
                    img = img.reshape(1,256,256,3)
                    img = preprocess_input(img/1.)
                    text = 'Classe predicte: '+str(np.argmax(model.predict(img)))
                    st.image(img_to_show, width=width, caption=text) # 130 for hanving all classes in one screen
                    #st.write('The predicted leaf class is: ', np.argmax(model.predict(img)))
            elif num_images_col1 < n <= num_images_col2:
                with col2:
                    img = plt.imread(class_path+image_name)
                    img_to_show =img
                    img = img.reshape(1,256,256,3)
                    img = preprocess_input(img/1.)
                    text = 'Classe predicte: '+str(np.argmax(model.predict(img)))
                    st.image(img_to_show, width=width, caption=text) # 130 for hanving all classes in one screen
                    #st.write('The predicted leaf class is: ', np.argmax(model.predict(img)))
            elif num_images_col2 < n:
                with col3:
                    img = plt.imread(class_path+image_name)
                    img_to_show =img
                    img = img.reshape(1,256,256,3)
                    img = preprocess_input(img/1.)
                    text = 'Classe predicte: '+str(np.argmax(model.predict(img)))
                    st.image(img_to_show, width=width, caption=text) # 130 for hanving all classes in one screen
                    #st.write('The predicted leaf class is: ', np.argmax(model.predict(img)))






    col1, col2, col3= st.columns([2,1,2])
    #col1, col2, col3, col4, col5, col6 = st.columns(6)
    #df_classes['Class']
    choices =[]
    for i in df_classes.index:
        choice = f'%02d'%i+'. '+df_classes['Class'][i]
        choices.append(choice)


    models_choices = ['CNN','VGG16 UnFr', 'ResNet50 UnFr', 'InceptionV3 UnFr']
    model_name_to_load= ['CNN_FDS_bs32_20e.h5',
                           'VGG16_UnFr4_FDS.h5',
                           'RN50_UnFr10_FDS.h5',
                           'INV3_UnFr23_FDS.h5' ]
    with col2:
        model_selection  = st.selectbox("Choisir un modèle",options = models_choices)

        ### CNN ###
        if model_selection == models_choices[0]:

            model= keras.models.load_model('modeles/'+model_name_to_load[0])
            preprocess_input = Rescaling(scale=1.0/255)

        ### VGG ###
        elif model_selection == models_choices[1]:
            model= keras.models.load_model('modeles/'+model_name_to_load[1])
            from tensorflow.keras.applications.vgg16 import preprocess_input

        ### ResNet ###
        elif model_selection == models_choices[2]:
            model= keras.models.load_model('modeles/'+model_name_to_load[2])
            from tensorflow.keras.applications.resnet50 import preprocess_input

        ### Inception ###
        elif model_selection == models_choices[3]:
            model= keras.models.load_model('modeles/'+model_name_to_load[3])
            from tensorflow.keras.applications.inception_v3  import preprocess_input


    #choices = df_classes.index
    with col1:
        class_selection_1  = st.selectbox("Choisir une première classe",options = choices)
        index_1 = int(class_selection_1[:2])
        class_path_1 = 'test_min/' + df_classes['Directory'][index_1]+'/'
        images_1 =[]
        while len(images_1) < num_images:
            image_name = random.choice([
            x for x in os.listdir(class_path_1)
            if os.path.isfile(os.path.join(class_path_1, x))])
            if image_name not in images_1:
                images_1.append(image_name)



    with col3:
        class_selection_2  = st.selectbox("Choisir une deuxième classe",options = choices)
        index_2 = int(class_selection_2[:2])
        class_path_2 = 'test_min/' + df_classes['Directory'][index_2]+'/'
        images_2 =[]
        while len(images_2) < num_images:
            image_name = random.choice([
            x for x in os.listdir(class_path_2)
            if os.path.isfile(os.path.join(class_path_2, x))])
            if image_name not in images_2:
                images_2.append(image_name)


    col1, col2, col3, col4, col5, col6,col7, col8, = st.columns(8)

    show_class(images_1, class_path_1, col1,col2,col3)

    show_class(images_2, class_path_2, col6,col7,col8)

    #st.write(images_1)
    #st.write(images_2)

if choix == liste_choix[1]:

    import streamlit as st
    from PIL import Image
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import tensorflow
    from skimage.transform import resize

    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.markdown('# Drag and Drop')
    #load model
    option = st.selectbox( 'Liste de modèles : ', ("CNN", "VGG16", "Resnet 50", "Inception V3"))


    if option == "CNN":
        model = load_model('modeles/CNN_FDS_bs32_20e.h5')
    elif option == "VGG16":
        model = load_model('modeles/VGG16_UnFr4_FDS.h5')
    elif option == "Resnet 50":
        model = load_model('modeles/RN50_UnFr10_FDS.h5')
    elif option == "Inception V3":
        model = load_model('modeles/INV3_UnFr23_FDS.h5')


    #create file uploader
    img_data = st.file_uploader(label='Load leaf for recognition', type = ['png', 'jpg', 'jpeg'])
    if img_data is not None:

        #display image
        uploaded_img = Image.open(img_data)
        st.image(uploaded_img)

        #load image file to predict and make prediction
        #img_path = f'G:/Mon Drive/DataScience/Datascientest/Git/Projet-Plant-Recognizion/{img_data.name}'
        #img = image.load_img(img_path, target_size=(256,256))
        img = image.img_to_array(uploaded_img)

        #img = resize(img, (1, 256, 256, 3))
        img = img.reshape(1,256,256,3)

        #display prediction
        class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

        st.title('The leaf is most likely the following:')
        pred = class_names[np.argmax(model.predict(img))]
        st.write(pred)
        st.write('with a probability of :')
        st.write(np.around(np.max(model.predict(img)), 6))

