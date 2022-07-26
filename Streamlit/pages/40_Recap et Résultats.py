import tensorflow as tf  # deep neural networks
import streamlit as st
import pandas as pd
from tensorflow import keras  # interface for TensorFlo
from tensorflow.keras.models import Sequential  # linear stack of layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D  # convolution and pooling
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3



# from tensorflow.python.keras.models import load_model  # to load a saved model

width_class_report = 500
width_matrix = 600

st.set_page_config(layout='wide')

st.sidebar.markdown("# Veuillez choisir un modèle")

liste_choix = ["Tableau synthèse", "CNN_32", "CNN_128", "VGG16 freezed", "VGG16 unfreezed (4 layers)",
         "ResNet50 freezed", "ResNet50 unfreezed (10 layers)", "InceptionV3 freezed", "InceptionV3 unfreezed (12 layers)", "InceptionV3 unfreezed (23 layers)"]

choix = st.sidebar.radio("Modèles", options=liste_choix)

if choix == liste_choix[0]:

    st.markdown('# Synthèse des modèles')

    df = pd.read_csv('Comparaison modèles.csv', sep=';')
    # df = pd.read_csv('Comparaison modèles virg.csv', sep=',')
    # df.epochs = df.epochs.astype('float')
    df.train_param = df.train_param.astype('int64')

    # def get_pretty(valeur):
    #     return '{:,}'.format(round(valeur)).replace(',', ' ')

    # df.train_param = df.train_param.apply(get_pretty)

    # df.info()
    # définir couleurs tableau par modèle

    def highlight_models(s):
        if 'CNN' in s.Model:
            return ['background-color: lightcyan']*len(s)
        elif 'VGG' in s.Model:
            return ['background-color: bisque']*len(s)
        elif 'ResNet' in s.Model:
            return ['background-color: lavender']*len(s)
        elif 'Inception' in s.Model:
            return ['background-color: lightblue']*len(s)

    st.dataframe(df.style.apply(highlight_models, axis=1).highlight_max(
        ['test_F1'], axis=0).highlight_min(['train_param'], axis=0), width=5000)

    st.subheader('''
        9 modèles testés, dont 7 transfer learning. Meilleur modèle : VGG16 unfreezed
           ''')

    # Autre façon de construire la table pour éviter le problème de largeur de colonne sur st.dataframe (qui est mal géré, en prenant le minimum
    # de largeur entre le titre et les valeurs dans la colonne => le libellé de la colonne est trunqué si les valeurs sont plus petits, c'est
    # notre cas avec la colonne epochs)

    # fig = plotly.graph_objects.Figure(data=[go.Table(
    #     header=dict(values=list(df.columns),
    #                 fill_color='paleturquoise',
    #                 align='left'),
    #     cells=dict(values=np.transpose(df.values),
    #                fill_color='lavender',
    #                align='left'))
    # ])

    # st.plotly_chart(fig)


if choix == liste_choix[1]:
    st.markdown('# Modèle naïf (batch_size = 32)')

#    st.markdown('---')

#     img_height = 256
#     img_width = 256

#     def build_model():

#         model = Sequential()

#         model.add(Conv2D(32, (7, 7), activation="relu", padding="same",
#                          input_shape=(img_height, img_width, 3)))
#         model.add(Conv2D(32, (7, 7), activation="relu", padding="same"))

#         model.add(MaxPooling2D(3, 3))

#         model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))
#         model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))

#         model.add(MaxPooling2D(3, 3))

#         model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
#         model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

#         model.add(MaxPooling2D(3, 3))

#         model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))
#         model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

#         model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))
#         model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

#         model.add(Flatten())

#         model.add(Dense(512, activation="relu"))

#         model.add(Dropout(0.5))

#         model.add(Dense(38, activation="softmax"))

#         # Define model optimizer (optimizes the input weights by comparing the prediction and the loss function)
#         # We choose Adam optimizer, with an initial learning_rate = 0.0001.
#         # Adam optimizer modifies the learning_rate as needed during training
#         opt = keras.optimizers.Adam(learning_rate=0.0001)

#         model.compile(
#             optimizer=opt,
#             # Computes the crossentropy loss between the labels and predictions.
#             loss="categorical_crossentropy",
#             # our labels are provided in a one_hot representation (dataset label_mode=categorical)
#             metrics=['accuracy'])

#         return model

#     code_str = '''
# model = Sequential()


# model.add(Conv2D(32, (7, 7), activation="relu", padding="same",
#                      input_shape=(img_height, img_width, 3)))
# model.add(Conv2D(32, (7, 7), activation="relu", padding="same"))


# model.add(MaxPooling2D(3, 3))



# model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))

# model.add(Conv2D(64, (5, 5), activation="relu", padding="same"))


# model.add(MaxPooling2D(3, 3))



# model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))

# model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))


# model.add(MaxPooling2D(3, 3))



# model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))

# model.add(Conv2D(256, (3, 3), activation="relu", padding="same"))


# model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

# model.add(Conv2D(512, (3, 3), activation="relu", padding="same"))

# model.add(Flatten())


# model.add(Dense(512, activation="relu"))

# model.add(Dropout(0.5))


# model.add(Dense(38, activation="softmax"))
# '''
#     model = build_model()

#     col01, col02 = st.columns(2)

#     with col01:
#         st.header("Model Code & Summary")
#         st.code(code_str)

#     with col02:
#         model.summary(print_fn=lambda x: st.text(x))

#     st.markdown('---')

    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/CNN FDS lr 0.0001 20e.png')

    st.markdown('---')

    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image('images/Conf matrix best CNN.png')

    with col2:
        st.image('images/Classif report best CNN.png')

    st.subheader('''Confusion principalement entre :
- les classes 7 et 9 (Corn_(maize)___Cercospora_leaf_spot Gray_leaf_ vs Corn_(maize)___Northern_Leaf_Blight 
- les maladies des tomates''')
    # st.header("Model Summary")
    # model.summary(print_fn=lambda x: st.text(x))

    # st.header("Loss & Accuracy")
    # st.image('images/CNN FDS lr 0.0001 20e.png')

    # st.image('images/Conf matrix best CNN.png', width=width_matrix)

    # st.header("Classification Report")
    # st.image('images/Classif report best CNN.png', width=width_class_report)

if choix == liste_choix[2]:

    st.markdown('# Modèle naïf (batch_size = 128)')

    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/CNN FDS lr 0.0001 20e batch 128.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image('images/Conf matrix CNN 128.png')

    with col2:
        st.image('images/Classif report CNN 128.png')

    st.subheader("Effet d'augmenter le batch_size => Dégradation dans la capacité du modèle à généraliser (plus grand overfitting et moindre précision sur les données test)")

##########################################################
### VGG16 Freezed ########################################
##########################################################
if choix == liste_choix[3]:
    st.markdown('# VGG16 Freezed')
    
    st.markdown('---')


    # ### Summary ###
    # n_class=38
    # img_height = 256
    # img_width = 256
    # base_model = VGG16(weights='imagenet', include_top=False,
    #                  classes= n_class, input_shape = (img_height,img_width,3)) 


    # # Freeze the layers of base_model
    # for layer in base_model.layers: 
    #     layer.trainable = False

    # model_VGG16 = Sequential()

    # model_VGG16.add(base_model)

    # model_VGG16.add(GlobalAveragePooling2D()) 
    # model_VGG16.add(Dense(1024,activation='relu'))
    # model_VGG16.add(Dropout(rate=0.2))
    # model_VGG16.add(Dense(512, activation='relu'))
    # model_VGG16.add(Dropout(rate=0.2))
    # model_VGG16.add(Dense(n_class, activation='softmax'))

    # st.header("Model Summary")
    # model_VGG16.summary(print_fn=lambda x: st.text(x))

    ### Results ###
    st.header("Loss & Accuracy")
    st.image('images/VGG_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")

    col1, col2 = st.columns([2,1])

    with col1:
        st.image('images/VGG_matrix.png')
        

    with col2:           
        st.image('images/VGG_class_report.png')


    st.markdown("#### Les résultats de cette version “freezed” sont déjà assez bons dans l'ensemble.")
    st.markdown("#### Les classes 7 et 9, ainsi que les maladies des tomates, restent moins faciles à classer.")

##########################################################
### VGG16 Unfreezed ######################################
##########################################################
if choix == liste_choix[4]:
    st.markdown('# VGG16 Unfreezed (4 layers)')


    # ### Summary ###
    # n_class=38
    # img_height = 256
    # img_width = 256
    # base_model = VGG16(weights='imagenet', include_top=False,
    #                  classes= n_class, input_shape = (img_height,img_width,3)) 

    # # Freeze the layers of base_model
    # for layer in base_model.layers: 
    #     layer.trainable = False

    # model_VGG16 = Sequential()

    # model_VGG16.add(base_model)

    # model_VGG16.add(GlobalAveragePooling2D()) 
    # model_VGG16.add(Dense(1024,activation='relu'))
    # model_VGG16.add(Dropout(rate=0.2))
    # model_VGG16.add(Dense(512, activation='relu'))
    # model_VGG16.add(Dropout(rate=0.2))
    # model_VGG16.add(Dense(n_class, activation='softmax'))

    # for layer in model_VGG16.layers:
    #     if "Functional" == layer.__class__.__name__:  
    #          #here you can iterate and choose the layers of your nested model
    #          for _layer in layer.layers[-4:]:
    #               # your logic with nested model layers
    #               _layer.trainable = True

    # st.header("Model Summary")
    # model_VGG16.summary(print_fn=lambda x: st.text(x))



    ### Results ###

    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/VGG_UnFr_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2,1])

    with col1:
        st.image('images/VGG_UnFr_matrix.png')
        

    with col2:           
        st.image('images/VGG_UnFr_class_report.png')




    st.markdown("#### L’unfreezing du dernier bloc de convolution permet d’améliorer les résultats.")
    st.markdown("#### Les classes 7 et 9, ainsi que les maladies des tomates, restent moins faciles à classer.")

    # font_bullet = 24
    # bullet = f"L’unfreezing du dernier bloque de convolution permet d’améliorer ultérieurement les résultats."
    # text = f'<p style="font-family:sans-serif; color: black ; font-size: {font_bullet}px;">{bullet}</p>'
    # st.write(text, unsafe_allow_html=True)
    # bullet = f"Le classes 7 et 9, ainsi que les maladies des tomates, restent un peu plus compliquées à classer."
    # text = f'<p style="font-family:sans-serif; color: black; font-size: {font_bullet}px;">{bullet}</p>'
    # st.write(text, unsafe_allow_html=True)
    
##########################################################
### ResNet50 Freezed #####################################
##########################################################
if choix == liste_choix[5]:
    st.markdown('# ResNet50 Freezed')
    


    # ### Summary ###
    # n_class=38
    # img_height = 256
    # img_width = 256
    # base_model = ResNet50(weights='imagenet', include_top=False,
    #                  classes= n_class, input_shape = (img_height,img_width,3)) 


    # # Freeze the layers of base_model
    # for layer in base_model.layers: 
    #     layer.trainable = False

    # model_ResNet50 = Sequential()

    # model_ResNet50.add(base_model)

    # model_ResNet50.add(GlobalAveragePooling2D()) 
    # model_ResNet50.add(Dense(1024,activation='relu'))
    # model_ResNet50.add(Dropout(rate=0.2))
    # model_ResNet50.add(Dense(512, activation='relu'))
    # model_ResNet50.add(Dropout(rate=0.2))
    # model_ResNet50.add(Dense(n_class, activation='softmax'))

    # st.header("Model Summary")
    # model_ResNet50.summary(print_fn=lambda x: st.text(x))

    ### Results ###

    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/ResNet_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2,1])

    with col1:
        st.image('images/ResNet_matrix.png')
        

    with col2:           
        st.image('images/ResNet_class_report.png')

    st.markdown("#### Les résultats de cette version “freezed” sont déjà très bons.")
    st.markdown("#### Les classes 7 et 9, ainsi que les maladies des tomates, restent moins faciles à classer.")
##########################################################
### ResNet50 Unfreezed ###################################
##########################################################
if choix == liste_choix[6]:
    st.markdown('# ResNet50 Unfreezed (10 layers)')
    


    ### Summary ###
    # n_class=38
    # img_height = 256
    # img_width = 256
    # base_model = ResNet50(weights='imagenet', include_top=False,
    #                  classes= n_class, input_shape = (img_height,img_width,3)) 


    # # Freeze the layers of base_model
    # for layer in base_model.layers: 
    #     layer.trainable = False

    # model_ResNet50 = Sequential()

    # model_ResNet50.add(base_model)

    # model_ResNet50.add(GlobalAveragePooling2D()) 
    # model_ResNet50.add(Dense(1024,activation='relu'))
    # model_ResNet50.add(Dropout(rate=0.2))
    # model_ResNet50.add(Dense(512, activation='relu'))
    # model_ResNet50.add(Dropout(rate=0.2))
    # model_ResNet50.add(Dense(n_class, activation='softmax'))

    # for layer in model_ResNet50.layers:
    #     if "Functional" == layer.__class__.__name__:  
    #          #here you can iterate and choose the layers of your nested model
    #          for _layer in layer.layers[-10:]:
    #               # your logic with nested model layers
    #               _layer.trainable = True

    # st.header("Model Summary")
    # model_ResNet50.summary(print_fn=lambda x: st.text(x))

    ### Results ###
    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/ResNet_UnFr_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2,1])

    with col1:
        st.image('images/ResNet_UnFr_matrix.png')
        

    with col2:           
        st.image('images/ResNet_UnFr_class_report.png')

    #st.markdown("### L’unfreezing du dernier “bottleneck building block” permet d'améliorer encore les résultats et d'atteindre une Test Accuracy de 99.4%.")
    st.markdown("#### L’unfreezing du dernier “bottleneck building block” permet d'améliorer les résultats.")
    st.markdown("#### Les classes 7 et 9, ainsi que les maladies des tomates, restent moins faciles à classer.")


    ##########################################################
    ### Inception Freezed #####################################
    ##########################################################
if choix == liste_choix[7]:

    st.markdown('# Inception V3 Freezed')

    # ### Summary ###
    # n_class = 38
    # img_height = 256
    # img_width = 256
    # n_class = 38

    # # Modèle Inception V3
    # # This model is freezed and weights are based on 'imagenet'.
    # base_model = InceptionV3(weights='imagenet',
    #                          include_top=False,
    #                          classes=n_class,
    #                          input_shape=(img_height, img_width, 3))

    # # Free the layers of base_model
    # for layer in base_model.layers:
    #     layer.trainable = False

    # # We can had some layers on the Inception model
    # model_Inception = Sequential()

    # model_Inception.add(base_model)

    # model_Inception.add(GlobalAveragePooling2D())
    # model_Inception.add(Dense(1024, activation='relu'))
    # model_Inception.add(Dropout(rate=0.2))
    # model_Inception.add(Dense(512, activation='relu'))
    # model_Inception.add(Dropout(rate=0.2))
    # model_Inception.add(Dense(n_class, activation='softmax'))  # softmax is the activation function recommanded



    # st.header("Model Summary")
    # model_Inception.summary(print_fn=lambda x: st.text(x))


    ### Results ###
    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/Inception_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image('images/Inception_matrix.png')

    with col2:
        st.image('images/Inception_class_report.png')

    st.subheader("Accuracy élevée mais plus faible que les modèles précedents."
                 " Plus rapide à tourner avec moins de ressource.")

##########################################################
### Inception Unfreezed 1 ###################################
##########################################################
if choix == liste_choix[8]:
    st.markdown('# Inception Unfreezed (12 layers)')

    # ### Summary ###
    # n_class = 38
    # img_height = 256
    # img_width = 256
    # n_class = 38

    # # Modèle Inception V3
    # # This model is freezed and weights are based on 'imagenet'.
    # base_model = InceptionV3(weights='imagenet',
    #                          include_top=False,
    #                          classes=n_class,
    #                          input_shape=(img_height, img_width, 3))

    # # Free the layers of base_model
    # for layer in base_model.layers:
    #     layer.trainable = False

    # # We can had some layers on the Inception model
    # model_Inception = Sequential()

    # model_Inception.add(base_model)

    # model_Inception.add(GlobalAveragePooling2D())
    # model_Inception.add(Dense(1024, activation='relu'))
    # model_Inception.add(Dropout(rate=0.2))
    # model_Inception.add(Dense(512, activation='relu'))
    # model_Inception.add(Dropout(rate=0.2))
    # model_Inception.add(Dense(n_class, activation='softmax'))  # softmax is the activation function recommanded

    # for layer in model_Inception.layers:
    #     if "Functional" == layer.__class__.__name__:
    #         # here you can iterate and choose the layers of your nested model
    #         for _layer in layer.layers[-12:]:
    #             # your logic with nested model layers
    #             _layer.trainable = True

    # st.header("Model Summary")
    # model_Inception.summary(print_fn=lambda x: st.text(x))

    ### Results ###
    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/Inception_Un12_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image('images/Inception_Un12_matrix.png')

    with col2:
        st.image('images/Inception_Un12_class_report.png')

    st.subheader("On augmente très faiblement les résultats et le nombre de paramètres entrainé")

##########################################################
### Inception Unfreezed 2 ###################################
##########################################################
if choix == liste_choix[9]:
    st.markdown('# Inception Unfreezed (23 layers)')

    # ### Summary ###
    # n_class = 38
    # img_height = 256
    # img_width = 256
    # base_model = InceptionV3(weights='imagenet', include_top=False,
    #                       classes=n_class, input_shape=(img_height, img_width, 3))

    # # Free the layers of base_model
    # for layer in base_model.layers:
    #     layer.trainable = False

    # # We can had some layers on the Inception model
    # model_Inception = Sequential()

    # model_Inception.add(base_model)

    # model_Inception.add(GlobalAveragePooling2D())
    # model_Inception.add(Dense(1024, activation='relu'))
    # model_Inception.add(Dropout(rate=0.2))
    # model_Inception.add(Dense(512, activation='relu'))
    # model_Inception.add(Dropout(rate=0.2))
    # model_Inception.add(Dense(n_class, activation='softmax'))  # softmax is the activation function recommanded

    # for layer in model_Inception.layers:
    #     if "Functional" == layer.__class__.__name__:
    #         # here you can iterate and choose the layers of your nested model
    #         for _layer in layer.layers[-23:]:
    #             # your logic with nested model layers
    #             _layer.trainable = True

    # st.header("Model Summary")
    # model_Inception.summary(print_fn=lambda x: st.text(x))

    ### Results ###
    st.markdown('---')
    st.header("Loss & Accuracy")
    st.image('images/Inception_Un23_training.png')

    st.markdown('---')
    st.header("Confusion matrix & Classification report")
    col1, col2 = st.columns([2, 1])

    st.markdown('---')
    with col1:
        st.image('images/Inception_Un23_matrix.png')

    with col2:
        st.image('images/Inception_Un23_class_report.png')

    st.subheader("On augmente très faiblement les résultats et le nombre de paramètres entrainé. "
                 "Une simulation exploratoire montre qu'il faut aller au moins à la couche 50 pour améliorer encore le modèle.")


