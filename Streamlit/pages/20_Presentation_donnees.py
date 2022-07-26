import numpy as np  # numerical computing
import pandas as pd  # data structures and data analysis tools
import matplotlib.pyplot as plt  # plotting images
import streamlit as st
# import mpld3 # Pour une figure interactive
import streamlit.components.v1 as components

st.set_page_config(layout='wide')

st.header('Différentes BDD disponibles initialement')

st.image('images/Table synthèse databases.png')

st.subheader("Choix du dataset à la fois complet et équilibré : New Plant Diseases Dataset")
# df = pd.read_excel('Data_overview.xlsx')
# st.table(df)

# st.markdown('---')
# st.header("Datasets train & test")
# st.subheader("Train")

# class_names = ['Apple___Apple_scab',
#                'Apple___Black_rot',
#                'Apple___Cedar_apple_rust',
#                'Apple___healthy',
#                'Blueberry___healthy',
#                'Cherry_(including_sour)___Powdery_mildew',
#                'Cherry_(including_sour)___healthy',
#                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#                'Corn_(maize)___Common_rust_',
#                'Corn_(maize)___Northern_Leaf_Blight',
#                'Corn_(maize)___healthy',
#                'Grape___Black_rot',
#                'Grape___Esca_(Black_Measles)',
#                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                'Grape___healthy',
#                'Orange___Haunglongbing_(Citrus_greening)',
#                'Peach___Bacterial_spot',
#                'Peach___healthy',
#                'Pepper,_bell___Bacterial_spot',
#                'Pepper,_bell___healthy',
#                'Potato___Early_blight',
#                'Potato___Late_blight',
#                'Potato___healthy',
#                'Raspberry___healthy',
#                'Soybean___healthy',
#                'Squash___Powdery_mildew',
#                'Strawberry___Leaf_scorch',
#                'Strawberry___healthy',
#                'Tomato___Bacterial_spot',
#                'Tomato___Early_blight',
#                'Tomato___Late_blight',
#                'Tomato___Leaf_Mold',
#                'Tomato___Septoria_leaf_spot',
#                'Tomato___Spider_mites Two-spotted_spider_mite',
#                'Tomato___Target_Spot',
#                'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#                'Tomato___Tomato_mosaic_virus',
#                'Tomato___healthy']


# # TRAIN DATASET volumétrie, architecture

# nb_img_train_jpg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 213, 235, 0, 0, 0,
#                     0, 0, 0, 19, 0, 0, 0, 0, 1, 0, 1, 0, 0, 146, 1, 0, 0, 0, 0, 0, 0, 1831]

# nb_img_train_JPG = [2016, 1987, 2008, 1760, 1826, 1816, 1907, 1920, 1683, 1692, 1888, 1429, 1673, 2010, 1722, 1728, 1838,
#                     2022, 1824, 1717, 1939, 1939, 1781, 1913, 1987, 1824, 1773, 1702, 1920, 1705, 1925, 1745, 1741, 1827, 1882, 1961, 1790, 28]

# nb_img_test_TOTAL = [2016, 1987, 1760, 2008, 1816, 1683, 1826, 1642, 1907, 1908, 1859, 1888, 1920, 1722, 1692, 2010, 1838,
#                      1728, 1913, 1988, 1939, 1939, 1824, 1781, 2022, 1736, 1774, 1824, 1702, 1920, 1851, 1882, 1745, 1741, 1827, 1961, 1790, 1926]


# # Bar plot for number of .jpg and .JPG images
# fig, ax = plt.subplots(figsize=(20, 3))

# ax.bar(class_names, nb_img_train_jpg, label='jpg')
# ax.bar(class_names, nb_img_train_JPG, label='JPG',
#         bottom=nb_img_train_jpg)  # bottom parameter to avoid superposition
# plt.xticks(rotation=90)
# plt.xlabel('Folder')
# plt.ylabel('Number of images')
# plt.title('Number of .jpg and .JPG images per folder')
# plt.legend(loc='best')

# st.pyplot(fig)

# # TEST DATASET volumétrique, architecture

# nb_img_test_jpg = [0, 0, 0, 0, 0, 0, 0, 0, 0, 451, 0, 0, 54, 55, 0, 0,
#                    0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 1, 0, 0, 0, 30, 0, 0, 0, 0, 0, 0, 0]


# nb_img_test_JPG = [496, 504, 440, 502, 423, 480, 456, 421, 454, 14, 472, 477, 423, 355, 430, 503, 432,
#                    459, 445, 478, 505, 456, 497, 485, 426, 485, 443, 456, 425, 480, 433, 481, 435, 457, 448, 470, 436, 490]

# nb_img_test_TOTAL = [504, 496, 440, 502, 454, 421, 456, 410, 477, 477, 465, 472, 480, 430, 423, 503, 459,
#                      432, 478, 497, 485, 485, 456, 445, 505, 434, 444, 456, 425, 480, 463, 470, 436, 435, 457, 490, 448, 481]


# # Bar plot for number of .jpg and .JPG images

# fig, ax = plt.subplots(figsize=(20, 3))

# ax.bar(class_names, nb_img_test_jpg, label='jpg')
# ax.bar(class_names, nb_img_test_JPG, label='JPG',
#         bottom=nb_img_test_jpg)  # bottom parameter to avoid superposition
# plt.xticks(rotation=90)
# plt.xlabel('Folder')
# plt.ylabel('Number of images')
# plt.title('Number of .jpg and .JPG images per folder')
# plt.legend(loc='best')

# st.subheader("Test")
# st.pyplot(fig)

# # Pour une figure interactive
# # fig_html = mpld3.fig_to_html(fig)
# # components.html(fig_html, height=600)
