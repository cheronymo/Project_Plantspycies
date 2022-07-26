import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import time


st.set_page_config(layout="wide")


#df_classes
df_classes = pd.read_excel('classes.xlsx',sheet_name = 'Sheet1', index_col = 0)

tot_train_by_class = df_classes['Train Capital'] + df_classes['Train lower']
tot_test_by_class = df_classes['Test Capital'] + df_classes['Test lower']
n_classes = df_classes.shape[0]
tot_image_train = tot_train_by_class.sum()
tot_image_test = tot_test_by_class.sum()
#tot_image_train = df_classes['Train Capital'].sum()+df_classes['Train lower'].sum()
#tot_image_test = df_classes['Test Capital'].sum()+df_classes['Test lower'].sum()
average_train = tot_image_train/n_classes
average_test = tot_image_test/n_classes 

delta_train_perc_dataset = (tot_train_by_class - average_train )/tot_image_train
delta_test_perc_dataset = (tot_test_by_class - average_test )/tot_image_test

delta_train_perc_average = (tot_train_by_class - average_train )/average_train
delta_test_perc_average = (tot_test_by_class - average_test )/average_test


label_DS_overview = 'Structure de la Base de Données' 
#label_dataset = 'Le Dataset : Espèces et Maladies'
label_dataset = 'Espèces et Maladies'
label_train_set = 'Train Set'
label_test_set = 'Test Set'
label_train_set_full = 'Train Set' #'Train Set "Full"'
label_test_set_full = 'Test Set'#'Test Set "Full"'
label_images = 'Les Images'
label_prep = 'Préparation des Données'

#### Function for charting Classes #######
def class_chart(set, y_lim,result_full, flag_perc = False):
    
    base_color = "Darkgreen"
    ticks_color = base_color
    fig = plt.figure(figsize=(24,8))
    plt.xticks(rotation = 90, color=ticks_color, fontsize = 12)
    plt.yticks(color=ticks_color)
    plt.title(set + " Set", fontsize=80, color=base_color)
    plt.xlabel("Classes", fontsize=50, color=base_color)
    

    if flag_perc:
        plt.ylabel("Delta %", fontsize=50, color=base_color)
        plt.ylim([-0.15, 0.15])
        if set == "Train":
            y = [delta_train_perc_average, delta_train_perc_dataset]
        elif set == "Test":
            y = [delta_test_perc_average, delta_test_perc_dataset]
        plt.bar(df_classes['Class'],y[0] , color = [base_color] , label='Delta comme % de la moyenne', alpha=0.7)
        plt.bar(df_classes['Class'],y[1] , color = ['red'] , label='Delta comme % du dataset', alpha=0.7)
        plt.legend()
    else:
        plt.ylabel("Images", fontsize=50, color=base_color)
        plt.ylim([0, y_lim])    
        plt.bar(df_classes['Class'],df_classes[set + ' Capital'] , color = df_classes['Color'] , label='JPG', alpha=1)
        if result_full:
            plt.bar(df_classes['Class'],df_classes[set + ' lower'] , color = df_classes['Color'] ,bottom= df_classes[ set +' Capital'], label='jpg', alpha=1)
    
    st.pyplot(fig)



## Sidebar
#st.sidebar.image(image, caption='PLANTSPycies', width = 200)
#pages = ['0','1', '2', '3','4', '5']
page = 12
#page = st.sidebar.radio("Menu", options = pages)
choices = [
            label_DS_overview,
            label_dataset, 
            label_images,  
#            label_train_set, 
#            label_test_set, 
            label_train_set_full, 
            label_test_set_full, 
            label_prep]


 


viz_selection  = st.sidebar.radio("",options = choices) 


col1, col2, col3, col4, col5= st.columns(5)
#col1, col2, col3, col4, col5, col6 = st.columns(6)


#with col1:  viz_selection  = st.selectbox("",options = choices) 
if (viz_selection == label_train_set) | (viz_selection == label_test_set):
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        
    
with col1:  
    st.write("")
    st.write("")
    if viz_selection == label_images: test = st.button('Nouvelles Images') 
    if viz_selection == label_train_set_full: test = st.button('Delta en percentage de la moyenne')
    if viz_selection == label_test_set_full: test = st.button('Delta en percentage de la moyenne') 

#with col2:
#    st.write("")
#    st.write("")
#    if viz_selection == label_train_set_full: dummy = st.button('Valeurs') 
#    if viz_selection == label_test_set_full: dummy = st.button('Valeurs') 
if viz_selection == label_DS_overview :
    st.title(label_DS_overview)
    #st.write("")
    col1, col2 = st.columns([1,10])
    with col2:
        test_size_img = 380
        image_data = Image.open('images/data.png')
        st.image(image_data, width = test_size_img)


if viz_selection == label_dataset :
    st.title('Le Dataset : Espèces et Maladies')

    classes = df_classes['Class']
    colors = df_classes['Color']

    col1, col2, col3 = st.columns(3)
    for i in range(df_classes.shape[0]):
        class_ = classes[i]
        color =  colors[i]
        n_train =  tot_train_by_class[i]
        n_test =  tot_test_by_class[i]
        text_1 = f'<p style="font-family:sans-serif; color:{color}; font-size: 16px;">- {i}: {class_}' 
        text_2 = f'<span style="font-family:sans-serif; color:{color}; font-size: 11px;"<em> ({n_train},{n_test})</p>'
        if i <=10:
            with col1:
                st.markdown(text_1+ text_2, unsafe_allow_html=True)
        if 10< i <=23:
            with col2:
                st.markdown(text_1+ text_2, unsafe_allow_html=True)
        if 23< i:
            with col3:
                st.markdown(text_1+ text_2, unsafe_allow_html=True)
    summary = f'Espèces représentées : 14  ------>  Classes : 38'
    text = f'<p style="font-family:sans-serif; color: black ; font-size: 20px;">{summary}</p>'
    st.write(text, unsafe_allow_html=True)

    summary = f"Nombre total d'images : {tot_image_train+tot_image_test} (80% Train Set, 20% Test Set)"
    text = f'<p style="font-family:sans-serif; color: black ; font-size: 20px;">{summary}</p><hr />'
    st.write(text, unsafe_allow_html=True)

    legenda_1 = f'Format :  Espèce___Maladie/Saine '
    legenda_2 = f'(nombre images dans le "Train Folder", nombre images dans le "Test Folder")'
    text_1 = f'<p style="text-align: right; font-family:sans-serif; color: black ; font-size: 14px;">{legenda_1}'
    text_2 = f'<span style="ext-align: right; font-family:sans-serif; color: black ; font-size: 12px;"><em>{legenda_2}'
    st.write(text_1 + text_2, unsafe_allow_html=True)




if (viz_selection == label_train_set) | (viz_selection == label_train_set_full):
    st.write("")
    st.write("")
    if viz_selection == label_train_set_full:
        if test:
            class_chart("Train", 0.2,viz_selection == label_train_set_full,True)    
        else:
            class_chart("Train", 2100,viz_selection == label_train_set_full)
    else:
        class_chart("Train", 2100,viz_selection == label_train_set_full)
    


if (viz_selection == label_test_set)| (viz_selection == label_test_set_full):
    st.write("")
    st.write("")
    if viz_selection == label_test_set_full:
        if test:
            class_chart("Test", 0.2,viz_selection == label_test_set_full,True)    
        else:
            class_chart("Test", 600,viz_selection == label_test_set_full)
    else:
        class_chart("Test", 600,viz_selection == label_test_set_full)



if viz_selection == label_images: 
    import os
    import random 
    images =[]
    captions =[]
    for n,(class_, dir_class) in enumerate(zip(df_classes['Class'],df_classes['Directory'])):
        class_path = 'test_min/'+dir_class+'/'
        random_filename = random.choice([
        x for x in os.listdir(class_path)
        if os.path.isfile(os.path.join(class_path, x))])
        images.append(class_path+random_filename)
        caption = f'{str(n)}:{class_}'
        #captions.append('Class '+str(n)+" "+class_)
        captions.append(caption)
    st.image(images, width=150, caption=captions) # 130 for hanving all classes in one screen





if viz_selection == label_prep:
    st.title('Préparation des Données') 
    col1, col2 = st.columns([1,2])
    with col1:
        test_size_img = 300
        image_data = Image.open('images/data.png')
        st.image(image_data, width = test_size_img)
    with col2:
        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = ['Training Set', 'Validation Test' 'Test Set']
        sizes = [tot_image_train, tot_image_test  ]
        explode = (0, 0.1)
        colors = ['r','Darkgreen']
        fig1,ax1 = plt.subplots()
        #fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels,
            autopct='%1.1f%%', 
            colors =colors,
            textprops={'fontsize': '20'},
            shadow=False, startangle=90)
        #plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #st.pyplot(fig1)


        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = ['Training Set', 'Validation Test' , 'Test Set']
        sizes = [tot_image_train*0.8, tot_image_train*0.2, tot_image_test]
        explode = (0, 0, 0.1)
        colors = ['r', 'gold', 'Darkgreen']
        fig2,ax2 = plt.subplots()
        #fig2, ax2 = plt.subplots()
        ax2.pie(sizes, explode=explode, labels=labels,
            autopct='%1.1f%%', 
            colors =colors,
            textprops={'fontsize': '20'},
            shadow=False, startangle=90)
        #plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #st.pyplot(fig2)

        split = f"Séparation de la base en trois datasets : Training, Validation, Test"
        text = f'<p style="font-family:sans-serif; color: Darkgreen ; font-size: 20px;"><u>{split}</p>'
        st.write(text, unsafe_allow_html=True)
    
        #image_data = Image.open('images/pie_data.png')
        image_data = Image.open('images/pie2.png')
        st.image(image_data, width = 710)
        

        st.markdown('#')

        
        reduced_DS = f"Création d'une base de données réduite"
        text = f'<p style="font-family:sans-serif; color: Darkgreen ; font-size: 20px;"><u>{reduced_DS}</p>'
        st.write(text, unsafe_allow_html=True)

        image_data = Image.open('images/reduced_DS2.png')
        st.image(image_data, width = 610)
        
        pre_pro = f"Preprocessing "
        text = f'<p style="font-family:sans-serif; color: Darkgreen ; font-size: 20px;"><u>{pre_pro}</p>'
        #st.write(text, unsafe_allow_html=True)
    
        
        st.markdown('#')

    
if viz_selection == 0: 
    st.header('Liste des points possibles (work in progress)')
    st.markdown('image de la structure des directories ????')
    st.markdown('train valid Test')
    st.markdown('preprocessing')
    st.markdown('Reduced Dataset')
    st.markdown('??????')
    code_str = '''


    # Define Batch Size and image size
batch_size_train =  128 
img_height = 256
img_width = 256 


### Split the contents of the folder "train" in Training and Validation Set  ###
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  label_mode = 'categorical', validation_split=0.2,  
  subset="training",  seed=123,
  image_size=(img_height, img_width), batch_size=batch_size_train)

val_ds = tf.keras.utils.image_dataset_from_directory(
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_train,
  label_mode = 'categorical', validation_split=0.2,  
  subset="validation",  seed=123,
  image_size=(img_height, img_width), batch_size=batch_size_train)


### Test Set ###
test_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir_test,
  label_mode = 'categorical',
  image_size=(img_height, img_width),
  batch_size= 512)

  '''
    st.code(code_str)