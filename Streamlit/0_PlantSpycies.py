import streamlit as st
from PIL import Image


# Titre
st.markdown("<h1 style='text-align: center; color: grey;'>PlantSpycies</h1>", unsafe_allow_html=True)

#Image
image = Image.open('images/PlantSpycies.jpeg')
col1, col2, col3 = st.columns([0.1, 10, 0.1])
col2.image(image, use_column_width=True)



# Architecture
## Slidebar menu

st.sidebar.image("images/datascientest.png", use_column_width=True)

add_textbox = st.sidebar.write("_Team_")

add_textbox = st.sidebar.write("**BECCHERE Giovanni**")

add_textbox = st.sidebar.write("**CHERO Guillaume** ")

add_textbox = st.sidebar.write("**PARRA CARRASCOSA Valentin Miguel**")

add_textbox = st.sidebar.write("*Datascientest - Mai 2022*")


