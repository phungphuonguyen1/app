import streamlit as st
from langchain.llms import OpenAI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import joblib
from PIL import Image
import sklearn

# Config function
img=Image.open("nano.jpg")
st.set_page_config(page_title='Nanoparticles distribution prediction',page_icon=img)

# hide main menu and footer
hide_menu_style= """
    <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu_style,unsafe_allow_html=True)

def user_input_features():
    ponp=st.sidebar.slider('Interaction between polymers and nanoparticles: ',0.0,2.5, 0.4)
    npnp=st.sidebar.slider('Interaction between nanoparticles and nanoparticles: ',0.0,2.5, 0.4)
    d=st.sidebar.slider('Diameter of nanoparticles: ',1,10,4)
    phi=st.sidebar.slider('Number of particles: ',0.001,0.1,0.02)
    cLength=st.sidebar.slider('Length of polymer chain: ',5,100,20)
    st.sidebar.subheader('Distance Range')
    distance_str_min = st.sidebar.text_input('Minimum distance in nm: ','0.075')
    distance_str_width_range = st.sidebar.text_input('Width range in nm: ','150')
    distance_min= float(distance_str_min)
    distance_range= float(distance_str_width_range)
    Po_NP=pd.DataFrame({'Po_NP':[ponp]*2000})
    NP_NP=pd.DataFrame({'NP_NP':[npnp]*2000})
    D_aim=pd.DataFrame({'D_aim':[d]*2000})
    Phi=pd.DataFrame({'Phi':[phi]*2000})
    Chain_length=pd.DataFrame({'Chain_length':[cLength]*2000})
    distance = pd.DataFrame({'distance': np.linspace(distance_min, distance_min + distance_range, 2000)})
    features=pd.concat([Po_NP,NP_NP,D_aim,Phi,Chain_length,distance], axis=1)
    features.columns = ['Po_NP','NP_NP','D_aim','Phi','Chain length','distance']
    #features=data
    return features


#try:
    #model = joblib.load("/mount/src/app/model.pkl")
#except Exception as e:
    #st.error(f"An error occurred while loading the model: {e}")
with st.sidebar:
    st.title("Input parameters")
    st.info("Please enter inputs for the caculation.")
df = user_input_features()
 
st.title("DISTRIBUTION OF NANOPARTICLES IN A POLYMER MATRIX PREDICTION")
st.header("Problem description")
st.write("""Polymer nanocomposites (PNC) offer a broad range of properties that are intricately 
         connected to the spatial distribution of nanoparticles (NPs) in polymer matrices. 
         Understanding and controlling the distribution of NPs in a polymer matrix is a significantly challenging task.
         We aim to address this challenge via machine learning. In this website, we use Decision Tree Regression to predict the distribution of nanoparticles in a polymer matrix.""")
image=Image.open("polymer_nanoparticle.jpg")
st.image(image,caption="nanoparticle in a polymer matrix, distribution diagram of nanoparticle")
st.image("https://editor.analyticsvidhya.com/uploads/210362021-07-18%20(2).png",caption="artificial neural network")
#st.write("For more information, please read this article: ")
#st.link("https://pubs.rsc.org/en/content/articlelanding/2023/sm/d3sm00567d/unauth",caption="nanoNET: machine learning platform for predicting nanoparticles distribution in a polymer matrix")
st.write("For more information, please read this article:  [nanoNET: machine learning platform for predicting nanoparticles distribution in a polymer matrix](https://pubs.rsc.org/en/content/articlelanding/2023/sm/d3sm00567d/unauth)")
if st.sidebar.button("Predict!"):
    st.subheader('User input parameter')
    col1, col2, col3 = st.columns(3)
    col1.metric("Interaction: polymer-NP", "{}".format(max(df['Po_NP'])))
    col2.metric("Interaction: NP-NP", "{}".format(max(df['NP_NP'])))
    col3.metric("Diameter of NP", "{}".format(max(df['D_aim'])))
    col4,col5=st.columns(2)
    col4.metric("Number of NP", "{}".format(max(df['Phi'])))
    col5.metric("Length of polymer chain", "{}".format(max(df['Chain length'])))
    st.metric("Distance range: ", "From {} nm to {} nm".format(min(df['distance']),max(df['distance'])))
    #st.write("""
            #Interaction between polymers and nanoparticles: {}\n
            #Interaction between nanoparticles and nanoparticles: {}\n
            #Diameter of nanoparticles: {}\n
            #Number of particle: {}\n
            #Length of polymer chain: {}\n
            #Distance range: {} - {} nm\n
             #""".format(max(df['Po_NP']),max(df['NP_NP']),max(df['D_aim']),max(df['Phi']),max(df['Chain length']),min(df['distance']),max(df['distance'])))
    # Load the model
    model = joblib.load('/mount/src/app/model.pkl')

    # Print the type and structure of the loaded object

    predictions1=model.predict(df)
    st.write('max predicts: ',max(predictions1))
    st.write('min predicts: ',min(predictions1))
    st.subheader('Prediction')

    fig, ax = plt.subplots()
    ax.scatter(df['distance'],predictions1)
    ax.set_xlabel('distance')
    ax.set_ylabel('density')
    ax.set_title('Prediction')

    # Display the plot in Streamlit
    st.pyplot(fig)

    # -- Allow data download
    download = df
    df = pd.DataFrame(download)
    """csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    fn =  str(max(df['Po_NP']))+' - ' +str(max(df['NP_NP']))+str(max(df['D_aim']))+str(max(df['Phi']))+str(max(df['Chain length']))+' - '+str(min(df['distance']))+'/'+str(max(df['distance'])) + '.csv'
    href = f'<a href="data:file/csv;base64,{b64}" download="{fn}">Download Data as CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)"""
    st.download_button(
        "Download data csv",
        download,
        file_name=fn, 
        mime='text/csv',
        on_click= st.write("Successful download data!")
    )
