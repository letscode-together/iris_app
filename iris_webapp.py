import streamlit as st
import joblib
import numpy as np
st.set_page_config(page_title="Iris Identification")
st.title("Iris Flower Classification")
st.image("iris_type.png", caption="Iris Image")
def get_model():
    return joblib.load('iris.joblib')
spl = st.text_input("Enter Sepal Length: ","")
spw = st.text_input("Enter Sepal Width: ","")
pel = st.text_input("Enter Petal Length: ","")
pew = st.text_input("Enter Petal Width: ","")
# sepal length, sepal width, petal length, petal width
if st.button('Classify'):
    values = [spl,spw,pel,pew]
    flaot_values = []
    for x in values:
        flaot_values.append(float(x))
    input_values = np.asarray(flaot_values).reshape(1,-1) # transform 2D
    mymodel = get_model()
    pred = mymodel.predict(input_values) # [0]
    pred = int(pred[0])
    if pred == 0:
        st.write("Your flower is Serosa")
    elif pred == 1:
        st.write("Your flower is Versicolor")
    elif pred == 2:
        st.write("Your flower is Virginica")
    
    
        