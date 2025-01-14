import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize

#Function
def load_model():
    model = tf.keras.models.load_model("Trained_model.keras", compile=False)
    return model


#loading and preprocessing audio file
def load_and_preprocess_file(file_path,target_shape=(150,150)):
    data = []
    audio_data,sample_rate = librosa.load(file_path,sr=None)
    chunk_duration=4
    overlap_duration=2
    #convert duration to sample
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    #calculate the number of chunks
    num_chunks = int(np.ceil((len(audio_data)-chunk_samples)/(chunk_samples-overlap_samples)))+1

    #iterate 
    for i in range (num_chunks):
        start = i*(chunk_samples-overlap_samples)
        end = start+chunk_samples
        chunk = audio_data[start:end]
        spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                
        #Resize matrix based on target shape
        spectrogram = resize(np.expand_dims(spectrogram,axis=-1),target_shape)
        data.append(spectrogram)

    return np.array(data)

def model_prediction(X_test):
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements,counts = np.unique(predicted_categories,return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return  max_elements[0]

# Load the model globally
model = load_model()

##Streamlit UI

#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

## Main Page
if(app_mode=="Home"):
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(''' ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧''')
    image_path = "music_genre_home.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""
**Developed with passion and precision, this project represents the culmination of my journey in data science and deep learning. Dive into the world of AI-driven music analysis and experience the power of cutting-edge technology!**

### How It Works
1. **Upload Audio:** Go to the **Genre Classification** page and upload an audio file.
2. **Analysis:** Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. **Results:** View the predicted genre along with related information.

### Why This System Stands Out?
- **Accuracy You can trust:** Powered by state-of-the-art deep learning techniques and trained on the renowned GTZAN dataset.
- **Seamless Design:**  A user-friendly interface built with Streamlit, designed to make interaction intuitive.
- **Fast and Efficient:** Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the **Genre Classification** page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### Behind the Scenes

####This capstone project is entirely self-developed, showcasing expertise in:
- Audio Data Processing: Leveraging libraries like Librosa for extracting audio features.
- Model Development: Designing and training deep learning models tailored for genre classification.
-Interactive Deployment: Bringing everything together in a real-time web application.
""")



#About Project
elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Music has always been a universal language, yet understanding its intricate patterns and what differentiates one genre from another has been a challenge for decades. This project dives deep into the heart of sound analysis, aiming to uncover the unique characteristics that define different genres.

                By leveraging state-of-the-art deep learning techniques and visualizing audio as Mel Spectrograms, this system bridges the gap between sound and machine intelligence, offering insights into the fascinating world of music.


                ### About Dataset
                #### Content
                1. **genres original** - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. **List of Genres** - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. **images original** - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. **2 CSV files** - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                
                ### Why GTZAN?
                By leveraging state-of-the-art deep learning techniques and visualizing audio as Mel Spectrograms, this system bridges the gap between sound and machine intelligence, offering insights into the fascinating world of music.

                This project demonstrates how to transform raw audio into meaningful visualizations and predictive insights, showcasing the potential of AI in understanding and analyzing sound.
                """)

    

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
    if test_mp3 is not None:
            filepath = 'Test_Music/'+test_mp3.name
            

    #Show Button
    if(st.button("Play Audio")):
        st.audio(test_mp3)
    
    #Predict Button
    if(st.button("Predict")):
      with st.spinner("Please Wait.."):       
        X_test = load_and_preprocess_file(filepath)
        result_index = model_prediction(X_test)
        st.balloons()
        label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
        st.markdown("**:blue[Model Prediction:] It's a  :red[{}] music**".format(label[result_index]))

       