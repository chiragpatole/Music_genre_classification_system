# Music_genre_classification_system
In this project, a robust and accurate machine learning model for music genre classification is developed.
Convolutional Neural Networks (CNNs) are utilized to analyze audio data, with a focus on Mel spectrograms,
which are employed as visual representations of the audio's frequency content over time. By training the CNN
on a labeled dataset of music tracks, patterns and features characteristic of diﬀerent genres are learned by
the model. The ultimate goal is that new, unseen music tracks can be automatically categorized into their
respective genres. A user-friendly Streamlit web application is further developed to facilitate the use of this
trained model.

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
