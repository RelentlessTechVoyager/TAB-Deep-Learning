# TAB-Deep-Learning
This repo consists of work related to projects done under Deep Learning domain of Technical Advisory Board (TAB-2024) of TRC.

## Minor Project
### Title- Taxi Fare Prediction 
**Objective**: To develop a machine learning model for predicting taxi fares based 
on input features such as trip duration, distance traveled, number of 
passengers, base fare, tips, miscellaneous fees, and surge pricing.
<br>Our goal was to create a regression model capable of estimating taxi fares 
accurately. After data preprocessing, we used the Scikit-Learn library to train 
and evaluate the model. Key evaluation metrics included Mean Absolute Error 
(MAE), Mean Squared Error (MSE), and the R-squared score
Significant factors of Taxi Fare: 
- Distance travelled,
- Duration,
- No. of passengers,
- Fare,
- Tips,
- Misc. Fee,
- Surge applied

## Major Project
### Title- Music Genre Classification
**Objective**: Develop a deep learning model to classify music tracks into genres using raw audio data, enabling accurate and automated categorization. This improves music recommendation systems, cataloging, and playlist generation by capturing complex audio patterns.<br>
<br> **Approach**:To develop a robust and accurate deep learning model for music genre classification, we will follow a structured approach consisting of data preparation, model design, training, evaluation, and fine-tuning. Use the GTZAN dataset, which includes 1000 audio tracks evenly distributed across 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock). Load audio tracks using a library such as Librosa. Convert raw audio into Mel-spectrograms to capture time-frequency representations. Accept Mel-spectrogram or MFCC images. Stack multiple convolutional layers with ReLU activation functions. Use Max Pooling layers to reduce dimensionality while retaining essential features. Apply after each convolutional layer 
to normalize outputs and accelerate convergence. Add LSTM layers to capture sequential dependencies and 
temporal patterns in the audio data. Optionally use bidirectional LSTMs to capture dependencies in both 
forward and backward directions. Use fully connected layers to integrate features from CNN and LSTM layers. 
Use a softmax activation function to output probability distributions over the 10 genres. Analyze the confusion 
matrix to understand misclassifications and identify areas for improvement. Use TensorFlow or PyTorch for 
building and training the neural network models. Use Librosa for audio processing and feature extraction. Use 
Scikit-learn for evaluation metrics and cross-validation. Deploy the trained model as a web service or integrate 
it into a music application for real-time genre classification.
