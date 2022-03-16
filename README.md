# AI-Chatbot

## Chatbot-model.py
This python file performs the necessary pre-processing on the FAQs and converts them to machine readable **One Hot Encoded** data for the neural network model which is implemented using Tensorflow.

## Intents.json
Contains the .json file of intents for the AI Chatbot which will answer FAQ.


## Libraries required
> All installation codes are for Python 3.X
### Numpy
Numpy library will be required to convert data into machine feedable arrays.
Install numpy by running the following code in terminal:
`pip3 install numpy`

### Tensorflow
In order to build our neural network, tensorflow will play a vital role,
since it provides a user friendly way to construct neural networks based
on our requirements.
Install tensorflow by running the following code in terminal:
`pip3 install tensorflow`

### NLTK
For sentence and word stemming, NLTK module will be used.
Install NLTK by running the following code in terminal:
`pip3 install nltk`

### Pickle
Pickle is used to save the final neural network model and the input in a file, which can be used
instead of rerunning the model everytime prediction needs to be made for a given input.
>Need not be installed, comes pre installed with Python3.x


