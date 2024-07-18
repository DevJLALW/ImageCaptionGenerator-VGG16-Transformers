# ImageCaptionGenerator-VGG16-Transformers
# WIP

This project aims to generate descriptive captions for images using a combination of Convolutional Neural Networks (CNN) and Transformers. Specifically, we use a pre-trained VGG16 model for extracting image features and a Transformer-based model for generating captions.

# Dataset
The project utilizes the Flickr8k dataset, which contains 8,000 images along with five captions per image. Additionally, GloVe embeddings are used for initializing the word embeddings in the caption generator.

Flickr8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k
GloVe Embeddings: https://www.kaggle.com/datasets/rtatman/glove-global-vectors-for-word-representation

# Model Architecture
Feature Extraction using VGG16
We use the VGG16 model pre-trained on ImageNet to extract features from images. The last fully connected layer is removed, and the second last layer's output is used as the feature representation.

# Caption Generation using Transformers
We implement a Transformer model for generating captions from the extracted image features and tokenized captions.

# Steps to Train the Model

# Extract Image Features:
Use the VGG16 model to extract features from each image in the Flickr8k dataset.
Save the extracted features for later use.

# Load Captions:
Read the captions from the provided captions.txt file.
Preprocess and tokenize the captions.

# Prepare GloVe Embeddings:
Load the GloVe embeddings and create an embedding matrix for the tokenizer vocabulary.

# Define the Model Architecture:
Use a Transformer model to process the tokenized captions.
Combine the image features and caption embeddings to generate the final output.


# Train the Model:
Train the model using the extracted image features and tokenized captions.
Save the trained model for later inference.
Generating Captions
After training, use the trained model to generate captions for new images by following these steps:

# Load the Trained Model:
Load the saved model and tokenizer.

# Extract Image Features:
Use the VGG16 model to extract features from the new image.

# Generate Caption:
Use the trained Transformer model to generate a caption from the extracted features.
