# Image Classification App with TensorFlow and Streamlit

This application is an image classification tool built using TensorFlow and Streamlit. It leverages a Convolutional Neural Network (CNN) model to classify images into 10 categories based on the CIFAR-10 dataset. Users can upload images for classification, view prediction results, and explore the training history of the model.

## Features
- **Image Classification**: Classifies uploaded images into one of the 10 categories from the CIFAR-10 dataset: 
  - Airplane
  - Automobile
  - Bird
  - Cat
  - Deer
  - Dog
  - Frog
  - Horse
  - Ship
  - Truck
- **Model Training**: Automatically trains a CNN model using the CIFAR-10 dataset, saving both the model checkpoints and training history.
- **Training History**: Visualizes training and validation accuracy and loss over time via interactive charts.
- **Prediction Confidence**: Displays the predicted class along with the confidence score of the model's prediction.

## Technologies Used
- **TensorFlow**: For building, training, and evaluating the CNN model.
- **Streamlit**: For creating the interactive web interface.
- **Pandas**: For managing and saving training history data.
- **NumPy**: For handling numerical operations and data manipulation.
- **PIL**: For image preprocessing and manipulation.

## Installation

Follow these steps to set up the application on your local machine:

### 1. Clone the repository:
   ```bash
   git clone https://github.com/arrkpong/Image-Classification-Convolutional-Neural-Network-CNN.git
   ```
2. Navigate into the project directory:
   ```bash
   cd Image-Classification-Convolutional-Neural-Network-CNN
   ```
3. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Alternative: Install with uv

This repository supports [uv](https://docs.astral.sh/uv/), a fast Python package manager. To get started with uv:

1. Install uv (see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Create a virtual environment and sync dependencies:
   ```bash
   uv venv
   uv sync
   ```

3. Add new dependencies (optional):
   ```bash
   uv add <package-name>
   ```

4. Run the application with uv:
   ```bash
   uv run streamlit run main.py
   ```

Running the Application
To start the application, run the following command:
   ```bash
   streamlit run app.py
   ```
Once the app starts, you can upload an image and the model will classify it. If the model has already been trained, it will use the trained weights to predict the class of the uploaded image.

## Model Training
The model is trained using the CIFAR-10 dataset, which is automatically loaded during initialization.
The CNN architecture includes:
Conv2D layers: For feature extraction from the input images.
BatchNormalization: For faster convergence and reducing overfitting.
LeakyReLU: For non-linear activations.
MaxPooling: For downsampling and reducing dimensionality.
Fully connected Dense layers: For classification of the extracted features.
The model is trained using the Adam optimizer with categorical crossentropy loss, and the training process is saved to a CSV file for easy access.

## Model Checkpoints
The app saves model checkpoints during training to prevent losing progress in case of interruption.
The best model (based on the lowest validation loss) is saved as model_checkpoint.keras for future predictions.
##Training History

The training history, including accuracy and loss for both the training and validation datasets, is saved as a CSV file (training_history.csv).
Interactive charts are displayed on the app to show the training and validation performance over time.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
CIFAR-10 Dataset: The CIFAR-10 dataset is widely used for training machine learning models on image classification tasks.
TensorFlow: An open-source platform for machine learning that powers the model in this app.
Streamlit: A framework for creating interactive web applications with minimal effort.

## Troubleshooting
Error during model training: Make sure your system has sufficient resources (memory and GPU support) to handle model training.
Image upload issues: The app currently supports only .jpg image formats. Please convert your images before uploading.
Training logs not appearing: If you're not seeing training logs, check the logs directory to ensure TensorBoard is logging correctly.
