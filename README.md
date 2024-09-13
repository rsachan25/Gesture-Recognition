# Gesture Recognition for Smart Televisions
## Project Overview:
This project focuses on creating a **gesture recognition system** for smart televisions, utilizing deep learning to provide an intuitive and contactless user experience. The system allows users to control their television sets using hand gestures captured by the webcam, eliminating the need for traditional remote controls.

The system aims to recognize five key hand gestures:
- **Thumbs up** (e.g., increase volume)
- **Thumbs down** (e.g., decrease volume)
- **Left swipe** (e.g., navigate backward)
- **Right swipe** (e.g., navigate forward)
- **Stop** (e.g., pause playback)
## Objectives:
**1. Data Preprocessing:**
   
- Ensure that the generator can efficiently preprocess batches of videos. Tasks include cropping, resizing, and normalizing the input video frames.
  
**2. Model Development:**
  
- Create a model capable of training without errors.
- Focus on minimizing the number of parameters to reduce inference time and enhance real-time performance.
- Train the model on a small dataset initially, before scaling to the complete dataset.
  
**3. Write-Up:**
  
- Provide a comprehensive description of the base model and the process of its evolution through various modifications and experiments.
- Justify every design decision, explaining the selection of metrics and hyperparameters, and how these were used to optimize the model's performance.
## Dataset:
The dataset used in this project comprises hundreds of short videos, each containing sequences of 30 frames. These frames are labeled based on the specific gesture performed in the video. The dataset is categorized into five classes representing the five different gestures.

## Generator:
The generator plays a key role in handling video data by reading and preprocessing video sequences, normalizing them, and ensuring that batches are created properly for both training and validation. It also incorporates data augmentation techniques during training to improve generalization and prevent overfitting.

The primary functions of the generator include:

- **Reading video sequences** from the dataset.
- **Resizing video frames** to a consistent resolution (120x120 pixels) for input into the model.
- **Normalization** of pixel values to improve model convergence during training.
## Model Architecture:
This project utilizes deep learning models built with **Keras** to perform 3D convolutional operations on the video data.

Key Layers Used:
- **Conv3D:** Essential for capturing spatiotemporal features across video frames.
- **MaxPooling3D:** Reduces the dimensionality of the features, retaining the most relevant ones.
- **TimeDistributed Layers:** Enables the model to treat the sequence of video frames effectively when combined with recurrent layers like GRU or LSTM.
- **ConvLSTM2D:** Used in later experiments for sequential processing, effectively capturing both spatial and temporal patterns in the video data.
- **Dropout:** Incorporated to prevent overfitting by randomly dropping some neurons during training.
## Key Experiments:
**Experiment 1:**

- Model: Conv3D
- Batch Size: 20, Epochs: 15
- Results: Train accuracy of 1.00 but a validation accuracy of 0.77, suggesting overfitting.
- Solution: Additional layers were added in subsequent models to improve generalization.
  
**Experiment 4:**

- Model: Conv3D
- Batch Size: 10, Epochs: 30
- Results: Validation accuracy of 0.90, one of the best performances with over 700k parameters.

**Experiment 11:**

- Model: Time Distributed + ConvLSTM2D
- Batch Size: 15, Epochs: 20
- Results: Validation accuracy of 0.78 with only 13,589 parameters, making this the most efficient model in terms of performance and size.
  
**Final Model:**

The best-performing model combines **ConvLSTM2D** with **TimeDistributed layers**, allowing it to capture both spatial and temporal features efficiently. This model is lightweight (only 226KB) and performs well on validation data, making it ideal for deployment in real-time systems like smart televisions.

## Training Process:
**GPU Utilization:**

- Maximize GPU utilization by experimenting with different batch sizes and monitoring performance. Start with a lower batch size and gradually increase it until the GPU is fully utilized without causing errors.
  
**Hyperparameter Tuning:**

- Parameters like learning rate, batch size, and the number of epochs were carefully tuned through multiple experiments to achieve optimal performance.
  
**Metrics:**

- The primary metrics used to evaluate the model's performance are accuracy and loss on both training and validation datasets.
- Regular checks for overfitting were conducted, leading to the addition of Dropout layers in later experiments.
## How to Run the Project:
**Environment Setup:**

- Ensure you have **TensorFlow, Keras**, and other dependencies installed.
- This project is optimized for GPU usage, so ensure a compatible GPU is available and configured correctly.
  
**Data Preprocessing:**

- Use the provided generator to preprocess video data before feeding it into the model. Ensure that the dataset is properly structured into training and validation folders.
  
**Model Training:**

- Run the training script to start model training. You can modify hyperparameters such as learning rate, batch size, and the number of epochs to see their effect on performance.
  
**Evaluation:**

- The final model can be evaluated on validation data to assess its real-time performance.
## Conclusion:
This project successfully demonstrates the implementation of a gesture recognition system for smart televisions using deep learning models. The optimal model leverages ConvLSTM2D, achieving a balance between accuracy and computational efficiency. With further fine-tuning, the model is ready for real-world deployment in smart television systems, offering users a seamless and intuitive control interface.
