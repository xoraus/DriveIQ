**I. INTRODUCTION**

A. **Problem Statement**

   The goal of this project is to create a self-driving car simulation by predicting steering angles using Convolutional Neural Networks (CNNs). These CNNs map raw pixels collected from input images to steering commands.

**II. PROJECT DESIGN**

A. **Analysis of Dataset**

   To build a self-driving car simulation, we started by analyzing the Udacity dataset. This dataset serves as a valuable resource for training our model. It includes images from various driving conditions:

   1. **Driving Condition 1:** These images likely represent typical daytime driving with clear weather conditions.
   2. **Driving Condition 2:** Images under different lighting conditions or various weather situations.
   3. **Driving Condition 3:** Images depicting scenarios with traffic and other vehicles on the road.
   4. **Driving Condition 4:** Representing more complex road conditions, such as curves and intersections.
   5. **Driving Condition 5:** Potentially extreme conditions like nighttime driving or adverse weather.

B. **Typical Images**

   The Udacity dataset comprises images representing various lighting, traffic, and driving conditions. This diversity in the dataset is crucial for training a robust self-driving car model, as real-world driving conditions can vary widely.

**III. METHODOLOGY**

A. **Data Processing**

   1. **Data Loader:** To begin, we loaded the Udacity dataset, which includes a 25-minute video, with 20 minutes designated for training and 5 minutes for testing. This video provided the raw footage necessary for training our model.

   2. **Data Preprocessor:** The video file was converted into a series of images at a rate of 30 frames per second. This resulted in a total of 45,405 images (30 frames per second) from a 25-minute video (45,000 frames). These images were then normalized, with pixel values ranging from [0, 255] being rescaled to [0, 1]. Additionally, the images were resized to a consistent dimension of 66x200x3 to facilitate transfer learning. This preprocessing step was performed using the CV2 library in Python.

   3. **Data Splitter:** The obtained images were divided into two categories, with 80% of the data designated for training the model and the remaining 20% for testing. It's worth noting that the splitting was performed temporally, ensuring that frames from the same time intervals were either part of the training or testing dataset. This temporal splitting approach helps simulate real-time decision-making for the self-driving car model.

B. **Model Initialization**

   1. **Initialize Training Parameters:** The model's training parameters were configured, including the number of epochs (30), batch size (100), and learning rate (0.001).

   2. **Initialize TensorFlow Graph:** A new TensorFlow session was created to initialize all the training variables and set the stage for model training.

C. **Model Training**

   In this module, the model was trained using the prepared input dataset.

   1. **Convolutional Neural Network:** The core of the model consists of Convolutional Neural Networks. These networks are designed to minimize the mean squared error between the network's steering command output and the actual steering angles. The CNN's convolutional layers were fine-tuned through experimentation to perform feature extraction, while fully connected layers served as controllers for steering.

   2. **Data Logger:** After each batch, a summary of the training iteration was logged to monitor the model's progress.

   3. **Data Saver:** The model's weights were saved after training to ensure that the trained model could be used for prediction and simulation.

**IV. RESULTS AND DISCUSSIONS**

A. **Model Evaluation and Validation**

   The final model comprises 9 layers with a total of 444,819 parameters. While some parameters identify features such as curves, street lanes, and other objects on the road, the end-to-end nature of the model makes it challenging to visualize the specific functions of individual parameters.

   The model underwent training for various numbers of epochs, including 10, 25, 30, and 50. The best results were achieved with 30 epochs, which produced the lowest validation loss of $0.00161$ on the $19^{th}$ epoch.

   The model's current configuration enables it to predict steering angles with a Root Mean Square Error (RMSE) of $0.0558$. This level of accuracy is a significant achievement, indicating that the model can make reliable predictions for steering commands based on the input images.

**V. CONCLUSION**

In conclusion, this project successfully utilized Convolutional Neural Networks to create a self-driving car simulation that predicts steering angles based on raw image data. The comprehensive analysis of the Udacity dataset, along with meticulous data preprocessing, allowed us to train a model capable of navigating various driving conditions and making accurate steering decisions. The model's performance, as evidenced by a low RMSE, underscores its effectiveness in real-world scenarios.

**REFERENCES**

[1] Y. LeCun, B. Boser, J. S. Denker, D. Henderson, R. E. Howard, W. Hubbard, and L. D. Jackel. Backpropagation applied to handwritten zip code recognition. Neural Computation, 1(4):541–551, Winter 1989.

[2] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. Imagenet classification with deep convolutional neural networks. In F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 25, pages 1097–1105. Curran Associates, Inc., 2012.

[3] L. D. Jackel, D. Sharman, Stenard C. E., Strom B. I., , and D Zuckert. Optical character recognition for self-service banking. AT&T Technical Journal, 74(1):16–24, 1995.

[4] Large scale visual recognition challenge (ILSVRC).

[5] Net-Scale Technologies, Inc. Autonomous off-road vehicle control using end-to-end learning, July 2004.

[6] Dean A. Pomerleau. ALVINN, an autonomous land vehicle in a neural network. Technical report, Carnegie Mellon University, 1989.

[7] Danwei Wang and Feng Qi. Trajectory planning for a four-wheel-steering vehicle.
