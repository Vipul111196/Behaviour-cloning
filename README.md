# Behavioral Cloning

## Introduction 

- The objective of this project is to create a model for an autonomous car that can drive using the data collected from a human driving behavior. 
- The data collected consists of images from three different cameras mounted on the vehicle along with the corresponding steering angles.
- To achieve this, a Convolutional Neural Network (CNN) has been developed using the Keras library, which provides a high-level API for deep learning networks, with Tensorflow as the backend.

## Project files description

There are several files included in this repository:

- **drive.py**: a Python script used to drive the car autonomously, it receives images as input for the CNN and sends back the predicted steering angle and speed.
- **Behavioral_Cloning_main.ipynb**: a python notebook to create and train the CNN, and output the trained model.
- **model.h5**: contains the trained convolutional network.
- **video.py**: a script to create a video of the autonomous vehicle.

To use drive.py, the trained model must be saved as an h5 file, i.e. model.h5. This can be done using the "model.save(filepath)" command. Then, the model can be used with drive.py by running "python drive.py model.h5". The predictions made by the model will be sent back to the server via a websocket connection.

The video of the autonomous agent can be saved using "python drive.py model.h5 run1". The images seen by the agent will be saved in the specified directory, in this case "run1". The video.py script can then be used to create a video from these images, "python video.py run1". The FPS of the video can be specified as an optional argument.

## Project Aim:

The aim of the project was to train a Deep Network to replicate human steering behavior, allowing the vehicle to drive autonomously on the simulator provided by Udacity. The network inputs image data from the front camera and predicts the steering direction at each moment.

## Major steps:

The steps involved in the project are:

- **Data Collection**: Collecting driving data using the Udacity simulator in training mode.
- **Data Processing**: Analyzing, augmenting and processing the collected images.
- **Setting up a Neural Network**: Designing, training and validating a model that predicts steering angles from image data. Model is inspired from the following Nvidia model (https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
- **Training and Validation**: Using the model to drive the vehicle autonomously around the track, keeping the vehicle on the road for an entire loop.

### Data Collection
Training data is gathered from three simulated dashboard cameras with different angles of view. Our training dataset comprised of a mixture of Udacity's data and our manually collected data. The total training dataset entered was 8628 datapoints. Data collected in training mode include three camera images matched with steering angle, throttle, brake and speed. Data is gathered by manually driving the simulated vehicle. At each point in time, the simulated dashboard cameras generate and store three simultaneous images demonstared below:

<img src="output_images/left_2016_12_01_13_31_12_937.jpg" width="250" align="center" /> <img src="output_images/center_2016_12_01_13_31_12_937.jpg" width="250" align="center" /> <img src="output_images/right_2016_12_01_13_31_12_937.jpg" width="250" align="center" />


To include all three cameras, an angle correction was used. This is explained best in the image below. A correction of negative .15 and positive .15 was added to the steering angle corresponding to left and right images to enable us to utilises all three cameras. A fixed correction of .15 was applied as it yielded the best results. 

<p align="center">
    <img src="output_images/carnd-using-multiple-cameras.png" width="300" width="250" height="200" align="center" />
</p> 

The data captured was very biased towards a steering angle of 0 as we might expect given the stretches of straght roads. The hitogram below demonstrates that. Although not done in this project, we could have limited datapoints with 0 steering angle to prevent the network being overly biased towards 0 and to support generalization.

<p align="center">
    <img src="output_images/output_1_0.png" width="250" width="300" height="200" align="center" />
</p> 

### Data Processing
We employ two image preprocessing methods that support a more generalised CNN, seamlessly increase the dataset size and accelerate network training. These methods include:

* Normalization
* Image Flipping
* Lambda Cropping

Normalization was used to achieve a well-conditioned problem with zero mean and equal variance. Each pixel was divided by 255 and 0.5 was subtracted so that the range was between -.5 and +.5.

TTo account for bias in the dataset and double our training data, we horizontally flipped each image and corresponding steering angle (i.e. multiply by -1). The images below demonstrate normal and flipped images from the center (top), left (middle) and right (bottom) cameras.

<p align="center">
    <td> <img src="output_images/output_2_1.png" alt = "Drawing" style = "width: 100px;"/> </td>
    <td> <img src="output_images/output_2_2.png" alt = "Drawing" style = "width: 100px;"/> </td>
    <td> <img src="output_images/output_2_3.png" alt = "Drawing" style = "width: 100px;"/> </td>
</p> 

Examining the camera images further, we can see that a large proportion of the top half of the image includes sky, trees etc. This is largely irrelevant for our CNN which should focus on road marking, road texture, lane lines etc. As a result, we removed the top half from each image. We also removed the bottom portion as the vehicle bonnet is also an irrelevant feature. Example copeed images are shown below:

<p align="center">
    <img src="output_images/output_3_1.png" alt = "Drawing" style = "width: 750px;"/> 
</p>

### Setting up a Neural Network
In this project we employed NVIDIA's CNN architecture shown in the image below. The same network is used for both steering and velocity predictions. It comprises of nine layers. This includes a normalization layer at the beginning, five convolutional layers and finally three fully connected layers. Relu activation function is used for each convolutional layer. The first three convolutions use a 2x2 stride and 5x5 kernels whereas the last two use 3x3 kernels, also shown in the image below.

<p align="center">
    <img src="output_images/CNN-Architecture.png" width="400" height="400" align="center" />
</p>

## 3. Training and Validation

Given the size of the dataset, we employed `fit_generator` from the Keras library. We use a Python generator to generate data for training insteading of storing the training data. This is was a necessity for our networks. We use the adam optimizer which is similar to stochastic gradient descent with a learning rate of .001.

To monitor and prevent overfitting, the dataset was split with 80% as training and 20% as validation. To prevent overfitting the model, we monitored the validation accuracy changes over each epoch. Evidently, the validation loss began to rise at epoch 4 so we settled with 3 epochs. This is shown in the figures below:

<p align="center">
    <img src="output_images/test_val_acc2.png" width="250" align="center" />
</p>

Additionally, we could have included dropout layers in our network to support generalization. However, for the purpose of this project, this network sufficed.

## 4. Results
***
The video files for track 1, 2 and 3 are provided above. The vehicle successfully navigated each track with minor incidents such as swerving or abnomrmal braking.

The command line to begin the simulation is:

`python drive_new.py model_nvid_angle.h5 model_nvid_speed.h5`

