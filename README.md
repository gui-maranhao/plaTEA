# EmotionRAM - Faces

![header](front.gif)

This repository contains a minimal working example of one of our experiments from EmotionRAM for facial expression recognition. This means that we are focusing entirely on facial expressions in this case, without concern for context or body language.

This demo is based on EfficientNet-B0 architecture and was trained on AfeW. So, for my fellow LatinX colleagues, some problems with fairness are expected, as AfeW is not a culturally representative dataset. We will be publishing our dataset for brazilian emotion recognition soon to solve this limitation.

## Usage

Refer to the installinstructions.txt file for the commands to install the packages. It is recommended to use an Anaconda environment to configure a virtual environment. Afterwards, just run `webcam.py` from the terminal or other IDE. You can choose the webcam ID and the device (CPU or GPU) by using the --cameraID and --device parameters.

Similarly, inference.py allows you to predict emotions on a video stored locally. This file looks for an `input.mp4` into the folder and saves the video as `output.mp4`. Please notice that you need to install [ffmpeg](https://ffmpeg.org/download.html) for this to work properly.

## Setup Instructions

```Bash
# Create and activate a Conda environment
conda create -n emotionram python=3.8 -y
conda activate emotionram

# Install all the required Python packages
pip install -r requirements.txt
```

### Running the Application

To run real-time analysis from your webcam:

```Bash
# Use your GPU (MPS on Mac, CUDA on others)
python webcam.py --device gpu

# Or force it to use CPU
python webcam.py --device cpu

# To use a different camera, specify its ID
python webcam.py --cameraID 1
```

To run inference on a local video file:

Place your video in the main project folder and name it input.mp4.

Run the script:

```Bash
python inference.py
```
