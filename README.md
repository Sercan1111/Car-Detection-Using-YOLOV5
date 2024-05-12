# Car Detection using YOLOv5

## Overview
This project focuses on detecting cars in images using the YOLOv5 object detection model. The repository contains all the necessary scripts and instructions to train a YOLOv5 model on a custom dataset of car images. This project leverages Python and utilizes libraries such as PyTorch, OpenCV, and Matplotlib for processing and analyzing images.

## Dataset
The dataset consists of images containing cars with corresponding bounding boxes that define the location of each car in the images. The data is split into training and validation sets to ensure the model can generalize well to new, unseen images.

## Preprocessing Steps
- Loading and visualizing data: Images are loaded and displayed with bounding boxes to ensure correct labeling.
- Splitting data: Images are split into training and validation datasets.
- Normalization and transformation: Images are resized and normalized to fit the input requirements of YOLOv5.

## Model Configuration
- YOLOv5 is configured to detect only one class (car).
- The model architecture is adapted from the YOLOv5 repository to suit the single-class detection.

## Training the Model
- The model is trained using the custom dataset with specified hyperparameters.
- Training progress can be monitored using TensorBoard.

## Files in the Repository
- `train_solution_bounding_boxes.csv`: Ground truth bounding boxes for training images.
- `sample_submission.csv`: A sample submission file in the correct format.
- `train.py`: Script to train the YOLOv5 model.
- `detect.py`: Script for detecting cars in new images using the trained model.
- `requirements.txt`: Lists all the dependencies to run the scripts.

## How to Run
1. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
2- Run the training script:
python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --cfg models/yolov5l.yaml --weights yolov5l.pt --name yolov5l_car_results
3-To detect cars in new images:
python detect.py --weights runs/train/yolov5l_car_results/weights/best.pt --img 640 --conf 0.25 --source data/images/

#Results
-The trained model achieves high accuracy in detecting cars with precision and recall values over 95%. These metrics were validated using the validation dataset included in the training process.

# Future Work
-Expanding the dataset to include more varied conditions like night-time images or different weather conditions.
-Experimenting with different YOLOv5 models like YOLOv5s or YOLOv5m for faster performance or higher accuracy.
-Implementing additional features like car model and color recognition.

#Contributions
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate.

