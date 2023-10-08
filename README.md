Part 1: Image Classification using Bag-of-Words (50 pts)


1.1 Training Phase


1.2 Testing Phase

2.1 Feature Extraction and Description
* Q2.1: Extract SIFT descriptors from training datasets based on keypoints. Show two images from each of the five classes (draw the circles with the size of keypoints). (10-pts).`**  

2.2 Building Visual Vocabulary
* Q2.2 Building Visual Vocabulary (running kmeans)
2.3 Encoding Features Using Visual Vocabulary
2.4 Representing images by frequencies of visual words
* Q2.4: Representing images by frequencies of visual words (5-pts)
2.5 Classification
* Q2.5: Classification (5-pts)
2.6 Evaluation
* Q2.6: Evaluation and Discussion (30-pts)



Part 2: Image Classification using Convolutional Neural Networks

1. Image Classification on CIFAR-100
* Q1.1 Test dataloader and show the images of each class of CIFAR-100 (3-pts)
* Q1.2: Architecture understanding. Implement architecture of TwolayerNet and ConvNet (4-pts)
* Q1.3: Preparation of training. Create Dataloader yourself and define the transform function and optimizer. (8-pts)
* Complement Cifar100_loader() (2-pts)
* Complement Transform function and Optimizer (2-pts)
* Train the TwolayerNet and ConvNet with CIFAR100_loader, transform and optimizer you implemented and compare the results (4-pts)
* Q1.4: Setting up the hyperparameters (10-pts)
2. Finetuning the ConvNet
* 2.1 STL-10 Dataset
    * Q2.1 Create the STL10_dataset (5-pts)
* 2.2 Fine-tuning ConvNet
    * Q2.2 Finetuning from ConvNet (10-pts)
3. Bonus (optional): Extra points for rank in class leaderboard
