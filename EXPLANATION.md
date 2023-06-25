# Explanation


## Technologies and features we used
* Python 3.9
* Pytorch
* Summarywriter (to create logs)
* Pre-trained models (alexnet)

## Training models
Starting from [main.py](main.py), [train.py](train.py), 
[test.py](test.py),[dataset.py](dataset.py),[model.py](model.py),
are part of the training progress. [config.py](config.py) contains
all the training settings, such as batch size, epochs and some more.
At the end of the epochs we create a .pth file to save the trained weights
and reuse them in our backend [backend2.py](backend2.py).

## Results
After training bunch of models we took out the top 5 best models. 
* 1st: AlexNet 30 Epochs (with cleaned dataset) = 59%
* 2nd: AlexNet 60 Epochs (with cleaned dataset) = 56%
* 3rd: AlexNet 30 Epochs (with UNCLEAN dataset) = 55%
* 4th: AlexNet 100 Epochs (with cleaned dataset) = 54%
* 5th: ResNet 30 Epochs (with cleaned dataset) = 54%

After testing out all the models manually and automatically, we found out that
all the models have different strenghts. Here is an example:

#### 1st: AlexNet (avg: 59% accuracy)
*   angry: 33%
*	happy: 81%
*	sad: 80%
*	relaxed: 43%

vs

#### 3rd: AlexNet (avg: 55% accuracy) 
*	angry: 46%
*	happy: 54%
*	sad: 61%
*	relaxed: 59%

We can see that both models have different strenghts. The first model is
good at the Happy and Sad emotions while the other AlexNet is a lot better
with the Angry and Relaxed emtions.

## Improving Accuracy 
After we found out the different strenghts of 
the models, we tried to improve our highest accuracy (59%).
The first thing we did was a voting system. So every model would predict
a emotion. The prediction which occurs most often would be the final
prediction. With this approach we could get an accuracy increase by
1,5%. 

#### Ensemble: AlexNet1,AlexNet2,AlexNet3,AlexNett4,ResNet: (avg: 60,5% accuracy)
*	angry: 46%
*	happy: 73%
*	sad: 75%
*	relaxed: 48%

but then with voting + calculating avg confidence sums we got the best restults

##### Ensemlbe + Confidence Values:AlexNet1,AlexNet2,AlexNet3,AlexNet4,ResNet: (avg: 62% accuracy)

*	angry: 50%
*	happy: 72%
*	sad: 76%
*	relaxed: 50%

## Scrips to manually test images

### First script
[ensembling.py](ensembling.py) 
It calculates the average confidence values for each model and 
determines the final prediction based on the highest average 
confidence value. It tracks the total number of correct predictions
and the total number of images processed to calculate accuracy.

### Second script
[final_man_ensembled_weighted_voting.py](final_man_ensembled_weighted_voting.py)
 uses a dictionary, model_performance, to track the performance 
of each model separately. For each model, it tracks the number 
of correct predictions, total predictions, sum of correct 
confidence values, and sum of incorrect confidence values. 
It calculates accuracy based on the total number of correct 
predictions and total images processed. It also tracks the average
confidence values for correct and incorrect predictions for each 
model.

## Backend
We have two Backends. The first [backend.py](backend.py) is using
the best AlexNet to predict the emotions. Our second [backend2.py](backend2.py)
is using the voting + avg confidence values implementation to predict
the emotions. 


