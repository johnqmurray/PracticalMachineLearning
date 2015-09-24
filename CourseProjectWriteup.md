# Course Project Writeup
##  Practical Machine Learning, September 2015 
### John Q. Murray (johnqmurray@gmail.com)

# Overview

The course project was to create a machine learning algorithm using the dataset from Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. 
Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. Velloso et. al. tackled an interesting problem. "In activity recognition using on-body sensing, a large body of work has investigated automatic techniques to discriminate which activity was performed.
So far, only little work has focused on the problem of quantifying how (well) an activity was performed. We refer to the latter as 'qualitative activity recognition.'" As described in their paper, they learned that they could effectively model correct and incorrect activity using 17 data points from on-body sensors. 

The Course Project involved predicting the "classe" values of a predefined Test set of 20 samples, after building a model using a Training set of 19622 samples from this Velloso data. The "classe" variable represented a categorical value indicating the person's qualititative performance 
of the Unilateral Dumbbell Biceps Curl: "exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E)."

# Building the Model

The caret package in R is unbelievably powerful and contains advanced modeling methods. The approach was to explore the data and then try several of the modelling functions,
with minimal processing of the data, to see which approach might be the most promising, and then fine-tune the most promising modelling approach. 

The Velloso research paper suggested the possibility of identifying the 17 most important variables among the datapoints provided in the Train and Test datasets. It was not possible to positively identify these variables based on the documentation, but examining the data demonstrated that much of the Test data was Not Available (NA). 
To reduce the size of the dataset and improve performance of the caret modeling functions, the training and the testing datasets were processed identically, pared down to only those variables in the test dataset that contained valid data. This was larger than 17, but reduced the number of data points from 160 to 50.

```{r}
library(caret);
inTrain <- read.csv("pml-training.csv", na.strings="NA");
inTest <- read.csv("pml-testing.csv", na.strings="NA");
View(inTest)
# data in most columns in the Test dataset appear as blank or NA; get rid of index to start
inTrain <- inTrain[,-1]
inTest <- inTest[,-1]
# preserve only those data points that have actual data in the Test set
inTrainMinCols <- inTrain[,c(7:10, 36:48, 59:67, 83:85, 101, 115:123, 139, 150:159)]
# perform same processing on Training set and Testing set
inTestMinCols <- inTest[,c(7:10, 36:48, 59:67, 83:85, 101, 115:123, 139, 150:159)]
colnames(inTrainMinCols)

[1] "roll_belt"            "pitch_belt"           "yaw_belt"             "total_accel_belt"     "gyros_belt_x"       
[6] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"         "accel_belt_z"       
[11] "magnet_belt_x"        "magnet_belt_y"        "magnet_belt_z"        "roll_arm"             "pitch_arm"          
[16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"        
[21] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"         "magnet_arm_y"       
[26] "magnet_arm_z"         "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
[31] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"  
[36] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"        "yaw_forearm"          "total_accel_forearm"
[41] "gyros_forearm_x"      "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"    
[46] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"     "magnet_forearm_z"     "classe"         
colnames(inTestMinCols)
# same except problem_id in place of classe
```

A standard CART approach was first tried using the RPART method. The classe value was compared against all other remaining variables. 
However, this approach produced accuracy of only about .50, equivalent to a coin flip:

```{r}
set.seed(1235)
myrpartfit <- train(classe ~ ., method="rpart", data=inTrainMinCols) 
myrpartfit$finalModel
n= 19622 

node), split, n, loss, yval, (yprob)
      * denotes terminal node

 1) root 19622 14042 A (0.28 0.19 0.17 0.16 0.18)  
   2) roll_belt< 130.5 17977 12411 A (0.31 0.21 0.19 0.18 0.11)  
     4) pitch_forearm< -33.95 1578    10 A (0.99 0.0063 0 0 0) *
     5) pitch_forearm>=-33.95 16399 12401 A (0.24 0.23 0.21 0.2 0.12)  
      10) magnet_dumbbell_y< 439.5 13870  9953 A (0.28 0.18 0.24 0.19 0.11)  
        20) roll_forearm< 123.5 8643  5131 A (0.41 0.18 0.18 0.17 0.061) *
        21) roll_forearm>=123.5 5227  3500 C (0.077 0.18 0.33 0.23 0.18) *
      11) magnet_dumbbell_y>=439.5 2529  1243 B (0.032 0.51 0.043 0.22 0.19) *
   3) roll_belt>=130.5 1645    14 E (0.0085 0 0 0 0.99) *
> 
19622 samples
   49 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E'

plot(myrpartfit$finalModel, uniform=TRUE, main="Classification Tree")
text(myrpartfit$finalModel, use.n=TRUE, all=TRUE, cex=.8)
```

The resulting Classification Tree did not even appear to select the D option! On reflection, this may have been due to use of the entire training set; perhaps the A values 
are bunched in the beginning of the training data. Another attempt could use createDataPartition to split the training data itself on the classe variable.

# Using Cross Validation (CV)

The Random Forests method () automatically incorporates cross-validation into the model. As Leo Breiman and Adele Cutler write (www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm): 
"In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the run, as follows:
Each tree is constructed using a different bootstrap sample from the original data." Because the RF method performs its own CV, the entire training dataset was used during the processing.

```{r}
set.seed(33833)
myRfFitMinCols <- train(classe~.,data=inTrainMinCols, method="rf", prox=TRUE);
Random Forest

No pre-processing
Resampling: Bootstrapped (25 reps)
Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ...
Resampling results across tuning parameters:
 
  mtry  Accuracy   Kappa      Accuracy SD  Kappa SD  
   2    0.9926805  0.9907386  0.001164525  0.001475985
  25    0.9927514  0.9908290  0.001199482  0.001516755
  49    0.9842237  0.9800374  0.004252516  0.005380508
 
Accuracy was used to select the optimal model using the largest value.
The final value used for the model was mtry = 25.

Call:
randomForest(x = x, y = y, mtry = param$mtry, proximity = TRUE)
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 25
```` 

# Calculating the Expected Out of Sample Error

The OOB value estimates the out of sample error rate. In this experiment, the initial run was surprisingly successful, with a rate less than 1 percent:

        OOB estimate of  error rate: 0.43%
Confusion matrix:
     A    B    C    D    E  class.error
A 5577    2    0    0    1 0.0005376344
B   20 3772    5    0    0 0.0065841454
C    0    9 3402   11    0 0.0058445354
D    0    0   24 3190    2 0.0080845771
E    0    0    4    6 3597 0.0027723870

If there had been issues with the data, the Random Forest would have been re-run using only the most important variables.
Breiman writes: "Another useful option is to do an automatic rerun using only those variables that were most important in the original run."
The varImp method lists the most important variables:

varImp(myRfFitMinCols)
rf variable importance
 
  only 20 most important variables shown (out of 49)
 
                     Overall
roll_belt            100.000
pitch_forearm         57.836
yaw_belt              56.543
pitch_belt            46.488
magnet_dumbbell_y     44.043
magnet_dumbbell_z     43.530
roll_forearm          43.100
accel_dumbbell_y      22.404
accel_forearm_x       17.782
magnet_dumbbell_x     17.257
roll_dumbbell         16.894
accel_dumbbell_z      14.760
accel_belt_z          14.513
magnet_forearm_z      13.947
magnet_belt_z         13.879
total_accel_dumbbell  13.605
yaw_arm               11.806
magnet_belt_y         11.602
gyros_belt_z          11.500
magnet_belt_x          9.782


# Predicting 20 Different Test Cases

The initial model fit using the Random Forest method was used to predict values in the Test dataset, as follows:

```{r}
myPredict <- predict(myRfFitMinCols$finalModel, newdata=inTestMinCols)
> myPredict
1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B
Levels: A B C D E
````

All 20 values were accepted as correct during the initial submission, so no further processing was performed. 

# Conclusion

Use caret! 
