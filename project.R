# Coursera: Practical Machine Learning Project
# classify dumb-bell exercise errors by metrics from various devices

# Tried doing rpart, but when it finally returned, there was no object! 
# Because call to train was in a function and thus output is not returned!

library ("labeling", lib.loc="packages")
library ("digest", lib.loc="packages")
library ("ggplot2", lib.loc="packages")
library ("e1071", lib.loc="packages")
library ("caret", lib.loc="packages")

library ("randomForest", lib.loc="packages")

Pml <- read.csv ("data/pml-training.csv");
PmlTest <- read.csv ("data/pml-testing.csv");

set.seed (2112)

# X is the index of the data!
# Strip out X, user_name, 3 timestamps, 2 window vars.
PmlSub <- Pml[8:dim(Pml)[2]]

# Filter out skewness and kurtosis and other useless variables.
PmlSub <- PmlSub[,-grep("kurt", colnames (PmlSub))]
PmlSub <- PmlSub[,-grep("skew", colnames (PmlSub))]

# Remove the ones that don't look promising from the melt plot.
# subset technique via http://stackoverflow.com/questions/9845929/removing-a-list-of-columns-from-a-data-frame-using-subset
PmlSub <- subset( PmlSub, select = -c(amplitude_yaw_belt, var_yaw_belt, max_yaw_belt, min_yaw_belt, amplitude_roll_belt, amplitude_yaw_belt, stddev_yaw_belt, var_yaw_belt, max_yaw_dumbbell, min_yaw_dumbbell, amplitude_yaw_dumbbell, gyros_dumbbell_x, max_yaw_forearm, min_yaw_forearm, amplitude_yaw_forearm, gyros_forearm_y, gyros_forearm_z) )

# Remove the ones with 19612 NAs. Whoa, so many. No wonder the knn-impute 
# took so long!
PmlSub <- subset( PmlSub, select = -c(max_roll_belt, max_picth_belt, min_roll_belt, min_pitch_belt, amplitude_pitch_belt, var_total_accel_belt, avg_roll_belt, stddev_roll_belt, var_roll_belt, avg_pitch_belt, stddev_pitch_belt, var_pitch_belt, avg_yaw_belt, var_accel_arm, avg_roll_arm, stddev_roll_arm, var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm, max_roll_arm, max_picth_arm, max_yaw_arm, min_roll_arm, min_pitch_arm, min_yaw_arm, amplitude_roll_arm, amplitude_pitch_arm, amplitude_yaw_arm, max_roll_dumbbell, max_picth_dumbbell, min_roll_dumbbell, min_pitch_dumbbell, amplitude_roll_dumbbell, amplitude_pitch_dumbbell, var_accel_dumbbell, avg_roll_dumbbell, stddev_roll_dumbbell, var_roll_dumbbell, avg_pitch_dumbbell, stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell, stddev_yaw_dumbbell, var_yaw_dumbbell, max_roll_forearm, max_picth_forearm, min_roll_forearm, min_pitch_forearm, amplitude_roll_forearm, amplitude_pitch_forearm, var_accel_forearm, avg_roll_forearm, stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm) )


# Do all the same with PmlTest
PmlTestSub <- PmlTest[8:dim(PmlTest)[2]]

# Filter out skewness and kurtosis and other useless variables.
PmlTestSub <- PmlTestSub[ , -grep("kurt", colnames(PmlTestSub))]
PmlTestSub <- PmlTestSub[ , -grep("skew", colnames(PmlTestSub))]

# Remove the ones that don't look promising from the melt plot.
# subset technique via http://stackoverflow.com/questions/9845929/removing-a-list-of-columns-from-a-data-frame-using-subset
PmlTestSub <- subset( PmlTestSub, select = -c(amplitude_yaw_belt, var_yaw_belt, max_yaw_belt, min_yaw_belt, amplitude_roll_belt, amplitude_yaw_belt, stddev_yaw_belt, var_yaw_belt, max_yaw_dumbbell, min_yaw_dumbbell, amplitude_yaw_dumbbell, gyros_dumbbell_x, max_yaw_forearm, min_yaw_forearm, amplitude_yaw_forearm, gyros_forearm_y, gyros_forearm_z) )

# Remove the ones with 19612 NAs. Whoa, so many. No wonder the knn-impute 
# took so long!
PmlTestSub <- subset( PmlTestSub, select = -c(max_roll_belt, max_picth_belt, min_roll_belt, min_pitch_belt, amplitude_pitch_belt, var_total_accel_belt, avg_roll_belt, stddev_roll_belt, var_roll_belt, avg_pitch_belt, stddev_pitch_belt, var_pitch_belt, avg_yaw_belt, var_accel_arm, avg_roll_arm, stddev_roll_arm, var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm, max_roll_arm, max_picth_arm, max_yaw_arm, min_roll_arm, min_pitch_arm, min_yaw_arm, amplitude_roll_arm, amplitude_pitch_arm, amplitude_yaw_arm, max_roll_dumbbell, max_picth_dumbbell, min_roll_dumbbell, min_pitch_dumbbell, amplitude_roll_dumbbell, amplitude_pitch_dumbbell, var_accel_dumbbell, avg_roll_dumbbell, stddev_roll_dumbbell, var_roll_dumbbell, avg_pitch_dumbbell, stddev_pitch_dumbbell, var_pitch_dumbbell, avg_yaw_dumbbell, stddev_yaw_dumbbell, var_yaw_dumbbell, max_roll_forearm, max_picth_forearm, min_roll_forearm, min_pitch_forearm, amplitude_roll_forearm, amplitude_pitch_forearm, var_accel_forearm, avg_roll_forearm, stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm) )

# The test set doesn't have classe, so need to make our own.
InTrain = createDataPartition (PmlSub3$classe, p = .8)[[1]]

Training = PmlSub3[ InTrain, ]
Test = PmlSub3[ -InTrain, ]

############################################################################ Some more exploratory analysis from here:
# https://www.r-bloggers.com/how-to-use-data-analysis-for-machine-learning-example-part-1/
###########################################################################

library ("reshape", lib.loc="packages")

Melt.PmlSub <- melt (PmlSub)
ggplot (data = Melt.PmlSub, aes (x=value)) + stat_density() + facet_wrap (~variable, scales = "free")


############################################################################ LDA 
############################################################################

# lgreski says: "One's ability to run these models in parallel is often 
# the difference between using a highly effective algorithm like random forest 
# versus a less effective but more computationally efficient algorithm (such as 
# linear discriminant analysis)."

# Indeed, runs pretty quickly, but low accuracy.

ModFitLda <- train (classe ~., method = "lda", data = Training)
PredLda <- predict (ModFitLda, Test)


# Gets 67.86% accuracy. Ugh.

###########################################################################
# Simple Classification Tree (rpart)
###########################################################################
# ModFitTree <- train (classe ~., method = "rpart", data = Training)
# PredTree <- predict (ModFitTree, Test)
# confusionMatrix (Test$classe, PredTree)

# Gets 52.79% accuracy. EVEN WORSE!

###########################################################################
# KNN
###########################################################################

# Needs preprocessing.

# PreProc <- preProcess (Training [,-dim(PmlSub3)[2]], method = c ("center", "scale")
                       
# PreProc <- preProcess (Training [,-dim(Training)[2]], method = "knnImpute")
# TrainPreProc <- predict (PreProc, Training[, -dim(Training)[2]])

# library ("RANN", lib.loc = "packages/")
# TrainPreProc <- predict (PreProcImpute, Training[, -dim(Training)[2]])

# TrainPreProc$classe <- Training$classe 
# ModFitKnn <- train (classe ~., method = "knn", data = TrainPreProc)

###########################################################################
# Random Forest
###########################################################################

ModFitRf <- train (classe ~., method = "rf", data = Training)

# This takes a few hours to train, so it's nice to have a notification when it's done.
system ("say model complete!")

PredRf <- predict (ModFitRf, Test)
confusionMatrix (Test$classe, PredRf)
