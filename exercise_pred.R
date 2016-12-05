setwd("Google Drive/coursera/machine learning/course project") # set working directory

# download training and test set
train <- read.csv("pml-training.csv", na.strings = c("", "NA"))
#train <- read.csv("pml-training.csv", na.strings = c(""))
#train <- read.csv("pml-training.csv")
test <- read.csv("pml-testing.csv")

# install libraries
library(caret)
library(dplyr)

#create training, testing, validating set
set.seed(1126)
inTrain <- createDataPartition(y=train$classe, p=0.7, list = FALSE)
testing <- train[-inTrain,] # validation set
training <- train[inTrain,] # use to make test and train data

#examine training set
dim(training)
head(training)
str(training)
training <- training[-c(1, 3:4)]

#examining the variables with NA
training$max_roll_belt[1:100]
training$max_roll_belt[1:100][18] # first non NA value
training$kurtosis_roll_belt[1:100]
class(training$kurtosis_roll_belt)

#remove columns with DIV/0!
summary(training)
str(training)
levels(training$amplitude_yaw_belt)
str(training[80:160])
str(training[1:80])
training <- subset(training, select = -c(kurtosis_yaw_belt, skewness_yaw_belt, amplitude_yaw_belt,
                                         kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, amplitude_yaw_dumbbell,
                                         kurtosis_yaw_forearm, amplitude_yaw_forearm, skewness_yaw_forearm
                                         ))
# convert factors with NA to numeric
str(training)
training$kurtosis_roll_belt <- as.numeric(levels(training$kurtosis_roll_belt))[training$kurtosis_roll_belt]
training$kurtosis_picth_belt <- as.numeric(levels(training$kurtosis_picth_belt))[training$kurtosis_picth_belt]
training$skewness_roll_belt <- as.numeric(levels(training$skewness_roll_belt))[training$skewness_roll_belt]
training$skewness_roll_belt.1 <- as.numeric(levels(training$skewness_roll_belt.1))[training$skewness_roll_belt.1]
training$max_yaw_belt <- as.numeric(levels(training$max_yaw_belt))[training$max_yaw_belt]
training$min_yaw_belt <- as.numeric(levels(training$min_yaw_belt))[training$min_yaw_belt]
training$kurtosis_roll_arm <- as.numeric(levels(training$kurtosis_roll_arm))[training$kurtosis_roll_arm]
training$kurtosis_picth_arm <- as.numeric(levels(training$kurtosis_picth_arm))[training$kurtosis_picth_arm]
training$kurtosis_yaw_arm <- as.numeric(levels(training$kurtosis_yaw_arm))[training$kurtosis_yaw_arm]
training$skewness_roll_arm <- as.numeric(levels(training$skewness_roll_arm))[training$skewness_roll_arm]
training$skewness_pitch_arm <- as.numeric(levels(training$skewness_pitch_arm))[training$skewness_pitch_arm]
training$skewness_yaw_arm <- as.numeric(levels(training$skewness_yaw_arm))[training$skewness_yaw_arm]
training$kurtosis_roll_dumbbell <- as.numeric(levels(training$kurtosis_roll_dumbbell))[training$kurtosis_roll_dumbbell]
training$kurtosis_picth_dumbbell <- as.numeric(levels(training$kurtosis_picth_dumbbell))[training$kurtosis_picth_dumbbell]
training$skewness_roll_dumbbell <- as.numeric(levels(training$skewness_roll_dumbbell))[training$skewness_roll_dumbbell]
training$skewness_pitch_dumbbell <- as.numeric(levels(training$skewness_pitch_dumbbell))[training$skewness_pitch_dumbbell]
training$max_yaw_dumbbell <- as.numeric(levels(training$max_yaw_dumbbell))[training$max_yaw_dumbbell]
training$min_yaw_dumbbell <- as.numeric(levels(training$min_yaw_dumbbell))[training$min_yaw_dumbbell]
training$min_yaw_forearm <- as.numeric(levels(training$min_yaw_forearm))[training$min_yaw_forearm]
training$max_yaw_forearm <- as.numeric(levels(training$max_yaw_forearm))[training$max_yaw_forearm]
training$skewness_pitch_forearm <- as.numeric(levels(training$skewness_pitch_forearm))[training$skewness_pitch_forearm]
training$skewness_roll_forearm <- as.numeric(levels(training$skewness_roll_forearm))[training$skewness_roll_forearm]
training$kurtosis_roll_forearm <- as.numeric(levels(training$kurtosis_roll_forearm))[training$kurtosis_roll_forearm]
training$kurtosis_picth_forearm <- as.numeric(levels(training$kurtosis_picth_forearm))[training$kurtosis_picth_forearm]

## look at the datasets separately. First as columns with no NA, then the sumstats
#subset that has measurements for each time step
noNA <- training[ , colSums(is.na(training)) == 0]
summary(noNA)
#subset containing only summary stats
sumstats <- training[,colSums(is.na(training)) != 0]
sumstats$classe <- training$classe
head(sumstats)
str(sumstats)
sumstats <- na.omit(sumstats)

#look for near zero variance
nearZeroVar(training)

#find correlations in noNA
cordata <- noNA[5:56]
descrCorr <- cor(cordata)
head(descrCorr)
highCor <- findCorrelation(descrCorr, 0.9)
corOut <- cordata[,-highCor]
newdat <- cbind(noNA[c(57)], corOut)

#try model with newdata
# RF 
# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(newdat)-1)
tunegrid <- expand.grid(.mtry=mtry)
modRF <- train(classe ~ ., data = newdat, method = "rf", metric=metric, tuneGrid=tunegrid, trControl=control)
head(newdat)
sum(is.na(newdat))

#figure out how to impute NAs
head(training)
str(training)


#find correlations in sumstats
cordata <- sumstats[-92]
descrCorr <- cor(cordata)
head(descrCorr)
highCor <- findCorrelation(descrCorr, 0.9)
corOut <- cordata[,-highCor]
newdatsum <- cbind(sumstats[c(92)], corOut)
sumstats$logical <- ifelse(sumstats$classe == "A", "yes", "no")
newdatlog <- cbind(sumstats[c(93)], corOut)
#subset training set into variables that are summary statistics
head(sumstats)
mtry <- sqrt(ncol(sumstats)-1)
modRF <- train(logical ~ ., data = newdatlog, method = "rf", metric=metric, tuneGrid=tunegrid, trControl=control)
modRF <- train(classe ~ ., data = newdatsum, method = "rf", metric=metric, tuneGrid=tunegrid, trControl=control)
modRF <- train(classe ~ ., data = sumstats, method = "rf", metric=metric, tuneGrid=tunegrid, trControl=control)
sumstats <- na.omit(sumstats)
sumstats$min_yaw_forearm[1:25]
### function to turn #DIV/0! into NA (need to make)

#examine noNA data
featurePlot(x=noNA[,2:5], y = noNA$classe, plot = "pairs", auto.key=list(columns=5))
featurePlot(x=noNA[,6:9], y = noNA$classe, plot = "pairs", auto.key=list(columns=5))
featurePlot(x=noNA[,10:13], y = noNA$classe, plot = "pairs", auto.key=list(columns=5))
featurePlot(x=noNA[,14:17], y = noNA$classe, plot = "pairs", auto.key=list(columns=5))

#RF model with PCA
ctrl <- trainControl(preProcOptions = list(thresh = 0.85))
modPCA <- train(classe ~ ., data = corOut, preProcess = "pca",
                trControl = ctrl, method = "rf")

# RF without PCA
modRF <- train(classe ~ ., data = corOut, method = "rf")

#RF on all noNA data
modRF2 <- train(classe ~ ., data = noNA, method = "rf")

#RF on all data
modGLM <- train(classe ~ ., data = training, method = "glm")
# try imputing NAs?

## try two different data sets. one with noNA, one with only summary stats
#GLM model with PCA with noNA
ctrl <- trainControl(preProcOptions = list(thresh = 0.85))
modrf <- train(classe ~ ., data = noNA, preProcess = "pca",
               trControl = ctrl, method = "rf")



### looking at data
belt <- grep("belt", names(training), value = TRUE)
broll <- grep("roll", belt, value = TRUE)
bpitch <- grep("pitch", belt, value = TRUE)
byaw <- grep("yaw", belt, value = TRUE)
bell <- grep("dumbbell", names(training), value = TRUE)
armband <- grep("forearm", names(training), value = TRUE)
glove <- grep("_arm", names(training), value = TRUE)

beltna <- grep("belt", names(noNA), value = TRUE)
bellna <- grep("dumbbell", names(noNA), value = TRUE)
armbandna <- grep("forearm", names(noNA), value = TRUE)
glovena <- grep("_arm", names(noNA), value = TRUE)


