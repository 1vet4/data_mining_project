library(dplyr)
library(tibble)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(lubridate)
library(caret)
library(xgboost)
library(data.table)

set.seed(42) 

# DATA PREPROCESSING

# Read data
health <- read.csv("healthcare_noshows_appointments.csv", header=TRUE) %>%
  data.frame()
health$Showed_up <- factor(health$Showed_up)

# Downsampling technique
health <- downSample(
  x = health[, -which(names(health) == "Showed_up")],  # Visi kintamieji, išskyrus "Showed_up"
  y = health$Showed_up
)

colnames(health)[ncol(health)] <- "Showed_up"

# Target variable preparation for Logistic Regression
health$Showed_up <- factor(health$Showed_up)
levels(health$Showed_up) <- make.names(levels(health$Showed_up))

levels(health$Showed_up) <- c("No", "Yes")

continuous_vars <- c("Age")

health[continuous_vars] <- scale(health[continuous_vars])

# Remove rows with mistakes
health <- health[health$Date.diff >= 0, ]

# Remove redundant columns
health <- health[, !colnames(health) %in% c("PatientId", "AppointmentID", "Gender", "Alcoholism",
                                            "ScheduledDay", "AppointmentDay", "Neighbourhood", "Handcap")]
head(health)

# SPLITTING THE DATA

split <- sample.split(health$Showed_up, SplitRatio = 0.8)
train_data <- subset(health, split == TRUE)
test_data <- subset(health, split == FALSE)


# 5-FOLD CROSS VALIDATION

train_control <- trainControl(
  method = "cv", 
  number = 5,
  verboseIter = TRUE,
  summaryFunction = twoClassSummary,
  classProbs = TRUE
)

# LOGISTIC REGRESSION

# DEFINE MODEL 

log_model <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "glm", 
  family = binomial(),
  metric = "ROC",
  trControl = train_control
)

log_model$results

print(log_model)

# RANDOM FOREST

rf_model <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "rf",
  metric = "ROC",
  trControl = train_control
)

print(rf_model$results)

print(rf_model)


# XGBOOST

target_col <- "Showed_up"
features <- setdiff(names(train_data), target_col)

train_matrix <- data.matrix(train_data[, features])
test_matrix <- data.matrix(test_data[, features])

train_label <- as.numeric(train_data[[target_col]])
test_label <- as.numeric(test_data[[target_col]])

params <- list(
  objective = "binary:logistic", 
  eval_metric = "logloss",       
  max_depth = 6,                 
  eta = 0.1,                     
  nthread = 2                     
)

nrounds <- 100  
nfold <- 5     

train_control <- trainControl(
  method = "cv", 
  number = nfold, 
  classProbs = TRUE,  
  summaryFunction = twoClassSummary,  
  verboseIter = TRUE, 
  savePredictions = "final"
)

model_xgb <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "xgbTree",  
  metric = "ROC",  
  trControl = train_control,  
  
)
print(model_xgb$results)

print(model_xgb)



# DATA PREPROCESSING WITH FEATURE ENGINEERING 

health <- read.csv("healthcare_noshows_appointments.csv", header=TRUE) %>%
  data.frame()

health$Showed_up <- factor(health$Showed_up)

health <- downSample(
  x = health[, -which(names(health) == "Showed_up")]
  y = health$Showed_up
)
colnames(health)[ncol(health)] <- "Showed_up"

health$Showed_up <- factor(health$Showed_up)
levels(health$Showed_up) <- make.names(levels(health$Showed_up))

levels(health$Showed_up) <- c("No", "Yes")

continuous_vars <- c("Age")

health[continuous_vars] <- scale(health[continuous_vars])


health <- health[health$Date.diff >= 0, ]

# Additional "Visit Count" feature
visit_counts <- as.data.frame(table(health$PatientId))
colnames(visit_counts) <- c("PatientId", "VisitCount")
health <- merge(health, visit_counts, by = "PatientId", all.x = TRUE)

# Additional "Day of week" feature
health$Day_of_week <- wday(health$AppointmentDay)

head(health)

health <- health[, !colnames(health) %in% c("PatientId", "AppointmentID", "Gender", "Alcoholism", "Handcap",
                                            "ScheduledDay","AppointmentDay", "Neighbourhood")]



# SPLITTING THE DATA

split <- sample.split(health$Showed_up, SplitRatio = 0.8)
train_data <- subset(health, split == TRUE)
test_data <- subset(health, split == FALSE)

#5-FOLD CROSS VALIDATION

train_control <- trainControl(
  method = "cv", 
  number = 5,
  verboseIter = TRUE,
  summaryFunction = twoClassSummary,
  classProbs = TRUE
)



# DEFINE MODEL 

log_model <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "glm", 
  family = binomial(),
  metric = "ROC",
  trControl = train_control
)

log_model$results

print(log_model)

# RANDOM FOREST

rf_model <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "rf",
  metric = "ROC",
  trControl = train_control
)

print(rf_model$results)

print(rf_model)

# XGBOOST

target_col <- "Showed_up"
features <- setdiff(names(train_data), target_col)

train_matrix <- data.matrix(train_data[, features])
test_matrix <- data.matrix(test_data[, features])
train_matrix
train_label <- as.numeric(train_data[[target_col]])
test_label <- as.numeric(test_data[[target_col]])
params <- list(
  objective = "binary:logistic", 
  eval_metric = "logloss",       
  max_depth = 6,                 
  eta = 0.1,                     
  nthread = 2                     
)

nrounds <- 100  
nfold <- 5     

train_control <- trainControl(
  method = "cv", 
  number = nfold, 
  classProbs = TRUE,  
  summaryFunction = twoClassSummary,  
  verboseIter = TRUE, 
  savePredictions = "final"
)

model_xgb <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "xgbTree",  
  metric = "ROC",  
  trControl = train_control,  
  
)
print(model_xgb$results)

print(model_xgb)