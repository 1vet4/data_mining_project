install.packages("caTools")
install.packages("smotefamily")
install.packages("pROC")

library(caTools)
library(dplyr)
library(tibble)
library(tidyr)
library(tidyverse)
library(ggplot2)
library(lubridate)
library(caret)
library(xgboost)
library(data.table)
library(smotefamily)
library(pROC)



set.seed(123) 

# DATA PREPROCESSING

health <- read.csv("C:/Users/amand/Desktop/healthcare_noshows_appointments.csv", header=TRUE) %>%
  data.frame()

attach(health)
names(health)

# Removing rows with mistakes
health <- health[health$Date.diff >= 0, ]

health <- health[, !colnames(health) %in% c("PatientId", "AppointmentID", "Gender", "Alcoholism", "Handcap",
                                            "ScheduledDay", "AppointmentDay", "Neighbourhood")]

# Target variable preparation for Logistic Regression
health$Showed_up <- factor(health$Showed_up)
levels(health$Showed_up) <- make.names(levels(health$Showed_up))

levels(health$Showed_up) <- c("No", "Yes")

continuous_vars <- c("Age")

health[continuous_vars] <- scale(health[continuous_vars])

head(health)

# SMOTE IMPUTATION

smote_output <- SMOTE(health[, !names(health) %in% "Showed_up"], health$Showed_up, K = 5)
balanced_data <- smote_output$data

balanced_data$Showed_up <- as.factor(balanced_data$class)
balanced_data$class <- NULL  

table(balanced_data$Showed_up)




#SPLIT THE BALANCED DATA

split <- sample.split(balanced_data$Showed_up, SplitRatio = 0.8)
train_data <- subset(balanced_data, split == TRUE)
test_data <- subset(balanced_data, split == FALSE)


#5-FOLD CROSS VALIDATION

train_control <- trainControl(
  method = "cv", 
  number = 5,
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

pred <- predict(log_model, test_data, type = "prob") 
roc_curve <- roc(test_data$Showed_up, pred[, "Yes"], levels = c("No", "Yes"))
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression")

auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 3)))



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

feature_importance <- varImp(rf_model)
print(feature_importance)
plot(feature_importance, main = "Feature Importance - Random Forest")


feature_importance <- varImp(rf_model)
print(feature_importance)
plot(feature_importance, main = "Feature Importance - Random Forest")



pred <- predict(rf_model, test_data, type = "prob") 
roc_curve <- roc(test_data$Showed_up, pred[, "Yes"], levels = c("No", "Yes"))
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Random Forest")

auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 3)))



# XGBOOST

target_col <- "Showed_up"
features <- setdiff(names(train_data), target_col)

train_matrix <- data.matrix(train_data[, features])
test_matrix <- data.matrix(test_data[, features])

train_label <- as.numeric(train_data[[target_col]])
test_label <- as.numeric(test_data[[target_col]])
train_label

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


pred <- predict(model_xgb, test_data, type = "prob") 
roc_curve <- roc(test_data$Showed_up, pred[, "Yes"], levels = c("No", "Yes"))
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for XGBOOST")

auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 3)))



# ADDING FEATURE ENGINEERING DATA

health <- read.csv("C:/Users/amand/Desktop/healthcare_noshows_appointments.csv", header=TRUE) %>%
  data.frame()

attach(health)
names(health)

health <- health[health$Date.diff >= 0, ]

visit_counts <- as.data.frame(table(health$PatientId))
colnames(visit_counts) <- c("PatientId", "VisitCount")
health <- merge(health, visit_counts, by = "PatientId", all.x = TRUE)

health$Day_of_week <- wday(health$AppointmentDay)

health <- health[, !colnames(health) %in% c("PatientId", "AppointmentID", "Gender", "Alcoholism", "Handcap",
                                            "ScheduledDay", "AppointmentDay", "Neighbourhood")]

health$Showed_up <- factor(health$Showed_up)
levels(health$Showed_up) <- make.names(levels(health$Showed_up))

levels(health$Showed_up) <- c("No", "Yes")

continuous_vars <- c("Age", "VisitCount")

health[continuous_vars] <- scale(health[continuous_vars])

head(health)


#SMOTE 

smote_output <- SMOTE(health[, !names(health) %in% "Showed_up"], health$Showed_up, K = 5)
balanced_data <- smote_output$data

balanced_data$Showed_up <- as.factor(balanced_data$class)
balanced_data$class <- NULL  

table(balanced_data$Showed_up)


# SPLIT THE BALANCED DATA

split <- sample.split(balanced_data$Showed_up, SplitRatio = 0.8)
train_data <- subset(balanced_data, split == TRUE)
test_data <- subset(balanced_data, split == FALSE)


#5-FOLD CROSS VALIDATION

train_control <- trainControl(
  method = "cv", 
  number = 5,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE, 
  classProbs = TRUE
)

# LOGISTIC REGRESSION

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


predictions <- predict(log_model, test_data, type = "prob")  
roc_curve <- roc(test_data$Showed_up, predictions[, "Yes"], levels = c("No", "Yes"))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Logistic Regression")

auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 3)))



# RANDOM FOREST

tune_grid <- expand.grid(
  mtry = 5  # You can also use a range, e.g., mtry = c(2, 3, 4, 5)
)

# Train the Random Forest model using caret
rf_model <- train(
  Showed_up ~ ., 
  data = train_data, 
  method = "rf",
  metric = "ROC",  # ROC as the evaluation metric
  trControl = train_control,  # Apply cross-validation
  tuneGrid = tune_grid  # Pass the grid of hyperparameters for tuning
)

print(rf_model$results)

print(rf_model)


predictions <- predict(rf_model, test_data, type = "prob")  
roc_curve <- roc(test_data$Showed_up, predictions[, "Yes"], levels = c("No", "Yes"))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Random Forest")

auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 3)))




# XGBOOST

target_col <- "Showed_up"
features <- setdiff(names(train_data), target_col)

train_matrix <- data.matrix(train_data[, features])
test_matrix <- data.matrix(test_data[, features])
train_matrix
train_label <- as.numeric(train_data[[target_col]]) 
test_label <- as.numeric(test_data[[target_col]]) 
train_label

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

predictions <- predict(model_xgb, test_data, type = "prob")  
roc_curve <- roc(test_data$Showed_up, predictions[, "Yes"], levels = c("No", "Yes"))

plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve for Xgboost")

auc_value <- auc(roc_curve)
print(paste("AUC Value:", round(auc_value, 3)))


