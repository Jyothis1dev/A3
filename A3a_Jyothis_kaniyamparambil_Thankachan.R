# Load necessary libraries
library(tidyverse)
library(caret)
library(pROC)
library(rpart)

# Install and load the rpart.plot package
install.packages("rpart.plot")
library(rpart.plot)

# Load the dataset
data <- read.csv("D:/Assignments_SCMA632/lung_cancer_examples.csv")

# Display the first few rows and structure of the dataset
print(head(data))
print(str(data))

# Inspect the Outcome column
print("Unique values in Outcome column:")
print(unique(data$Outcome))
print("Counts of each value in Outcome column:")
print(table(data$Outcome))

# Check for NA values in the Outcome column
na_count <- sum(is.na(data$Outcome))
print(paste("Number of NA values in Outcome column:", na_count))

# If there are NA values, remove them
if (na_count > 0) {
  data <- na.omit(data)
}

# Ensure Outcome is a factor with at least two levels
if (length(unique(data$Outcome)) < 2) {
  stop("The Outcome column must have at least two unique values.")
}

data$Outcome <- as.factor(data$Outcome)

# Check the Outcome column again
print("Unique values in Outcome column after processing:")
print(unique(data$Outcome))
print("Counts of each value in Outcome column after processing:")
print(table(data$Outcome))

# Logistic Regression Analysis
set.seed(123)

# Load the caret library correctly
library(caret)

trainIndex <- createDataPartition(data$Outcome, p = .8, 
                                  list = FALSE, 
                                  times = 1)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# Fit the logistic regression model
log_model <- glm(Outcome ~ ., data = trainData, family = binomial)

# Summarize the model
print(summary(log_model))

# Predict on the test data
log_pred <- predict(log_model, testData, type = "response")
log_pred_class <- ifelse(log_pred > 0.5, 1, 0)

# Confusion Matrix
conf_matrix <- confusionMatrix(as.factor(log_pred_class), as.factor(testData$Outcome))
print(conf_matrix)

# ROC Curve
roc_curve <- roc(testData$Outcome, log_pred)
plot(roc_curve, col = "blue")
print(auc(roc_curve))

# Decision Tree Analysis
# Fit the decision tree model
tree_model <- rpart(Outcome ~ ., data = trainData, method = "class")

# Summarize the model
print(summary(tree_model))

# Predict on the test data
tree_pred <- predict(tree_model, testData, type = "class")

# Confusion Matrix
tree_conf_matrix <- confusionMatrix(tree_pred, as.factor(testData$Outcome))
print(tree_conf_matrix)

# ROC Curve
tree_roc_curve <- roc(as.numeric(testData$Outcome), as.numeric(tree_pred))
plot(tree_roc_curve, col = "red")
print(auc(tree_roc_curve))

# Compare the models
log_auc <- auc(roc_curve)
tree_auc <- auc(tree_roc_curve)

print(paste("Logistic Regression AUC:", log_auc))
print(paste("Decision Tree AUC:", tree_auc))

# Plot ROC curves together
plot(roc_curve, col = "blue", main = "ROC Curves Comparison")
plot(tree_roc_curve, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), 
       col = c("blue", "red"), lwd = 2)

