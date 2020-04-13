#importing the dataset 
credit_card <- read.csv('D:\\Free Learning\\Credit Card Fraud Detection System\\creditcardfraud\\creditcard.csv')

#view structure of the dataset
str(credit_card)

#convert Class column to a factor variable
credit_card$Class <- factor(credit_card$Class, levels = c(0, 1))

#summary of dataset
summary(credit_card)

#count missing values
sum(is.na(credit_card))

#------------------------------------------------------------------------------------------------------------------------------------

#get distribution of fraud and legit transactions in the dataset
table(credit_card$Class)

#get the percentage of fraud and legit transactions in the dataset
prop.table(table(credit_card$Class))


#pie chart of all transactions
labels <- c("legit", "fraud")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)),2))
labels <- paste0(labels, "%")

pie(table(credit_card$Class), labels, col = c("green", "red"),
      main = "Pie chart for credit card transactions")


#-------------------------------------------------------------------------------------------------------------------------------------

#no model predictions

predictions <- rep.int(0, nrow(credit_card))
predictions <- factor(predictions, levels = c(0, 1))

library(caret)
confusionMatrix(data = predictions, reference = credit_card$Class)


#-------------------------------------------------------------------------------------------------------------------------------------

library(dplyr)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)

library(ggplot2)

ggplot(data = credit_card, aes(x = V1, y = V2, col = Class)) +
        geom_point() +
        theme_bw() +
        scale_color_manual(values = c("green", "red"))

#-------------------------------------------------------------------------------------------------------------------------------------

# creating training and test sets for fraud detection model

library(caTools)

set.seed(123)

data_sample = sample.split(credit_card$Class,SplitRatio = 0.80)

train_data = subset(credit_card, data_sample==TRUE)

test_data = subset(credit_card, data_sample==FALSE)

dim(train_data)
dim(test_data)

#-------------------------------------------------------------------------------------------------------------------------------------

# Random over-sampling (ROS)

n_legit <- 22750
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit

library(ROSE)
oversampling_result <- ovun.sample(Class ~ .,
                                   data = train_data,
                                   method = "over",
                                   N = new_n_total,
                                   seed = 2019)

oversampled_credit <- oversampling_result$data

table(oversampled_credit$Class)

ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c("green", "red"))

#-----------------------------------------------------------------------------------------------------------------------------------

# Random under-sampling (RUS)

table(train_data$Class)

n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud

# library(ROSE)
undersampling_result <- ovun.sample(Class ~ .,
                                    data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)

undersampled_credit <- undersampling_result$data

table(undersampled_credit$Class)

ggplot(data = undersampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  theme_bw() +
  scale_color_manual(values = c("green", "red"))

#-------------------------------------------------------------------------------------------------------------------------------------

# ROS & RUS 

n_new <- nrow(train_data)
fraction_fraud_new <- 0.50

sampling_result <- ovun.sample(Class ~ .,
                               data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)

sampled_credit <- sampling_result$data

table(sampled_credit$Class) 

prop.table(table(sampled_credit$Class))

ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) +
  theme_bw() +
  scale_color_manual(values = c("green", "red"))

#------------------------------------------------------------------------------------------------------------------------------------

# using SMOTE to balance the dataset

library(smotefamily)

table(train_data$Class)

# setting the number of legit and fraud cases and the desired percentage of legit transactions
n0 <- 22750
n1 <- 35
r0 <- 0.6

# calculate the value for dup_size parameter of SMOTE
ntimes <- ((1 - r0)/r0) * (n0/n1) - 1

smote_output = SMOTE(X = train_data[ , -c(1, 31)],
                     target = train_data$Class,
                     K = 5,
                     dup_size = ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

# class distribution for original dataset
ggplot(train_data, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  scale_color_manual(values = c("green", "red"))

# class distribution for over-sampled dataset using SMOTE
ggplot(credit_smote, aes(x = V1, y = V2, col = Class)) +
  geom_point() +
  scale_color_manual(values = c("green", "red"))


#-------------------------------------------------------------------------------------------------------------------------------------

library(rpart)
library(rpart.plot)

CART_model <- rpart(Class ~ ., credit_smote)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2 )


# predict fraud cases on test_data
predicted_val <- predict(CART_model, test_data, type = 'class')

# build confusion matrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)


#------------------------------------------------------------------------------------------------------------------------------------

# predict fraud cases on the credit_card dataset
predicted_val <- predict(CART_model, credit_card, type = 'class')

# build confusion matrix
library(caret)
confusionMatrix(predicted_val, credit_card$Class)

#----------------------------------------------------------------------------------------------------------------------------------------
#========================================================================================================================================


# Decision tree without SMOTE and viewing the results of the predictions
CART_model <- rpart(Class ~ ., train_data)

rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2 )


# predict fraud cases on test_data
predicted_val <- predict(CART_model, test_data, type = 'class')

# build confusion matrix
library(caret)
confusionMatrix(predicted_val, test_data$Class)

# predicting on the credit_card dataset now
predicted_val <- predict(CART_model, credit_card, type = 'class')
confusionMatrix(predicted_val, credit_card$Class)


#======================================================================================================================================



























