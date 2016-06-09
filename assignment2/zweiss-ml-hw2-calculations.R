
# ==================================================================================================================
# Assignment 2 by Zarah Weiß
# Machine Learning for CL course
# SS 16
# ==================================================================================================================

# enter your own directory!
dat <- read.csv('/Users/zweiss/Documents/Uni/9_SS16/iscl-s9_ss16-machine_learning/assignment2/data/language-stats.csv')

# ==================================================================================================================
# Task 1
# Fit a logistic regression classifier distinguishing the sentences in German from the sentences in the other two 
# languages using the number of words in a sentence as the only predictor. Do not use any regularization for exercises 
# 1 to 3 (See the notes at the end of this document).
# ==================================================================================================================

# introduce new column for binary language classification
dat$isGerman <- ifelse(dat$language=='German', TRUE, FALSE)
str(dat)

# 1.1 create model on training set and 
# 1.2 look at results for coefficients
# ==================================================================================================================

glm.1 <- glm(isGerman ~ sent_length, family=binomial(link='logit'), dat)
summary(glm.1)

# 1.3 get accuracy, precision, recall and F1-score of the fitted model on the training data
# ==================================================================================================================

accuracy <- function(tp, fp, tn, fn) {
  return((tp + tn) / (tp + tn + fp + fn))
}

precision <- function(tp, fp) {
  return(ifelse((tp + fp) > 0, tp / (tp + fp), 0))
}

recall <- function(tp, fn) {
  return(ifelse((tp + fn) > 0, tp / (tp + fn), 0))
}

f1Score <- function(tp, fp, fn) {
  return(
    ifelse(precision(tp, fp) + recall(tp, fn) > 0, 
           2 * (precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn)), 
           0))
}

# predict training data
dat.pred.glm1 <- predict(glm.1, newdata = dat, type = 'response')
dat$IsGermanPred.glm1 <- ifelse(dat.pred.glm1 > 0.5, TRUE, FALSE)
# visually inspect results
plot(x = dat$sent_length, y = dat.pred.glm1, ylab = "Predicted probability of sentence being German", xlab = "Sentence length")

# get true positives, false positives, false negatives and true negatives
tp.glm1 <- nrow(dat[dat$isGerman==TRUE & dat$IsGermanPred.glm1==TRUE,])
fp.glm1 <- nrow(dat[dat$isGerman==FALSE & dat$IsGermanPred.glm1==TRUE,])
fn.glm1 <- nrow(dat[dat$isGerman==TRUE & dat$IsGermanPred.glm1==FALSE,])
tn.glm1 <- nrow(dat[dat$isGerman==FALSE & dat$IsGermanPred.glm1==FALSE,])


# calculate accuracy, precision, recall, and fscore
accuracy.glm1 <- accuracy(tp.glm1, fp.glm1, tn.glm1, fn.glm1)
accuracy.glm1
precision.glm1 <- precision(tp.glm1, fp.glm1)
precision.glm1
recall.glm1 <- recall(tp.glm1, fn.glm1)
recall.glm1
f1.glm1 <- f1Score(tp.glm1, fp.glm1, fn.glm1)
f1.glm1

# ==================================================================================================================
# Task 2
# The model you fit in exercise 1 suffers from the class imbalance problem (besides the poor predictor). A workaround 
# for this problem is to decide for a probability threshold other than 0.5.
# ==================================================================================================================

# 2.1 Find and report the best threshold value that maximizes the F1-score

findBestThreshold <- function(df, predictions, thresholds, PLOT = TRUE) {
  f1 <- c()
  i = 1
  for (t in thresholds) {
    # get prediction labels based on new threshold t
    df$predicted <- ifelse(predictions > t, TRUE, FALSE)
    # get true positives, false positives, false negatives and true negatives
    tp.tmp <- nrow(df[df$expected==TRUE & df$predicted==TRUE,])
    fp.tmp <- nrow(df[df$expected==FALSE & df$predicted==TRUE,])
    fn.tmp <- nrow(df[df$expected==TRUE & df$predicted==FALSE,])
    # get f1
    f1[i] <- f1Score(tp.tmp, fp.tmp, fn.tmp)
    i = i + 1
  }
  if(PLOT) {
    plot(thresholds, f1, type="b", xlab="Threshold", ylab="F1 Score")
  }
  # return best value
  tBest = -1
  for (j in 1:length(thresholds)) {
    if(f1[j] == max(f1)) {
      tBest = thresholds[j]
      print("Best f-Score:")
      print(f1[j])
    }
  }
  return(tBest)
}

expected <- dat$isGerman
# find best threshold
tBest <- findBestThreshold(data.frame(expected), dat.pred.glm1, c(0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 
                                                                  0.36, 0.37, 0.38, 0.39, 0.4, 0.45, 0.5))
tBest
# best threshold is 0.35!
dat$IsGermanPred.glm1.bestThreshold <- ifelse(dat.pred.glm1 > tBest, TRUE, FALSE)

# 2.2 Write down the discriminant function (the function f(X) whose value is positive for the positive instances 
# (sentences in German) and negative for the negative instances)

# f(x) = 0 if x < 0.35, else 1

# 2.3 Report the accuracy, precision, recall and F1-score at the best threshold value

# get the new true positives, false positives, false negatives and true negatives
tp.glm1.35 <- nrow(dat[dat$isGerman==TRUE & dat$IsGermanPred.glm1.bestThreshold==TRUE,])
fp.glm1.35 <- nrow(dat[dat$isGerman==FALSE & dat$IsGermanPred.glm1.bestThreshold==TRUE,])
fn.glm1.35 <- nrow(dat[dat$isGerman==TRUE & dat$IsGermanPred.glm1.bestThreshold==FALSE,])
tn.glm1.35 <- nrow(dat[dat$isGerman==FALSE & dat$IsGermanPred.glm1.bestThreshold==FALSE,])
# calculate new measures
accuracy.glm1.35 <- accuracy(tp.glm1.35, fp.glm1.35, tn.glm1.35, fn.glm1.35)
accuracy.glm1.35
precision.glm1.35 <- precision(tp.glm1.35, fp.glm1.35)
precision.glm1.35
recall.glm1.35 <- recall(tp.glm1.35, fn.glm1.35)
recall.glm1.35
f1.glm1.35 <- f1Score(tp.glm1.35, fp.glm1.35, fn.glm1.35)
f1.glm1.35


# ==================================================================================================================
# Task 3
# This time, besides the sentence length, use 15 additional predictors that indicate the relative frequencies of POS 
# unigrams within the sentence. Evaluate your model with probability threshold of 0.5, and report accuracy, precision, 
# recall and F1-score values. Does this model suffer from the class imbalance problem equally?
# ==================================================================================================================s

# 3.1 create more complex model
# get rid of sentence number, actual language and glm.1 prediction
dat.reduced <- dat[,-c(16, 18, 20, 21, 22)]

# train model with all features (only POS and sentence length left)
glm.2 <- glm(isGerman ~ ., family=binomial(link='logit'), dat.reduced)
summary(glm.2)

# predict training data
dat.pred.glm2 <- predict(glm.2, newdata = dat.reduced, type = 'response')
dat$IsGermanPred.glm2 <- ifelse(dat.pred.glm2 > 0.5, TRUE, FALSE)

# get true positives, false positives, false negatives and true negatives
tp.glm2 <- nrow(dat[dat$isGerman==TRUE & dat$IsGermanPred.glm2==TRUE,])
fp.glm2 <- nrow(dat[dat$isGerman==FALSE & dat$IsGermanPred.glm2==TRUE,])
fn.glm2 <- nrow(dat[dat$isGerman==TRUE & dat$IsGermanPred.glm2==FALSE,])
tn.glm2 <- nrow(dat[dat$isGerman==FALSE & dat$IsGermanPred.glm2==FALSE,])

# calculate accuracy, precision, recall, and fscore
accuracy.glm2 <- accuracy(tp.glm2, fp.glm2, tn.glm2, fn.glm2)
accuracy.glm2
precision.glm2 <- precision(tp.glm2, fp.glm2)
precision.glm2
recall.glm2 <-recall(tp.glm2, fn.glm2)
recall.glm2
f1.glm2 <- f1Score(tp.glm2, fp.glm2, fn.glm2) 
f1.glm2

# Does the model suffer from the class inbalance problem equally?
# No it doesn't. This can be seen at the results for accuracy, precision, recall and F1 being quite similar.
# This might be due to the fact, that we have a number of good predictors now, that can identify the classes
# not just by a majority vote


# ==================================================================================================================
# Task 4
# In this exercise you will fit two three-way classifiers predicting the language, rather than the binary 
# distinction between German and others, using the same predictors as in exercise 3.
# ==================================================================================================================

library(LiblineaR)

# 4.1 Fit two separate models to the complete data, one with L1, the other one with L2 regularization. For both 
# models, use the regularization parameter = 50.

dat.reduced2 <- dat[, -c(16, 18, 19, 20, 21, 22)]
linM.l2 <- LiblineaR(data = dat.reduced2, target = dat$language, type = 0, cost = 50)
linM.l2

linM.l1 <- LiblineaR(data = dat.reduced2, target = dat$language, type = 6, cost = 50)
linM.l1

# 4.3 Calculate and compare accuracy of both L1 and L2 regularized models.

# predict training data
dat.pred.linM.l2 <- predict(linM.l2, newx = dat.reduced2, proba = TRUE)
dat.pred.linM.l1 <- predict(linM.l1, newx = dat.reduced2)

# get true positives, false positives, false negatives and true negatives
tp.l2.english <- nrow(dat[dat$language=='English' & dat.pred.linM.l2$predictions=='English',])
fp.l2.english <- nrow(dat[dat$language!='English' & dat.pred.linM.l2$predictions=='English',])
fn.l2.english <- nrow(dat[dat$language=='English' & dat.pred.linM.l2$predictions!='English',])
tn.l2.english <- nrow(dat[dat$language!='English' & dat.pred.linM.l2$predictions!='English',])

tp.l2.german <- nrow(dat[dat$language=='German' & dat.pred.linM.l2$predictions=='German',])
fp.l2.german <- nrow(dat[dat$language!='German' & dat.pred.linM.l2$predictions=='German',])
fn.l2.german <- nrow(dat[dat$language=='German' & dat.pred.linM.l2$predictions!='German',])
tn.l2.german <- nrow(dat[dat$language!='German' & dat.pred.linM.l2$predictions!='German',])

tp.l2.japanese <- nrow(dat[dat$language=='Japanese' & dat.pred.linM.l2$predictions=='Japanese',])
fp.l2.japanese <- nrow(dat[dat$language!='Japanese' & dat.pred.linM.l2$predictions=='Japanese',])
fn.l2.japanese <- nrow(dat[dat$language=='Japanese' & dat.pred.linM.l2$predictions!='Japanese',])
tn.l2.japanese <- nrow(dat[dat$language!='Japanese' & dat.pred.linM.l2$predictions!='Japanese',])

tp.l1.english <- nrow(dat[dat$language=='English' & dat.pred.linM.l1$predictions=='English',])
fp.l1.english <- nrow(dat[dat$language!='English' & dat.pred.linM.l1$predictions=='English',])
fn.l1.english <- nrow(dat[dat$language=='English' & dat.pred.linM.l1$predictions!='English',])
tn.l1.english <- nrow(dat[dat$language!='English' & dat.pred.linM.l1$predictions!='English',])

tp.l1.german <- nrow(dat[dat$language=='German' & dat.pred.linM.l1$predictions=='German',])
fp.l1.german <- nrow(dat[dat$language!='German' & dat.pred.linM.l1$predictions=='German',])
fn.l1.german <- nrow(dat[dat$language=='German' & dat.pred.linM.l1$predictions!='German',])
tn.l1.german <- nrow(dat[dat$language!='German' & dat.pred.linM.l1$predictions!='German',])

tp.l1.japanese <- nrow(dat[dat$language=='Japanese' & dat.pred.linM.l1$predictions=='Japanese',])
fp.l1.japanese <- nrow(dat[dat$language!='Japanese' & dat.pred.linM.l1$predictions=='Japanese',])
fn.l1.japanese <- nrow(dat[dat$language=='Japanese' & dat.pred.linM.l1$predictions!='Japanese',])
tn.l1.japanese <- nrow(dat[dat$language!='Japanese' & dat.pred.linM.l1$predictions!='Japanese',])




# calculate accuracy, precision, recall, and fscore
accuracy.l2.english <- accuracy(tp.l2.english, fp.l2.english, tn.l2.english, fn.l2.english)
accuracy.l2.english
accuracy.l2.german <- accuracy(tp.l2.german, fp.l2.german, tn.l2.german, fn.l2.german)
accuracy.l2.german
accuracy.l2.japanese <- accuracy(tp.l2.japanese, fp.l2.japanese, tn.l2.japanese, fn.l2.japanese)
accuracy.l2.japanese
accuracy.l2.overall <- accuracy(tp.l2.english + tp.l2.german + tp.l2.japanese, 
                                fp.l2.english + fp.l2.german + tp.l2.japanese, 
                                tn.l2.english + tn.l2.german + tn.l2.german, 
                                fn.l2.english + fn.l2.german + fn.l2.japanese)
accuracy.l2.overall

accuracy.l1.english <- accuracy(tp.l1.english, fp.l1.english, tn.l1.english, fn.l1.english)
accuracy.l1.english
accuracy.l1.german <- accuracy(tp.l1.german, fp.l1.german, tn.l1.german, fn.l1.german)
accuracy.l1.german
accuracy.l1.japanese <- accuracy(tp.l1.japanese, fp.l1.japanese, tn.l1.japanese, fn.l1.japanese)
accuracy.l1.japanese
accuracy.l1.overall <- accuracy(tp.l1.english + tp.l1.german + tp.l1.japanese, 
                                fp.l1.english + fp.l1.german + tp.l1.japanese, 
                                tn.l1.english + tn.l1.german + tn.l1.german, 
                                fn.l1.english + fn.l2.german + fn.l1.japanese)
accuracy.l1.overall

# 4.4 Tabulate the confusion matrix of the L2-regularized model you fit in the previous step.

# confusion matrix English

# actually english, predicted as: english, german, japanese
exp.english <- c(nrow(dat[dat$language=='English' & dat.pred.linM.l2$predictions=='English',]),
             nrow(dat[dat$language=='English' & dat.pred.linM.l2$predictions=='German',]),
             nrow(dat[dat$language=='English' & dat.pred.linM.l2$predictions=='Japanese',]))

# actually german, predicted as: english, german, japanese
exp.german <- c(nrow(dat[dat$language=='German' & dat.pred.linM.l2$predictions=='English',]),
             nrow(dat[dat$language=='German' & dat.pred.linM.l2$predictions=='German',]),
             nrow(dat[dat$language=='German' & dat.pred.linM.l2$predictions=='Japanese',]))

# actually englisjapaneseh, predicted as: english, german, japanese
exp.japanese <- c(nrow(dat[dat$language=='Japanese' & dat.pred.linM.l2$predictions=='English',]),
             nrow(dat[dat$language=='Japanese' & dat.pred.linM.l2$predictions=='German',]),
             nrow(dat[dat$language=='Japanese' & dat.pred.linM.l2$predictions=='Japanese',]))

conf.matrix <- rbind(exp.english, exp.german, exp.japanese)
colnames(conf.matrix) <- c('pred.english', 'pred.german', 'pred.japanese')
conf.matrix

# conf.matrix.english <- data.frame(c(tp.l2.english, fp.l2.english), c(fn.l2.english, tn.l2.english), row.names = c('predicted positives', 'predicted negatives'))
# colnames(conf.matrix.english) <- c('expected positives', 'expexted negatives')
# conf.matrix.english
# 
# # confusion matrix German
# conf.matrix.german <- data.frame(c(tp.l2.german, fp.l2.german), c(fn.l2.german, tn.l2.german), row.names = c('predicted positives', 'predicted negatives'))
# colnames(conf.matrix.german) <- c('expected positives', 'expexted negatives')
# conf.matrix.german
# 
# # confusion matrix Japanese
# conf.matrix.japanese <- data.frame(c(tp.l2.japanese, fp.l2.japanese), c(fn.l2.japanese, tn.l2.japanese), row.names = c('predicted positives', 'predicted negatives'))
# colnames(conf.matrix.japanese) <- c('expected positives', 'expexted negatives')
# conf.matrix.japanese
# 
# # confusion matrix overall
# conf.matrix.overall <- data.frame(c(tp.l2.english + tp.l2.german + tp.l2.japanese, 
#                                     fp.l2.english + fp.l2.german + fp.l2.japanese), 
#                                   c(fn.l2.english + fn.l2.german + fn.l2.japanese, 
#                                     tn.l2.english + tn.l2.german + tn.l2.japanese), 
#                                   row.names = c('predicted positives', 'predicted negatives'))
# colnames(conf.matrix.overall) <- c('expected positives', 'expexted negatives')
# conf.matrix.overall



# ==================================================================================================================
# Task 5
# Using the same model in exercise 4 with L2 regularization, evaluate the model accuracy using 10-fold cross 
# validation, and report the average accuracy and its standard error.
# ==================================================================================================================


# cross validation

start <- c(0, 0, 0)
# dat is sorted: first all English, then all German, then Japanese, i.e. German starts after the last English row, etc.
end <- c(0, nrow(dat[dat$language=='English',]), nrow(dat[dat$language=='English',])+nrow(dat[dat$language=='German',]))
chunk <- c(trunc(nrow(dat[dat$language=='English',])/10), trunc(nrow(dat[dat$language=='German',])/10), 
           trunc(nrow(dat[dat$language=='Japanese',])/10))

accuracies <- c()
for (k in 1:10) {

  # update indices for chunks for each language in the data
  for (l in 1:3) {
    start[l] <- end[l] + 1
    # if it is the last chunk, just take the rest
    if (k == 10 & l == 3) {
      end <- c(nrow(dat[dat$language=='English',]), 
               nrow(dat[dat$language=='English',])+nrow(dat[dat$language=='German',]), 
               nrow(dat[dat$language=='English',])+nrow(dat[dat$language=='German',])+nrow(dat[dat$language=='Japanese',]))
    } else {
      end[l] <- start[l] + chunk[l] -1
    }
  }
  
#   print(start)
#   print(end)
#   print(ranges)
  
  # set up test and training
  dat.train <- dat.reduced2[-c(start[1]:end[1], start[2]:end[2], start[3]:end[3]),]
  lang.train <- dat$language[-c(start[1]:end[1], start[2]:end[2], start[3]:end[3])]
  
  dat.test <- dat.reduced2[c(start[1]:end[1], start[2]:end[2], start[3]:end[3]),]
  expected <- dat$language[c(start[1]:end[1], start[2]:end[2], start[3]:end[3])]
  lang.test <- as.data.frame(expected)
   
  # build model
  linM.l2.tmp <- LiblineaR(data = dat.train, target = lang.train, type = 0, cost = 50)
  # test model
  linM.l2.tmp.pred <- predict(linM.l2.tmp, newx = dat.test, proba = TRUE)
  lang.test$predicted <- linM.l2.tmp.pred$predictions
  
  # get accuracy
  tp.e <- nrow(lang.test[lang.test$expected=='English' & lang.test$predicted=='English',])
  tp.g <- nrow(lang.test[lang.test$expected=='German' & lang.test$predicted=='German',])
  tp.j <- nrow(lang.test[lang.test$expected=='Japanese' & lang.test$predicted=='Japanese',])
  tp.tmp <- tp.e + tp.g + tp.j
  
  fp.e <- nrow(lang.test[lang.test$expected!='English' & lang.test$predicted=='English',])
  fp.g <- nrow(lang.test[lang.test$expected!='German' & lang.test$predicted=='German',])
  fp.j <- nrow(lang.test[lang.test$expected!='Japanese' & lang.test$predicted=='Japanese',])
  fp.tmp <- fp.e + fp.g + fp.j
  
  tn.e <- nrow(lang.test[lang.test$expected!='English' & lang.test$predicted!='English',])
  tn.g <- nrow(lang.test[lang.test$expected!='German' & lang.test$predicted!='German',])
  tn.j <- nrow(lang.test[lang.test$expected!='Japanese' & lang.test$predicted!='Japanese',])
  tn.tmp <- tn.e + tn.g + tn.j
  
  fn.e <- nrow(lang.test[lang.test$expected=='English' & lang.test$predicted!='English',])
  fn.g <- nrow(lang.test[lang.test$expected=='German' & lang.test$predicted!='German',])
  fn.j <- nrow(lang.test[lang.test$expected=='Japanese' & lang.test$predicted!='Japanese',])
  fn.tmp <- fn.e + fn.g + fn.j
    
  accuracies[k] <- accuracy(tp.tmp, fp.tmp, tn.tmp, fn.tmp)
}

avgAcc <- mean(accuracies)
stdErr <- sd(accuracies) / sqrt(length(accuracies))




# can't get a standard error here!
# linM.l2.acc <- LiblineaR(data = dat.reduced2, target = dat$language, type = 0, cost = 50, cross = 10)
# linM.l2.acc


