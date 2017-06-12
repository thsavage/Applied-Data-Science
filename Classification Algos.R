# 6/12 UPDATE: MXNET hasn't been compiled for R 3.2.2.  Perhaps someone should let them know.  
# I will redo this in Python using Tensorflow and Keras.


# This sheet addresses the many flavors of neural networks.
# If you have already fitted a logit model, then you have fitted the simplest neural network:
# the feedforward multilayered perceptron with one hidden layer.  
# NNs are powerful tools for classification.

# Load the various R packages and start simply.

library(neuralnet) # Basic MLP package with a nice graphics interface but slow
library(mxnet) # Flexible package that requires more specification of hyperparameters and fast
library(mlbench) # Has a useful logic gate dataset
library(imager) # This has images

# Let's start with the task: mapping the square root function, f(x) = x^(1/2).  

set.seed(10102014) # Set the seed for replication.
input <- as.data.frame(runif(2000, min=0, max=15)) # Generate 2000 random numbers uniformly distributed between 0 and 15.
output <- input**(1/2) # This is the functional mapping.

inputdata <- cbind(input, output) # Column bind the data into one variable.
colnames(inputdata) <- c("Input", "Output") # And give names.

ann <- neuralnet(Output~Input, data=inputdata, hidden=c(5,5), threshold=0.1, rep=10, 
                 lifesign="full", err.fct="sse", act.fct="logistic")

plot(ann, rep="best") # Plot the neural network with the best fit based on SSE

# This is an MLP with two hidden layers and five perceptrons per layer.
# The values from each of the nodes is the same as the raw coefficient that would be generated from a single logit model.
# As such, they have no real-world interpretation.
# Each of the blue numbers are the weights that tie the perceptrons together, except for the right most.
# It is just a bias adjustment term, like the constant in regression.

testdata <- as.data.frame(1:15) # Generate some test data for input to the ANN, the integers between 1 and 10.
ann.results <- compute(ann, testdata) # Run the test data through the ANN.
err <- as.data.frame(ann.results$net.result) - testdata**(1/2) # Measure the error as the difference between the test input and the ANN output.
cleanedoutput <- cbind(testdata, testdata**(1/2), as.data.frame(ann.results$net.result), err)
colnames(cleanedoutput) <- c("Test Input", "Expected Output", "ANN Output", "Error")

cleanedoutput  # Output the results in a clean table.

# Now let's examine out of sample.
testdata <- as.data.frame(15:50) # Generate some test data for input to the ANN, the integers between 1 and 10.
ann.results <- compute(ann, testdata) # Run the test data through the ANN.
err <- as.data.frame(ann.results$net.result) - testdata**(1/2) 
cleanedoutput <- cbind(testdata, testdata**(1/2), as.data.frame(ann.results$net.result), err)
colnames(cleanedoutput) <- c("Test Input", "Expected Output", "ANN Output", "Error")

cleanedoutput

# As can be seen in the table, as we move outside of the range of data used to train the NN, 
# its accuracy breaks down monotonically.  This is neither a novel problem nor a novel observation.  
# Our capacity to predict out of sample, which is all that is done here, breaks down.
# This is as true for linear regression, as it is for more sophisticated techniques, like NNs.
# Remember Hastie et al. "There has been a great deal of hype surrounding neural networks, 
# making them seem magical and mysterious... they are just nonlinear statistical models."

# Let's use the same package on a classic problem, the XOR logic gate.
# Look at the graph.  When the two features have the same sign, the label is red (or 1).
# When the two features have a different sign, the label is black (or 0).
# No linear regression can fit this, nor could a logit classifier.

set.seed(1492)
xor.data <- mlbench.xor(1000, 2)
plot(xor.data, pch=16, cex=0.5)
xor.data <- as.data.frame(xor.data)
xor.data$classes <- as.numeric(xor.data$classes) - 1

f <- classes ~ x.1 + x.2

summary(lm(f, xor.data)) # linear model is a crap shoot
summary(xor.data)
logit <- neuralnet(f, xor.data, hidden=1, rep=10,
                   err.fct = "sse", linear.output = TRUE)
net.xor <- neuralnet(f, xor.data, hidden=c(5,5), rep=10, 
                     err.fct = "sse", linear.output = TRUE)

plot(logit, rep="best") # Examine the betas and the weights, as well as the size of the error
plot(net.xor, rep="best") # NN does a fine job

test.data <- as.data.frame(cbind(-0.5, 0.5)) # Create test data of (-0.5, 0.5), which should be 0.
test.data

result <- compute(logit, test.data) # Run the test data through the logit.
result$net.result # Really bad

result <- compute(net.xor, test.data) # Run the test data through the ANN.
result$net.result # Basically zero

# If it can be digitized, it can be analyzed.
# Character recognition was an early success story for machine learning.
# Digital images are simply pixels, so lots of ones and zeros.  
# Linear or logit models don't work.
# NNs do.

# We'll start with Yann LeCun's famous NMIST dataset of handwritten images.
train <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_train.csv')
test <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')

# Display the images.
par(mfrow=c(3,4))
for(i in 1:10){
  image(t(apply(matrix(as.vector(as.matrix(train[(i-1)*500+50,-1])),ncol=28,nrow=28,byrow=T),2,rev)),col=grey(seq(0,1,length.out=256)))
}


# Preprocess data in particular normalize the pixels to between 0 and 1.
train <- as.matrix(train)
test <- as.matrix(test)
train.x <- train[,-1] / 255
train.y <- train[,1]
test.x <- test[,-1] / 255
test.y <- test[,1]
table(train.y)


# Set up parameters for deep neural network: 3 layers with large number of perceptrons.
# Softmax output layer is set at 10 because there are 10 digits with the softmax being the most likely prediction.
data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name="fc1", num_hidden=128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu")
fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
fc3 <- mx.symbol.FullyConnected(act1, name="fc3", num_hidden=64)
act3 <- mx.symbol.Activation(fc3, name="relu3", act_type="relu")
fc4 <- mx.symbol.FullyConnected(act2, name="fc4", num_hidden=10)
softmax <- mx.symbol.SoftmaxOutput(fc3, name="sm")

devices <- mx.cpu()

mx.set.seed(1066)
model <- mx.model.FeedForward.create(softmax, X=train.x, y=train.y,
                                     ctx=devices, num.round=25, array.batch.size=100,
                                     learning.rate=0.07, momentum=0.9, eval.metric=mx.metric.accuracy,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.log.train.metric(100))

pred = predict(model, test.x, ctx=mx.cpu())
pred.label <- max.col(t(pred)) - 1
table(test.y, pred.label)
sum(diag(table(test.y, pred.label)))/nrow(test)


# Now compare with decision tree models.  
library(xgboost)
library(Matrix)

train <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_train.csv')
test <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')

# Preprocess the data, in particular express as sparse matrices for speed.
train[,-1] <- train[,-1] / 255
test[,-1] <- test[,-1] / 255
train.mx <- sparse.model.matrix(label~.-1, data = train)
test.mx <- sparse.model.matrix(label~.-1, data = test)
dtrain <- xgb.DMatrix(train.mx, label=train$label)
dtest <- xgb.DMatrix(test.mx, label=test$label)

params <- list(objective="multi:softmax", num_class=10, eval_metric="mlogloss", eta=0.3, 
               max_depth=10, subsample=1, colsample_bytree=0.5)
watchlist <- list(train = dtrain, test = dtest)

xgb <-xgb.train(params = params, data = dtrain, nrounds = 70, watchlist = watchlist)

y.pred <- predict(xgb, dtest)
table(test$label, y.pred)
sum(diag(table(test$label, y.pred)))/nrow(test)


# Unsupervised learning: cluster analysis and dimensionality reduction.
# Recall we have a sparse matrix of data: wide data set that contains mostly zeros.
# We could use an unsupervised method called k-means clustering.
# Ideally we could also reduce this sparse but wide matrix to a narrow set of features for classification.
# Start with k-means using 10 clusters (because I know ahead that there are 10 digits).
# PCA is the standard method to use for dimensionality reduction.  
# It is a linear method that has been around for 100 years.
# There is a new computationally-intense method.
# See https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding

train <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')
attach(train)
table(label)
train[,-1] <- train[,-1] / 255

colors <- rainbow(length(unique(label)))
names(colors) <- unique(label)


# First K-Means
kmeans <- kmeans(train[, -1], centers = 10, n = 20)
library(cluster)
library(fpc)
plotcluster(train[,-1], kmeans$cluster)

# Next PCA
# Plot labels with respect to first two PCs.
pca <- princomp(train[,-1])$scores[, 1:2]
plot(pca[, 1:2], t='n')
text(pca, labels = label, col = colors[label])

# Finally, t-distributed stochastic neighbor embedding.
# Plot labels with respect to first two PCs.
library(tsne)
ecb = function(x,y){ plot(x,t='n'); text(x, labels = label, col = colors[label]) }
tsne = tsne(train[,-1], epoch_callback = ecb, perplexity = 50, max_iter = 1000)
plot(tsne, t='n')
text(tsne, labels = label, col = colors[label])