# Load all necessary libararies
# If you're using R for the fist time, you will need to install.packges("X") 
# for all of the libraries below.
library(MASS)
library(R.utils)
# library(R2jags)
library(manipulate)
library(MCMCpack)
library(R2WinBUGS)
library(mvtnorm)  

# Bayesian Inference Using Coin Flipping Experiment
# Must have R Studio for this code to work
# If you understand this visual, then you will understand the core of the Bayesian concept of 
# Prior Probability
p <- seq(from=0.005, to=0.995, by=0.005)
manipulate( 
{plot(p, dbeta(p, alpha.hyper, beta.hyper), 
      col="blue", lwd=2, type="l", las=1, bty="n", 
      ylim=c(0, 8), ylab="Density", xlab="Pr(Heads)",
      main="Prior Probability")
 polygon(c(p, rev(p)), c(dbeta(p, alpha.hyper, beta.hyper), 
                         rep(0, length(p))), col=rgb(0, 0, 1, 0.2), border=NA)}, 
alpha.hyper=slider(0.1, 10, step=0.1, initial=1), 
beta.hyper=slider(0.1, 10, step=0.1, initial=1))

# Flip a coin 10 times
set.seed(12345)
p <- seq(from=0.005, to=0.995, by=0.005)  # Support for outcome
p.true <- 0.5  # This is the true probability of heads.  Increase if you want an unbalanced coin.
N <- 10  # Number of flips.  Increase if you want a larger set of outcomes
y <- rbinom(N, size=1, prob=p.true)  # Generate flips following binomial random variable with parameter p.
table(y) # Outcomes

# Graph the likelihood of the observed outcomes given a fair coin assuming the binomial distribution.
likelihood <- sapply(p, function(p) { prod(p^y * (1-p)^(1-y)) } )
# plot(p, likelihood, lwd=2, las=1, bty="n", type="l", xlab="Coin Flip", ylab="Likelihood")
like.rescale <- N * p.true * likelihood/max(likelihood)  # This rescales the likelihood for visual presentation.
plot(p, like.rescale, lwd=2, las=1, bty="n", type="l", xlab="Coin Flip", ylab="Likelihood", main="Likelihood of Observed Data")

# Examine posterior using outcomes from the flips.  Prior and posteriors are conjugate beta distributions.
manipulate(
{plot(p, like.rescale, lwd=2, las=1, bty="n", 
      ylim=c(0,8), type="l", ylab="Density", xlab="",
      main="Posterior (red) is proportional to Likelihood (black) x Prior (blue)")
 alpha.hyper.post <- alpha.hyper + sum(y)
 beta.hyper.post <- beta.hyper + N - sum(y)
 lines(p, dbeta(p, alpha.hyper, beta.hyper), col="blue", lwd=2)
 polygon(c(p, rev(p)), c(dbeta(p, alpha.hyper, beta.hyper), 
                         rep(0, length(p))), col=rgb(0, 0, 1, 0.2), border=NA)
 lines(p, dbeta(p, alpha.hyper.post, beta.hyper.post), col="red", lwd=2)
 polygon(c(p, rev(p)), c(dbeta(p, alpha.hyper.post, beta.hyper.post), 
                         rep(0, length(p))), col=rgb(1, 0, 0, 0.2), border=NA)
 lines(p, like.rescale, lwd=2)}, 
alpha.hyper=slider(0.1, 10, step=0.1, initial=1), 
beta.hyper=slider(0.1, 10, step=0.1, initial=1))

# Posterior distributions frequently do not have closed-form solutions, so it is necessary to 
# simulate them using MCMC methods.  
# Gibbs MCMC sampler for bivariate normal pdf with correlation 0.5 

# Start with a graphic example of what we're doing.
sigma <- matrix(c(1,.5,.5,1), ncol=2)  ## This is the covariance matrix: variances to 1; correlation is therefore 0.5.
a <- rmvnorm(n=100000, mean=c(0,0), sigma=sigma) ## 1,000 draws from a bivariate normal with mean zero and correlation sigma.
x <- a[,1]
y <- a[,2]

# What does this look like in "3D"?
X <- kde2d(a[,1], a[,2], n = 100)
persp(X, phi = 45, theta = 30, shade = .1, border = NA)

# As a cross section
plot(x,y, pch=16, col="darkblue",ylim=c(-4,4), xlim=c(-4,4), xlab="x values", ylab="y values", main="Bivariate Normal with Correlation 0.5") ## This is what we are 
rm(list=ls())

# Now do the Gibbs algorithm.
rep <- 10000  ## Number of replications
x=matrix(-10, rep)
y=matrix(-10, rep)  ## Start the sampling process at (-10, -10) 
for(j in 2:rep) { 
  x[j]=rnorm(1, mean=(.5*y[j-1]), sd=sqrt(1-.5*.5)) 
  y[j]=rnorm(1, mean=(.5*x[j]), sd=sqrt(1-.5*.5)) 
}

# Plot out Gibbs simulations to show of coverage of support of bivariate normal.
plot(x[1:5], y[1:5], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 5 points
plot(x[1:10], y[1:10], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 10 points
plot(x[1:50], y[1:50], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 50 points
plot(x[1:500], y[1:500], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 500 points
plot(x[1:1000], y[1:1000], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 1000 points
plot(x[1:2000], y[1:2000], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 2000 points
plot(x[1:5000], y[1:5000], type="l", ylim=c(-10,4), xlim=c(-10,4), xlab="x values", ylab="y values")  ## First 5000 points

plot(x[500:10000], y[500:10000], type="l", ylim=c(-4,4), xlim=c(-4,4), xlab="x values", ylab="y values", main="Gibbs MCMC Sampler After Burn-in")  ## After 5000 burn in 
hist(x[500:10000], breaks=100, col="darkblue", xlab="x values", freq=F, main="Marginal of x")  ## Distribution of x
hist(y[500:10000], breaks=100, col="darkblue", xlab="y values", freq=F, main="Marginal of y")  ## Distribution of y

# The marginals are simply univariate normals with mean zero and variance one.
mean(x[500:10000])  
quantile(x[500:10000], c(.025, .975))  
mean(y[500:10000])  
quantile(y[500:10000], c(.025, .975))  

rm(list=ls())

# Bayesian inference regarding correlation coefficient in bivariate normal
sigma <- matrix(c(1,.5,.5,1), ncol=2)  ## Set variances to 1 and correlation to 0.5
ss <- 1000  ## Sample Size needed for log posterior distribution
a <- rmvnorm(n=ss, mean=c(0,0), sigma=sigma)  ## Data are mean zero, variance I, correlation 0.5
x <- a[,1]
y <- a[,2]

# This is the log posterior of bivariate normal with mean 0, variance I, and correlation r.  
# See wikipedia for the formula of bivariate normal.
# http://en.wikipedia.org/wiki/Multivariate_normal_distribution
lnpost<-function(r) 
{-((ss+3)/2)*log(1-r**2)-.5*(sum(x**2)-2*r*sum(x*y)+sum(y**2))/(1-r**2)}

draws <- 100000 ## Number of Metropolis Hastings draws
corr=matrix(0,draws) ## 10,000 MH draws from log posterior
acctot=0 

# Metropolis Hastings MCMC sampler using sample of x and y.
# Gibbs MCMC sampler does not work well in this example.
for(i in 2:draws) 
{ 
  corr[i]=corr[i-1]+runif(1,min=-.07,max=.07) 
  acc=1 
  if(abs(corr[i])>1){acc=0; corr[i]=corr[i-1]} 
  if((lnpost(corr[i])-lnpost(corr[i-1])) <log(runif(1,min=0,max=1))) 
  {corr[i]=corr[i-1]; acc=0} 
  acctot=acctot+acc 
##  if(i%%100==0){print(c(i,corr[i],acctot/i))} ## If you want to see some output, unremark this line
}

plot(corr[1:100], type="l", xlab="", ylab="Posterior Value", main="Trace of first 100 MH draws")  ## MH draws from posterior.  Note that the convergence is very fast.
hist(corr[5001:draws], breaks=100, col="darkblue", freq=F, xlab="", main="Posterior of Correlation Coefficient")

cor(x,y)  ## This is the correlation in our generated sample.
mean(corr[5001:draws])  ## Bayes average after burnin
sd(corr[5001:draws])  ## Bayes standard deviation after burnin
quantile(corr[5001:draws], c(.025, .975))  ## Bayes 95% credible interval after burnin
100*table(corr[5001:draws]>=.5)/(draws-5000)  ## "Hypothesis Test" that correlation >= 0.5

rm(list=ls())

# Do the CAPM using Bayes linear regression
library(xts)
library(Quandl)
library(MCMCpack)
library(stargazer)

# Grab data using Quandl
AAPL.Q <- Quandl("YAHOO/AAPL", start_date="2010-01-01", end_date="2015-09-18", type="xts")  ## Apple
NASDAQ.Q <- Quandl("NASDAQOMX/COMP", start_date="2010-01-01", end_date="2015-09-18", type="xts")  ## NASDAQ
AAPL <- AAPL.Q[, "Adjusted Close"]
NASDAQ <- NASDAQ.Q[, "Index Value"]
data <- merge(as.zoo(AAPL), as.zoo(NASDAQ))  ## Merge into single time-series dataset
names <- c("AAPL", "NASDAQ")
colnames(data) <- names

data.level <- as.xts(data)  ## Levels 
data.returns <- diff(log(data.level), lag=1)  ## Log returns
data.returns <- na.omit(data.returns)  ## Dump missing values

hist(data.returns$AAPL, breaks=100, col="darkblue", probability=T, main="Histogram of AAPL Daily Returns Since 2010", xlab="Daily Returns")  
# Histogram of returns.  Do they look normally distributed?  Lots of work in finance depends on normally distributed returns.
plot.ts(y=data.returns$NASDAQ, x=data.returns$AAPL, pch=16, col="darkblue", main="CAPM Data", xlab="Log Returns of NASDAQ", xlim=c(-.1,.1), ylab="Log Returns of AAPL", ylim=c(-.1,.1))  ## time series plot in R  
abline(lm(data.returns$AAPL ~ data.returns$NASDAQ), col="red")  ## I added the best fit line so the graph looks similar to that presented in class for Apple.

capm.ols <- lm(AAPL ~ NASDAQ, data=data.returns)  ## Estimate the OLS CAPM

data.returns.df <- as.data.frame(as.matrix(data.returns))
capm.bayes <-MCMCregress(AAPL ~ NASDAQ, data=data.returns.df, 
                         burnin=5000, mcmc=15000, seed=10302014, verbose=0)  ## Estimate Bayesian linear model assuming normality

stargazer(capm.ols, ci.level=.95, ci=TRUE, type="text")
summary(capm.bayes)

plot(capm.bayes)


# Now let's look at the Griliches data if we have time
library(foreign)
griliches.data <- read.dta("E:\\Big Data\\GX 5004\\griliches.dta")

griliches.ols <- lm(lw~rns+mrt+smsa+med+iq+kww+age+s+expr, data=griliches.data)
griliches.bayes <-MCMCregress(lw~rns+mrt+smsa+med+iq+kww+age+s+expr, data=griliches.data, 
                              burnin=5000, mcmc=15000, seed=10302014, verbose=0)  ## Estimate Bayesian linear model assuming normality
summary(griliches.ols)
summary(griliches.bayes)


# Union data and Bayesian Logit if we have time
union.data <- read.dta("E:\\Big Data\\GX 5004\\union.dta")

union.logit <- glm(union ~ age+grade+smsa+south+black+year, data = union.data, family = "binomial") 
union.bayes <- MCMClogit(union ~ age+grade+smsa+south+black+year, data = union.data, 
                         burnin=5000, mcmc=15000, seed=10302014, verbose=0)

summary(union.logit)
summary(union.bayes)