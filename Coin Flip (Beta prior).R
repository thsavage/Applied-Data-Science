# Flip a coin 10 times.
# p is the support of the outcome.
# p.true is the true probabilty of heads.  Move away from 0.5 for an unbalanced coin.
# N is the number of flips.  Increase if you want a larger more experimental outcomes (sample size).
# Flip are generated from binomial RV with parameter p.true.
# Table shows the outcomes.
p <- seq(from=0.005, to=0.995, by=0.005)  
p.true <- 0.5  
N <- 10  
y <- rbinom(N, size=1, prob=p.true)  
table(y)  

# Set the hyperparameters on the beta prior.  1,1 is uniform.  10,10 is bell shaped.  .1, .1 is Jeffrey's prior.
alpha.prior <- 1
beta.prior <- 1

# This is the posterior distribution as beta(alpha.prior, beta.prior)  
plot(p, dbeta(p, alpha.prior, beta.prior),
     col="blue", lwd=2, type="l", las=1, bty="n", 
     ylim=c(0, 8), ylab="Density", xlab="Support for P(Heads)",
     main="Prior Probability Model for Coin Flip")

# Graph the likelihood of the observed outcomes assuming the binomial distribution.
# Note that I rescale the likelihood so that it can be graphed together with prior and posterior.
likelihood <- sapply(p, function(p) { prod(p^y * (1-p)^(1-y)) } )
# plot(p, likelihood, lwd=2, las=1, bty="n", type="l", xlab="Coin Flip", ylab="Likelihood")
like.rescale <- N * p.true * likelihood / max(likelihood)  ## This rescales the likelihood for visual presentation.
plot(p, like.rescale, lwd=2, las=1, bty="n", type="l", xlab="Coin Flip", ylab="Likelihood")

# Graph everything together.  Note how the likelihood "reshapes" the prior.
plot(p, like.rescale, lwd=2, las=1, bty="n", type="l", 
     main="Posterior (red) is Prior (blue) x Likelihood (black)", xlab="Coin Flip", ylab="Likelihood")
lines(p, dbeta(p, alpha.prior, beta.prior), col="blue", lwd=2)
alpha.post <- alpha.prior + sum(y)
beta.post <- beta.prior + sum(1-y)
lines(p, dbeta(p, alpha.post, beta.post), col="red", lwd=2)