install.packages('remotes')
remotes::install_version("Matrix", version="1.6.0")
#remotes::install_version("quantreg", version="5.94")
remotes::install_version("rms", version="6.3-0")
remotes::install_version("CalibrationCurves", version="0.1.5")
install.packages('pmcalibration')

library(CalibrationCurves)
library(rms)
library(pmcalibration)
library(readr)

options('repos') 

#"http://nexus-tvstre.uksouth.cloudapp.azure.com/repository/r-proxy/" 
#options(repos = c(CRAN = "https://cloud.r-project.org"))


df <- read_csv("C:/Users/5lJC/Desktop/fitval_may2024/results/check-cal-in-r/predictions.csv")
y <- df$y_true
p <- df$y_prob

# Harrel's function: intercept 0.666, slope 1.050
val.prob(p, y)

# CalibrationCurves: intercept 0.52, slope 1.050 -- intercept is calc differently
val.prob.ci.2(p, y, smooth="loess", CL.smooth="fill")

# pmcurves: intercept 0.52, slope 1.050 -- intercept is calc differently
# it's estmated from formula  y ~ 1 + offset(logit(y_prob)); ?offset
# which means that predicted logits are set as fixed, and only intercept is estimated
# which measures a constant by which you'd have to adjust the predicted logits to be close to true logits
# log(p_true/(1-p_true)) = alpha + log(y_prob/(1-y_prob))
# log(p_true/(1-p_true) * (1-y_prob)/(y_prob)) = alpha

m <- logistic_cal(y, p)
alpha <- m$calibration_intercept$coefficients

oe_ratio <- mean(y) / mean(p)
print(oe_ratio)
print(exp(alpha))
