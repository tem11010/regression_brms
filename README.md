---
title: "Bayesian Regression Analysis in R using brms"
author: "TEMoore"
graphics: yes
output:
  prettydoc::html_pretty:
    theme: cayman
    highlight: github
editor_options:
  chunk_output_type: console
---

```{r setup, include = F}
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@ Knitr Options
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Set root directory to the project directory
knitr::opts_knit$set(root.dir = rprojroot::find_rstudio_root_file())

# Set default knitr options: 
# Suppress warnings and messages, cache chunks, 
#  set default figure size to 6x8 at 300 dpi, and save a png and pdf
knitr::opts_chunk$set(warning = F, message = F, collapse = T, cache = T,
    fig.height = 6, fig.width = 8, dpi = 120,  
    dev = 'png')

```

#### Keywords: Bayesian, brms, looic, model selection, multiple regression, posterior probability check, weighted model averaging


## Introduction

There are many [good reasons](http://andrewgelman.com/2016/12/13/bayesian-statistics-whats/ "good reasons") to analyse your data using Bayesian methods.Historically, however, these methods have been computationally intensive and difficult to implement, requiring knowledge of sometimes challenging coding platforms and languages, like [WinBUGS](https://www.mrc-bsu.cam.ac.uk/software/bugs/the-bugs-project-winbugs/ "WinBUGS"), [JAGS](http://mcmc-jags.sourceforge.net/ "JAGS"), or [Stan](http://mc-stan.org/ "Stan"). Newer R packages, however, including, [r2jags](https://cran.r-project.org/web/packages/R2jags/index.html "r2jags"), [rstanarm](https://cran.r-project.org/web/packages/rstanarm/index.html "rstanarm"), and [brms](https://cran.r-project.org/web/packages/brms/index.html "brms") have made building Bayesian regression models in R relatively [straightforward](https://thinkinator.com/2016/01/12/r-users-will-now-inevitably-become-bayesians/ "straightforward"). For some background on Bayesian statistics, there is a Powerpoint presentation [here](https://www.k-state.edu/cnap/Bayesian%20statistics%20powerpoint.pptx "here").

Here I will introduce code to run some simple regression models using the brms package. This package offers a little more flexibility than rstanarm, although the both offer many of the same functionality. I encourage you to check out the extremely helpful [vignettes](https://github.com/paul-buerkner/brms/tree/master/vignettes "vignettes") written by Paul Buerkner. Paul's [Github](https://github.com/paul-buerkner/brms "Github") page is also a useful resource. I won't go into too much detail on prior selection, or demonstrating the full flexibility of the brms package (for that, check out the vignettes), but I will try to add useful links where possible. I will also go a bit beyond the models themselves to talk about model selection using loo, and model averaging




## Packages
First, lets load the packages, the most important being brms.

```{r packages}
require(brms)
require(ggplot2)
require(gdata)
require(dplyr)
require(parallel)
require(cowplot)

```

## The Data
For this analysis, I am going to use the diamonds dataset, from ggplot2. Because it is pretty large, I am going to subset it.

```{r loading_data}
set.seed(2018)
data(diamonds)

diamonds.full <- na.omit(diamonds) %>% drop.levels # remove missing data

diamonds.full$rows <- as.numeric(rownames(diamonds.full))

diamonds.train <- diamonds.full %>% group_by(color, clarity) %>% sample_n(size = 30) # subset
  
diamonds.keep <-   filter(diamonds.full, !rows %in% diamonds.train$rows) # remove row in training set

diamonds.test <- diamonds.keep[sample(nrow(diamonds.keep), 20000), ]

# get rid of zero values and outliers

diamonds.train <- diamonds.train %>% filter( x> 0, y >0, z > 0, price > 0, price < max(diamonds$price))

diamonds.test <- diamonds.test %>% filter( x> 0, y >0, z > 0, price > 0, price < max(diamonds$price))





```

## Setting the number of cores
Because these analyses can sometimes be a little sluggish, it is recommended to set the number of cores you use to the maximum number available. You can check how many cores you have available with the following code. 

```{r setting_cores}
getOption("mc.cores", 1)

```

We'll use this bit of code again when we are running our models and doing model selection.

## Examining and visualizing data
What I am interested in is how well the properties of a diamond predict it's price. Does the size of the diamond matter? What is the relative importance of color vs clarity?  Given that the answer to both of these questions is almost certainly yes, let's see if the models tell us the same thing. To get a description of the data, let's use the help function.

```{r data_description}
help(diamonds)

```


Let's take a look at the data. First let's plot price as a function carat, a well-know metric of diamond quality. Here I plot the raw data and then both variables log-transformed.

```{r visualize_data}

plot_grid(
  ggplot()+
    
    geom_point(data = diamonds.train, aes(y = price, x = carat))+
    
    geom_smooth(), 

  ggplot()+
    
    geom_point(data = diamonds.train, aes(y = log(price), x = log(carat)))+
    
    geom_smooth(), ncol = 2

  )

```

We might considering logging price before running our models with a Gaussian family, or consider using a different link function (e.g. log). First, let's visualize how clarity and color influence price. Here I will first plot boxplots of price by level for clarity and color, and then price vs carat, with colors representing levels of clarity and color.

```{r visualize_data_3}

plot_grid(
  
  ggplot(data = diamonds.train, aes(y = log(price), x = clarity)) + geom_boxplot(), 
  
  ggplot(data = diamonds.train, aes(y = log(price), x = color)) + geom_boxplot(), 
  
  ggplot(data = diamonds.train, aes(y = log(price), x = log(carat), color = clarity)) + 
    
         geom_point() + geom_smooth(method = "lm") + 
         
         ylim(c(min(log(diamonds$price)),  max(log(diamonds$price)))), 
  
  ggplot(data = diamonds.train, aes(y = log(price), x = log(carat), color = color)) + 
         
         geom_point() + geom_smooth(method = "lm") + 
    
         ylim(c(min(log(diamonds$price)), max(log(diamonds$price)))), 

  ncol = 2
)


```

From these plots, it looks as if there may be differences in the intercepts and slopes (especially for clarity) between color and clarity classes. We can model this using a mixed effects model. But let's start with simple multiple regression.

## Fitting models
For this first model, we will look at how well diamond 'carat' correlates with price. 

```{r fitting_models_1}
brm.1 <- brm(log(price) ~ log(carat), 
             
             brmsfamily("gaussian"), 
     
             data = na.omit(diamonds.train), 
             
             chains = 4, #specify the number of Markov chains
             
             cores = getOption("mc.cores", 1),
             
             iter = 3000, warmup = 1500, thin = 5,
             
             prior =  c(prior(normal(0, 3), "b"), # set normal prior on regression coefficients (mean of 0, location of 3)
                        
                        prior(normal(0, 3), "Intercept"))) # set normal prior on intercept (mean of 0, location of 3)
```

This might take a few minutes to run, depending on the speed of your machine.

A really fantastic tool for interrogating your model is using the 'launch_shinystan' function, which you can call as: 

'launch_shinystan(brm.1)'.

For now, we will take a look at a summary of the models in R, as well as plots of the posterior distributions and the Markov chains. We can see from the summary that our chains have converged sufficiently (rhat = 1).

We can also get an R-squared estimate for our model, thanks to a newly-developed method from Andrew Gelman, Ben Goodrich, Jonah Gabry and Imad Ali, with an explanation [here](http://www.stat.columbia.edu/~gelman/research/unpublished/bayes_R2.pdf "here").

```{r fitting_models_2}
summary(brm.1) # note Population-Level Effects = 'fixed effects'

plot(brm.1)

stanplot(brm.1, pars = c("b_Intercept", "b_logcarat")) + xlim(c(0, 9))

bayes_R2(brm.1)

```

Note that log(carat) clearly explains a lot of the variation in diamond price (as we'd expect), with a significantly positive slope (1.52 +- 0.01).


## Model validation using approximate leave-one-out cross-validation
Another way to get at the model fit is approximate leave-one-out cross-validation, via the [loo](http://mc-stan.org/loo/reference/loo-package.html "loo") package, developed by Vehtari, Gelman, and Gabry ( [2017a](https://link.springer.com/article/10.1007%2Fs11222-016-9696-4 "2017a"), [2017b](https://arxiv.org/abs/1507.02646 "2017b") ). Using loo, we can compute a LOOIC, which is similar to an [AIC](https://en.wikipedia.org/wiki/Akaike_information_criterion "AIC"), which some readers may be familiar with. 

```{r fitting_models_3}
loo(brm.1, cores = getOption("mc.cores", 1))
plot(loo(brm.1, cores = getOption("mc.cores", 1)))

```

The plot of the loo shows the Pareto shape k parameter for each data point. This parameter is used to test the reliability and convergence rate of the PSIS-based estimates. For more details, check out the help and the references above. For our purporses, we want to ensure that no data points have too high values of this parameter. The default threshold for a high value is k > 0.7.

## Mixed effects models
We can also run models including group-level effects (also called random effects). Here I will run models with clarity and color as grouping levels, first separately and then together in an 'overall' model.

Here's the model with clarity as the group-level effect. I have also run the function 'loo', so that we can compare models. The model with the lowest LOOIC is the better model. 

```{r fitting_models_4}
brm.2 <- brm(log(price) ~ log(carat) + (1|clarity), 
             
             brmsfamily("gaussian"), 
     
             data = na.omit(diamonds.train), 
             
             chains = 4, #specify the number of Markov chains
             
             cores = getOption("mc.cores", 1),
             
             iter = 3000, warmup = 1500, thin = 5,
             
             prior =  c(prior(normal(0, 3), "b"), # set normal prior on regression coefficients (mean of 0, location of 3)
                        
                        prior(normal(0, 3), "Intercept"))) # set normal prior on intercept (mean of 0, location of 3)

loo(brm.1, brm.2)

```

And now with color.

```{r fitting_models_5}
brm.3 <- brm(log(price) ~ log(carat)  + (1|color), 
             
             brmsfamily("gaussian"), 
     
             data = na.omit(diamonds.train), 
             
             chains = 4, #specify the number of Markov chains
             
             cores = getOption("mc.cores", 1),
             
             iter = 3000, warmup = 1500, thin = 5,
             
             prior =  c(prior(normal(0, 3), "b"), # set normal prior on regression coefficients (mean of 0, location of 3)
                        
                        prior(normal(0, 3), "Intercept"))) # set normal prior on intercept (mean of 0, location of 3)

loo(brm.1, brm.2, brm.3)

```


And here's a model with the log of carat as the fixed effect and color and clarity as group-level effects.

```{r fitting_models_6}
brm.4 <- brm(log(price) ~ log(carat) + (1|color) + (1|clarity), 
             
             brmsfamily("gaussian"), 
     
             data = na.omit(diamonds.train), 
             
             chains = 4, #specify the number of Markov chains
             
             cores = getOption("mc.cores", 1),
             
             iter = 3000, warmup = 1500, thin = 5,
             
             prior =  c(prior(normal(0, 3), "b"), # set normal prior on regression coefficients (mean of 0, location of 3)
                        
                        prior(normal(0, 3), "Intercept"))) # set normal prior on intercept (mean of 0, location of 3)

loo(brm.1, brm.2, brm.3, brm.4)

bayes_R2(brm.4)

summary(brm.4)
coef(brm.4)

```

All of the mixed effects models we have looked at so far have only allowed the intercepts of the groups to vary, but, as we saw when we were looking at the data, it seems as if different levels of our groups could have different slopes too. We can specify a model that allow the slope of the price~carat relationship to cary by both color and clarity.

```{r fitting_models_7}
brm.5 <- brm(log(price) ~ log(carat)  + (1 + log(carat)|color) + (1 + log(carat)|clarity), 
             
             brmsfamily("gaussian"), 
     
             data = na.omit(diamonds.train), 
             
             chains = 4, #specify the number of Markov chains
             
             cores = getOption("mc.cores", 1),
             
             iter = 3000, warmup = 1500, thin = 5,
             
             prior =  c(prior(normal(0, 3), "b"), # set normal prior on regression coefficients (mean of 0, location of 3)
                        
                        prior(normal(0, 3), "Intercept"))) # set normal prior on intercept (mean of 0, location of 3)

```


We can now compare our models using 'loo'.

```{r fitting_models_8}
loo(brm.1, brm.2, brm.3, brm.4, brm.5, cores = getOption("mc.cores", 1))
```

It looks like the final model we ran is the best model. Let's take a look at the Bayesian R-squared value for this model, and take a look at the model summary. We can also get more details on the coefficients using the 'coef' function. Similarly we could use 'fixef' for population-level effects and 'ranef' from group-level effects. 

```{r fitting models_8}
bayes_R2(brm.4)

summary(brm.4)

coef(brm.4)

```

Clearly, the variables we have included have a really strong influence on diamond price! 

## Posterior probability checking using pp_check
The [pp_check](https://cran.r-project.org/web/packages/bayesplot/vignettes/graphical-ppcs.html, "pp_check") allows for graphical posterior predictive checking. We can generate figures to compare the observed data to simulated data from the posterior predictive distribution. This is a great graphical way to evaluate your model. 

There are many different options of plots to choose from. In the first plot I use density plots, where the observed y values are plotted with expected values from the posterior distribution.

```{r posterior_checking}
pp_check(brm.5, nsamples = 100)

```

It is good to see that our model is doing a fairly good job of capturing the slight bimodality in logged diamond prices, althogh specifying a different family of model might help to improve this.

Here, 'nsamples' refers to the number of draws from the posterior distribution to use to calculate yrep values. 

We can aslo look at the fit based on groups. Here, for example, are scatteplots with the observed prices (log scale) on the y-axis and the average (across all posterior samples) on the x-axis.


```{r posterior_checking_2}
pp_check(brm.5, type = "scatter_avg_grouped", group = "color") + 
  
  geom_abline(intercept = 0, slope = 1 , color = "red", lty = 2)

```

## Model prediction
Finally, we can evaluate how well our model does at predicting diamond data that we held out. We can use the 'predict' function (as we would with a more standard model). We can also get estimates of error around each data point! We can plot the prediction using ggplot2.

```{r prediction}

pred.1 <- predict(brm.5, newdata = diamonds.test)

r.sq <- as.character(round(summary(lm(log(diamonds.test$price) ~ pred.1[, 1]))$r.squared, 2))

lb1 <- paste("R^2 == ", r.sq)

ggplot()+
  
  geom_point(aes(x = pred.1[,1], y = log(diamonds.test$price))) +
  
  geom_errorbarh(aes(x = pred.1[,1], y = log(diamonds.test$price), 
                     
                     xmin = pred.1[,1] - pred.1[, 2], 
                     xmax = pred.1[,1] + pred.1[, 2])) + 
  
  geom_smooth(aes(x = pred.1[,1], y = log(diamonds.test$price)), method = "lm", color = "red", lty = 2)+
  
  geom_text(aes(x=6, y=9.5, label = lb1, size = 8), parse=TRUE, show.legend = F)
            
            
```

That's all for this post.

Thanks for reading!

##Contact Me

Please check out my personal website at [timothyemoore.com](https://www.timothyemoore.com/ "timothyemoore.com")
