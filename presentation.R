## ----setup, include=FALSE,message=FALSE----------------------------------
library(mlr)
library(dplyr)
library(ggplot2)
load("./results/tuning.RData")
mieszkania <- na.omit(read.csv("https://raw.githubusercontent.com/STWUR/STWUR-2017-06-07/master/data/mieszkania_dane.csv", fileEncoding = "UTF-8")) 

## ------------------------------------------------------------------------
listLearners()

## ------------------------------------------------------------------------
predict_price <- makeRegrTask(id = "affordableApartments", 
                              data = mieszkania, target = "cena_m2")

learnerNN <- makeLearner("regr.nnet")

## ------------------------------------------------------------------------
all_params <- makeParamSet(
  makeDiscreteParam("size", values = c(1, 3, 4, 5)),
  makeDiscreteParam("decay", values = seq(0.3, 0.8, length.out = 5))
)

set.seed(1792)

res <- tuneParams(learnerNN, task = predict_price,
                  resampling = makeResampleDesc("CV", iters = 10L),
                  par.set = all_params,
                  control =  makeTuneControlGrid())

as.data.frame(res[["opt.path"]]) %>% 
  mutate(blad_cena = sqrt(mse.test.mean)) %>% 
  ggplot(aes(x = size, y = blad_cena, color = as.factor(decay))) +
  geom_point() +
  theme_bw()

## ------------------------------------------------------------------------
res

chosen_predictor <- train(makeLearner("regr.nnet", size=3, decay=0.55), predict_price)

## ------------------------------------------------------------------------
predict(chosen_predictor, newdata = data.frame(n_pokoj = 3, 
                                               metraz = 55, 
                                               rok = 1920, 
                                               pietro = 3, 
                                               pietro_maks = 7, 
                                               dzielnica = "Stare Miasto"))

