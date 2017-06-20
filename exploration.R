library(dplyr)
library(mlr)

mieszkania <- na.omit(read.csv("https://raw.githubusercontent.com/STWUR/STWUR-2017-06-07/master/data/mieszkania_dane.csv",
                               fileEncoding = "UTF-8")) %>%
  mutate(cena = metraz*cena_m2,
         tanie = factor(cena < 300000)) %>%
  select(-cena)


predict_affordable <- makeClassifTask(id = "affordableApartments", 
                                      data = mieszkania, target = "tanie")

predict_price <- makeRegrTask(id = "affordableApartments", 
                              data = mutate(mieszkania, tanie = factor(tanie)), target = "cena_m2")

learnerNN <- makeLearner("regr.nnet")


all_params <- makeParamSet(
  makeDiscreteParam("size", values = 1L:10),
  makeDiscreteParam("decay", values = seq(0, 0.1, by=0.005))
)

res <- tuneParams(learnerNN, task = predict_affordable, 
                  resampling = makeResampleDesc("CV", iters = 5L),
                  par.set = all_params, 
                  control =  makeTuneControlGrid())

