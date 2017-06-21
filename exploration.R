library(dplyr)
library(mlr)
library(ggplot2)

mieszkania <- na.omit(read.csv("https://raw.githubusercontent.com/STWUR/STWUR-2017-06-07/master/data/mieszkania_dane.csv",
                               fileEncoding = "UTF-8")) 

#' na które mieszkania stać Wrocławian? ---------------------------------
#' 
#' Według GUS (http://wroclaw.stat.gov.pl/) przeciętne (średnie)
#' wyngrodzenie we Wrocławiu wyniosło 4532,14 zł brutto.
#' To jest 3 223,81 zł netto
pensja <- 3223.81
#odjmiemy 1500 złotych na życie (jedzenie, ubrania, rozrywkę)
max_rata <- pensja - 1500

#zalozmy kredyt na 5% rocznie 

# rata = kwota kredytu * (1 + procent/12)^ liczba_rat * ((1 + procent/12)-1)/((1 + procent/12)^liczba_rat-1)
liczba_rat <- 25*12
r <- 0.05
max_rata/(1 + r/12)^liczba_rat/((1 + r/12)-1)*((1 + r/12)^liczba_rat-1)

#wychodzi 300K kredytu

tanie_mieszkania <- mieszkania %>%
  mutate(cena = metraz*cena_m2,
         tanie = factor(cena < 300000)) %>%
  select(-cena, -cena_m2)

predict_affordable <- makeClassifTask(id = "affordableApartments",
                                      data = tanie_mieszkania, target = "tanie")

learnerNN <- makeLearner("classif.nnet", predict.type = "prob")


all_params <- makeParamSet(
  makeDiscreteParam("size", values = c(1, 3, 4, 5)),
  makeDiscreteParam("decay", values = seq(0.3, 0.8, length.out = 5))
)

set.seed(1792)

affordable_res <- tuneParams(learnerNN, task = predict_affordable, 
                             resampling = makeResampleDesc("CV", iters = 10L),
                             par.set = all_params, 
                             control =  makeTuneControlGrid(),
                             measures = list(auc))



as.data.frame(affordable_res[["opt.path"]]) %>% 
  ggplot(aes(x = size, y = auc.test.mean, color = as.factor(decay))) +
  geom_point() +
  theme_bw()

chosen_predictor <- train(makeLearner("classif.nnet", size=3, decay=0.3), predict_affordable)

predict(chosen_predictor, newdata = data.frame(n_pokoj = 3, 
                                               metraz = 55, 
                                               rok = 1920, 
                                               pietro = 3, 
                                               pietro_maks = 7, 
                                               dzielnica = "Stare Miasto"))

save(res, affordable_res, file = "./results/tuning.RData")

# do sprawdzenia: głębokie sieci neuronowe i lasy losowe
makeLearner("classif.dbnDNN", predict.type = "prob")
makeLearner("classif.saeDNN", predict.type = "prob")
makeLearner("classif.randomForest", predict.type = "prob")