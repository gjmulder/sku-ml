# library(lubridate)
library(jsonlite)
# library(forecast)
library(tidyverse)
library(purrrlyr)

ts_to_json <- function(row) {
  names(row) <- NULL
  idx <- row[1]
  
  json <- (paste0(toJSON(
    list(
      start = start_date,
      target = na.omit(unlist(target[4:length(row)])),
      feat_static_cat = c(idx)
    ),
    auto_unbox = TRUE
  ), "\n"))
  return(json)
}

sample1 <- read.csv("sample1.csv")

sample1 %>%
  mutate(idx = ) %>%
  by_row(..f = ts_to_json) %>%
  select(.out) ->
  sample1_json