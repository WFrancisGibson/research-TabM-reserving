##########################################
#########  TabM bCCNN reserving: project setup
#########  every analysis script and the tests source this file first
##########################################

suppressPackageStartupMessages({
  library(data.table)
  library(ChainLadder)
  library(ggplot2)
})

## settings (paths, hyper-parameters, run lengths) from config.yml;
## R_CONFIG_ACTIVE=quick selects the quick profile
cfg <- config::get(file = here::here("config.yml"))
paths <- lapply(cfg$paths, here::here)
for (p in paths) dir.create(p, showWarnings = FALSE, recursive = TRUE)
cat("config profile:", Sys.getenv("R_CONFIG_ACTIVE", "default"),
    "| tables in", paths$tables, "\n")

## Keras 3 runs on torch through reticulate: both variables are read when
## keras3 loads, so they are set before the functions in R/ are sourced
Sys.setenv(RETICULATE_PYTHON = cfg$keras$python,
           KERAS_BACKEND = cfg$keras$backend)
options(keras.view_metrics = FALSE, keras.fit_verbose = 0)

## functions in R/
for (f in list.files(here::here("R"), pattern = "\\.R$", full.names = TRUE)) {
  source(f)
}
