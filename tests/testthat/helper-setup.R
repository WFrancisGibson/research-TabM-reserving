##########################################
#########  shared fixtures of the checks in tests/testthat
#########  the claims (~65 MB) and the triangles are built once per run
##########################################

library(testthat)
source(here::here("analysis", "00_setup.R"))

claims_path <- file.path(paths$raw, "simulated_claims.csv")
quick_tests <- nzchar(Sys.getenv("TABM_QUICK_TESTS"))
fixtures <- new.env()

skip_without_data <- function() {
  skip_if_not(file.exists(claims_path),
              paste("no simulated claims at", claims_path))
}

## the claims, the six LoBs' triangles and their ccODP fits
get_claims <- function() {
  skip_without_data()
  if (is.null(fixtures$claims)) fixtures$claims <- fread(claims_path)
  fixtures$claims
}
get_lob_data <- function() {
  if (is.null(fixtures$lob_data)) {
    fixtures$lob_data <- lob_triangles(get_claims(), cfg$data)
  }
  fixtures$lob_data
}
get_odps <- function() {
  if (is.null(fixtures$odps)) {
    fixtures$odps <- lapply(get_lob_data(), function(d) ccodp_fit(d$full))
  }
  fixtures$odps
}

## Keras checks: skipped with TABM_SKIP_KERAS=1 or without the torch venv
needs_keras <- function() {
  skip_if(nzchar(Sys.getenv("TABM_SKIP_KERAS")), "TABM_SKIP_KERAS set")
  skip_if_not(file.exists(cfg$keras$python), "torch venv not found")
  # starting Python warns about py_require() packages this venv does not need
  suppressWarnings({
    library(keras3)
    config_backend()
  })
}

## hyper-parameters of configuration 'label', built as the scripts build them
test_param <- function(label, k = cfg$tabm$k, seed = 0L) {
  param <- list(hidden = cfg$bccnn$hidden,
                dropout = cfg$bccnn$dropout,
                learning_rate = cfg$bccnn$learning_rate,
                rho = cfg$bccnn$rho,
                epsilon = cfg$bccnn$epsilon,
                k = k,
                seed = seed)
  modifyList(param, cfg$configs[[label]])
}

## chain-ladder reserves by accident year (ChainLadder package)
cl_reserves <- function(tri) {
  n <- nrow(tri$incremental)
  cum <- incr2cum(as.triangle(ifelse(tri$train_mask, tri$incremental, NA)))
  predict(chainladder(cum))[, n] - getLatestCumulative(cum)
}
