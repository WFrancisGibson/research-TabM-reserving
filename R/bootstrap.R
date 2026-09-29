##########################################
#########  Prediction uncertainty: parametric ODP bootstrap
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Sections 2.3,
#########  3.3.2 and 3.3.4
##########################################

## conditional RMSEP (15): process variance phi * reserve (16) plus the
## parameter error, the standard deviation of the bootstrap reserves
bootstrap_result <- function(reserves, reserve, phi, n_failed) {
  parameter_error <- if (length(reserves) > 1) sd(reserves) else NA_real_
  process_variance <- phi * reserve
  list(reserves = reserves,
       reserve = reserve,
       phi = phi,
       n_failed = n_failed,
       parameter_error = parameter_error,
       process_variance = process_variance,
       rmsep = sqrt(process_variance + parameter_error^2))
}

## one bootstrap triangle from assumption (1): Y* = phi * Poisson(mu / phi)
## on the observed cells (mean mu, variance phi * mu), 0 on the future cells
sample_triangle <- function(mu, phi, mask) {
  y <- matrix(rpois(length(mu), pmax(mu, 0) / phi) * phi, nrow(mu), ncol(mu))
  triangle(ifelse(mask, y, 0), mask)
}

## ccODP bootstrap (Paper C Sec. 2.3): the ccODP refitted on every replicate
bootstrap_ccodp <- function(tri, mu, phi, reserve, replicates, seed) {
  set.seed(seed)
  reserves <- numeric()
  n_failed <- 0L
  for (r in seq_len(replicates)) {
    boot <- sample_triangle(mu, phi, tri$train_mask)
    fit <- tryCatch(ccodp_fit(boot), error = function(e) NULL)
    if (is.null(fit)) {
      n_failed <- n_failed + 1L
    } else {
      reserves <- c(reserves, fit$reserve)
    }
  }
  bootstrap_result(reserves, reserve, phi, n_failed)
}

## network bootstrap (Paper C Sec. 3.3.4) around the fitted mean mu with the
## reduced dispersion phi: every replicate refits the ccODP and then the
## network (all k submodels) for 'epochs' steps;
## the replicates are drawn up front because set_random_seed() also reseeds
## R, and replicate r trains with seed * 1000 + r - 1
bootstrap_nn <- function(tri, mu, phi, reserve, epochs, param, replicates,
                         seed) {
  set.seed(seed)
  draws <- lapply(seq_len(replicates),
                  function(r) sample_triangle(mu, phi, tri$train_mask))
  reserves <- numeric()
  n_failed <- 0L
  for (r in seq_len(replicates)) {
    param$seed <- as.integer(seed * 1000L + r - 1L)
    fit <- tryCatch({
      odp <- ccodp_fit(draws[[r]])
      nn <- nn_fit(odp, draws[[r]], epochs, param,
                   curves = FALSE, snapshots = epochs)
      collective_reserve(nn$snapshots[[as.character(epochs)]],
                         tri$train_mask)
    }, error = function(e) {
      message("replicate ", r, " failed: ", conditionMessage(e))
      NULL
    })
    if (is.null(fit)) {
      n_failed <- n_failed + 1L
    } else {
      reserves <- c(reserves, fit)
    }
    # Keras keeps the graphs of every model built: clear them now and then
    if (r %% 5L == 0L) {
      clear_session()
      invisible(gc())
      cat("    replicate", r, "of", replicates, "\n")
    }
  }
  bootstrap_result(reserves, reserve, phi, n_failed)
}

## network dispersion (Paper C Sec. 3.3.2): the ccODP phi reduced by the
## relative decrease (%) of the validation deviance, an increase counts as 0
## (INFERRED: time-aware split, collective validation deviance at the stop)
reduced_phi <- function(phi, decrease_pct) {
  phi * (1 - max(decrease_pct, 0) / 100)
}
