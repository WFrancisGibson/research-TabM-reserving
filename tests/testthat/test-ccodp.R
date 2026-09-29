##########################################
#########  checks of the ccODP model (R/fit ODP GLM.R)
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Section 2
#########  the ccODP reserves are the chain-ladder reserves (Remark 2.3)
##########################################

test_that("ccODP reproduces the chain-ladder total reserve", {
  for (d in get_lob_data()) {
    odp <- ccodp_fit(d$full)
    cl <- sum(cl_reserves(d$full))
    expect_lt(abs(odp$reserve - cl) / cl, 1e-10)
  }
})

test_that("ccODP reproduces the chain-ladder reserves by origin", {
  for (d in get_lob_data()) {
    odp <- ccodp_fit(d$full)
    cl <- unname(cl_reserves(d$full))
    keep <- cl > 0
    expect_equal(odp$reserve_o[keep], cl[keep], tolerance = 1e-9)
  }
})

test_that("corner parametrisation", {
  for (odp in get_odps()) {
    expect_identical(odp$alpha[1], 0)
    expect_identical(odp$beta[1], 0)
    expect_length(odp$alpha, 12L)
    expect_length(odp$beta, 12L)
  }
})

test_that("deviance-based dispersion on 78 - 23 = 55 degrees of freedom", {
  for (odp in get_odps()) {
    expect_lt(abs(odp$phi - odp$deviance / 55), 1e-12)
  }
})

test_that("dispersion is over-dispersed", {
  for (odp in get_odps()) expect_gt(odp$phi, 1)
})

test_that("fitted means match the observations in aggregate", {
  for (d in get_lob_data()) {
    odp <- ccodp_fit(d$full)
    y <- d$full$incremental[d$full$train_mask]
    fitted <- odp$mu[d$full$train_mask]
    expect_lt(abs(sum(fitted) - sum(y)) / sum(y), 1e-9)
  }
})

test_that("Poisson deviance is zero at saturation", {
  y <- c(3.0, 0.0, 17.5)
  expect_equal(poisson_deviance(y, y), 0, tolerance = 1e-12)
})
