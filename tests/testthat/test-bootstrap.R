##########################################
#########  checks of R/bootstrap.R
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Sections 2.3,
#########  3.3.2 and 3.3.4
##########################################

test_that("sample_triangle has mean mu and variance phi * mu", {
  set.seed(3)
  mask <- runoff_mask(12)
  mu <- matrix(2000, 12, 12)
  phi <- 8
  draws <- replicate(2000, sample_triangle(mu, phi, mask)$incremental[1, 1])
  expect_equal(mean(draws), 2000, tolerance = 0.02)
  expect_equal(var(draws), phi * 2000, tolerance = 0.10)
  # 0 on the future cells
  expect_true(all(sample_triangle(mu, phi, mask)$incremental[!mask] == 0))
})

test_that("reduced_phi floors a negative validation decrease at zero", {
  expect_equal(reduced_phi(10, 20), 8)
  expect_equal(reduced_phi(10, -5), 10)
  expect_equal(reduced_phi(10, 0), 10)
})

test_that("bootstrap_ccodp gives the chain-ladder scale of uncertainty", {
  d <- get_lob_data()[[1]]
  odp <- get_odps()[[1]]
  bs <- bootstrap_ccodp(d$full, odp$mu, odp$phi, odp$reserve, 40L, 1L)
  expect_equal(bs$n_failed, 0L)
  expect_length(bs$reserves, 40L)
  expect_true(bs$rmsep > 0 && bs$rmsep < 0.2 * odp$reserve)
  expect_equal(bs$process_variance, odp$phi * odp$reserve)
})

test_that("bootstrap_nn runs end to end and is reproducible", {
  needs_keras()
  d <- get_lob_data()[[1]]
  odp <- get_odps()[[1]]
  param <- test_param("tabm_tanh_signs", 4L)
  a <- bootstrap_nn(d$full, odp$mu, odp$phi, odp$reserve, 5L, param, 2L, 9L)
  b <- bootstrap_nn(d$full, odp$mu, odp$phi, odp$reserve, 5L, param, 2L, 9L)
  expect_equal(a$n_failed, 0L)
  expect_length(a$reserves, 2L)
  expect_equal(a$reserves, b$reserves, tolerance = 1e-6)
  expect_true(is.finite(a$rmsep))
})
