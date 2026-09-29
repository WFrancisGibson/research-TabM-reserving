##########################################
#########  checks of the ccODP start and the training of the networks
#########  TabM bCCNN (T6): every submodel and the mean start in the ccODP;
#########  Paper C (14) for the single bCCNN (R/nn_models.R)
##########################################

tol <- 1e-5
k_test <- if (quick_tests) 4L else 20L
lobs <- if (quick_tests) 1L else 1:6

test_that("(T6) every submodel and the mean start at the ccODP means", {
  needs_keras()
  odps <- get_odps()
  lob_data <- get_lob_data()
  for (m in lobs) {
    for (label in names(cfg$configs)) {
      set_random_seed(m)
      model <- nn_model(odps[[m]], test_param(label, k_test))
      mu_sub <- nn_predict(model, odps[[m]]$n)
      for (l in seq_len(dim(mu_sub)[1])) {
        gap <- max(abs(mu_sub[l, , ] - odps[[m]]$mu) / odps[[m]]$mu)
        expect_lt(gap, tol, label = paste("LoB", m, label, "submodel", l))
      }
      reserve <- collective_reserve(mu_sub, lob_data[[m]]$full$train_mask)
      expect_equal(reserve,
                   odps[[m]]$reserve,
                   tolerance = tol,
                   label = paste("LoB", m, label, "initial reserve"))
    }
  }
})

test_that("zero steps leave the ccODP model and record its deviance", {
  needs_keras()
  d <- get_lob_data()[[1]]
  odp <- get_odps()[[1]]
  for (label in c("tabm_tanh_signs", "packed_tanh", "single_bccnn")) {
    h <- nn_fit(odp, d$full, 0L, test_param(label, k_test))$history
    expect_length(h$train_collective, 1L)
    expect_equal(h$train_collective[1],
                 odp$deviance,
                 tolerance = 1e-4,
                 label = label)
    expect_equal(h$train_individual[1],
                 odp$deviance,
                 tolerance = 1e-4,
                 label = label)
    expect_length(h$validation_collective, 0L)
  }
})

test_that("training lowers the deviance, respects Jensen, keeps alpha, beta", {
  needs_keras()
  d <- get_lob_data()[[1]]
  split <- time_aware_split(d$full, 2L)
  odp <- ccodp_fit(split$train)
  fit <- nn_fit(odp,
                split$train,
                150L,
                test_param("tabm_tanh_signs", k_test),
                vali = split$validation,
                snapshots = c(50L, 150L))
  h <- fit$history
  expect_length(h$train_collective, 151L)
  expect_lt(h$train_collective[151], h$train_collective[1])
  # Jensen: the deviance of the mean is never above the mean deviance
  expect_true(all(h$validation_collective <= h$validation_individual + 1e-6))
  expect_true(all(h$train_collective <= h$train_individual + 1e-6))
  # step 0 is the ccODP deviance of the training triangle
  expect_equal(h$train_collective[1], odp$deviance, tolerance = 1e-4)
  # snapshots at the requested steps, the last one is the trained model
  expect_setequal(names(fit$snapshots), c("50", "150"))
  expect_equal(dim(fit$snapshots[["150"]]), c(k_test, 10L, 10L))
  expect_equal(fit$snapshots[["150"]],
               nn_predict(fit$model, 10L),
               tolerance = 1e-6)
  # the embeddings alpha_i, beta_j stay frozen
  expect_equal(as.numeric(get_weights(get_layer(fit$model, "AY_embed"))[[1]]),
               odp$alpha,
               tolerance = 1e-6)
  expect_equal(as.numeric(get_weights(get_layer(fit$model, "DY_embed"))[[1]]),
               odp$beta,
               tolerance = 1e-6)
  # the adapters other than the first one moved away from 1 (TabM)
  expect_true(any(get_weights(get_layer(fit$model, "be2"))[[2]] != 1))
})

test_that("snapshots equal separate runs stopped at those steps", {
  needs_keras()
  d <- get_lob_data()[[3]]
  odp <- get_odps()[[3]]
  p <- test_param("tabm_tanh_signs", 4L, seed = 7L)
  one_run <- nn_fit(odp, d$full, 40L, p, curves = FALSE,
                    snapshots = c(15L, 40L))
  short_run <- nn_fit(odp, d$full, 15L, p, curves = FALSE, snapshots = 15L)
  expect_equal(one_run$snapshots[["15"]],
               short_run$snapshots[["15"]],
               tolerance = 1e-5)
})

test_that("k = 1 packed with a random head is a plain 20/15/10 tanh network", {
  needs_keras()
  odp <- get_odps()[[1]]
  n <- odp$n
  set_random_seed(5L)
  p <- test_param("packed_tanh", 1L)
  p$dropout <- 0
  model <- nn_model(odp, p)
  head <- get_weights(get_layer(model, "head"))
  head[[1]] <- array(rnorm(10), dim = c(1L, 10L, 1L))
  head[[2]] <- matrix(0.3, 1L, 1L)
  set_weights(get_layer(model, "head"), head)
  # accident and development year of every cell, row by row
  ay <- rep(1:n - 1L, each = n)
  dy <- rep(1:n - 1L, times = n)
  a <- odp$alpha[ay + 1L]
  b <- odp$beta[dy + 1L]
  z <- cbind(a, b)
  for (layer in c("packed1", "packed2", "packed3")) {
    w <- get_weights(get_layer(model, layer))
    z <- tanh(z %*% w[[1]][1, , ] +
                matrix(w[[2]][1, ], nrow(z), ncol(w[[1]][1, , ]), byrow = TRUE))
  }
  plain <- exp(a + b + 0.3 + as.numeric(z %*% head[[1]][1, , 1]))
  expect_equal(nn_predict(model, n)[1, , ],
               matrix(plain, n, n, byrow = TRUE),
               tolerance = 1e-5)
})

test_that("the single bCCNN goes through the same interface", {
  needs_keras()
  d <- get_lob_data()[[2]]
  odp <- get_odps()[[2]]
  fit <- nn_fit(odp,
                d$train,
                20L,
                test_param("single_bccnn", seed = 1L),
                vali = d$validation,
                snapshots = 20L)
  expect_equal(dim(fit$snapshots[["20"]]), c(1L, 12L, 12L))
  expect_equal(fit$history$train_individual, fit$history$train_collective)
  expect_equal(fit$history$validation_individual,
               fit$history$validation_collective)
  expect_length(fit$history$validation_collective, 21L)
  # Paper C's weight w on the skip connection is trainable here
  w <- get_weights(get_layer(fit$model, "Response"))
  expect_true(w[[1]][1, 1] != 1)
})
