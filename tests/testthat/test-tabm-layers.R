##########################################
#########  checks of the custom TabM layers and the TabM bCCNN builder
#########  of R/nn_models.R
#########  TabM: Gorishniy, Kotelnikov & Babenko (2025), (T2) and (T5)
##########################################

## number of trainable parameters (count_params() also counts the frozen
## embedding weights)
n_trainable <- function(model) {
  sum(sapply(model$trainable_weights, function(w) length(as.array(w))))
}

test_that("BatchEnsembleLinear equals the per-submodel matrices (T2)", {
  needs_keras()
  set.seed(11)
  k <- 4L
  q_in <- 6L
  q_out <- 5L
  lyr <- layer_batch_ensemble(q_out = q_out,
                              k = k,
                              r_initial = matrix(rnorm(k * q_in), k, q_in))
  z <- array(rnorm(7 * k * q_in), dim = c(7L, k, q_in))
  invisible(lyr(op_convert_to_tensor(z, dtype = "float32")))  # builds it
  w <- get_weights(lyr)
  # weights in the order b, r, s, c
  expect_equal(lapply(w, dim),
               list(c(q_in, q_out), c(k, q_in), c(k, q_out), c(k, q_out)))
  # adapters s and c away from 1 and 0, then the (T2) form:
  # submodel l has the matrix b * outer(r_l, s_l) and the bias c_l
  w[[3]] <- matrix(rnorm(k * q_out), k, q_out)
  w[[4]] <- matrix(rnorm(k * q_out), k, q_out)
  set_weights(lyr, w)
  out <- as.array(lyr(op_convert_to_tensor(z, dtype = "float32")))
  for (l in seq_len(k)) {
    w_l <- w[[1]] * outer(w[[2]][l, ], w[[3]][l, ])
    expected <- z[, l, ] %*% w_l + matrix(w[[4]][l, ], 7L, q_out, byrow = TRUE)
    expect_equal(out[, l, ], expected, tolerance = 1e-5)
  }
})

test_that("PackedLinear is k linear maps with Glorot-bounded kernels", {
  needs_keras()
  set.seed(12)
  k <- 20L
  q_in <- 15L
  q_out <- 10L
  lyr <- layer_packed_linear(q_out = q_out, k = k)
  z <- array(rnorm(9 * k * q_in), dim = c(9L, k, q_in))
  out <- as.array(lyr(op_convert_to_tensor(z, dtype = "float32")))
  w <- get_weights(lyr)
  expect_equal(dim(w[[1]]), c(k, q_in, q_out))
  expect_equal(dim(w[[2]]), c(k, q_out))
  for (l in c(1L, 7L, k)) {
    expected <- z[, l, ] %*% w[[1]][l, , ] +
      matrix(w[[2]][l, ], 9L, q_out, byrow = TRUE)
    expect_equal(out[, l, ], expected, tolerance = 1e-5)
  }
  # TabM's initialiser: U(-b, b) with Glorot's b on every q_in x q_out
  # slice (Keras's glorot_uniform on the 3-D kernel has k times the fans)
  b <- sqrt(6 / (q_in + q_out))
  expect_true(max(abs(w[[1]])) <= b)
  expect_true(abs(sd(w[[1]]) / (b / sqrt(3)) - 1) < 0.15)
  expect_true(all(w[[2]] == 0))
})

test_that("the TabM bCCNN starts as TabM prescribes (T5)", {
  needs_keras()
  odp <- get_odps()[[1]]
  k <- 20L
  set_random_seed(3L)
  model <- nn_model(odp, test_param("tabm_tanh_signs", k))
  be <- lapply(1:3, function(d) get_weights(get_layer(model, paste0("be", d))))
  # the first adapter r in {-1, +1}, the other two exactly 1
  expect_setequal(unique(as.numeric(be[[1]][[2]])), c(-1, 1))
  for (d in 2:3) expect_true(all(be[[d]][[2]] == 1))
  # s exactly 1 and c exactly 0
  for (d in 1:3) {
    expect_true(all(be[[d]][[3]] == 1))
    expect_true(all(be[[d]][[4]] == 0))
  }
  # heads: kernels 0, biases c_ODP
  head <- get_weights(get_layer(model, "head"))
  expect_true(all(head[[1]] == 0))
  expect_equal(as.numeric(head[[2]]), rep(odp$intercept, k), tolerance = 1e-6)
  expect_false(get_layer(model, "AY_embed")$trainable)
  expect_false(get_layer(model, "DY_embed")$trainable)
  expect_equal(as.numeric(get_weights(get_layer(model, "AY_embed"))[[1]]),
               odp$alpha,
               tolerance = 1e-6)
  expect_equal(n_trainable(model), 490 + k * 138)
  #
  # 2 k distinct N(0, 1) draws in the first adapter
  model_n <- nn_model(odp, test_param("tabm_tanh_normal", k))
  r1 <- get_weights(get_layer(model_n, "be1"))[[2]]
  expect_equal(length(unique(as.numeric(r1))), 2L * k)
  #
  # parameter counts: deep ensemble, one packed network and the single
  # bCCNN (Paper C's count, with w)
  packed <- nn_model(odp, test_param("packed_tanh", k))
  expect_equal(n_trainable(packed), k * 546)
  expect_equal(n_trainable(nn_model(odp, test_param("packed_tanh", 1L))), 546)
  expect_equal(n_trainable(nn_model(odp, test_param("single_bccnn"))), 547)
  #
  relu <- nn_model(odp, test_param("tabm_relu_signs", 4L))
  act <- get_layer(relu, "act1")
  expect_true(grepl("relu", act$activation$`__name__`, ignore.case = TRUE) ||
                identical(act$get_config()$activation, "relu"))
})

test_that("the custom layers' weights keep their add_weight order and names", {
  needs_keras()
  model <- nn_model(get_odps()[[2]], test_param("tabm_tanh_signs", 3L))
  nms <- sapply(get_layer(model, "be2")$weights, function(w) w$name)
  expect_equal(nms, c("B", "R", "S", "C"))
  nms <- sapply(get_layer(model, "head")$weights, function(w) w$name)
  expect_equal(nms, c("kernel", "bias"))
})
