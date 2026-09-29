##########################################
#########  bCCNN and TabM bCCNN: networks started in the ccODP model
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Section 3, Listing 4
#########  TabM: Gorishniy, Kotelnikov & Babenko (2025), Sections 3.2-3.3
#########  equations (T1)-(T8): README.md, model notes
##########################################

## BatchEnsemble layer, TabM (T3), on tensors of shape (cells, k, q):
## the shared matrix b, rescaled by the adapters r (input) and s (output),
## plus the bias c; one row of r, s and c per submodel (T2)
layer_batch_ensemble <- keras3::Layer(
  "BatchEnsembleLinear",
  initialize = function(q_out,
                        k,
                        r_initial = NULL,
                        ...) {
    super$initialize(...)
    self$q_out <- as.integer(q_out)
    self$k <- as.integer(k)
    self$r_initial <- r_initial
  },
  build = function(input_shape) {
    q_in <- as.integer(tail(unlist(input_shape), 1))
    self$b <- self$add_weight(shape = shape(q_in, self$q_out),
                              initializer = "glorot_uniform",
                              name = "B")
    # r starts at 1, or at r_initial (the draw of the first adapter)
    r_init <- if (is.null(self$r_initial)) "ones" else
      initializer_constant(matrix(as.numeric(self$r_initial), self$k, q_in))
    self$r <- self$add_weight(shape = shape(self$k, q_in),
                              initializer = r_init,
                              name = "R")
    self$s <- self$add_weight(shape = shape(self$k, self$q_out),
                              initializer = "ones",
                              name = "S")
    self$c <- self$add_weight(shape = shape(self$k, self$q_out),
                              initializer = "zeros",
                              name = "C")
  },
  call = function(inputs) {
    op_matmul(inputs * self$r, self$b) * self$s + self$c
  }
)

## k separate linear maps in one einsum: every layer of the deep ensemble
## ("packed") and the k heads; kernels U(-bound, bound) with Glorot's bound
## per submodel (Keras's glorot_uniform on the 3-D kernel multiplies the fans
## by k)
layer_packed_linear <- keras3::Layer(
  "PackedLinear",
  initialize = function(q_out,
                        k,
                        kernel_initializer = NULL,
                        bias_initializer = "zeros", ...) {
    super$initialize(...)
    self$q_out <- as.integer(q_out)
    self$k <- as.integer(k)
    self$kernel_initializer <- kernel_initializer
    self$bias_initializer <- bias_initializer
  },
  build = function(input_shape) {
    q_in <- as.integer(tail(unlist(input_shape), 1))
    kernel_init <- self$kernel_initializer
    if (is.null(kernel_init)) {
      bound <- sqrt(6 / (q_in + self$q_out))
      kernel_init <- initializer_random_uniform(minval = -bound,
                                                maxval = bound)
    }
    self$kernel <- self$add_weight(shape = shape(self$k, q_in, self$q_out),
                                   initializer = kernel_init,
                                   name = "kernel")
    self$bias <- self$add_weight(shape = shape(self$k, self$q_out),
                                 initializer = self$bias_initializer,
                                 name = "bias")
  },
  call = function(inputs) {
    op_einsum("nkq,kqo->nko", inputs, self$kernel) + self$bias
  }
)

## single bCCNN, Paper C (13)-(14), Listing 4: the skip connection
## w * (alpha_i + beta_j) plus a network on (alpha_i, beta_j),
## started in the ccODP model w = 1, c = c_ODP, B = 0 (14)
bccnn_model <- function(odp, param) {
  n <- odp$n
  q0 <- param$hidden
  #
  # declare AY and DY features to be categorical
  accyear <- layer_input(shape = c(1L), dtype = "int32", name = "AccYear")
  devyear <- layer_input(shape = c(1L), dtype = "int32", name = "DevYear")
  #
  # embedding layers holding the ccODP alpha_i and beta_j, non-trainable
  ay_embed <- accyear %>%
    layer_embedding(
      input_dim = n,
      output_dim = 1L,
      trainable = FALSE,
      embeddings_initializer = initializer_constant(matrix(odp$alpha, n, 1L)),
      name = "AY_embed"
    ) %>%
    layer_flatten(name = "AY_flat")
  dy_embed <- devyear %>%
    layer_embedding(
      input_dim = n,
      output_dim = 1L,
      trainable = FALSE,
      embeddings_initializer = initializer_constant(matrix(odp$beta, n, 1L)),
      name = "DY_embed"
    ) %>%
    layer_flatten(name = "DY_flat")
  #
  # concatenate the embeddings for the NN part, add them for the CC part
  concate0 <- list(ay_embed, dy_embed) %>% layer_concatenate(name = "concate0")
  cc0 <- list(ay_embed, dy_embed) %>% layer_add(name = "CC0")
  #
  # the 3 tanh hidden layers of the NN part, Keras's glorot_uniform kernels
  # (INFERRED: Listing 4 does not name the initialiser)
  nn0 <- concate0 %>%
    layer_dense(units = q0[1], activation = "tanh", name = "hidden1") %>%
    layer_dropout(rate = param$dropout, name = "dropout1") %>%
    layer_dense(units = q0[2], activation = "tanh", name = "hidden2") %>%
    layer_dropout(rate = param$dropout, name = "dropout2") %>%
    layer_dense(units = q0[3], activation = "tanh", name = "hidden3") %>%
    layer_dropout(rate = param$dropout, name = "dropout3")
  #
  # the bCCNN with the skip connection for the CC part, started in the ccODP
  # model (14): weight 1 on CC0, 0 on the q0[3] NN0 neurons, bias c_ODP
  response <- list(cc0, nn0) %>%
    layer_concatenate(name = "concate1") %>%
    layer_dense(
      units = 1L,
      activation = "exponential",
      kernel_initializer = initializer_constant(matrix(c(1, rep(0, q0[3])))),
      bias_initializer = initializer_constant(odp$intercept),
      name = "Response"
    )
  model <- keras_model(inputs = list(accyear, devyear), outputs = response)
  #
  # Keras's poisson loss and RMSprop (Listing 4)
  model %>% compile(
    loss = "poisson",
    optimizer = optimizer_rmsprop(learning_rate = param$learning_rate,
                                  rho = param$rho,
                                  epsilon = param$epsilon)
  )
  model
}

## TabM bCCNN, TabM (T1)-(T5): k submodels in one model, submodel l has the
## mean exp(alpha_i + beta_j + eta_l) with its own head eta_l; hidden layers
## "tabm" (BatchEnsemble, shared b) or "packed" (k separate networks, the
## deep ensemble); every submodel starts in the ccODP model (T6)
tabm_model <- function(odp, param) {
  n <- odp$n
  k <- as.integer(param$k)
  q0 <- param$hidden
  #
  # declare AY and DY features to be categorical
  accyear <- layer_input(shape = c(1L), dtype = "int32", name = "AccYear")
  devyear <- layer_input(shape = c(1L), dtype = "int32", name = "DevYear")
  #
  # embedding layers holding the ccODP alpha_i and beta_j, non-trainable
  ay_embed <- accyear %>%
    layer_embedding(
      input_dim = n,
      output_dim = 1L,
      trainable = FALSE,
      embeddings_initializer = initializer_constant(matrix(odp$alpha, n, 1L)),
      name = "AY_embed"
    ) %>%
    layer_flatten(name = "AY_flat")
  dy_embed <- devyear %>%
    layer_embedding(
      input_dim = n,
      output_dim = 1L,
      trainable = FALSE,
      embeddings_initializer = initializer_constant(matrix(odp$beta, n, 1L)),
      name = "DY_embed"
    ) %>%
    layer_flatten(name = "DY_flat")
  #
  # the network input (alpha_i, beta_j) and the skip connection
  # alpha_i + beta_j (weight 1, no adapter: thesis (4.3))
  z0 <- list(ay_embed, dy_embed) %>% layer_concatenate(name = "z0")
  skip <- list(ay_embed, dy_embed) %>% layer_add(name = "skip")
  #
  # k copies of the input, shape (cells, k, 2): axis 2 is the submodel axis
  # (keras3 ops count axes from 1)
  z <- z0 %>%
    layer_reshape(c(1L, 2L), name = "z0_row") %>%
    layer_lambda(function(x) op_repeat(x, k, axis = 2L),
                 output_shape = c(k, 2L),
                 name = "copy_to_submodels")
  #
  # the hidden layers (T1); dropout draws a mask per cell, submodel and neuron
  for (d in seq_along(q0)) {
    if (param$kind == "tabm") {
      # TabM Sec. 3.2 and App. A.2: the first adapter r starts at random
      # signs or N(0, 1), every other adapter at 1
      # (INFERRED: drawn from R's RNG with the run's seed)
      r1 <- NULL
      if (d == 1) {
        r1 <- if (param$r1_init == "signs") {
          sample(c(-1, 1), k * 2, replace = TRUE)
        } else {
          rnorm(k * 2)
        }
        r1 <- matrix(r1, k, 2)
      }
      z <- z %>%
        layer_batch_ensemble(q_out = q0[d],
                             k = k,
                             r_initial = r1,
                             name = paste0("be", d))
    } else {
      z <- z %>%
        layer_packed_linear(q_out = q0[d], k = k, name = paste0("packed", d))
    }
    z <- z %>%
      layer_activation(param$activation, name = paste0("act", d)) %>%
      layer_dropout(rate = param$dropout, name = paste0("dropout", d))
  }
  #
  # k heads, TabM (T5): kernels 0 and biases c_ODP, so every eta_l = c_ODP
  eta <- z %>%
    layer_packed_linear(
      q_out = 1L,
      k = k,
      kernel_initializer = "zeros",
      bias_initializer = initializer_constant(matrix(odp$intercept, k, 1L)),
      name = "head"
    ) %>%
    layer_flatten(name = "eta")
  #
  # submodel means exp(alpha_i + beta_j + eta_l): the skip connection enters
  # the exponential, broadcast over the k submodels (T4)
  mu <- list(skip, eta) %>%
    layer_lambda(function(x) op_exp(x[[1]] + x[[2]]),
                 output_shape = c(k),
                 name = "mu")
  model <- keras_model(inputs = list(accyear, devyear), outputs = mu)
  #
  # Keras's poisson loss on y repeated in k columns is the mean of the k
  # individual losses (T7); RMSprop (Listing 4)
  model %>% compile(
    loss = "poisson",
    optimizer = optimizer_rmsprop(learning_rate = param$learning_rate,
                                  rho = param$rho,
                                  epsilon = param$epsilon)
  )
  model
}

## the network of a configuration (param$kind: single, tabm or packed)
nn_model <- function(odp, param) {
  if (param$kind == "single") {
    bccnn_model(odp, param)
  } else {
    tabm_model(odp, param)
  }
}

## submodel means over the whole n x n rectangle, dropout off:
## array (k, n, n), k = 1 for the single bCCNN
nn_predict <- function(model, n) {
  # accident and development year of each cell, from 0 for the embedding
  # layers, row by row (accident year slowest)
  x <- list(as.matrix(rep(1:n - 1L, each = n)),
            as.matrix(rep(1:n - 1L, times = n)))
  mu <- as.array(model(x, training = FALSE))    # one column per submodel
  aperm(array(t(mu), dim = c(ncol(mu), n, n)), c(1, 3, 2))
}

## full-batch gradient descent from the ccODP start (Listing 4); history
## entry e + 1: the collective and individual (TabM (T8)) deviances after e
## steps on tri and vali; submodel means kept at the steps in snapshots
nn_fit <- function(odp, tri, epochs, param,
                   vali = NULL, curves = TRUE, snapshots = integer(0)) {
  set_random_seed(as.integer(param$seed))       # Keras's and R's RNG
  model <- nn_model(odp, param)
  k <- if (param$kind == "single") 1L else param$k
  cells <- train_cells(tri)
  cells_v <- if (!is.null(vali)) train_cells(vali)
  # observed cells, row by row; y in k columns (TabM (T7))
  x <- list(as.matrix(cells$i), as.matrix(cells$j))
  y <- matrix(cells$y, length(cells$y), k)
  #
  # deviances and submodel means after 'step' gradient descent steps
  history <- list(train_collective = numeric(0),
                  train_individual = numeric(0),
                  validation_collective = numeric(0),
                  validation_individual = numeric(0))
  mu_snap <- list()
  record <- function(step) {
    if (curves || step %in% snapshots) {
      mu_sub <- nn_predict(model, odp$n)
      if (step %in% snapshots) mu_snap[[as.character(step)]] <<- mu_sub
    }
    if (curves) {
      mu_bar <- apply(mu_sub, c(2, 3), mean)
      e <- step + 1
      history$train_collective[e] <<- triangle_deviance(cells, mu_bar)
      history$train_individual[e] <<- individual_deviance(cells, mu_sub)
    }
    if (curves && !is.null(vali)) {
      history$validation_collective[e] <<- triangle_deviance(cells_v, mu_bar)
      history$validation_individual[e] <<- individual_deviance(cells_v, mu_sub)
    }
  }
  # the step count is kept here (keras3 passes the epoch from 1)
  steps <- 0L
  after_step <- callback_lambda(
    on_train_begin = function(logs) record(0L),
    on_epoch_end = function(epoch, logs) {
      steps <<- steps + 1L
      record(steps)
    }
  )
  callbacks <- if (curves || length(snapshots) > 0) list(after_step) else NULL
  if (epochs > 0) {
    model %>% fit(x, y,
                  epochs = as.integer(epochs),
                  batch_size = nrow(y),
                  shuffle = FALSE,
                  verbose = 0,
                  callbacks = callbacks)
  } else {
    record(0L)
  }
  list(model = model,
       history = history,
       snapshots = mu_snap)
}

## reserve of the ensemble: the mean over the submodels, summed over the
## cells outside the mask (the lower triangle)
collective_reserve <- function(mu_sub, mask) {
  sum(apply(mu_sub, c(2, 3), mean)[!mask])
}
