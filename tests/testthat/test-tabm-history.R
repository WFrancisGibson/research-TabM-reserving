##########################################
#########  checks of the recorded curves against Keras's own loss
#########  Keras's poisson loss is the mean over cells and submodels of
#########  mu - y log(mu), so the recorded mean individual deviance (T8) is
#########  2 n loss + 2 sum(y log(y) - y) (R/nn_models.R)
##########################################

test_that("the individual deviance is 2 n Keras poisson loss + constant", {
  needs_keras()
  d <- get_lob_data()[[1]]
  split <- time_aware_split(d$full, 2L)
  odp <- ccodp_fit(split$train)
  k <- 4L
  p <- test_param("tabm_tanh_signs", k)
  h <- nn_fit(odp, split$train, 30L, p, vali = split$validation)$history
  #
  # the same run by hand (same seed, inputs and fit settings), with Keras's
  # poisson loss on the validation cells after every step; not through
  # fit(validation_data = ): on torch its evaluation draws from torch's
  # RNG, which the dropout masks of the next steps come from
  train <- train_cells(split$train)
  vali <- train_cells(split$validation)
  x <- list(as.matrix(train$i), as.matrix(train$j))
  y <- matrix(train$y, length(train$y), k)
  x_vali <- list(as.matrix(vali$i), as.matrix(vali$j))
  y_vali <- matrix(vali$y, length(vali$y), k)
  set_random_seed(0L)
  model <- nn_model(odp, p)
  loss <- numeric(0)
  after_step <- callback_lambda(on_epoch_end = function(epoch, logs) {
    mu <- model(x_vali, training = FALSE)
    loss[length(loss) + 1] <<- mean(as.array(loss_poisson(y_vali, mu)))
  })
  model %>% fit(x, y,
                epochs = 30L,
                batch_size = nrow(y),
                shuffle = FALSE,
                verbose = 0,
                callbacks = list(after_step))
  const <- 2 * sum(ifelse(vali$y > 0, vali$y * log(vali$y) - vali$y, 0))
  from_keras <- 2 * length(vali$y) * loss + const
  expect_equal(from_keras, h$validation_individual[-1], tolerance = 1e-4)
  #
  # evaluate() gives the same loss after the last step
  ev <- model %>% evaluate(x_vali, y_vali, verbose = 0)
  expect_equal(ev$loss, loss[30], tolerance = 1e-6)
})
