##########################################
#########  checks of the claims split and the triangles (R/triangles.R)
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Listings 2 and 3
##########################################

test_that("split is balanced within LoB and accident year", {
  halves <- split_claims(get_claims())
  ay <- cfg$data$first_ay + 1:cfg$data$n_dev - 1
  count <- function(x) {
    table(factor(x$LoB, 1:cfg$data$n_lob), factor(x$AY, ay))
  }
  gap <- abs(count(halves$train) - count(halves$validation))
  expect_lte(max(gap), 1)
})

test_that("split partitions the claims", {
  # ClNr restarts at 1 in each Simulation.Machine() call:
  # a row id is the only unique key
  claims <- copy(get_claims())
  claims$row_id <- seq_len(nrow(claims))
  halves <- split_claims(claims)
  expect_equal(nrow(halves$train) + nrow(halves$validation), nrow(claims))
  expect_length(intersect(halves$train$row_id, halves$validation$row_id), 0)
})

test_that("halves sum to the full triangle", {
  claims <- get_claims()
  halves <- split_claims(claims)
  expect_equal(aggregate_claims(halves$train, cfg$data) +
                 aggregate_claims(halves$validation, cfg$data),
               aggregate_claims(claims, cfg$data),
               tolerance = 1e-12)
})

test_that("triangle shape and observed cell count", {
  n <- cfg$data$n_dev
  for (d in get_lob_data()) {
    for (tri in d[c("full", "train", "validation")]) {
      expect_equal(dim(tri$incremental), c(n, n))
      expect_equal(sum(tri$train_mask), 78L)
    }
  }
})

test_that("all aggregated payments are non-negative", {
  for (d in get_lob_data()) {
    expect_gte(min(d$full$incremental), 0)
    expect_gte(min(d$train$incremental), 0)
    expect_gte(min(d$validation$incremental), 0)
  }
})

test_that("scaling to thousands", {
  claims <- get_claims()
  pay <- sprintf("Pay%02d", 1:cfg$data$n_dev - 1)
  raw <- sum(as.matrix(claims[, pay, with = FALSE]))
  agg <- sum(aggregate_claims(claims, cfg$data))
  expect_lt(abs(agg - raw / 1000) / abs(raw), 1e-9)
})

test_that("run-off mask is the upper triangle", {
  mask <- runoff_mask(12)
  for (i in 0:11) {
    for (j in 0:11) expect_equal(mask[i + 1, j + 1], i + j < 12)
  }
  expect_equal(sum(mask), 78L)
})

test_that("train_cells is row-major and future_cells is its complement", {
  tri <- triangle(matrix(1:144, 12, 12, byrow = TRUE), runoff_mask(12))
  cells <- train_cells(tri)
  expect_equal(cells$i[1:13], c(rep(0L, 12), 1L))
  expect_equal(cells$j[1:13], c(0:11, 0L))
  future <- future_cells(tri)
  expect_equal(length(cells$y) + length(future$y), 144L)
  expect_equal(sum(future$y), sum(tri$incremental[!tri$train_mask]))
})
