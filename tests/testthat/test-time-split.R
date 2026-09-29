##########################################
#########  checks of the time-aware validation split (R/triangles.R)
#########  the Python project's tests/test_time_split.py
##########################################

test_that("counts and geometry on a 12 x 12 triangle", {
  inc <- matrix(as.numeric(1:144), 12, 12, byrow = TRUE) + 1
  full <- triangle(inc, runoff_mask(12))
  s <- time_aware_split(full, 2)
  expect_equal(dim(s$train$incremental), c(10L, 10L))
  expect_equal(s$n_train, 55L)          # 10 x 11 / 2
  expect_equal(s$n_validation, 17L)     # 9 cells on diagonal 10, 8 on 11
  #
  v <- train_cells(s$validation)
  expect_setequal(unique(v$i + v$j), c(10L, 11L))
  expect_true(max(v$i) <= 9L && max(v$j) <= 9L)
  expect_equal(s$excluded_cells,
               data.frame(i = c(0L, 0L, 1L, 10L, 10L, 11L),
                          j = c(10L, 11L, 10L, 0L, 1L, 0L)))
  # every validation cell is observed in the full triangle and none is a
  # training cell
  expect_true(all(full$train_mask[cbind(v$i + 1L, v$j + 1L)]))
  expect_false(any(s$train$train_mask & s$validation$train_mask))
  # the values are the full triangle's values
  expect_equal(v$y, inc[cbind(v$i + 1L, v$j + 1L)])
  tr <- train_cells(s$train)
  expect_true(all(tr$i + tr$j <= 9L))
  expect_equal(tr$y, inc[cbind(tr$i + 1L, tr$j + 1L)])
})

test_that("the training triangle's ccODP is its chain ladder", {
  d <- get_lob_data()[[1]]
  s <- time_aware_split(d$full, 2)
  odp <- ccodp_fit(s$train)
  expect_equal(odp$n, 10L)
  # Poisson score equations: the fitted column sums are the observed ones
  cells <- train_cells(s$train)
  for (dy in 0:9) {
    keep <- cells$j == dy
    expect_equal(sum(odp$mu[cbind(cells$i[keep] + 1L, dy + 1L)]),
                 sum(cells$y[keep]),
                 tolerance = 1e-8)
  }
  # the chain ladder of the 10 x 10 triangle two years earlier
  expect_equal(odp$reserve, sum(cl_reserves(s$train)), tolerance = 1e-8)
})
