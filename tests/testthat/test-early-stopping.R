##########################################
#########  checks of the stopping rules (R/early stopping.R)
#########  against brute-force versions of their definitions
##########################################

## 0-based step of the lowest centred moving average, window by window
brute_pick <- function(v, smooth) {
  half <- smooth %/% 2L
  centres <- (half + 1L):(length(v) - half)
  means <- sapply(centres, function(t) mean(v[(t - half):(t + half)]))
  centres[which.min(means)] - 1L
}

test_that("pick_epoch is the argmin of the centred moving average", {
  set.seed(1)
  v <- 100 * exp(-(0:399) / 80) + 20 + rnorm(400, 0, 0.3) + (0:399) * 0.02
  expect_equal(pick_epoch(v, 51L), brute_pick(v, 51L))
  # too short to smooth: the raw argmin
  expect_equal(pick_epoch(c(5, 3, 4), 51L), 1L)
})

test_that("pooled_curve normalises each LoB by its own start", {
  a <- c(10, 5, 4)
  b <- c(100, 90, 80)
  expect_equal(pooled_curve(list(a, b)), (a / 10 + b / 100) / 2)
  expect_equal(pooled_curve(list(a, b))[1], 1)
})

test_that("at_edge flags minima inside the last smoothing window", {
  expect_true(at_edge(5975L, 6000L, 51L))
  expect_false(at_edge(5000L, 6000L, 51L))
})

test_that("smoothed keeps the length and pads the edges", {
  v <- as.numeric(1:200)
  s <- smoothed(v, 51L)
  expect_length(s, 200L)
  expect_equal(s[26], mean(1:51))                  # first interior value
  expect_equal(s[1:25], rep(s[26], 25))            # left pad
  expect_equal(s[176:200], rep(s[175], 25))        # right pad
  expect_equal(smoothed(v[1:10], 51L), v[1:10])    # too short: unchanged
})

test_that("elbow_epoch matches a brute-force scan of its definition", {
  steps <- 0:5999
  for (drift in c(0, 0.02, 0.05)) {
    set.seed(2)
    curve <- 1 - 0.55 * (1 - exp(-steps / 250)) - drift * steps / 6000 +
      rnorm(6000, 0, 0.005)
    s <- smoothed(curve, 51L)
    found <- NA
    for (tt in seq_len(length(s) - 200L)) {
      gained <- s[1] - s[tt]
      ahead <- s[tt] - s[tt + 200L]
      if (gained > 0 && ahead <= 0.01 * gained) {
        found <- tt - 1L
        break
      }
    }
    expect_equal(elbow_epoch(curve, 0.01, 200L, 51L), found)
    # the elbow of the exponential is near step 1000
    expect_true(abs(found - 1000) < 350)
  }
})

test_that("elbow_epoch returns the last step when the curve never flattens", {
  curve <- 1 - (0:999) / 1000
  expect_equal(elbow_epoch(curve, 0.01, 200L, 51L), 999L)
  # shorter than the window
  expect_equal(elbow_epoch(c(1, 0.9, 0.8), 0.01, 200L, 51L), 2L)
})

test_that("averaged_curves averages every curve name over seeds", {
  runs <- list(list(a = c(1, 2), b = c(10, 20)),
               list(a = c(3, 4), b = c(30, 40)))
  expect_equal(averaged_curves(runs), list(a = c(2, 3), b = c(20, 30)))
})
