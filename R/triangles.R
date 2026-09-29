##########################################
#########  Claims reserving triangles
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Listings 2 and 3
#########  time-aware split: the last calendar years held out
#########  (the Python project's src/time_split.py)
##########################################

## the n x n incremental payments (future cells included: simulated data) and
## the mask of the observed cells; the models only see train_cells()
triangle <- function(incremental, mask) {
  list(incremental = incremental, train_mask = mask)
}

## observed cells at the valuation date: 0-based i + j < n, which is Paper C's
## i + j <= I (78 cells for n = 12)
runoff_mask <- function(n) {
  outer(seq_len(n) - 1L, seq_len(n) - 1L, "+") < n
}

## (i, j, y) of the cells of mask, 0-based, in row-major order (i slowest):
## the order of numpy's np.where and of the Python batches
train_cells <- function(tri, mask = tri$train_mask) {
  n <- nrow(tri$incremental)
  grid <- expand.grid(j = seq_len(n) - 1L, i = seq_len(n) - 1L)
  keep <- mask[cbind(grid$i + 1L, grid$j + 1L)]
  list(i = grid$i[keep],
       j = grid$j[keep],
       y = tri$incremental[cbind(grid$i[keep] + 1L, grid$j[keep] + 1L)])
}

## the future cells (lower triangle): the back-test, never fitted
future_cells <- function(tri) train_cells(tri, !tri$train_mask)

## Paper C Listing 2:
## the claims, ordered by LoB and accident year, go alternately to a
## training and a validation half; order() is stable, so each (LoB, AY)
## group is split as evenly as its size allows
split_claims <- function(claims) {
  ordered <- claims[order(claims$LoB, claims$AY), ]
  half <- rep(c(1, 2), length.out = nrow(ordered))
  list(train = ordered[half == 1, ], validation = ordered[half == 2, ])
}

## Paper C Listing 3: array LoB x accident year x development year of the
## incremental payments in units of scale; rows = accident years from
## first_ay, columns = Pay00, Pay01, ...
aggregate_claims <- function(claims, data_cfg) {
  n <- data_cfg$n_dev
  pay <- sprintf("Pay%02d", 1:n - 1)
  totals <- as.data.table(claims)[, lapply(.SD, sum),
                                  by = c("LoB", "AY"),
                                  .SDcols = pay]
  out <- array(0, dim = c(data_cfg$n_lob, n, n))
  for (r in seq_len(nrow(totals))) {
    out[totals$LoB[r], totals$AY[r] - data_cfg$first_ay + 1, ] <-
      as.numeric(totals[r, pay, with = FALSE])
  }
  out / data_cfg$scale
}

## the triangles of every LoB: all claims (full) and the two halves of
## Listing 2 (train, validation), with the same run-off mask
lob_triangles <- function(claims, data_cfg) {
  halves <- split_claims(claims)
  full <- aggregate_claims(claims, data_cfg)
  train <- aggregate_claims(halves$train, data_cfg)
  vali <- aggregate_claims(halves$validation, data_cfg)
  mask <- runoff_mask(data_cfg$n_dev)
  lapply(seq_len(data_cfg$n_lob), function(m) {
    list(lob = m,
         full = triangle(full[m, , ], mask),
         train = triangle(train[m, , ], mask),
         validation = triangle(vali[m, , ], mask))
  })
}

## time-aware split: training = the n0 x n0 triangle 'holdout' calendar years
## earlier (n0 = n - holdout); validation = the held-out cells inside it (the
## others have no ccODP parameter): 55 and 17 cells for n = 12, holdout = 2
time_aware_split <- function(full, holdout) {
  n0 <- nrow(full$incremental) - holdout
  i <- row(full$incremental) - 1L        # 0-based accident year
  j <- col(full$incremental) - 1L        # 0-based development year
  held_out <- full$train_mask & i + j >= n0
  vali <- held_out & i < n0 & j < n0
  excluded <- which(held_out & !vali, arr.ind = TRUE) - 1L
  excluded <- data.frame(i = excluded[, 1], j = excluded[, 2])
  excluded <- excluded[order(excluded$i, excluded$j), ]
  rownames(excluded) <- NULL
  square <- full$incremental[1:n0, 1:n0]
  train <- triangle(square, runoff_mask(n0))
  validation <- triangle(square, vali[1:n0, 1:n0])
  list(train = train,
       validation = validation,
       excluded_cells = excluded,
       n_train = sum(train$train_mask),
       n_validation = sum(validation$train_mask))
}
