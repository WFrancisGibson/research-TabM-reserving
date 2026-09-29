##########################################
#########  Loss functions
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), eq. (4)
##########################################

## Poisson deviance loss D(y, mu), Paper C eq. (4) with phi = 1:
## twice the sum of y * log(y / mu) - (y - mu), where y * log(y / mu) = 0
## for y = 0
poisson_deviance <- function(y, mu) {
  positive <- y > 0
  term <- numeric(length(y))
  term[positive] <- y[positive] * log(y[positive] / mu[positive])
  2 * sum(term - (y - mu))
}

## deviance of the n x n means mu on the cells (i, j, y) of train_cells();
## the collective deviance of an ensemble: mu = the mean of its submodels
triangle_deviance <- function(cells, mu) {
  poisson_deviance(cells$y, mu[cbind(cells$i + 1L, cells$j + 1L)])
}

## TabM (T8): mean over the submodels l of their own deviances, from the
## k x n x n submodel means mu_sub (the scale the training loss works on)
individual_deviance <- function(cells, mu_sub) {
  idx <- cbind(cells$i + 1L, cells$j + 1L)
  mean(vapply(seq_len(dim(mu_sub)[1]),
              function(l) poisson_deviance(cells$y, mu_sub[l, , ][idx]),
              numeric(1)))
}
