##########################################
#########  Early stopping: stopping rules on the validation curves
#########  thesis rule: minimum of the smoothed curve (Gabrielli 2020);
#########  slope elbow: first step after which the gain flattens (INFERRED)
##########################################

## a curve holds entry 1 = the ccODP start (0 steps) and entry e + 1 = after
## e gradient descent steps; the rules return 0-based step counts

## step with the lowest centred moving average (window smooth) of the curve;
## the edges where the window does not fit are NA and ignored by which.min()
pick_epoch <- function(validation, smooth) {
  validation <- as.numeric(validation)
  if (smooth > 1 && length(validation) > smooth) {
    validation <- filter(validation, rep(1 / smooth, smooth), sides = 2)
  }
  which.min(validation) - 1L
}

## pooled curve: mean over the LoBs of the curves, each divided by its
## ccODP start, so that every LoB weighs the same
pooled_curve <- function(curves) {
  colMeans(do.call(rbind, lapply(curves, function(v) {
    as.numeric(v) / as.numeric(v)[1]
  })))
}

## TRUE when the step is within the last smoothing window of the search:
## the curve still descends at the end of the window
at_edge <- function(epoch, epochs, smooth) epoch >= epochs - smooth

## centred moving average of the same length as the curve: the edges are
## padded with the first and last interior value
smoothed <- function(curve, window) {
  curve <- as.numeric(curve)
  steps <- length(curve)
  if (window <= 1 || steps <= window) return(curve)
  pad <- window %/% 2
  inner <- as.numeric(filter(curve, rep(1 / window, window), sides = 2))
  inner <- inner[(pad + 1):(steps - pad)]
  c(rep(inner[1], pad),
    inner,
    rep(inner[length(inner)], steps - length(inner) - pad))
}

## slope elbow (INFERRED): the first step t at which the next 'window' steps
## gain less than tol of the gain so far on the smoothed pooled curve s,
## s(t) - s(t + window) <= tol * (s(0) - s(t)); the last step if never
elbow_epoch <- function(pooled, tol, window, smooth) {
  s <- smoothed(pooled, smooth)
  steps <- length(s)
  if (steps <= window + 1) return(steps - 1L)
  idx <- seq_len(steps - window)        # entry t + 1 of s is step t
  gained <- s[1] - s[idx]
  ahead <- s[idx] - s[idx + window]
  ok <- ahead <= tol * gained & gained > 0
  if (any(ok)) which(ok)[1] - 1L else steps - 1L
}

## pointwise mean over the seeds of the histories in runs (lists of curves
## with the same names)
averaged_curves <- function(runs) {
  out <- lapply(names(runs[[1]]), function(curve) {
    colMeans(do.call(rbind, lapply(runs, function(r) r[[curve]])))
  })
  names(out) <- names(runs[[1]])
  out
}
