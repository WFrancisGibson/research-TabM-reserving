##########################################
#########  ccODP: cross-classified over-dispersed Poisson model
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Section 2
##########################################

## E[Y_ij] = mu_ij = exp(c + alpha_i + beta_j), Var(Y_ij) = phi * mu_ij
## (Paper C eq. (1)) with alpha_1 = beta_1 = 0 (R's treatment contrasts),
## fitted on the observed cells of tri; the quasi-Poisson MLE gives the
## chain-ladder reserves (Paper C Remark 2.3)
ccodp_fit <- function(tri) {
  cells <- train_cells(tri)
  n <- nrow(tri$incremental)
  dat <- data.frame(y = cells$y,
                    ay = factor(cells$i, levels = seq_len(n) - 1L),
                    dy = factor(cells$j, levels = seq_len(n) - 1L))
  d_glm <- glm(y ~ ay + dy,
               data = dat,
               family = quasipoisson(),
               control = glm.control(epsilon = 1e-14, maxit = 100))
  b <- coef(d_glm)
  # coefficients by position: c, alpha_2..alpha_n, beta_2..beta_n
  intercept <- unname(b[1])
  alpha <- c(0, unname(b[2:n]))
  beta <- c(0, unname(b[(n + 1):(2 * n - 1)]))
  # fitted means of the whole rectangle
  mu <- exp(intercept + outer(alpha, beta, "+"))
  #
  # dispersion (Paper C eq. (5)): deviance over the observed cells divided by
  # (cells - parameters), 78 - 23 = 55 for n = 12
  deviance <- poisson_deviance(cells$y, mu[cbind(cells$i + 1L, cells$j + 1L)])
  phi <- deviance / (length(cells$y) - length(b))
  #
  # reserves: the fitted means of the future cells
  forecast <- ifelse(tri$train_mask, 0, mu)
  list(intercept = intercept,
       alpha = alpha,
       beta = beta,
       n = n,
       mu = mu,
       deviance = deviance,
       phi = phi,
       reserve_o = rowSums(forecast),
       reserve = sum(forecast))
}
