##########################################
#########  Data, ccODP and chain ladder of the six lines of business
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Listings 2 and 3,
#########  Table 2; time-aware split of the observed triangles
##########################################

source(here::here("analysis", "00_setup.R"))

n <- cfg$data$n_dev

##########################################
#########  load data
##########################################

## the individual claims (the seed-75 simulation, Paper C Table 1), the
## triangles of the six LoBs (Listings 2 and 3) and their time-aware splits
claims <- fread(file.path(paths$raw, "simulated_claims.csv"))
lob_data <- lob_triangles(claims, cfg$data)
splits <- lapply(lob_data, function(d) time_aware_split(d$full, cfg$holdout))

##########################################
#########  ccODP and chain ladder per LoB (Paper C Table 2)
##########################################

## ccODP on the valuation-date triangle (78 cells), on Paper C's training
## half of the claims (78 cells) and on the time-aware training triangle
## (the triangle 'holdout' years earlier, 55 cells); in thousands
table2 <- do.call(rbind, lapply(seq_along(lob_data), function(m) {
  d <- lob_data[[m]]
  s <- splits[[m]]
  odp <- ccodp_fit(d$full)
  odp_half <- ccodp_fit(d$train)
  odp_sub <- ccodp_fit(s$train)
  #
  # the ccODP reserves are the chain-ladder reserves (Paper C Remark 2.3)
  cum <- incr2cum(as.triangle(ifelse(d$full$train_mask,
                                     d$full$incremental,
                                     NA)))
  cl <- sum(predict(chainladder(cum))[, n] - getLatestCumulative(cum))
  stopifnot(abs(cl - odp$reserve) / odp$reserve < 1e-8)
  #
  true_reserves <- sum(d$full$incremental[!d$full$train_mask])
  data.frame(
    lob = d$lob,
    true_reserves = true_reserves,
    cl_reserves = odp$reserve,
    bias_cl = odp$reserve - true_reserves,
    in_sample_loss = triangle_deviance(train_cells(d$full), odp$mu),
    out_of_sample_loss = triangle_deviance(future_cells(d$full), odp$mu),
    dispersion = odp$phi,
    claim_half_cl_reserves = odp_half$reserve,
    claim_half_dispersion = odp_half$phi,
    time_aware_train_cl_reserves = odp_sub$reserve,
    time_aware_train_dispersion = odp_sub$phi,
    time_aware_validation_deviance_ccodp =
      triangle_deviance(train_cells(s$validation), odp_sub$mu)
  )
}))
fwrite(table2, file.path(paths$tables, "table2_ccodp.csv"))

## Table 2 per LoB and the total reserves
round(table2, 1)
round(colSums(table2[, c("true_reserves", "cl_reserves", "bias_cl")]))

##########################################
#########  time-aware split geometry
##########################################

## the same for every LoB: the cells of the first one
s <- splits[[1]]
vali <- train_cells(s$validation)
geometry <- list(
  holdout = cfg$holdout,
  n_train_cells = s$n_train,
  n_validation_cells = s$n_validation,
  training_triangle = sprintf("%d x %d",
                              nrow(s$train$incremental),
                              ncol(s$train$incremental)),
  validation_calendar_index = sort(unique(vali$i + vali$j)),
  excluded_cells = s$excluded_cells,
  claim_split_cells = sum(lob_data[[1]]$train$train_mask)
)
jsonlite::write_json(geometry,
                     file.path(paths$tables, "time_aware_geometry.json"),
                     auto_unbox = TRUE,
                     pretty = TRUE,
                     digits = NA,
                     na = "null")

c(holdout = cfg$holdout,
  training_cells = s$n_train,
  validation_cells = s$n_validation,
  excluded_cells = nrow(s$excluded_cells))
