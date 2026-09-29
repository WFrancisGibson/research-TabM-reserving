##########################################
#########  validation curves of the six configurations
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Sec. 3.3.2, Figure 2
#########  Rscript "analysis/02_validation curves/validation curves.R"
#########    [split] [configuration or all] [torch threads]
##########################################

source(here::here("analysis", "00_setup.R"))
library(keras3)

##########################################
#########  settings
##########################################

## positional arguments (run_all.ps1): [1] split, [2] configuration or "all",
## [3] torch threads; else config.yml, all six configurations and torch's
## default number of threads (every core)
args <- commandArgs(trailingOnly = TRUE)
split <- if (length(args) >= 1) args[1] else cfg$split
labels <- if (length(args) >= 2 && args[2] != "all") {
  args[2]
} else {
  names(cfg$configs)
}
if (length(args) >= 3) {
  reticulate::import("torch")$set_num_threads(as.integer(args[3]))
}
epochs <- cfg$epochs[[split]]      # gradient descent steps (search window)
seeds <- cfg$seeds
holdout <- cfg$holdout
k <- cfg$tabm$k

## bCCNN hyper-parameters (Paper C Listing 4) and TabM's k submodels
param <- list(hidden = cfg$bccnn$hidden,
              dropout = cfg$bccnn$dropout,
              learning_rate = cfg$bccnn$learning_rate,
              rho = cfg$bccnn$rho,
              epsilon = cfg$bccnn$epsilon,
              k = cfg$tabm$k)

## one checkpoint per configuration, named by the run settings
tag <- if (split == "time-aware") {
  sprintf("e%d_s%d_h%d_K%d", epochs, seeds, holdout, k)
} else {
  sprintf("e%d_s%d_K%d", epochs, seeds, k)
}
ckpt_dir <- file.path(paths$processed, paste0(cfg$prefix[[split]], "_curves"))
dir.create(ckpt_dir, showWarnings = FALSE)

##########################################
#########  load data
##########################################

## the six LoBs' triangles (Paper C Listings 2 and 3) with the training and
## validation triangles of the split: time-aware (the last 'holdout'
## calendar years) or claim-level (Paper C's 50/50 claims split)
claims <- fread(file.path(paths$raw, "simulated_claims.csv"))
lob_data <- lob_triangles(claims, cfg$data)
splits <- lapply(lob_data, function(d) {
  if (split == "time-aware") time_aware_split(d$full, holdout) else d
})

## ccODP on each training triangle: the start of every network (Paper C (14))
odps <- lapply(splits, function(s) ccodp_fit(s$train))

##########################################
#########  validation curves (Paper C Figure 2)
##########################################

## per configuration, LoB and seed: 'epochs' full-batch steps from the ccODP
## start with the deviances after every step, averaged over the seeds;
## a configuration whose checkpoint exists is skipped (unless force)
for (label in labels) {
  ckpt <- file.path(ckpt_dir, paste0(label, "_", tag, ".rds"))
  if (file.exists(ckpt) && !cfg$force) {
    cat(label, ": curves already at ", basename(ckpt), "\n", sep = "")
    next
  }
  curves <- list()
  per_seed <- list()
  lob_meta <- list()
  for (m in seq_along(lob_data)) {
    started <- Sys.time()
    runs <- list()
    for (s in seq_len(seeds)) {
      p <- modifyList(param, cfg$configs[[label]])
      p$seed <- s - 1L
      runs[[s]] <- nn_fit(odps[[m]],
                          splits[[m]]$train,
                          epochs,
                          p,
                          vali = splits[[m]]$validation)$history
    }
    lob <- as.character(m)
    curves[[lob]] <- averaged_curves(runs)
    # collective validation curve of every seed, one row per seed
    val <- lapply(runs, function(h) h$validation_collective)
    per_seed[[lob]] <- do.call(rbind, val)
    # the ccODP of the training triangle and the cell counts
    lob_meta[[lob]] <- list(
      cl_reserve_training_triangle = odps[[m]]$reserve,
      dispersion_training_triangle = odps[[m]]$phi,
      deviance_training_triangle = odps[[m]]$deviance,
      n_train = sum(splits[[m]]$train$train_mask),
      n_validation = sum(splits[[m]]$validation$train_mask)
    )
    clear_session()
    invisible(gc())
    seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    cat("  ", label, " LoB ", m, ": ", seeds, " seeds x ", epochs,
        " steps in ", round(seconds), " s\n", sep = "")
  }
  meta <- list(label = label,
               split = split,
               epochs = epochs,
               seeds = seeds,
               holdout = if (split == "time-aware") holdout else NA_integer_,
               K = k,
               keras3 = as.character(packageVersion("keras3")),
               keras_python = reticulate::import("keras")$`__version__`,
               torch = reticulate::import("torch")$`__version__`,
               date = format(Sys.time(), "%Y-%m-%d %H:%M"))
  saveRDS(list(meta = meta,
               curves = curves,
               per_seed_validation = per_seed,
               lob_meta = lob_meta),
          ckpt)
  cat(label, ": wrote ", basename(ckpt), "\n", sep = "")
}
