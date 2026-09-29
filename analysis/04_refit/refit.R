##########################################
#########  Refits on the full triangle at the stopping points
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Section 3.3.3
#########  TabM: Gorishniy, Kotelnikov & Babenko (2025)
##########################################

source(here::here("analysis", "00_setup.R"))
library(keras3)

##########################################
#########  settings
##########################################

## positional arguments (run_all.ps1): [1] split, [2] configuration or "all",
## [3] torch threads; else config.yml, all six configurations and torch's
## default number of threads
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
epochs <- cfg$epochs[[split]]
seeds <- cfg$seeds
prefix <- cfg$prefix[[split]]

## bCCNN hyper-parameters (Paper C Listing 4, thesis Sec. 4.3) and the
## number k of submodels of the ensembles (TabM)
param <- list(hidden = cfg$bccnn$hidden,
              dropout = cfg$bccnn$dropout,
              learning_rate = cfg$bccnn$learning_rate,
              rho = cfg$bccnn$rho,
              epsilon = cfg$bccnn$epsilon,
              k = cfg$tabm$k)

## one checkpoint per configuration, tagged like the validation curves
tag <- if (split == "time-aware") {
  sprintf("e%d_s%d_h%d_K%d", epochs, seeds, cfg$holdout, param$k)
} else {
  sprintf("e%d_s%d_K%d", epochs, seeds, param$k)
}
ckpt_dir <- file.path(paths$processed, paste0(prefix, "_refits"))
dir.create(ckpt_dir, showWarnings = FALSE)

##########################################
#########  load data
##########################################

## the stopping points (03 stopping rules), the six LoBs' triangles
## (Paper C Listing 3) and the ccODP on each full upper triangle
selection <- jsonlite::read_json(file.path(paths$processed,
                                           paste0(prefix, "_selection.json")),
                                 simplifyVector = TRUE)
lob_data <- lob_triangles(fread(file.path(paths$raw, "simulated_claims.csv")),
                          cfg$data)
odps <- lapply(lob_data, function(d) ccodp_fit(d$full))

##########################################
#########  refits at the stopping points (Paper C Sec. 3.3.3)
##########################################

## every network started in the ccODP of the full triangle; one run per seed
## to the largest stop keeps the submodel means at every stop on the way
## (the same seed gives the same random stream up to each stop)
for (label in labels) {
  sel <- selection$configs[[label]]
  # the paper's 300 is capped at the search window (quick profile) and
  # named "300", as the Python tables name it
  stops <- c(protocol = as.integer(sel$shared_epoch),
             elbow = as.integer(sel$elbow_epoch),
             "300" = as.integer(min(cfg$stopping$fixed, epochs)))
  ckpt <- file.path(ckpt_dir, paste0(label, "_refit_", tag, ".rds"))
  # reuse a checkpoint only if it was made for the same stopping points
  if (file.exists(ckpt) && !cfg$force) {
    old <- readRDS(ckpt)
    if (identical(unlist(old$meta$stops), unlist(as.list(stops)))) {
      cat(label, "refits already at", basename(ckpt), "\n")
      next
    }
  }
  p <- modifyList(param, cfg$configs[[label]])
  lobs <- list()
  for (m in seq_along(lob_data)) {
    started <- Sys.time()
    tri <- lob_data[[m]]$full
    odp <- odps[[m]]
    reserves <- lapply(stops, function(e) numeric(seeds))
    mu_seed0 <- list()
    for (s in seq_len(seeds)) {
      p$seed <- s - 1L
      fit <- nn_fit(odp, tri, max(stops), p,
                    curves = FALSE, snapshots = unique(stops))
      for (rule in names(stops)) {
        mu_sub <- fit$snapshots[[as.character(stops[[rule]])]]
        reserves[[rule]][s] <- collective_reserve(mu_sub, tri$train_mask)
        if (s == 1) mu_seed0[[rule]] <- mu_sub
      }
    }
    ## true, chain-ladder (ccODP) and network reserves; in-sample and
    ## out-of-sample (true lower triangle) deviance of the ccODP
    lobs[[as.character(m)]] <- list(
      true_reserves = sum(tri$incremental[!tri$train_mask]),
      cl_reserves = odp$reserve,
      in_sample_loss_ccodp = triangle_deviance(train_cells(tri), odp$mu),
      out_of_sample_loss_ccodp = triangle_deviance(future_cells(tri), odp$mu),
      dispersion_ccodp = odp$phi,
      stops = setNames(lapply(names(stops), function(rule) {
        list(epoch = stops[[rule]],
             reserves = reserves[[rule]],
             mu_sub_seed0 = mu_seed0[[rule]])
      }), names(stops))
    )
    clear_session()
    invisible(gc())
    cat(label, "LoB", m, ":", seeds, "seeds x", max(stops), "steps in",
        round(as.numeric(difftime(Sys.time(), started, units = "secs"))),
        "s\n")
  }
  saveRDS(list(meta = list(label = label,
                           split = split,
                           seeds = seeds,
                           K = param$k,
                           epochs_searched = epochs,
                           stops = as.list(stops),
                           date = format(Sys.time(), "%Y-%m-%d %H:%M")),
               lobs = lobs),
          ckpt)

  ## reserves (seed 0) at the stops against the true and chain-ladder
  ## reserves, by LoB
  print(round(sapply(lobs, function(x) {
    c(true = x$true_reserves,
      chain_ladder = x$cl_reserves,
      sapply(x$stops, function(y) y$reserves[1]))
  })))
}
