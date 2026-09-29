##########################################
#########  Prediction uncertainty of the ensembles: parametric bootstrap
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Sections 2.3,
#########  3.3.2 and 3.3.4, Table 5
##########################################

source(here::here("analysis", "00_setup.R"))
library(keras3)

##########################################
#########  settings
##########################################

## positional arguments (run_all.ps1): [1] LoB (absent or 0: every LoB),
## [2] torch threads; the rule, the configurations and the replicates come
## from config.yml (INFERRED: elbow stop, the protocol stop sits at the end
## of the 6000-step window on the time-aware split)
args <- commandArgs(trailingOnly = TRUE)
lob <- if (length(args) >= 1) as.integer(args[1]) else 0L
if (length(args) >= 2) {
  reticulate::import("torch")$set_num_threads(as.integer(args[2]))
}
split <- cfg$bootstrap$split
rule <- cfg$bootstrap$rule
replicates <- cfg$bootstrap$replicates
labels <- cfg$bootstrap$configs
prefix <- cfg$prefix[[split]]

## bCCNN hyper-parameters (Paper C Listing 4, thesis Sec. 4.3) and the
## number k of submodels of the ensembles (TabM)
param <- list(hidden = cfg$bccnn$hidden,
              dropout = cfg$bccnn$dropout,
              learning_rate = cfg$bccnn$learning_rate,
              rho = cfg$bccnn$rho,
              epsilon = cfg$bccnn$epsilon,
              k = cfg$tabm$k)

## one result per (configuration, LoB), tagged with the run of 02 to 05
## whose means are bootstrapped
tag <- if (split == "time-aware") {
  sprintf("e%d_s%d_h%d_K%d",
          cfg$epochs[[split]], cfg$seeds, cfg$holdout, param$k)
} else {
  sprintf("e%d_s%d_K%d", cfg$epochs[[split]], cfg$seeds, param$k)
}
ckpt_dir <- file.path(paths$processed, paste0(prefix, "_bootstrap"))
dir.create(ckpt_dir, showWarnings = FALSE)
odp_files <- file.path(ckpt_dir,
                       paste0("ccodp_lob", seq_len(cfg$data$n_lob),
                              "_r", replicates, ".json"))
result_files <- outer(labels, seq_len(cfg$data$n_lob), function(label, m) {
  file.path(ckpt_dir,
            paste0(label, "_", rule, "_lob", m, "_r", replicates, "_", tag,
                   ".json"))
})
rownames(result_files) <- labels

##########################################
#########  load data
##########################################

## stopping points and validation decrease (03), steps and seed-0 submodel
## means (05), the six LoBs' triangles and their ccODP fits
selection <- jsonlite::read_json(file.path(paths$processed,
                                           paste0(prefix, "_selection.json")),
                                 simplifyVector = TRUE)
lob_table <- fread(file.path(paths$tables, paste0(prefix, "_table.csv")))
means <- readRDS(file.path(paths$processed, paste0(prefix, "_means.rds")))
lob_data <- lob_triangles(fread(file.path(paths$raw, "simulated_claims.csv")),
                          cfg$data)
odps <- lapply(lob_data, function(d) ccodp_fit(d$full))
lobs <- if (lob == 0) seq_along(lob_data) else lob

##########################################
#########  ccODP bootstrap (Paper C Sec. 2.3)
##########################################

## the ccODP refitted on the replicates, seed + m - 1 for LoB m
for (m in lobs) {
  odp <- odps[[m]]
  if (!file.exists(odp_files[m])) {
    boot <- bootstrap_ccodp(lob_data[[m]]$full,
                            odp$mu,
                            odp$phi,
                            odp$reserve,
                            replicates,
                            cfg$bootstrap$seed + m - 1L)
    jsonlite::write_json(list(lob = m,
                              replicates = replicates,
                              dispersion_ccodp = odp$phi,
                              ccodp_process_se = sqrt(boot$process_variance),
                              ccodp_parameter_se = boot$parameter_error,
                              ccodp_rmsep = boot$rmsep,
                              failed = boot$n_failed),
                         odp_files[m],
                         auto_unbox = TRUE,
                         pretty = TRUE,
                         digits = NA,
                         na = "null")
  }
}

##########################################
#########  network bootstrap (Paper C Sec. 3.3.4)
##########################################

## around the collective mean mu_bar of seed 0 at the stop, with the reduced
## dispersion (Paper C Sec. 3.3.2): the ccODP phi less the validation
## decrease at that stop (INFERRED: none at the paper's 300 steps);
## seed + 50 + m - 1 for LoB m
for (m in lobs) {
  tri <- lob_data[[m]]$full
  odp <- odps[[m]]
  for (label in labels) {
    out <- result_files[label, m]
    if (file.exists(out) && !cfg$force) {
      cat(label, "LoB", m, ": cached\n")
      next
    }
    steps <- lob_table[[paste0("epochs_", label, "_", rule)]]
    epochs <- as.integer(steps[lob_table$lob == m])
    mu_bar <- apply(means[[label]][[as.character(m)]][[rule]], c(2, 3), mean)
    reserve <- sum(mu_bar[!tri$train_mask])
    true_reserve <- sum(tri$incremental[!tri$train_mask])
    sel <- selection$configs[[label]]
    decrease <- if (rule == "elbow") {
      sel$validation_decrease_pct_by_lob_elbow[[as.character(m)]]
    } else if (rule == "protocol") {
      sel$validation_decrease_pct_by_lob[[as.character(m)]]
    } else {
      0
    }
    phi <- reduced_phi(odp$phi, decrease)
    p <- modifyList(param, cfg$configs[[label]])
    cat(label, "LoB", m, ":", replicates, "replicates x", epochs, "steps\n")
    started <- Sys.time()
    boot <- bootstrap_nn(tri,
                         mu_bar,
                         phi,
                         reserve,
                         epochs,
                         p,
                         replicates,
                         cfg$bootstrap$seed + 50L + m - 1L)
    seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
    jsonlite::write_json(list(config = label,
                              lob = m,
                              replicates = replicates,
                              K = param$k,
                              rule = rule,
                              epochs = epochs,
                              true_reserves = true_reserve,
                              cl_reserves = odp$reserve,
                              tabm_reserves = reserve,
                              bias_tabm = reserve - true_reserve,
                              dispersion_ccodp = odp$phi,
                              validation_decrease_pct = decrease,
                              dispersion_tabm = phi,
                              tabm_process_se = sqrt(boot$process_variance),
                              tabm_parameter_se = boot$parameter_error,
                              tabm_rmsep = boot$rmsep,
                              tabm_bootstrap_reserve_mean = mean(boot$reserves),
                              failed_replicates = boot$n_failed,
                              seconds = round(seconds, 1)),
                         out,
                         auto_unbox = TRUE,
                         pretty = TRUE,
                         digits = NA,
                         na = "null")
  }
}

##########################################
#########  assemble (Paper C Table 5 analogue)
##########################################

## run without a LoB (run_all.ps1's last call), once every (configuration,
## LoB) result exists: RMSEP of the ccODP and of the networks per LoB
if (lob == 0 && all(file.exists(c(odp_files, result_files)))) {
  boot_table <- NULL
  for (m in seq_len(cfg$data$n_lob)) {
    boot_odp <- jsonlite::read_json(odp_files[m], simplifyVector = TRUE)
    for (label in labels) {
      r <- c(jsonlite::read_json(result_files[label, m],
                                 simplifyVector = TRUE),
             boot_odp[c("ccodp_process_se", "ccodp_parameter_se",
                        "ccodp_rmsep")])
      # a JSON null (numeric NA) reads back as NULL: keep the column numeric
      r <- lapply(r, function(v) if (is.null(v)) NA_real_ else v)
      boot_table <- rbind(boot_table, as.data.frame(r))
    }
  }
  fwrite(boot_table,
         file.path(paths$tables,
                   paste0(prefix, "_bootstrap_", rule, ".csv")))
  cols <- c("true_reserves", "cl_reserves", "tabm_reserves",
            "dispersion_ccodp", "dispersion_tabm", "ccodp_rmsep", "tabm_rmsep")
  print(data.frame(boot_table[c("config", "lob", "epochs")],
                   round(boot_table[cols], 1)))
} else {
  cat(sum(file.exists(result_files)), "of", length(result_files),
      "(configuration, LoB) results present\n")
}
