##########################################
#########  Stopping rules on the pooled validation curves
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Section 3.3.2,
#########  Figure 2 and Table 3; slope elbow rule INFERRED
##########################################

source(here::here("analysis", "00_setup.R"))

##########################################
#########  settings
##########################################

## positional argument (run_all.ps1): [1] split, else config.yml
args <- commandArgs(trailingOnly = TRUE)
split <- if (length(args) >= 1) args[1] else cfg$split
epochs <- cfg$epochs[[split]]
seeds <- cfg$seeds
k <- cfg$tabm$k
prefix <- cfg$prefix[[split]]
st <- cfg$stopping
labels <- names(cfg$configs)

## checkpoints of the validation curves (02 validation curves)
tag <- if (split == "time-aware") {
  sprintf("e%d_s%d_h%d_K%d", epochs, seeds, cfg$holdout, k)
} else {
  sprintf("e%d_s%d_K%d", epochs, seeds, k)
}
ckpt_dir <- file.path(paths$processed, paste0(prefix, "_curves"))

##########################################
#########  load checkpoints
##########################################

ckpts <- lapply(labels, function(label) {
  readRDS(file.path(ckpt_dir, paste0(label, "_", tag, ".rds")))
})
names(ckpts) <- labels

##########################################
#########  stopping rules
##########################################

## per configuration, on the pooled curve (every LoB's collective validation
## deviance relative to its ccODP start, averaged over the LoBs):
## protocol stop = smoothed minimum (Paper C's rule, one step count for all
## LoBs), elbow = first step after which the next elbow_window steps gain
## less than elbow_tol of the gain so far (INFERRED);
## relative decrease (%) of each LoB's validation deviance from the ccODP
## start at both stops (Paper C Table 3)
selection <- list(K = k,
                  split = split,
                  epochs_searched = epochs,
                  seeds_averaged = seeds,
                  smoothing_window = st$smooth,
                  elbow_tol = st$elbow_tol,
                  elbow_window = st$elbow_window,
                  holdout = if (split == "time-aware") cfg$holdout else NULL,
                  configs = list())
pooled_all <- list()
for (label in labels) {
  curves <- ckpts[[label]]$curves
  pooled <- pooled_curve(lapply(curves, function(cv) cv$validation_collective))
  shared <- pick_epoch(pooled, st$smooth)
  elbow <- elbow_epoch(pooled, st$elbow_tol, st$elbow_window, st$smooth)
  decrease <- function(step) {
    sapply(curves, function(cv) {
      v <- cv$validation_collective
      100 * (1 - v[step + 1L] / v[1])
    })
  }
  own <- sapply(curves, function(cv) {
    pick_epoch(cv$validation_collective, st$smooth)
  })
  selection$configs[[label]] <- list(
    shared_epoch = shared,
    pooled_validation_minimum = pooled[shared + 1L],
    own_epoch = as.list(own),
    at_edge = at_edge(shared, epochs, st$smooth),
    validation_decrease_pct_by_lob = as.list(decrease(shared)),
    elbow_epoch = elbow,
    pooled_validation_at_elbow = pooled[elbow + 1L],
    validation_decrease_pct_by_lob_elbow = as.list(decrease(elbow))
  )
  pooled_all[[label]] <- pooled
}
jsonlite::write_json(selection,
                     file.path(paths$processed,
                               paste0(prefix, "_selection.json")),
                     auto_unbox = TRUE,
                     pretty = TRUE,
                     digits = NA,
                     na = "null")

##########################################
#########  tables
##########################################

## all curves in long format (step 0 = ccODP start) and, for the figures of
## later scripts, the curves, the pooled curves and the run settings
long <- list()
for (label in labels) {
  curves <- ckpts[[label]]$curves
  for (lob in names(curves)) {
    for (s in names(curves[[lob]])) {
      v <- curves[[lob]][[s]]
      long[[length(long) + 1]] <- data.frame(config = label,
                                             lob = as.integer(lob),
                                             step = seq_along(v) - 1L,
                                             curve = s,
                                             deviance = v)
    }
  }
}
fwrite(do.call(rbind, long),
       file.path(paths$tables, paste0(prefix, "_curves.csv")))
saveRDS(list(curves = lapply(ckpts, "[[", "curves"),
             pooled = pooled_all,
             meta = lapply(ckpts, "[[", "meta")),
        file.path(paths$processed, paste0(prefix, "_curves.rds")))

## elbow stop by tolerance against the protocol stop
sensitivity <- NULL
for (label in labels) {
  elbows <- sapply(st$tols, function(tol) {
    elbow_epoch(pooled_all[[label]], tol, st$elbow_window, st$smooth)
  })
  names(elbows) <- paste0("elbow_tol_", st$tols)
  sensitivity <- rbind(sensitivity,
                       data.frame(config = cfg$pretty[[label]],
                                  protocol_minimum =
                                    selection$configs[[label]]$shared_epoch,
                                  t(elbows)))
}
fwrite(sensitivity,
       file.path(paths$tables, paste0(prefix, "_elbow_sensitivity.csv")))

## stopping steps and pooled relative validation deviance there
## (at_edge: the minimum sits at the end of the search window)
sel <- selection$configs
data.frame(protocol = sapply(sel, "[[", "shared_epoch"),
           at_edge = sapply(sel, "[[", "at_edge"),
           pooled_minimum =
             round(sapply(sel, "[[", "pooled_validation_minimum"), 4),
           elbow = sapply(sel, "[[", "elbow_epoch"),
           pooled_at_elbow =
             round(sapply(sel, "[[", "pooled_validation_at_elbow"), 4))

## smoothed minimum of each LoB's own validation curve
t(sapply(sel, function(x) unlist(x$own_epoch)))

## decrease (%) of the validation deviance at the elbow (Paper C Table 3)
round(t(sapply(sel, function(x) {
  unlist(x$validation_decrease_pct_by_lob_elbow)
})), 2)

sensitivity

##########################################
#########  figures
##########################################

## Paper C Figure 2 per configuration and the pooled curves of all
for (label in labels) {
  ggsave(paste0(prefix, "_figure2_", label, ".png"),
         curves_plot(ckpts[[label]]$curves,
                     c(protocol = sel[[label]]$shared_epoch,
                       elbow = sel[[label]]$elbow_epoch,
                       paper = st$fixed),
                     paste0(cfg$pretty[[label]], ", ", split, " split, K = ",
                            k, ", mean of ", seeds, " seeds"),
                     cfg$configs[[label]]$kind != "single"),
         path = paths$figures, width = 13, height = 7.5, dpi = 130)
}
ggsave(paste0(prefix, "_pooled.png"),
       pooled_plot(pooled_all,
                   sel,
                   cfg$pretty,
                   st$fixed,
                   paste0("Pooled collective validation curves, ", split,
                          " split (K = ", k, ")")),
       path = paths$figures, width = 10, height = 6.5, dpi = 130)
