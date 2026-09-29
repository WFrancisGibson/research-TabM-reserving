##########################################
#########  Reserves of the refits: comparison tables and figures
#########  Paper C: Gabrielli, Richman & Wuthrich (2020), Tables 2 and 4
#########  TabM: Gorishniy, Kotelnikov & Babenko (2025), Section 5.1
##########################################

source(here::here("analysis", "00_setup.R"))

## positional argument (run_all.ps1): [1] split, else config.yml
args <- commandArgs(trailingOnly = TRUE)
split <- if (length(args) >= 1) args[1] else cfg$split
epochs <- cfg$epochs[[split]]
seeds <- cfg$seeds
k <- cfg$tabm$k
prefix <- cfg$prefix[[split]]
labels <- names(cfg$configs)

##########################################
#########  load refits
##########################################

## refit checkpoints (04 refit) and the six LoBs' triangles
tag <- if (split == "time-aware") {
  sprintf("e%d_s%d_h%d_K%d", epochs, seeds, cfg$holdout, k)
} else {
  sprintf("e%d_s%d_K%d", epochs, seeds, k)
}
ckpt_dir <- file.path(paths$processed, paste0(prefix, "_refits"))
refits <- lapply(labels, function(label) {
  readRDS(file.path(ckpt_dir, paste0(label, "_refit_", tag, ".rds")))
})
names(refits) <- labels
stops <- lapply(refits, function(x) x$meta$stops)
lob_data <- lob_triangles(fread(file.path(paths$raw, "simulated_claims.csv")),
                          cfg$data)

##########################################
#########  tables
##########################################

## per LoB and (configuration, stop): steps, reserves of seed 0 and their
## bias, mean and standard deviation over the seeds, in-sample and
## out-of-sample deviance of the mean mu_bar, the submodels' own
## out-of-sample deviance (TabM (T8)) and the range of their reserves;
## the seed-0 submodel means are the input of the bootstrap
rows <- list()
means <- list()
sub_rows <- list()
for (m in seq_along(lob_data)) {
  tri <- lob_data[[m]]$full
  observed <- train_cells(tri)
  future <- future_cells(tri)
  base <- refits[[1]]$lobs[[as.character(m)]]
  rec <- list(lob = m,
              K = k,
              true_reserves = base$true_reserves,
              cl_reserves = base$cl_reserves,
              bias_cl = base$cl_reserves - base$true_reserves,
              in_sample_loss_ccodp = base$in_sample_loss_ccodp,
              out_of_sample_loss_ccodp = base$out_of_sample_loss_ccodp,
              dispersion_ccodp = base$dispersion_ccodp)
  for (label in labels) {
    lob <- refits[[label]]$lobs[[as.character(m)]]
    for (rule in names(lob$stops)) {
      st <- lob$stops[[rule]]
      mu_sub <- st$mu_sub_seed0
      mu_bar <- apply(mu_sub, c(2, 3), mean)
      key <- paste(label, rule, sep = "_")
      ensemble <- dim(mu_sub)[1] > 1
      # reserves of the k submodels individually
      sub <- apply(mu_sub, 1, function(mu) sum(mu[!tri$train_mask]))
      rec[[paste0("epochs_", key)]] <- st$epoch
      rec[[paste0("reserves_", key)]] <- st$reserves[1]
      rec[[paste0("bias_", key)]] <- st$reserves[1] - base$true_reserves
      rec[[paste0("reserves_", key, "_seed_mean")]] <- mean(st$reserves)
      rec[[paste0("reserves_", key, "_seed_sd")]] <-
        if (length(st$reserves) > 1) sd(st$reserves) else 0
      rec[[paste0("in_sample_loss_", key)]] <-
        triangle_deviance(observed, mu_bar)
      rec[[paste0("out_of_sample_loss_", key)]] <-
        triangle_deviance(future, mu_bar)
      rec[[paste0("out_of_sample_loss_individual_", key)]] <-
        if (ensemble) individual_deviance(future, mu_sub) else NA_real_
      rec[[paste0("submodel_reserve_min_", key)]] <- min(sub)
      rec[[paste0("submodel_reserve_max_", key)]] <- max(sub)
      rec[[paste0("submodel_reserve_range_pct_", key)]] <- if (ensemble) {
        100 * (max(sub) - min(sub)) / st$reserves[1]
      } else {
        NA_real_
      }
      means[[label]][[as.character(m)]][[rule]] <- mu_sub
      sub_rows[[length(sub_rows) + 1]] <- data.frame(config = label,
                                                     lob = m,
                                                     rule = rule,
                                                     epoch = st$epoch,
                                                     submodel = seq_along(sub),
                                                     reserve = sub)
    }
  }
  rows[[m]] <- as.data.frame(rec)
}
lob_table <- do.call(rbind, rows)
sub_reserves <- do.call(rbind, sub_rows)
fwrite(lob_table, file.path(paths$tables, paste0(prefix, "_table.csv")))
saveRDS(means, file.path(paths$processed, paste0(prefix, "_means.rds")))
fwrite(sub_reserves,
       file.path(paths$tables, paste0(prefix, "_submodel_reserves.csv")))

##########################################
#########  comparison tables
##########################################

## reserves per LoB and in total (the Python project's tabm_summary tables):
## protocol stop and 300 steps, and all three stopping rules
stem <- paste0("tabm_summary_", sub("^tabm_", "", prefix))
rule_sets <- list(c("protocol", "300"), c("protocol", "elbow", "300"))
names(rule_sets) <- paste0(stem, c("", "_all_rules"))
summaries <- lapply(rule_sets, function(rules) {
  cmp <- data.frame(lob = as.character(lob_table$lob),
                    true = lob_table$true_reserves,
                    "chain ladder" = lob_table$cl_reserves,
                    check.names = FALSE)
  for (rule in rules) {
    for (label in labels) {
      name <- paste0(cfg$pretty[[label]], " @", stops[[label]][[rule]], " (",
                     if (rule == "300") "paper 300" else rule, ")")
      cmp[[name]] <- lob_table[[paste0("reserves_", label, "_", rule)]]
    }
  }
  rbind(cmp,
        data.frame(lob = "total", lapply(cmp[-1], sum), check.names = FALSE))
})
for (f in names(summaries)) {
  fwrite(summaries[[f]], file.path(paths$tables, paste0(f, ".csv")))
}

##########################################
#########  figures
##########################################

## bias against the true reserves per LoB and in total, one panel per rule
rule_names <- c(protocol = "protocol (smoothed minimum)",
                elbow = "elbow",
                "300" = "paper 300")
lob_names <- c(paste("LoB", lob_table$lob), "total")
bias_long <- NULL
for (label in labels) {
  for (rule in names(rule_names)) {
    b <- lob_table[[paste0("bias_", label, "_", rule)]]
    bias_long <- rbind(bias_long,
                       data.frame(model = paste0(cfg$pretty[[label]], " (@",
                                                 stops[[label]][[rule]], ")"),
                                  rule = rule_names[[rule]],
                                  lob = lob_names,
                                  bias = c(b, sum(b))))
  }
}
for (rule in rule_names) {
  bias_long <- rbind(bias_long,
                     data.frame(model = "chain ladder",
                                rule = rule,
                                lob = lob_names,
                                bias = c(lob_table$bias_cl,
                                         sum(lob_table$bias_cl))))
}
bias_long$lob <- factor(bias_long$lob, levels = lob_names)
bias_long$rule <- factor(bias_long$rule, levels = rule_names)
ggsave(paste0(prefix, "_bias_comparison.png"),
       bias_plot(bias_long,
                 paste0("Bias against the true reserves, ", split,
                        " split, K = ", k)),
       path = paths$figures, width = 12, height = 9, dpi = 130)

## the ensembles' submodel reserves at the elbow stop against their mean,
## the true and the chain-ladder reserves
ensembles <- labels[sapply(cfg$configs, "[[", "kind") != "single"]
sub_long <- sub_reserves[sub_reserves$rule == "elbow" &
                           sub_reserves$config %in% ensembles, ]
sub_long$model <- unlist(cfg$pretty[sub_long$config])
sub_long$lob <- paste("LoB", sub_long$lob)
refs <- NULL
for (m in lob_table$lob) {
  collective <- unlist(lob_table[m, paste0("reserves_", ensembles, "_elbow")])
  refs <- rbind(refs,
                data.frame(lob = paste("LoB", m),
                           model = unlist(cfg$pretty[ensembles]),
                           what = "collective",
                           reserve = collective),
                data.frame(lob = paste("LoB", m),
                           model = NA,
                           what = c("true", "chain ladder"),
                           reserve = c(lob_table$true_reserves[m],
                                       lob_table$cl_reserves[m])))
}
ggsave(paste0(prefix, "_submodel_reserves.png"),
       submodel_plot(sub_long,
                     refs,
                     paste0("Submodel reserves at the elbow stop, ", split,
                            " split, K = ", k)),
       path = paths$figures, width = 13, height = 8, dpi = 130)

##########################################
#########  console
##########################################

## reserves (thousands) by LoB under the three stopping rules
all_rules <- summaries[[2]]
reserves <- round(t(as.matrix(all_rules[-1])))
colnames(reserves) <- all_rules$lob
reserves

## per (configuration, stop): total bias, mean absolute bias per LoB,
## out-of-sample deviance, mean seed standard deviation and, for the
## ensembles, the submodel diversity (TabM Sec. 5.1): gap (%) of the
## submodels' own deviance to that of their mean and range (%) of their
## reserves
overview <- data.frame(config = "chain ladder",
                       rule = NA,
                       epochs = 0,
                       total_bias = sum(lob_table$bias_cl),
                       mean_abs_bias = mean(abs(lob_table$bias_cl)),
                       out_of_sample_loss =
                         sum(lob_table$out_of_sample_loss_ccodp),
                       seed_sd = 0,
                       gap_pct = NA,
                       range_pct = NA)
for (label in labels) {
  for (rule in names(stops[[label]])) {
    key <- paste(label, rule, sep = "_")
    b <- lob_table[[paste0("bias_", key)]]
    loss <- lob_table[[paste0("out_of_sample_loss_", key)]]
    loss_sub <- lob_table[[paste0("out_of_sample_loss_individual_", key)]]
    seed_sd <- lob_table[[paste0("reserves_", key, "_seed_sd")]]
    range_pct <- lob_table[[paste0("submodel_reserve_range_pct_", key)]]
    overview <- rbind(overview,
                      data.frame(config = cfg$pretty[[label]],
                                 rule = rule,
                                 epochs = stops[[label]][[rule]],
                                 total_bias = sum(b),
                                 mean_abs_bias = mean(abs(b)),
                                 out_of_sample_loss = sum(loss),
                                 seed_sd = mean(seed_sd),
                                 gap_pct = mean(100 * (loss_sub / loss - 1)),
                                 range_pct = mean(range_pct)))
  }
}
overview[4:7] <- round(overview[4:7])
overview[8:9] <- round(overview[8:9], 2)
overview
