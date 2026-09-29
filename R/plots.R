##########################################
#########  Figures
#########  after Paper C (Gabrielli, Richman & Wuthrich 2020), Figure 2,
#########  and TabM (Gorishniy, Kotelnikov & Babenko 2025), Section 5.1
##########################################

## Paper C Figure 2 for one configuration: training and validation deviance
## after every gradient descent step, one panel per LoB, and the stopping
## points (protocol minimum, elbow, the paper's 300)
curves_plot <- function(curves, stops, title, show_individual) {
  series <- c(train_collective = "training, collective",
              validation_collective = "validation, collective",
              validation_individual = "validation, mean of submodels")
  if (!show_individual) series <- series[1:2]
  d <- NULL
  for (lob in names(curves)) {
    for (s in names(series)) {
      v <- curves[[lob]][[s]]
      d <- rbind(d, data.frame(lob = paste("LoB", lob),
                               step = seq_along(v) - 1L,
                               series = series[[s]],
                               deviance = v))
    }
  }
  p <- ggplot(d, aes(x = step, y = deviance, colour = series,
                     linetype = series)) +
    geom_line(linewidth = 0.5)
  # one vertical line per stopping rule, drawn in every panel
  rule_colour <- c(protocol = "black", elbow = "purple3", paper = "forestgreen")
  rule_line <- c(protocol = "dotted", elbow = "dashed", paper = "longdash")
  for (rule in names(stops)) {
    p <- p + geom_vline(xintercept = as.numeric(stops[[rule]]),
                        colour = rule_colour[[rule]],
                        linetype = rule_line[[rule]],
                        linewidth = 0.6)
  }
  p +
    facet_wrap(~lob, scales = "free_y", ncol = 3) +
    scale_colour_manual(values = c("training, collective" = "steelblue",
                                   "validation, collective" = "firebrick",
                                   "validation, mean of submodels" =
                                     "darkorange")) +
    scale_linetype_manual(values = c("training, collective" = "solid",
                                     "validation, collective" = "solid",
                                     "validation, mean of submodels" =
                                       "dashed"),
                          guide = "none") +
    labs(title = title,
         subtitle = paste0("stopping points: ",
                           paste(names(stops), "=", as.integer(stops),
                                 collapse = ", "),
                           "  (black dotted = protocol minimum, ",
                           "purple dashed = elbow, green = paper 300)"),
         x = "gradient descent step",
         y = "deviance",
         colour = NULL) +
    theme_bw(base_size = 10) + theme(legend.position = "bottom")
}

## pooled collective validation curves (each LoB relative to its ccODP start)
## of every configuration, with its protocol and elbow stops and the fixed
## step count of the paper
pooled_plot <- function(pooled, selection, pretty, fixed, title) {
  d <- NULL
  marks <- NULL
  for (label in names(pooled)) {
    sel <- selection[[label]]
    d <- rbind(d, data.frame(config = pretty[[label]],
                             step = seq_along(pooled[[label]]) - 1L,
                             relative = pooled[[label]]))
    marks <- rbind(marks,
                   data.frame(config = pretty[[label]],
                              rule = c("protocol (minimum)", "elbow"),
                              step = c(sel$shared_epoch, sel$elbow_epoch),
                              relative = c(sel$pooled_validation_minimum,
                                           sel$pooled_validation_at_elbow)))
  }
  ggplot(d, aes(x = step, y = relative, colour = config)) +
    geom_line(linewidth = 0.5) +
    geom_point(data = marks,
               aes(x = step,
                   y = relative,
                   colour = config,
                   shape = rule),
               size = 2.8,
               inherit.aes = FALSE) +
    geom_hline(yintercept = 1, colour = "grey50", linewidth = 0.4) +
    geom_vline(xintercept = fixed,
               colour = "forestgreen",
               linetype = "longdash",
               linewidth = 0.5) +
    labs(title = title,
         x = "gradient descent step",
         y = "collective validation deviance relative to ccODP",
         colour = NULL,
         shape = "stop") +
    theme_bw(base_size = 10) +
    theme(legend.position = "bottom", legend.box = "vertical")
}

## bias against the true reserves per LoB and in total, one panel per
## stopping rule
bias_plot <- function(bias_long, title) {
  ggplot(bias_long, aes(x = lob, y = bias, fill = model)) +
    geom_col(position = position_dodge(width = 0.85), width = 0.8) +
    geom_hline(yintercept = 0, colour = "black", linewidth = 0.4) +
    facet_wrap(~rule, ncol = 1, scales = "free_y") +
    labs(title = title,
         x = NULL,
         y = "reserve bias (thousands); negative = under-reserving",
         fill = NULL) +
    theme_bw(base_size = 10) + theme(legend.position = "bottom")
}

## the k submodel reserves against their mean, the true and the chain-ladder
## reserves, one panel per LoB (TabM Sec. 5.1: weak individually, strong
## collectively)
submodel_plot <- function(sub_long, refs, title) {
  ggplot() +
    geom_jitter(data = sub_long,
                aes(x = model, y = reserve),
                width = 0.12,
                height = 0,
                size = 1.2,
                colour = "firebrick",
                alpha = 0.7) +
    geom_point(data = refs[refs$what == "collective", ],
               aes(x = model, y = reserve),
               shape = 18,
               size = 3.5,
               colour = "steelblue") +
    geom_hline(data = refs[refs$what == "true", ],
               aes(yintercept = reserve),
               colour = "black",
               linewidth = 0.5) +
    geom_hline(data = refs[refs$what == "chain ladder", ],
               aes(yintercept = reserve),
               colour = "forestgreen",
               linetype = "dashed",
               linewidth = 0.5) +
    facet_wrap(~lob, scales = "free_y", ncol = 3) +
    labs(title = title,
         subtitle = paste("red: submodels individually; blue: their mean;",
                          "black: true reserves; green dashed: chain ladder"),
         x = NULL,
         y = "reserves (thousands)") +
    theme_bw(base_size = 10) +
    theme(axis.text.x = element_text(angle = 25, hjust = 1))
}
