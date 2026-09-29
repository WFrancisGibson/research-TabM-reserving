##########################################
#########  parity of the full time-aware run with the Python project's
#########  same data, ccODP start and initialiser family, other random
#########  streams: reserves agree within seed noise, not exactly
#########  (skipped until both projects have their time-aware results)
##########################################

ours_path <- file.path(paths$processed, "tabm_time_aware_selection.json")
theirs_path <- file.path(cfg$python_project,
                         "results",
                         "tabm_time_aware_selection.json")

skip_unless_both <- function() {
  skip_if_not(file.exists(ours_path), "no R time-aware results yet")
  skip_if_not(file.exists(theirs_path), "no Python time-aware results")
}

test_that("both projects hold out the same cells, same ccODP deviances", {
  needs_keras()
  skip_unless_both()
  ours <- jsonlite::read_json(ours_path, simplifyVector = TRUE)
  theirs <- jsonlite::read_json(theirs_path, simplifyVector = TRUE)
  expect_equal(ours$K, theirs$K)
  expect_equal(ours$holdout, theirs$holdout)
  curves_r <- readRDS(file.path(paths$processed, "tabm_time_aware_curves.rds"))
  np <- reticulate::import("numpy", convert = TRUE)
  z <- np$load(file.path(cfg$python_project,
                         "results",
                         "tabm_time_aware_curves.npz"))
  # step 0: the ccODP deviance of the training triangle on the 17
  # validation cells, identical in both projects
  for (label in names(cfg$configs)) {
    for (lob in 1:6) {
      r_curve <- curves_r$curves[[label]][[as.character(lob)]]
      py_curve <- reticulate::py_to_r(z[[paste0(label,
                                                "_lob",
                                                lob,
                                                "_validation")]])
      expect_equal(r_curve$validation_collective[1],
                   py_curve[1],
                   tolerance = 1e-5,
                   label = paste(label, "LoB", lob, "step 0"))
    }
  }
})

test_that("true and CL reserves agree, 300-step reserves within seed noise", {
  skip_unless_both()
  ours <- fread(file.path(paths$tables, "tabm_time_aware_table.csv"))
  theirs <- fread(file.path(cfg$python_project,
                            "results",
                            "tabm_time_aware_table.csv"))
  expect_equal(ours$true_reserves, theirs$true_reserves, tolerance = 1e-6)
  expect_equal(ours$cl_reserves, theirs$cl_reserves, tolerance = 1e-6)
  # the ensembles: every LoB within 3 %, the total within 1.5 %
  for (label in setdiff(names(cfg$configs), "single_bccnn")) {
    r300 <- ours[[paste0("reserves_", label, "_300")]]
    p300 <- theirs[[paste0("reserves_", label, "_300")]]
    rel <- abs(r300 - p300) / p300
    expect_true(all(rel < 0.03),
                label = paste(label, "per LoB, max", round(max(rel), 3)))
    expect_lt(abs(sum(r300) - sum(p300)) / sum(p300),
              0.015,
              label = paste(label, "total"))
  }
  # the single bCCNN: Keras's glorot start against torch's default, shown
  # side by side only
  print(c(r = sum(ours$reserves_single_bccnn_300),
          python = sum(theirs$reserves_single_bccnn_300)))
})
