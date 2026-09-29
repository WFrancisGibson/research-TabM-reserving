##########################################
#########  the claims data is the Python project's seed-75 simulation
#########  (Paper C Table 1); the Python data in cfg$python_project
##########################################

python_data <- file.path(cfg$python_project, "data")

## skipped when the Python data is missing or was simulated with other
## settings
manifests_agree <- function() {
  ours <- file.path(paths$raw, "simulation_manifest.json")
  theirs <- file.path(python_data, "simulation_manifest.json")
  if (!file.exists(ours) || !file.exists(theirs)) return(FALSE)
  a <- jsonlite::read_json(ours, simplifyVector = TRUE)
  b <- jsonlite::read_json(theirs, simplifyVector = TRUE)
  keys <- c("rng_flag", "seed", "V_large", "V_small", "std1", "std2", "quick")
  all(vapply(keys, function(k) identical(a[[k]], b[[k]]), logical(1)))
}
skip_unless_comparable <- function() {
  skip_if_not(file.exists(file.path(python_data, "simulated_claims.csv")),
              "Python simulated_claims.csv not available")
  skip_if_not(manifests_agree(), "simulation manifests differ")
}

test_that("the claims file is byte-identical to the Python project's", {
  skip_unless_comparable()
  ours <- unname(tools::md5sum(claims_path))
  theirs <- unname(tools::md5sum(file.path(python_data,
                                           "simulated_claims.csv")))
  expect_identical(ours, theirs)
})

test_that("claim counts match Paper C Table 1", {
  claims <- get_claims()
  expect_equal(nrow(claims), 1198888L)
  expect_equal(as.integer(table(claims$LoB)),
               c(250040L, 250197L, 99969L, 249683L, 249298L, 99701L))
})

test_that("the aggregated triangles are identical to the Python ones", {
  skip_unless_comparable()
  theirs <- fread(file.path(python_data, "simulated_claims.csv"))
  expect_equal(aggregate_claims(get_claims(), cfg$data),
               aggregate_claims(theirs, cfg$data),
               tolerance = 1e-12)
})
