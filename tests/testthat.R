##########################################
#########  run all checks of tests/testthat
#########  from the project root: Rscript tests/testthat.R
#########  TABM_QUICK_TESTS=1: fewer LoBs and submodels in the Keras checks
#########  TABM_SKIP_KERAS=1: skip the checks that need Python / Keras
##########################################

testthat::test_dir(here::here("tests", "testthat"))
