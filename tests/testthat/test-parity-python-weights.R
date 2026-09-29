##########################################
#########  parity with the Python (PyTorch) project: same weights, same
#########  predictions, for every configuration
#########  Python: scripts/16_export_r_parity.py -> results/r_parity/*.npz
#########  (the ccODP estimates and every tensor before and after 50 steps)
##########################################

## layers, keras3 <- torch:
## tabm be1..3 and packed packed1..3 <- hidden.0, hidden.3, hidden.6;
## head <- head.weight, head.bias; single hidden1..3 <- nn_part.stack.0/3/6
## and Response <- head (Keras kernel = the transposed torch weight)
parity_dir <- file.path(cfg$python_project, "results", "r_parity")
tol_pred <- 1e-5

parity_files <- function() {
  list.files(parity_dir,
             pattern = "_lob[0-9]+_seed[0-9]+_e[0-9]+\\.npz$",
             full.names = TRUE)
}

skip_without_parity <- function() {
  skip_if(length(parity_files()) == 0,
          "no results/r_parity/*.npz in the Python project")
}

load_npz <- function(path) {
  np <- reticulate::import("numpy", convert = TRUE)
  z <- np$load(path, allow_pickle = FALSE)
  keys <- reticulate::py_to_r(z$files)
  out <- lapply(keys, function(key) reticulate::py_to_r(z[[key]]))
  names(out) <- keys
  out
}

## torch (out, in) -> Keras (in, out); the bias as a 1-D array (a length-1
## R vector would reach Python as a scalar)
set_dense <- function(model, layer, weight, bias) {
  set_weights(get_layer(model, layer),
              list(t(weight), array(as.numeric(bias), dim = length(bias))))
}

## the Python tensors 'prefix' (init or trained) into the keras3 model
load_python_weights <- function(model, label, z, prefix) {
  g <- function(name) z[[paste0(prefix, "__", name)]]
  kind <- cfg$configs[[label]]$kind
  idx <- c(0, 3, 6)
  if (kind == "single") {
    for (d in 1:3) {
      set_dense(model,
                paste0("hidden", d),
                g(paste0("nn_part.stack.", idx[d], ".weight")),
                g(paste0("nn_part.stack.", idx[d], ".bias")))
    }
    set_dense(model, "Response", g("head.weight"), g("head.bias"))
  } else {
    for (d in 1:3) {
      hidden <- paste0("hidden.", idx[d], ".")
      if (kind == "tabm") {
        set_weights(get_layer(model, paste0("be", d)),
                    list(g(paste0(hidden, "B")),
                         g(paste0(hidden, "R")),
                         g(paste0(hidden, "S")),
                         g(paste0(hidden, "C"))))
      } else {
        set_weights(get_layer(model, paste0("packed", d)),
                    list(g(paste0(hidden, "weight")),
                         g(paste0(hidden, "bias"))))
      }
    }
    set_weights(get_layer(model, "head"),
                list(g("head.weight"), g("head.bias")))
  }
  invisible(model)
}

max_rel <- function(a, b) max(abs(a - b) / abs(b))

test_that("the ccODP estimates agree with the Python project's", {
  needs_keras()
  skip_without_parity()
  odps <- get_odps()
  for (path in parity_files()) {
    z <- load_npz(path)
    lob <- as.integer(sub(".*_lob([0-9]+)_.*", "\\1", basename(path)))
    expect_equal(odps[[lob]]$alpha,
                 as.numeric(z$alpha),
                 tolerance = 1e-7,
                 label = basename(path))
    expect_equal(odps[[lob]]$beta,
                 as.numeric(z$beta),
                 tolerance = 1e-7,
                 label = basename(path))
    expect_equal(odps[[lob]]$intercept,
                 as.numeric(z$intercept),
                 tolerance = 1e-7,
                 label = basename(path))
  }
})

test_that("with the Python weights every configuration gives its predictions", {
  needs_keras()
  skip_without_parity()
  odps <- get_odps()
  for (path in parity_files()) {
    z <- load_npz(path)
    label <- sub("_lob[0-9]+_seed[0-9]+_e[0-9]+\\.npz$", "", basename(path))
    lob <- as.integer(sub(".*_lob([0-9]+)_.*", "\\1", basename(path)))
    k <- as.integer(z$K)
    n <- odps[[lob]]$n
    set_random_seed(1L)
    model <- nn_model(odps[[lob]], test_param(label, k))
    #
    # untouched: R's own initialisation is the ccODP model, as Python's is
    expect_lt(max_rel(nn_predict(model, n), z$mu_init),
              tol_pred,
              label = paste(label, "init"))
    #
    # Python's initial tensors -> Python's initial predictions
    load_python_weights(model, label, z, "init")
    expect_lt(max_rel(nn_predict(model, n), z$mu_init),
              tol_pred,
              label = paste(label, "init weights"))
    #
    # negative control: the trained predictions differ from the initial ones
    expect_gt(max_rel(z$mu_trained, z$mu_init),
              1e-3,
              label = paste(label, "trained differs"))
    #
    # Python's trained tensors -> Python's trained predictions, per submodel
    load_python_weights(model, label, z, "trained")
    mu_r <- nn_predict(model, n)
    expect_equal(dim(mu_r), dim(z$mu_trained), label = label)
    gap <- max_rel(mu_r, z$mu_trained)
    expect_lt(gap, tol_pred, label = paste(label, "trained weights"))
    cat(sprintf("  parity %-18s K=%2d  max rel gap %.2e\n", label, k, gap))
  }
})
