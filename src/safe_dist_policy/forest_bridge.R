args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1L]]
read_matrix <- function(path) as.matrix(read.csv(path, header = FALSE))
write_matrix <- function(x, path) {
  write.table(x, path, row.names = FALSE, col.names = FALSE, sep = ",")
}

if (mode == "mean-fit") {
  if (!requireNamespace("grf", quietly = TRUE)) stop("Install R package grf.")
  train <- read_matrix(args[[2L]])
  destination <- args[[3L]]
  seed <- as.integer(args[[4L]])
  trees <- as.integer(args[[5L]])
  candidates <- as.integer(strsplit(args[[6L]], ",", fixed = TRUE)[[1L]])
  a <- train[, 1L]
  loss <- train[, 2L]
  utility <- train[, 3L]
  e <- train[, 4L]
  folds <- train[, 5L]
  x <- train[, -(1L:5L), drop = FALSE]

  fit_one <- function(y, rows, leaves) {
    grf::causal_forest(
      x[rows, , drop = FALSE], y[rows], a[rows], W.hat = e[rows],
      num.trees = trees, min.node.size = leaves, seed = seed,
      num.threads = 1L, compute.oob.predictions = FALSE
    )
  }
  risks <- numeric(length(candidates))
  targets <- if (all(utility == -loss)) list(loss) else list(loss, utility)
  for (i in seq_along(candidates)) {
    if (length(candidates) == 1L) break
    for (y in targets) {
      transformed <- ifelse(a == 1, y / e, -y / (1 - e))
      scale <- max(1, stats::var(transformed))
      for (k in unique(folds)) {
        keep <- which(folds != k)
        holdout <- which(folds == k)
        fit <- fit_one(y, keep, candidates[[i]])
        held_x <- x[holdout, , drop = FALSE]
        predicted <- predict(fit, held_x, num.threads = 1L)$predictions
        risks[[i]] <- risks[[i]] + mean((predicted - transformed[holdout])^2) / scale
      }
    }
  }

  size <- candidates[[which.min(risks)]]
  all_rows <- seq_along(loss)
  loss_fit <- fit_one(loss, all_rows, size)
  utility_fit <- if (length(targets) == 1L) NULL else fit_one(utility, all_rows, size)
  saveRDS(list(loss = loss_fit, utility = utility_fit, min_node_size = size), destination)
} else if (mode == "mean-predict") {
  if (!requireNamespace("grf", quietly = TRUE)) stop("Install R package grf.")
  fit <- readRDS(args[[2L]])
  x <- read_matrix(args[[3L]])
  loss <- predict(fit$loss, x, num.threads = 1L)$predictions
  utility <- if (is.null(fit$utility)) -loss else predict(fit$utility, x, num.threads = 1L)$predictions
  write_matrix(cbind(loss, utility), args[[4L]])
} else if (mode == "zicbcf-fit") {
  if (!requireNamespace("zicbcf", quietly = TRUE)) stop("Install R package zicbcf.")
  train <- read_matrix(args[[2L]])
  seed <- as.integer(args[[4L]])
  cutoff <- as.numeric(args[[5L]])
  burn <- as.integer(args[[6L]])
  draws <- as.integer(args[[7L]])
  trees <- as.integer(args[[8L]])
  set.seed(seed)
  x <- train[, -(1L:3L), drop = FALSE]
  fit <- zicbcf::zicbcf_smear(
    y = train[, 2L], z = train[, 1L], x_control = x, pihat = train[, 3L],
    nburn = burn, nsim = draws,
    ntree_control_hurdle = trees, ntree_moderate_hurdle = max(2L, trees %/% 4L),
    ntree_control_continuous = trees, ntree_moderate_continuous = max(2L, trees %/% 4L)
  )
  active0 <- pnorm(fit$mu_b)
  active1 <- pnorm(fit$mu_b + fit$tau_b)
  positive_tail <- function(log_mean) {
    if (cutoff == 0) return(matrix(1, nrow(log_mean), ncol(log_mean)))
    pnorm(sweep(log(cutoff) - log_mean, 1L, pmax(fit$sigma_c, 1e-8), "/"),
          lower.tail = FALSE)
  }
  pred <- cbind(
    colMeans(active0), colMeans(active1),
    colMeans(active0 * positive_tail(fit$mu_c)),
    colMeans(active1 * positive_tail(fit$mu_c + fit$tau_c)),
    colMeans(fit$mu0), colMeans(fit$mu1)
  )
  if (any(!is.finite(pred))) stop("zicbcf produced nonfinite projected targets.")
  write_matrix(pred, args[[3L]])
} else {
  stop("Unknown forest bridge operation.")
}
