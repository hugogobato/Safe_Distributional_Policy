args <- commandArgs(trailingOnly = TRUE)
source_dir <- normalizePath(args[[1]])
need_zic <- length(args) > 1L && identical(args[[2]], "zic")
repo <- "https://cran.r-project.org"
options(timeout = 900L, repos = c(CRAN = repo))
Sys.setenv(MAKEFLAGS = "-j1")

installed <- function(name, wanted) {
  requireNamespace(name, quietly = TRUE) &&
    (is.null(wanted) || utils::packageVersion(name) == package_version(wanted))
}

install_release <- function(name, wanted) {
  if (installed(name, wanted)) return(invisible(NULL))
  filename <- paste0(name, "_", wanted, ".tar.gz")
  relative_paths <- c(
    paste0("/src/contrib/", filename),
    paste0("/src/contrib/Archive/", name, "/", filename)
  )
  destination <- tempfile(fileext = ".tar.gz")
  on.exit(unlink(destination), add = TRUE)
  urls <- as.vector(outer(c(repo, "https://cloud.r-project.org"), relative_paths, paste0))
  downloaded <- FALSE
  for (url in urls) {
    unlink(destination)
    status <- tryCatch(
      suppressWarnings(utils::download.file(url, destination, mode = "wb", quiet = TRUE)),
      error = function(e) 1L
    )
    downloaded <- identical(as.integer(status), 0L) && file.exists(destination) && file.info(destination)$size >= 100L
    if (downloaded) break
  }
  if (!downloaded) {
    stop("Unable to download pinned package ", name, " ", wanted)
  }
  if (name %in% loadedNamespaces()) unloadNamespace(name)
  utils::install.packages(destination, repos = NULL, type = "source", Ncpus = 1L, INSTALL_opts = "--no-multiarch")
  if (!installed(name, wanted)) stop("Failed to install pinned package ", name, " ", wanted)
}

pins <- c(
  Rcpp = "1.1.2", RcppEigen = "0.3.4.0.2", DiceKriging = "1.6.1",
  zoo = "1.9-1", lmtest = "0.9-40", sandwich = "3.1-3", grf = "2.6.1"
)
if (need_zic) {
  pins <- c(pins, RcppArmadillo = "15.6.2-1", Rcereal = "1.3.2", truncnorm = "1.0-9",
            mvtnorm = "1.4-2", LaplacesDemon = "16.1.8", invgamma = "1.2", coda = "0.19-4.1")
}
if (!installed("Matrix", NULL)) stop("Install the R recommended Matrix package for this R version.")
if (need_zic && !installed("MASS", NULL)) stop("Install the R recommended MASS package for this R version.")
for (name in names(pins)) install_release(name, unname(pins[[name]]))
if (need_zic) {
  vendor <- file.path(source_dir, "vendor", "zicbcf")
  revision <- readLines(file.path(vendor, "SOURCE_REVISION"), n = 1L)
  target_lib <- .libPaths()[[1L]]
  marker <- file.path(target_lib, "zicbcf.SOURCE_REVISION")
  matched <- file.exists(marker) && identical(readLines(marker, n = 1L), revision) &&
    dir.exists(file.path(target_lib, "zicbcf")) && installed("zicbcf", "1.0.0")
  if (!matched) {
    if ("zicbcf" %in% loadedNamespaces()) unloadNamespace("zicbcf")
    utils::install.packages(vendor, lib = target_lib, repos = NULL, type = "source", Ncpus = 1L,
                            INSTALL_opts = c("--no-multiarch", "--preclean"))
    if (!dir.exists(file.path(target_lib, "zicbcf")) || !installed("zicbcf", "1.0.0")) {
      stop("Failed to install the vendored zicbcf package.")
    }
    writeLines(revision, marker)
  }
  cat("zicbcf source revision", revision, "\n")
}
# Include key recommended-package versions supplied by the selected R runtime.
print(utils::sessionInfo())
for (name in c(names(pins), "Matrix", if (need_zic) c("MASS", "zicbcf"))) {
  cat(name, as.character(utils::packageVersion(name)), "\n")
}
