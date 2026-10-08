library(pracma)
library(foreach)
library(xtable)
library(doParallel)
library(expm)
library(doSNOW)   # 用于进度条

main <- function(R2, sparsity, n, p, B = 1000,
                 out_dir = "weaksimu_revised",
                 out_file = "exam_main.txt") {

  cat("\n========== main() START ==========\n")
  cat("R2 =", R2, " sparsity =", sparsity, " n =", n, " p =", p, " B =", B, "\n")
  cat("getwd() =", getwd(), "\n")

  tau_n   <- R2/(1-R2)
  sigmaep <- 1
  sigmabeta <- 1
  rho_1 <- 1/2

  set.seed(1000)

  Sigma_1 <- matrix(0, n, n)
  for(i in 1:n)
    for(j in 1:n)
      Sigma_1[i,j] <- rho_1^(abs(i-j))
  sqrtSigma_1 <- sqrtm(Sigma_1)

  U2 <- randortho(p, type = c("orthonormal", "unitary"))
  eigen_Sigma_ep <- 0.5 + runif(n)
  sqrtSigma_ep <- diag(sqrt(eigen_Sigma_ep))

  eigen_Sigma_2 <- 0.5 + runif(p)
  sqrtSigma_2 <- U2 %*% diag(sqrt(eigen_Sigma_2)) %*% t(U2)

  # ---- local-safe cluster size ----
  Nslots <- suppressWarnings(as.integer(Sys.getenv("SLURM_JOB_CPUS_PER_NODE")))
  if (is.na(Nslots) || Nslots < 1) {
    Nslots <- max(1L, parallel::detectCores(logical = TRUE) - 1L)
  }
  cat("Using Nslots =", Nslots, "\n")

  cl <- parallel::makeCluster(Nslots, type = "PSOCK")
  doSNOW::registerDoSNOW(cl)

  # ---- 输出路径：改成绝对路径，避免“写到哪里去了” ----
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- normalizePath(file.path(out_dir, out_file), winslash = "\\", mustWork = FALSE)
  cat("Output will be written to:\n", out_path, "\n")

  # ---- 进度条 ----
  pb <- txtProgressBar(min = 0, max = B, style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)

  # ---- 并行模拟 ----
  final <- foreach(ii = 1:B, .combine = "rbind",
                   .packages = "glmnet",
                   .options.snow = opts,
                   .errorhandling = "stop") %dopar% {

    set.seed(1000 + ii)

    zero_loc <- (runif(p) > 1 - sparsity) + 0
    beta <- matrix(sqrt(1/sparsity)*sigmabeta*rnorm(p)/sqrt(p*tau_n^(-1))*zero_loc, nrow=p, ncol=1)

    ep <- sqrtSigma_ep %*% matrix(rnorm(n, sd=sigmaep), nrow=n, ncol=1)
    X  <- sqrtSigma_1 %*% matrix(rnorm(p*n), nrow=n, ncol=p) %*% sqrtSigma_2
    y  <- X %*% beta + ep

    n_oos <- 10000
    ep_oos <- matrix(rnorm(n_oos, sd=sigmaep), nrow=n_oos, ncol=1)
    X_oos  <- matrix(rnorm(p*n_oos), nrow=n_oos, ncol=p) %*% sqrtSigma_2
    y_oos  <- X_oos %*% beta + ep_oos

    fit_lasso <- cv.glmnet(X, y, alpha=1, intercept=FALSE)
    fit_ridge <- cv.glmnet(X, y, alpha=0, intercept=FALSE)
    betahat_lasso <- coef(fit_lasso, s="lambda.min")[-1]
    betahat_ridge <- coef(fit_ridge, s="lambda.min")[-1]

    DeltaLasso <- p*n^(-1)*tau_n^(-2) * (sum((sqrtSigma_2 %*% (betahat_lasso-beta))^2) - sum((sqrtSigma_2 %*% beta)^2))
    DeltaRidge <- p*n^(-1)*tau_n^(-2) * (sum((sqrtSigma_2 %*% (betahat_ridge-beta))^2) - sum((sqrtSigma_2 %*% beta)^2))

    R2oos_ridge <- 1 - mean((y_oos - predict(fit_ridge, X_oos, s="lambda.min"))^2) / mean(y_oos^2)
    R2oos_lasso <- 1 - mean((y_oos - predict(fit_lasso, X_oos, s="lambda.min"))^2) / mean(y_oos^2)

    c(R2, sparsity, R2oos_lasso, R2oos_ridge, DeltaLasso, DeltaRidge)
  }

  close(pb)

  # ---- 写文件（主进程写一次，避免并行冲突）----
  write.table(final, out_path, append = TRUE, col.names = FALSE, row.names = FALSE)
  cat("\nWrote", nrow(final), "rows to:\n", out_path, "\n")

  parallel::stopCluster(cl)
  cat("========== main() END ==========\n\n")

  invisible(final)
}

# ✅ 确保这里没有被注释掉
main(R2=0.05, sparsity=0.2, n=500, p=300, B=1000)
