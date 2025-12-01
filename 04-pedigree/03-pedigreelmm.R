# --- 1. Load Libraries ---
# install.packages("MASS") # for mvrnorm
# install.packages("Matrix") # for nearPD
# install.packages("nloptr") # for nloptr
library(MASS)
library(Matrix)
library(nloptr)

# --- 2. Setup and Load Data ---
pedigreeGRM <- Matrix::readMM("spatial-simulation.pedigree.grm.mtx")
pedigreeGRM <- as.matrix(pedigreeGRM)
pheno_path <- snakemake@input[["pheno"]]
pheno <- scan(pheno_path)
n <- length(pheno) 

# --- 3. Define Subsets and Partition Data ---
ss <- seq(1, n, 2)      # Training set indices
ss_out <- (1:n)[-ss]  # Prediction set indices

# Partition the phenotype vector
y1 <- pheno[ss] + rnorm(length(ss)) # Training phenotypes

# Partition the GRM
G11 <- pedigreeGRM[ss, ss]
G21 <- pedigreeGRM[ss_out, ss]

#cat("Training set size:", length(y1), "\n")
#cat("Prediction set size:", length(ss_out), "\n")

# --- 4. Eigendecomposition of Training GRM ---
eigen_G11 <- eigen(G11)
U1 <- eigen_G11$vectors    # Eigenvectors (U_1)
L1 <- eigen_G11$values     # Eigenvalues (Lambda_1)

# --- 5. Center and Rotate Training Data ---
mu_hat <- mean(y1)
y1_centered <- y1 - mu_hat
y1_star <- crossprod(U1, y1_centered) # t(U1) %*% y1_centered

# --- 6. Define the REML Log-Likelihood Function ---
neg_logL_REML <- function(params) {
  var_g <- params[1]
  var_e <- params[2]
  
  v_i <- L1 * var_g + var_e 
  
  # Check for non-positive variances which lead to log(negative)
  if (any(v_i <= 0)) {
    return(1e10) # Return a large number if parameters are invalid
  }
  
  logLik <- -0.5 * (sum(log(v_i)) + sum(y1_star^2 / v_i))
  return(-logLik)
}

# --- 6b. Define the Gradient Function ---
eval_g_REML <- function(params) {
  var_g <- params[1]
  var_e <- params[2]
  
  v_i <- L1 * var_g + var_e
  
  # Handle potential invalid parameters
  if (any(v_i <= 1e-9)) {
    # Return a gradient that pushes away from the boundary
    return(c(1e10, 1e10)) 
  }
  
  v_i_inv <- 1.0 / v_i
  v_i_sq_inv <- 1.0 / (v_i^2)
  
  common_term <- v_i_inv - (y1_star^2 * v_i_sq_inv)
  
  grad_g <- 0.5 * sum(L1 * common_term)
  grad_e <- 0.5 * sum(common_term)
  
  return(c(grad_g, grad_e))
}


# --- 7. Optimize Likelihood (with Gradient) ---
opt_results_nloptr <- nloptr(
  x0 = c(var_g = 1, var_e = 1),  # Starting parameters
  eval_f = neg_logL_REML,       # Objective function
  eval_grad_f = eval_g_REML,   # <-- ADDED: Gradient function
  lb = c(1e-6, 1e-6),           # Lower bounds (variances > 0)
  opts = list(
    "algorithm" = "NLOPT_LD_LBFGS", # This will now work
    "xtol_rel" = 1.0e-8
  )
)

# Extract the estimated variance components
var_g_hat <- opt_results_nloptr$solution[1]
var_e_hat <- opt_results_nloptr$solution[2]

#cat("\n--- Estimated Variance Components (nloptr + gradient) ---\n")
cat("Genetic Variance (sigma_g^2):", var_g_hat, "\n")
cat("Error Variance (sigma_e^2):", var_e_hat, "\n")

# --- 8. Predict Random Effects (g_2) for ss_out ---
lambda <- var_e_hat / var_g_hat
w <- y1_star / (L1 + lambda)
g2_hat <- G21 %*% (U1 %*% w)

# --- 9. Display Final Predictions ---
path <- snakemake@output[["blup"]]
write(g2_hat, path, ncolumns=nrow(g2_hat))
