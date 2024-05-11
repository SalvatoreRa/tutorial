############################################
#### Principal Component analysis
###########################################

PCA <- setRefClass(
  "PCA",
  # PCA class
  # Fit Method
  # Transform Method
  # Example usage
  # X <- matrix(rnorm(200), nrow=20, ncol=10)  
  # pca_model <- PCA$new()
  # pca_model$fit(X, n_components = 5) 
  # transformed_X <- pca_model$transform(X)
  #explained_variance <- pca_model$get_explained_variance()
  # print(transformed_X)
  # print(explained_variance)
  fields = list(
    mean = "numeric",               # Mean of each column
    rotation = "matrix",            # Eigenvectors (principal components)
    standard_deviation = "numeric", # Standard deviations (square roots of eigenvalues)
    n_components = "numeric"        # Number of components to retain
  ),
  
  methods = list(
    fit = function(X, n_components = NULL) {
      
      if (!is.matrix(X)) {
        stop("Input must be a matrix.")
      }
      
      
      if (!is.null(n_components)) {
        if (n_components > ncol(X) || n_components < 1) {
          stop("Number of components must be between 1 and the number of columns in X.")
        }
        .self$n_components <- n_components
      } else {
        .self$n_components <- ncol(X)
      }
      
      .self$mean <- colMeans(X)
      X_centered <- sweep(X, 2, .self$mean, "-")
      
      
      covariance_matrix <- cov(X_centered)
      
      
      eigen_decomp <- eigen(covariance_matrix)

      .self$rotation <- eigen_decomp$vectors[, 1:.self$n_components]
      .self$standard_deviation <- sqrt(eigen_decomp$values[1:.self$n_components])
      
      return(invisible(.self)) 
    },
    
    transform = function(X) {
      
      if (is.null(.self$rotation)) {
        stop("The PCA model has not been fitted yet. Call fit() first.")
      }
      
      
      X_centered <- sweep(X, 2, .self$mean, "-")
      
      
      X_transformed <- X_centered %*% .self$rotation
      return(X_transformed)
    },
    
    get_explained_variance = function() {
      # Check if PCA is already fitted
      if (is.null(.self$standard_deviation)) {
        stop("The PCA model has not been fitted yet. Call fit() first.")
      }
      
      # Calculate explained variance
      variances <- .self$standard_deviation^2
      total_variance <- sum(variances)
      explained_variance <- variances / total_variance
      return(explained_variance)
    }
  )
)



