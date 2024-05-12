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


############################################
#### Factor analysis
###########################################

FactorAnalysis <- setRefClass(
  # Factor analysis
  # Example usage
  # X <- matrix(rnorm(100), nrow=20, ncol=5)  # Random 20x5 matrix
  # fa_model <- FactorAnalysis$new()
  # fa_model$fit(X, n_factors = 2, rotation = "varimax")
  # loadings <- fa_model$get_loadings()
  # uniquenesses <- fa_model$get_uniquenesses()
  # print(loadings)
  # print(uniquenesses)
  "FactorAnalysis",
  fields = list(
    loadings = "ANY",             # To handle various types returned by factanal
    uniquenesses = "numeric",     # Uniquenesses for each variable
    n_factors = "numeric",        # Number of factors to retain
    rotation = "character"        # Type of rotation to apply
  ),
  
  methods = list(
    fit = function(X, n_factors, rotation = "varimax") {
      # Ensure matrix data
      if (!is.matrix(X) && !is.data.frame(X)) {
        stop("Input must be a matrix or data frame.")
      }
      
      # Validate and set the number of factors
      if (n_factors > ncol(X) || n_factors < 1) {
        stop("Number of factors must be between 1 and the number of columns in X.")
      }
      .self$n_factors <- n_factors
      .self$rotation <- rotation
      
      # Perform factor analysis
      fa_result <- factanal(X, factors = .self$n_factors, rotation = .self$rotation)
      
      # Store results
      .self$loadings <- fa_result$loadings
      .self$uniquenesses <- fa_result$uniqueness
      
      return(invisible(.self)) # Return self invisibly for method chaining
    },
    
    get_loadings = function() {
      # Check if Factor Analysis is already fitted
      if (is.null(.self$loadings)) {
        stop("Factor Analysis model has not been fitted yet. Call fit() first.")
      }
      # Return loadings as a matrix for easier handling in further analyses
      return(as.matrix(.self$loadings))
    },
    
    get_uniquenesses = function() {
      # Check if Factor Analysis is already fitted
      if (is.null(.self$uniquenesses)) {
        stop("Factor Analysis model has not been fitted yet. Call fit() first.")
      }
      return(.self$uniquenesses)
    }
  )
)


############################################
#### Latent Dirichlet Allocation (LDA)
###########################################

LDA <- setRefClass(
  "LDA",
  # Latent Dirichlet Allocation (LDA)
  # Example Document-Term Matrix (Sparse Representation)
  # dtm <- matrix(c(1,2,1,0,2,1,3,0,1), nrow=3, byrow=TRUE)
  # lda_model <- LDA$new()
  # lda_model$fit(dtm, K=2, alpha=0.1, beta=0.01, iterations=500)
  # print(lda_model$topics())
  # print(lda_model$topic_distribution())
  fields = list(
    K = "numeric",          # Number of topics
    alpha = "numeric",      # Hyperparameter for document-topic distributions
    beta = "numeric",       # Hyperparameter for topic-word distributions
    doc_topic = "matrix",   # Document-topic count matrix
    topic_word = "matrix",  # Topic-word count matrix
    topic_sum = "numeric",  # Total count of words assigned to topics
    doc_sum = "numeric"     # Total count of topics assigned to documents
  ),
  
  methods = list(
    fit = function(dtm, K, alpha, beta, iterations = 1000) {
      # Initialize
      .self$K <- K
      .self$alpha <- alpha
      .self$beta <- beta
      num_docs <- nrow(dtm)
      vocab_size <- ncol(dtm)
      
      .self$doc_topic <- matrix(0, num_docs, K)
      .self$topic_word <- matrix(0, K, vocab_size)
      .self$topic_sum <- numeric(K)
      .self$doc_sum <- numeric(num_docs)
      
      assignments <- list()
      
      # Randomly assign initial topics to each word in each document
      for (d in 1:num_docs) {
        assignments[[d]] <- numeric(length(dtm[d,]))
        for (w in which(dtm[d,] > 0)) {
          for (count in 1:dtm[d, w]) {
            topic <- sample(K, 1)
            assignments[[d]][w] <- topic
            .self$doc_topic[d, topic] <- .self$doc_topic[d, topic] + 1
            .self$topic_word[topic, w] <- .self$topic_word[topic, w] + 1
            .self$topic_sum[topic] <- .self$topic_sum[topic] + 1
          }
        }
        .self$doc_sum[d] <- sum(dtm[d,])
      }
      
      # Gibbs sampling
      for (iter in 1:iterations) {
        for (d in 1:num_docs) {
          for (w in which(dtm[d,] > 0)) {
            topic <- assignments[[d]][w]
            # Decrement counts
            .self$doc_topic[d, topic] <- .self$doc_topic[d, topic] - 1
            .self$topic_word[topic, w] <- .self$topic_word[topic, w] - 1
            .self$topic_sum[topic] <- .self$topic_sum[topic] - 1
            
            # Sample new topic
            prob <- (.self$doc_topic[d,] + alpha) *
              ((.self$topic_word[, w] + beta) / (.self$topic_sum + vocab_size * beta))
            topic <- which.max(rmultinom(1, 1, prob))
            assignments[[d]][w] <- topic
            
            # Increment counts
            .self$doc_topic[d, topic] <- .self$doc_topic[d, topic] + 1
            .self$topic_word[topic, w] <- .self$topic_word[topic, w] + 1
            .self$topic_sum[topic] <- .self$topic_sum[topic] + 1
          }
        }
      }
      return(invisible(.self))
    },
    
    topics = function(num_terms = 10) {
      apply(.self$topic_word, 1, function(x) order(x, decreasing = TRUE)[1:num_terms])
    },
    
    topic_distribution = function() {
      t(.self$doc_topic) / rowSums(.self$doc_topic)
    }
  )
)



############################################
#### Latent Dirichlet Allocation (LDA)
###########################################

NMF <- setRefClass(
  # Latent Dirichlet Allocation (LDA)
  # Assume V is the matrix used in fit
  # V <- matrix(rnorm(100), nrow = 10, ncol = 10)
  # V_new is the new data matrix you want to transform
  # V_new <- matrix(rnorm(30), nrow = 10, ncol = 3)
  # Create and fit the NMF model
  # nmf_model <- NMF$new()
  # nmf_model$fit(V, n_components = 2, max_iter = 200)
  # new_coefficients <- nmf_model$transform(V_new)
  # print("New Coefficients (H_new):")
  # print(new_coefficients)
  "NMF",
  fields = list(
    W = "matrix",  # Basis matrix
    H = "matrix",  # Coefficient matrix from the training
    n_components = "numeric"  # Number of components (features)
  ),
  
  methods = list(
    fit = function(V, n_components, max_iter = 100, tol = 1e-4) {
      # Validate inputs
      if (!is.matrix(V)) {
        stop("Input V must be a matrix.")
      }
      
      # Initialize dimensions and parameters
      n <- nrow(V)
      m <- ncol(V)
      .self$n_components <- n_components
      
      # Randomly initialize W and H
      .self$W <- matrix(runif(n * n_components), nrow = n, ncol = n_components)
      .self$H <- matrix(runif(n_components * m), nrow = n_components, ncol = m)
      
      # Begin iterations
      for (i in 1:max_iter) {
        # Update H
        H_new <- .self$H * ((t(.self$W) %*% V) / (t(.self$W) %*% .self$W %*% .self$H + tol))
        # Update W
        W_new <- .self$W * ((V %*% t(.self$H)) / (.self$W %*% .self$H %*% t(.self$H) + tol))
        
        # Check for convergence
        if (sqrt(sum((W_new - .self$W)^2)) < tol && sqrt(sum((H_new - .self$H)^2)) < tol) {
          message("Convergence achieved after ", i, " iterations.")
          break
        }
        
        # Update matrices
        .self$W <- W_new
        .self$H <- H_new
      }
      
      return(invisible(.self))
    },
    
    transform = function(V_new, max_iter = 100, tol = 1e-4) {
      if (!is.matrix(V_new)) {
        stop("Input V_new must be a matrix.")
      }
      
      # Initialize H_new with random values
      H_new <- matrix(runif(.self$n_components * ncol(V_new)), nrow = .self$n_components, ncol = ncol(V_new))
      
      # Update H_new to minimize |V_new - W*H_new|
      for (i in 1:max_iter) {
        H_update <- H_new * ((t(.self$W) %*% V_new) / (t(.self$W) %*% .self$W %*% H_new + tol))
        if (sqrt(sum((H_update - H_new)^2)) < tol) {
          message("Convergence achieved after ", i, " iterations.")
          break
        }
        H_new <- H_update
      }
      
      return(H_new)
    },
    
    get_features = function() {
      return(.self$W)
    },
    
    get_coefficients = function() {
      return(.self$H)
    }
  )
)

############################################
#### Truncated SVD
###########################################

TruncatedSVD <- setRefClass(
  "TruncatedSVD",
  # Truncated SVD
  # Example usage
  # X <- matrix(rnorm(100), nrow = 10, ncol = 10)
  # Create the Truncated SVD model and fit it
  #svd_model <- TruncatedSVD$new()
  #svd_model$fit(X, n_components = 3)
  # Transform the data using the fitted model
  # transformed_X <- svd_model$transform(X)
  # Retrieve components and singular values
  # components <- svd_model$get_components()
  # singular_values <- svd_model$get_singular_values()
  # print("Transformed Data:")
  # print(transformed_X)
  # print("Components (V):")
  # print(components)
  # print("Singular Values:")
  # print(singular_values)
  fields = list(
    U = "matrix",         # Left singular vectors
    S = "matrix",         # Diagonal matrix of singular values
    V = "matrix",         # Right singular vectors
    n_components = "numeric"  # Number of components to retain
  ),
  
  methods = list(
    fit = function(X, n_components) {
      # Validate inputs
      if (!is.matrix(X)) {
        stop("Input X must be a matrix.")
      }
      if (n_components > min(nrow(X), ncol(X))) {
        stop("Number of components cannot exceed the smaller dimension of X.")
      }
      
      # Compute the SVD
      svd_result <- svd(X)
      .self$U <- svd_result$u[, 1:n_components, drop = FALSE]
      .self$S <- diag(svd_result$d[1:n_components])
      .self$V <- svd_result$v[, 1:n_components, drop = FALSE]
      .self$n_components <- n_components
      
      return(invisible(.self))  # Return self invisibly for method chaining
    },
    
    transform = function(X) {
      # Check if SVD has been fitted
      if (is.null(.self$U)) {
        stop("Truncated SVD has not been fitted yet. Call fit() first.")
      }
      # Project X onto the first n_components singular vectors
      return(X %*% .self$V)
    },
    
    get_components = function() {
      return(.self$V)
    },
    
    get_singular_values = function() {
      return(diag(.self$S))
    }
  )
)






