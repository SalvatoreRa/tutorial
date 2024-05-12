############################################
#### Decision tree classifier
###########################################

DecisionTreeClassifier <- setRefClass(
  "DecisionTreeClassifier",
  # # Generate some example data
  # set.seed(123)
  # X <- matrix(rnorm(200), nrow = 100, ncol = 2)
  # y <- ifelse(X[, 1] + X[, 2] > 0, 1, 0)  
  # Create and fit the decision tree model
  # tree_model <- DecisionTreeClassifier$new(max_depth = 3)
  # tree_model$fit(X, y)
  # Predict using the fitted model
  # predictions <- tree_model$predict(X)
  # print("Predictions:")
  # print(predictions)
  fields = list(
    max_depth = "numeric",  # Maximum depth of the tree
    tree = "list"           # The structure of the tree, initialized as an empty list
  ),
  
  methods = list(
    initialize = function(max_depth = 5) {
      .self$max_depth <- max_depth
      .self$tree <- list()  # Ensure this is initialized as an empty list, not NULL
    },
    
    fit = function(X, y) {
      # Validate inputs
      if (!is.matrix(X)) {
        stop("Input X must be a matrix.")
      }
      if (!all(y %in% c(0, 1))) {
        stop("Input y must be a binary vector with values 0 and 1.")
      }
      
      # Convert y to factor and ensure levels are correctly set as "0" and "1"
      y <- factor(y, levels = c(0, 1))
      
      # Start building the tree
      .self$tree <- recursive_split(X, y, depth = 0)
    },
    
    recursive_split = function(X, y, depth) {
      # Handle case when there's no data to split
      if (nrow(X) == 0) {
        return(list(is_leaf = TRUE, class = NA))
      }
      # Determine stopping conditions
      if (depth >= .self$max_depth || nrow(X) < 2 || length(unique(y)) == 1) {
        # Return the mode of y as class
        return(list(is_leaf = TRUE, class = as.numeric(levels(y))[which.max(table(y))]))
      }
      
      # Initialize variables for the best split
      best_feature <- NULL
      best_threshold <- NULL
      best_score <- Inf
      
      # Iterate through each feature
      for (feature in seq(ncol(X))) {
        unique_values <- sort(unique(X[, feature]))
        thresholds <- head(unique_values, -1) + diff(unique_values) / 2
        
        for (threshold in thresholds) {
          left_idx <- X[, feature] <= threshold
          right_idx <- !left_idx
          left_y <- y[left_idx]
          right_y <- y[right_idx]
          score <- (length(left_y) / length(y)) * gini(left_y) +
            (length(right_y) / length(y)) * gini(right_y)
          
          if (score < best_score) {
            best_score <- score
            best_feature <- feature
            best_threshold <- threshold
          }
        }
      }
      
      # Check if a valid split was found
      if (!is.null(best_feature)) {
        left_idx <- X[, best_feature] <= best_threshold
        right_idx <- !left_idx
        left_tree <- recursive_split(X[left_idx, , drop = FALSE], y[left_idx], depth + 1)
        right_tree <- recursive_split(X[right_idx, , drop = FALSE], y[right_idx], depth + 1)
        
        return(list(is_leaf = FALSE, feature = best_feature, threshold = best_threshold, left = left_tree, right = right_tree))
      } else {
        # No valid split found, return a leaf node
        return(list(is_leaf = TRUE, class = as.numeric(levels(y))[which.max(table(y))]))
      }
    },
    
    predict = function(X) {
      if (is.null(.self$tree)) {
        stop("The model has not been fitted yet.")
      }
      apply(X, 1, function(x) predict_instance(x, .self$tree))
    },
    
    predict_instance = function(x, node) {
      if (node$is_leaf) {
        return(node$class)
      } else {
        if (x[node$feature] <= node$threshold) {
          return(predict_instance(x, node$left))
        } else {
          return(predict_instance(x, node$right))
        }
      }
    },
    
    gini = function(y) {
      p <- table(y) / length(y)
      return(1 - sum(p^2))
    }
  )
)

      

# Generate some example data
set.seed(123)
X <- matrix(rnorm(200), nrow = 100, ncol = 2)
y <- ifelse(X[, 1] + X[, 2] > 0, 1, 0)  # Simple linear classification boundary

# Create and fit the decision tree model
tree_model <- DecisionTreeClassifier$new(max_depth = 3)
tree_model$fit(X, y)

# Predict using the fitted model
predictions <- tree_model$predict(X)

print("Predictions:")
print(predictions)

