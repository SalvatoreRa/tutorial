
library(ggplot2)

#iris dataset
data(iris)

#colnames
names(iris)

#dimension of dataset
dim(iris)

#plot a dot-plot using the first two variables and color according to the specie
g <- ggplot(iris, aes(x =Sepal.Length, y =  Sepal.Width, col = Species)) +
  geom_point(size = 4) +
  labs(title = "Iris dataset")
g

#save it
jpeg(file = "iris.jpg", width =800, height =500, units = "px")
plot(g)
dev.off()


#### training a tree with 2 features ####
#import library for tree
install.packages("tree")
library(tree)

tree_2f <- tree(Species ~ Sepal.Width + Petal.Width, data = iris)
summary(tree_2f)

plot(tree_2f) #plot the tree
text(tree_2f) #add the text

#save it
jpeg(file = "iris_tree_2features.jpg", width =800, height =500, units = "px")
plot(tree_2f)
text(tree_2f)
dev.off()


### plot the partition
plot(iris$Petal.Width,iris$Sepal.Width,pch=19,col=as.numeric(iris$Species))
partition.tree(tree_2f,label="Species",add=TRUE)
legend(1.75,4.5,legend=unique(iris$Species),col=unique(as.numeric(iris$Species)),pch=19)


#save it
jpeg(file = "iris_tree_partition.jpg", width =800, height =500, units = "px")
plot(iris$Petal.Width,iris$Sepal.Width,pch=19,col=as.numeric(iris$Species))
partition.tree(tree_2f,label="Species",add=TRUE)
legend(1.75,4.5,legend=unique(iris$Species),col=unique(as.numeric(iris$Species)),pch=19)
dev.off()

#### using more variabele ###
tree <- tree(Species ~ Sepal.Width + Sepal.Length + Petal.Length + Petal.Width, data = iris)
#or this is the same
tree <- tree(Species ~ ., data = iris)
summary(tree)

plot(tree)
text(tree)

#save it
jpeg(file = "iris_tree_all_features.jpg", width =800, height =500, units = "px")
plot(tree_2f)
text(tree_2f)
dev.off()


#### alternative visualization ####
install.packages('rpart')
install.packages('rattle')


library(rpart)
library(rattle)

#training with all the features
tree_rpart <- rpart(Species ~ ., data=iris, method="class",)
tree_rpart

# plot decision tree
fancyRpartPlot(tree_rpart, main="Iris")

#save it
jpeg(file = "iris_tree_all_f_alternative_package.jpg", width =800, height =500, units = "px")
fancyRpartPlot(tree_rpart, main="Iris")
dev.off()


#train the moodel limiting the depth
tree_rpart <- rpart(Species ~ ., data=iris, method="class",
               control = list(maxdepth = 1))
tree_rpart


fancyRpartPlot(tree_rpart, main="Iris")

#save it
jpeg(file = "iris_tree_mindepth1.jpg", width =800, height =500, units = "px")
fancyRpartPlot(tree_rpart, main="Iris")
dev.off()

