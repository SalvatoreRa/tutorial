
#import libraries
library(ggplot2)


### simple example ###
#creating a simple dataset
#i am using this formula and the change a bit the values to introduce randomness
#y = (x* 1.7) + 2.1

x <- c(10, 12, 15, 18, 21, 23, 25, 28, 31, 35, 39, 45, 37, 51)
y <- c(19.1, 23.5, 27.2, 32.7, 36.8, 43.2, 44.6, 53.7, 54.5, 61.4, 68.7, 78.9, 65.3, 88.5)
df = data.frame(length = x, weight = y)

g <- ggplot(df, aes(length, weight )) +
  geom_point(col = "blue", size = 4)

#save it
jpeg(file = "r_regr_simple_dataset.jpg", width =800, height =500, units = "px")
g
dev.off()


#fit a simple regression
#lm([target variable y] ~ [imput variables X], data = [data frame])

#check the function
?lm

#fitting the model
regr = lm(weight ~ length, data = df )

#a summary of the regression model
summary(regr)

#plotted the fitted line
g1 <- ggplot(df, aes(length, weight )) +
  geom_point(col = "blue", size = 4) +
  geom_abline(slope = coef(regr)[["length"]], #slope
              intercept = coef(regr)[["(Intercept)"]]) #intercept


#save
jpeg(file = "fitted_simple_regression.jpg", width =800, height =500, units = "px")
g1
dev.off()

#residuals
plot(regr$residuals, pch = 16, col = "blue")

#save
jpeg(file = "residuals_simple_regression.jpg", width =800, height =500, units = "px")
plot(regr$residuals, pch = 16, col = "blue")
dev.off()


#plot cook distance
plot(cooks.distance(regr), pch = 16, col = "blue")


#save
jpeg(file = "cookdistance_simple_regression.jpg", width =800, height =500, units = "px")
plot(cooks.distance(regr), pch = 16, col = "blue")
dev.off()


### multivariate regression ###

df <- read.delim("https://raw.githubusercontent.com/SalvatoreRa/tutorial/main/datasets/Boston.csv", sep = ",")
df <- df[,-1]
head(df)

regr = lm(medv ~ lstat, data = df )
summary(regr)

#plotted the fitted line
g1 <- ggplot(df, aes(lstat, medv )) +
  geom_point(col = "blue", size = 4) +
  geom_abline(slope = coef(regr)[["lstat"]], #slope
              intercept = coef(regr)[["(Intercept)"]]) #intercept

#save
jpeg(file = "boston_simple_regression.jpg", width =800, height =500, units = "px")
g1
dev.off()

#linear regression using all the variable
#to use only a subset you can write:
#regr = lm(medv ~ crim + zn + indus, data = df )
regr = lm(medv ~ ., data = df )
summary(regr)


#residuals
plot(regr$residuals, pch = 16, col = "blue")

#save
jpeg(file = "residuals_boston_regression.jpg", width =800, height =500, units = "px")
plot(regr$residuals, pch = 16, col = "blue")
dev.off()

#plot cook distance
plot(cooks.distance(regr), pch = 16, col = "blue")

#save
jpeg(file = "cookdistance_boston_regression.jpg", width =800, height =500, units = "px")
plot(cooks.distance(regr), pch = 16, col = "blue")
dev.off()



