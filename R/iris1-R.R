print(iris)

install.packages(c("data.table","dplyr"))

library(data.table)
library(dplyr)


iris2 <- iris[which(Species == "setosa" | Species = "versicolor")]
