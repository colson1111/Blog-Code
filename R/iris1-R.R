
# install the dplyr package
install.packages("dplyr")

# load the dplyr package
library(dplyr)

# store iris data in a new dataframe
iris_work <- iris

# use dplyr filter and select functions with magrittr pipe operator to select rows and columns
iris_work <- iris %>% 
  filter(Species == "setosa" | Species == "versicolor") %>% 
  select(c(Species,Sepal.Length, Sepal.Width))

# what is iris_work?
class(iris_work) # data.frame

# what is the shape of iris_work?
dim(iris_work) # 100 rows by 3 columns

# plot the data
install.packages("ggplot2")
library(ggplot2)

p1 <- ggplot(iris_work, aes(x = Sepal.Width, y = Sepal.Length)) +  # create ggplot object 
  geom_point(aes(color = factor(Species))) +                       # define aesthetics of points, and color by Species
  scale_color_discrete(name = "Species",                           # change default legend values
                       labels = c("Setosa","Versicolor"))

p1 # show plot




