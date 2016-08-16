

library(dplyr)
library(ggplot2)
library(ggmap)
library(hexbin)

collision <- read.csv("C:\\Users\\Craig\\Documents\\Website\\Vehicle Accidents\\nypd_motor_vehicle_collisions.csv",
                   header=TRUE,
                   sep=",")

collision <- tbl_df(collision)

dim(collision) # 850,053 rows and 29 columns
head(collision)
sapply(collision, class)

# check the date range
collision$DATE <- as.Date(collision$DATE, format = "%m/%d/%Y") # convert date to date format
min(collision$DATE) # 2012-07-01
max(collision$DATE) # 2016-07-23


# time series chart of number of accidents
collision_by_day <- group_by(collision, DATE) %>% summarise(count = n())
collision_by_day$count <- as.numeric(collision_by_day$count)

plot <- ggplot(collision_by_day, aes(DATE, count)) +
  geom_line(na.rm=TRUE, color = "darkgreen") +
  scale_x_date() +
  xlab("") +
  ylab("Number of Accidents") +
  ggtitle("Daily Car Accidents")

# remove values without latitude and longitude
coll <- filter(collision, !is.na(LATITUDE) & !is.na(LONGITUDE))

coll$LATITUDE <- round(coll$LATITUDE, 3)
coll$LONGITUDE <- round(coll$LONGITUDE, 3)

# plot latitude and longitudes of car accidents
plot <- ggplot(coll, aes(x=LONGITUDE, y=LATITUDE)) +
  geom_point(size=0.05)
plot



# plot counts by lat and long
coll <- select(coll, LATITUDE, LONGITUDE) %>% group_by(LATITUDE,LONGITUDE) %>% summarise(count=n())
                      
plot2 <- ggplot(coll, aes(x=LONGITUDE, y=LATITUDE,color=count)) +
  geom_point(size=0.07) +
  scale_color_gradient(low="white", high="orange", trans="log")
plot2

plot2 <- ggplot(coll, aes(x=LONGITUDE, y=LATITUDE,color=count)) +
  geom_point(size=0.07) +
  scale_color_gradient(low="white", high="red", trans="log") +
  ggtitle("NYC Accident Density - 2012 to 2016")
plot2








