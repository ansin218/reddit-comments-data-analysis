# Add required libraries and packages
library(RSQLite)
library(dplyr)
library(ggplot2)

# Read the dataset using SQLite DB
db <- src_sqlite('data.sqlite', create=F)
sub <- "politics"

# Sort data with score above
dbsub <- db %>%
  tbl('January2015') %>%
  filter(subreddit==sub, score > 300) 
df <- collect(dbsub)

# Compute high scores for comments throughout days of the week
postday <- filter(df, nchar(body) > 30) %>% 
  select(created_utc) %>% 
  mutate(created_utc = as.POSIXct(created_utc, origin = "1970-01-01"), day = as.numeric(strftime(created_utc, "%u")))
  
# Generate plot for high scores for comments throughout days of the week
ggplot(posttimes, aes(x=day)) + geom_histogram(binwidth=0.20) + scale_x_continuous(breaks = 0:7) + coord_polar() + ggtitle(sub) + theme_bw()

# Compute high scores for comments throughout hours of the day
postday <- filter(df, nchar(body) > 30) %>% 
  select(created_utc) %>% 
  mutate(created_utc = as.POSIXct(created_utc, origin = "1970-01-01"), hour = as.numeric(strftime(created_utc, "%h")))
  
# Generate plot for high scores for comments throughout days of the week
ggplot(posttimes, aes(x=hour)) + geom_histogram(binwidth=0.20) + scale_x_continuous(breaks = 0:24) + coord_polar() + ggtitle(sub) + theme_bw()