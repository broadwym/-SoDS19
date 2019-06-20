# Load libraries 
library(tidyverse)
library(rvest)
library(stringr)
library(xml2)
library(purrr)
library(httr)
library(dplyr)

# Attempting to paginate
## I can loop over a range of years from 1919-2019 and append those dates to the end of the base URL. 

years <- seq(1919, 2019, by=1)

pages <- c("http://aviation-safety.net/database/dblist.php?Year=") %>%
  paste0(years) 

# Leaving out the category, location, operator, etc. nodes for sake of brevity 
read_table <- function(url){
  #add delay so that one is not attacking the host server (be polite)
  Sys.sleep(0.5)
  #read page
  page <- read_html(url)
  #extract the table out (the data frame is stored in the first element of the list)
  answer<-(page %>% html_nodes("table") %>%  html_table())[[1]]
  #convert the falatities column to character to make a standardize column type
  answer$fat. <-as.character(answer$fat.)
  answer
} 

# Writing to dataframe
aviation_df <- bind_rows(lapply(pages, read_table))

length(aviation_df)
aviation_df <- aviation_df[, -c(11:109)]
aviation_df <- aviation_df[, -c(7,8,10)]

aviation_df <- aviation_df %>% mutate_all(na_if,"")
aviation_df <- aviation_df %>% mutate_all(na_if,"unknown")
aviation_df <- aviation_df %>% mutate_all(na_if,"Unknown")


### Do not run 
# Looks like there's cells with a length of different blanks
## registration
sum(!grepl("^\\s+$|^$", aviation_df$registration))
aviation_df$registration <- sub("^\\s+$|^$", "NA", aviation_df$registration)
## Change to 'real' NAs
aviation_df$registration <- na_if(aviation_df$registration, 'NA') 
# location
aviation_df$location <- sub("^\\s+$|^$", "NA", aviation_df$location)
## Change to 'real' NAs
aviation_df$location <- na_if(aviation_df$location, 'NA') 

aviation_df$year <- regmatches(aviation_df$date, gregexpr("\\d{4}", aviation_df$date))
aviation_df$date <- gsub("?", "", aviation_df$date, fixed = TRUE)
aviation_df$type <- gsub("?", "", aviation_df$type, fixed = TRUE)
aviation_df$date <- gsub("--", "", aviation_df$date, fixed = TRUE)
aviation_df$date <- sub("^[^[:alnum:]]", "", aviation_df$date)
