library(dplyr)
library(tibble)
library(tidyr)
library(tidyverse)
library(ggplot2)

health <- read.csv("healthcare_noshows_appointments.csv", header=TRUE) %>%
  data.frame()%>%
  view()

dim(health)

attach(health)
names(health)
summary(health)

length(unique(health$ScheduledDay))
length(unique(health$AppointmentDay))

showed_up_percent <- sum(health$Showed_up==TRUE) / length(health$Showed_up)
showed_up_percent

'''
Only 27 unique  values for appointment day, but 110 for schedule day. Min of date_diff is also -6 which is weird, probably a mistake.
Most common value for date_diff = 0, meaning the appointment was on the same date that it was scheduled. Maybe this was for urgent care?
Almost 80% showed up, so pretty imbalanced dataset. Will probably need to do some specific split, bootstraping and cross validation.
No duplicates in AppointmentID. 81 different neighbourhoods might be too much for one-hot encoding.
'''

# Change dates to DateTime type

health$ScheduledDay <- as.Date(health$ScheduledDay, format="%Y-%m-%d")
health$AppointmentDay <- as.Date(health$AppointmentDay, format="%Y-%m-%d")

summary(health)

# Checking for missing values

sum(is.na(health))

# Investigating negative values in date diff 

health[health$Date.diff < 0, ]

# Clearly these are mistakes and should be removed

health <- health[health$Date.diff >= 0, ]

# Data looks clean, no missing values and no duplicates
# Now we will look into unique patients and the distribution of multiple visits

visit_counts <- as.data.frame(table(health$PatientId))
colnames(visit_counts) <- c("PatientId", "VisitCount")
multiple_visits <- visit_counts[visit_counts$VisitCount > 1, ]
multiple_visits <- multiple_visits[, "VisitCount", drop = FALSE]

barplot(table(multiple_visits[["VisitCount"]]),
        main = paste("Barplot of visit counts across unique patients"),
        col = "lightblue",
        xlab = 'Visits',
        ylab = "Count")

length(multiple_visits$VisitCount)

one_visit <- visit_counts[visit_counts$VisitCount == 1, ]
one_visit <- one_visit[, "VisitCount", drop = FALSE]

length(one_visit$VisitCount)

'''
36850 people had only one visit, meaning that almost 70 000 had more than one.
Could be possible to predict a no show based on previous behaviour in this case.
Vast majority of those came 2 times, maybe one inital consultation and then one check-up?
'''

'''
If we hypothesize that date difference between schedule day and appointment day might have a significant impact on a no-show, it makes sense to look at
percentage of a show-up and no-show across unique date differences.
The following results reveal that the mode - 0 has a show up of 95%, which makes sense. Across other date differences, the percentages vary from ~50 to ~75 but
we have to keep in mind the smaller counts. More interesting are bigger values, starting from 92 days. They are pretty rare but often have a 100% show-up.
Maybe it would be possible in the modeling stage to give different weights to intervals of dates, letting the model focus on extreme values. Or do
feature engineering using this information.

Another idea for feature engineering would be to create a season feature (summer, autumn, winter, spring) based on the date.
Create a cumulative sum for each patient of his show ups and no-shows.
'''
unique_date_diff <- unique(health$Date.diff)

date_diff_show_up_percentage <- data.frame(
  DateDiff = numeric(),
  ShowedUpCount = integer(),
  NoShowCount = integer(),
  ShowedUpPercentage = numeric(),
  NoShowPercentage = numeric(),
  stringsAsFactors = FALSE
)

for (date_diff in unique_date_diff) {
  showed_up_count <- sum(health$Date.diff == date_diff & health$Showed_up == TRUE)
  no_show_count <- sum(health$Date.diff == date_diff & health$Showed_up == FALSE)
  total_count <- showed_up_count + no_show_count
  show_up_percentage <- showed_up_count / total_count * 100.0
  no_show_percentage <- no_show_count / total_count * 100.0
  
  date_diff_show_up_percentage <- rbind(date_diff_show_up_percentage, data.frame(
    DateDiff = date_diff,
    ShowedUpCount = showed_up_count,
    NoShowCount = no_show_count,
    ShowedUpPercentage = show_up_percentage,
    NoShowPercentage = no_show_percentage
  ))
  
}

table<-arrange(date_diff_show_up_percentage, DateDiff)

install.packages("clipr")
library(clipr)
write_clip(table)

# Further analysing dates between appointments. Now looking into date differences between appointments across patients who had more than one visit.

health <- health %>%
  arrange(PatientId, AppointmentDay) %>%
  group_by(PatientId) %>%
  mutate(DaysSinceLastAppointment = as.numeric(AppointmentDay - lag(AppointmentDay)))

barplot(table(health[["DaysSinceLastAppointment"]]),
        main = paste("Barplot of days since last appointment across unique patients"),
        col = "lightblue",
        xlab = 'Days',
        ylab = "Count")

'''
Interesting to see that most people who had multiple appointments had them on the same day.
'''

# Visualising data

create_barplot <- function(column_name){
  barplot(table(health[[column_name]]),
          main = paste("Barplot of ",column_name),
          col = "lightblue",
          xlab = column_name,
          ylab = "Count")
}

create_boxplot <- function(column_name) {
  boxplot(health[[column_name]],
          main = paste("Boxplot of", column_name),
          col = "lightblue",
          xlab = column_name,
          ylab = "Values")
}

create_barplot("Gender")      
create_barplot("Age") 
create_barplot("Neighbourhood") 
create_barplot('Scholarship')
create_barplot("Hipertension") 
create_barplot("Diabetes") 
create_barplot("Alcoholism") 
create_barplot("Handcap")
create_barplot("SMS_received") 
create_barplot("Showed_up") 

create_boxplot("Age") 
create_boxplot("Date.diff") 

'''
Based on up to here, data seems clean. Very disbalanced across categories. No reasonable outliers(?) to remove, maybe later.
Most categories are very concentrated around mode.
'''


# Statistical testing
# We will perfom Chi-squared test for independence across cartesian product of columns

health <- health[, !(names(health) %in% c("PatientId", "AppointmentID"))]

health[] <- lapply(health, function(x) if (is.logical(x)) as.factor(x) else x)
categorical_columns <- sapply(health, is.factor) | sapply(health, is.character)

cat_cols <- names(health)[categorical_columns]

# Create all pairs of categorical columns
pairs <- combn(cat_cols, 2, simplify = FALSE)

results <- list()

# Perform Chi-square test for each pair
for (pair in pairs) {
  contingency_table <- table(health[[pair[1]]], health[[pair[2]]])
  test_result <- chisq.test(contingency_table)
  
  # Store results
  results[[paste(pair, collapse = " vs ")]] <- list(
    contingency_table = contingency_table,
    p_value = test_result$p.value,
    statistic = test_result$statistic,
    expected = test_result$expected
  )
}

for (name in names(results)) {
  cat("\nChi-square Test between", name, "\n")
  cat("Chi-square Statistic:", results[[name]]$statistic, "\n")
  cat("p-value:", results[[name]]$p_value, "\n")
  cat("\n")
}


'''
Reject almost all, except (Gender and Show_up), (Handcap, showed_up) and (showed_up, alcoholism)
meaning we wouldnt use these directly for predicting showing up?
Discard the gender overall, and for handicap, alcoholism and other similar features perform
clustering to get new features.
'''

