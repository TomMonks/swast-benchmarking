## Time series cross validation: Singular Spectrum Analysis
## Analysis conducted using Rssa version 1.0.2. and R 3.6.1
## r-base, r-forecast and r-svd dependencies managed via conda 


######## cran dependency management ############################################

rssaURL = 'https://cran.r-project.org/src/contrib/Archive/Rssa/Rssa_1.0.2.tar.gz'

#load Rssa if installed and return result (TRUE = loaded, FALSE not loaded.)
rssaInstalled = require('Rssa')
if(!rssaInstalled)
{
  ##install version 1.0.2 of Rssa
  install.packages(rssaURL, repos=NULL, type='source')
  library('Rssa')
}
################################################################################

#check working directory...
#code assumes call from swast-benchmarking/
getwd()

## read r-formatted data from file...
ts_r <- read.csv("./data/ts_r.csv", header=FALSE)

## Create a daily Date object - helps my work on dates
inds <- seq(as.Date("2013-12-30"), as.Date("2019-07-28"), by = "day")

## Create a time series object
myts <- ts(ts_r$V2[1:1279],     
           start = c(2013, as.numeric(format(inds[1], "%j"))),
           frequency = 365)

actual <- ts(ts_r$V2[1280:(1279+365)],
          start = c(2017, as.numeric(format(inds[1280], "%j"))),
          frequency = 365)

s <- ssa(myts)
summary(s)        # Show various information about the decomposition
plot(s)           # Show the plot of the eigenvalues
r <- reconstruct(s, groups = list(Trend = c(1, 4),
                                  Seasonality = c(2:3, 5:6))) # Reconstruct into 2 series
plot(r, add.original = TRUE)  # Plot the reconstruction

# Calculate 365-point forecast using first 6 components 
f <- forecast(s, groups = list(1:6), method = "recurrent", 
              interval='prediction', level=0.80, R=20, len = 365)

f <- forecast(s, groups = list(1:6), method = "recurrent", 
              len = 365)

# Plot the result (mean + prediction intervals) including the last 24 points of the series
plot(f, include = 24, shadecols = "orange", type = "l")
#add validation samples for illustration
lines(actual, col='green')

#** Cross Validation.
#create forecasts using rolling forecast origin
#retrain on each data set
max_horizon <- 365
origin <-seq(1279, 1279+549-max_horizon, 7)
levels <- c(0.80, 0.95)
n_boots <- 20
eigentriples <- 14

#run tscv twice.
for (level in levels)
{
  for (i in origin)
  {
    yr <- as.numeric(substring(inds[i], 1, 4))
    yr_val <- as.numeric(substring(inds[i+1], 1, 4))
    print(i)
    
    cv_ts <- ts(ts_r$V2[1:i],     
               start = c(yr, as.numeric(format(inds[i], "%j"))),
               frequency = 365)
    
    val_ts <- ts(ts_r$V2[(i+1):(i+max_horizon)],
                 start = c(yr_val, as.numeric(format(inds[i+1], "%j"))),
                 frequency = 365)
    
    #SSA analysis
    s <- ssa(cv_ts)
    f <- forecast(s, groups = list(1:eigentriples), method = "recurrent", 
                  interval='prediction', level=level, R=n_boots, len=max_horizon)
    
    #point forecast and predition intervals save to DF
    cv_data = data.frame(f$mean, f$lower, f$upper, val_ts)
    
    #save df to file ssa sub-directory for relevant PI.
    stage1 = './analysis/model_selection/stage1'
    write.csv(cv_data, paste(stage1, '/ssa/', level * 100, '_PI/cv_', i, '.csv', sep=""))
  }
}

