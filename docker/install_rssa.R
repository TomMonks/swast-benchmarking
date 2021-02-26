## Analysis is conducted using Rssa version 1.0.2. and R 3.6.1
## This is not available via conda or conda-forge.  Need to install
## via cran.
## r-base, r-forecast and r-svd dependencies managed via conda 


######## cran dependency management ############################################

#rssaURL = 'https://cran.r-project.org/src/contrib/Archive/Rssa/Rssa_1.0.2.tar.gz'

#load Rssa if installed and return result (TRUE = loaded, FALSE not loaded.)
##install version 1.0.2 of Rssa
install.packages("svd", repos='https://www.stats.bris.ac.uk/R/')
install.packages('Rssa', repos='https://www.stats.bris.ac.uk/R/')

##################################################################################
