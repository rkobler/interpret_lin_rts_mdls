
library(tidyverse)
library(ez)
library(RColorBrewer)
library(ggplot2)
library(latex2exp)
library(ggsignif)

root_dir = './outputs/simuls/'

fig_type = '.pdf'

color_cats <- c(
  "#56B4E9",# sky blue
  "#009D79",# blueish green
  "#E36C2F",  #vermillon
  "#EEA535",  # orange
  "#F0E442", #yellow
  "#0072B2", #blue
  "#CC79A7", #violet
  "#242424" # black
)