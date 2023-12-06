library("ggpubr")

library("readxl")

table = read_excel("C:/Users/jxe8989/My Drive/Research/diss/tests_wiiiii/test02/EvalMetrics_test02.xlsx")

table$SNR = factor(table$SNR, 
                   levels = c( "Inf", "35", "30", "25", "20", "15", "10", "5" ) 
)

table_regions = table[ table$Solver == "RegionPrior", ]

p <- ggviolin(table_regions, 
              x = "SNR", y = "LocalizationError",
              add = "mean_sd")
p



p <- ggboxplot(table, 
               x = "SNR", y = "LocalizationError",
               fill = "Solver",
)+ grids(linetype = "solid")

p

##
ggboxplot(table, 
          x = "SNR", y = "LocalizationError",
          fill = "Solver",
)+ grids(linetype = "solid")

##
ggboxplot(table, 
          x = "SNR", y = "HalfMax",
          fill = "Solver",
)+ grids(linetype = "solid")

##
ggboxplot(table, 
          x = "SNR", y = "Algorithm Time",
          fill = "Solver",
)+ grids(linetype = "solid") +
  scale_y_continuous(trans='log10')


##
ggboxplot(table, 
          x = "SNR", y = "Parameter Tuning Time",
          fill = "Solver",
)+ grids(linetype = "solid") +
  scale_y_continuous(trans='log10')
