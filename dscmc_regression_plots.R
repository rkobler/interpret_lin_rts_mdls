# load libraries and define parameters ------------------------------------

rm(list = ls())

source("./config.r")

root_dir = './outputs/'
file_tag = 'ds-cmc'

data_root_dir = paste(root_dir, file_tag, '/', sep='')

# load the dataframe ------------------------------------------------------

df <- read.csv(paste(data_root_dir,  "scores.csv", sep = ""), header = TRUE)
df$method <- factor(df$method, levels = c('diag', 'spoc', 'riemann'))

# select the parameters

df_title = 'regression score'
df_subtitle <- 'CMC dataset'

myplot <- df%>%
  group_by(method, components) %>%
  summarise(r2_mu = mean(r2), r2_sem = sd(r2)/sqrt(length(r2))) %>%
  ungroup() %>%
  ggplot(aes(x=components,y=r2_mu, color = method)) +
  geom_line(size = 1.5) +
  geom_ribbon(aes(ymin=r2_mu+r2_sem, ymax=r2_mu-r2_sem, fill = method, 
                  color = NULL), alpha=0.2) +
  theme_classic(base_size = 10) +
  theme(panel.grid.major.y = element_line(colour="lightgray", size=0.5)) +
  labs(title = df_title,subtitle = df_subtitle) +
  labs(y = TeX("R^2")) +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  geom_hline(yintercept = 0.0, color = "black", linetype = "dotted", size = 1) +
  ylim(-0.4, 0.75)
myplot
ggsave(filename=paste(data_root_dir, "r2_components.pdf", sep = ""), 
       plot=myplot, width=12, height=8, units="cm")


# statistical comparisons---------------------------------------------

maxcomps <- df %>% 
  group_by(components, method) %>%
  summarise(r2 = mean(r2)) %>% 
  group_by(method) %>% 
  filter(r2 == max(r2)) %>%
  filter(components == max(components)) %>%
  select(-r2) %>%
  ungroup() %>%
  spread(method, components)

dfcomp <- df%>%
  filter((method == "riemann" & components == maxcomps$riemann) | 
           (method == "spoc" & components == maxcomps$spoc) |
           (method == "diag" & components == maxcomps$diag))


dftests <- list()

combs <- combn(unique(dfcomp$method),2)

for (i in 1:dim(combs)[2]){
  
  tst <- dfcomp %>% 
    filter(method %in% combs[,i]) %>%
    t.test(r2 ~ method, paired = TRUE, data = .)  
  
  t <- tibble(method1 = combs[1,i], method2 = combs[2,i], 
              pval = tst$p.value, tval = tst$statistic)
  
  dftests[[length(dftests)+1]] <- t  
  
}

dftests <- bind_rows(dftests)

myplot <- dfcomp %>%
  mutate(title = factor(method, levels = method, 
                        label = paste(method, components))) %>%
  ggplot(aes(x=title,y=r2, fill=title)) +
  geom_boxplot(outlier.alpha = 0) +
  labs(title = df_title,subtitle = df_subtitle) +
  geom_point(pch = 21, position = position_jitterdodge(jitter.width = 0.25), 
             alpha = 0.4, size = 1) +
  theme_classic(base_size = 7) +
  theme(panel.grid.major.y = element_line(colour="lightgray", size=0.5)) +
  labs(y = TeX("R^2"), x= "") +
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  geom_hline(yintercept = 0.0, color = "black", linetype = "dotted", size = 1) +
  ylim(-0.5, 1.) +
  dftests %>%
  mutate(y_pos = 0.8 + 0.1 * (method1 == "diag" & method2 == "riemann")) %>%
  mutate(x_pos1 = 1 + 1 * (method1 == "spoc") + 2 * (method1 == "riemann")) %>%
  mutate(x_pos2 = 1 + 1 * (method2 == "spoc") + 2 * (method2 == "riemann")) %>%  
  geom_signif(data = ., 
              aes(xmin=x_pos1, xmax=x_pos2, 
                  annotations=formatC(pval, digits = 2), 
                  y_position = y_pos, fill = NULL), textsize = 2.5, 
              vjust = -0.1, manual = TRUE)

myplot

ggsave(filename=paste(data_root_dir, "comparisons.pdf", sep = ""), 
       plot=myplot, width=8, height=8, units="cm")
write.csv(dftests, paste(data_root_dir, "comparisons.csv", sep = ""))

