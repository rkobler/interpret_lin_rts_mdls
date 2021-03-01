# load libraries and define parameters ------------------------------------

rm(list = ls())

source("./config.r")

n_cv = 10
file_tag = 'patternnoise'

data_root_dir = paste(root_dir, file_tag, '/', sep='')

# load the dataframe ------------------------------------------------------

df <- read.csv(paste(data_root_dir,  "scores.csv", sep = ""), header = TRUE)
df$method <- factor(df$method, levels = c('dummy', 'diag', 'spoc', 'riemann'))

df <- df %>%
  group_by(noise_A) %>%
  mutate(target_score_sd = target_score_sd/target_score_mu[method == 'dummy']) %>%
  mutate(target_score_mu = target_score_mu/target_score_mu[method == 'dummy']) %>%
  ungroup()

df<- df %>%
  mutate(target_score_sem = target_score_sd/sqrt(n_cv)) %>%
  mutate(pattern_score_sem = pattern_score_sd/sqrt(n_cv))


df<- df %>%
  filter(!method %in% 'dummy') %>% droplevels


df_title = 'regression score'
df_subtitle <- paste("f = : ", unique(df$nonlinearity), ", sigma: ", unique(df$sigma), ", n_sources: ", unique(df$n_sources), ", n_components: ", unique(df$n_compo))


myplot <- df%>%
  ggplot(aes(x=noise_A,y=target_score_mu, group=method, color=method)) +
  labs(title = df_title,subtitle = df_subtitle) +
  geom_errorbar(aes(ymin=target_score_mu-target_score_sem, ymax=target_score_mu+target_score_sem), width=.05) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 2, shape = 21)+
  theme_classic(base_size = 7) +
  theme(panel.grid.major.y = element_line(colour="lightgray", size=0.5)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  labs(x = TeX("$\\mu$"), y = "normalized mae")+
  ylim(-0.01, 1.1)+
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL) +
  geom_hline(yintercept = 1., color = "black", linetype = "dotted",
             size = 1)  
myplot
ggsave(filename=paste(data_root_dir, file_tag, "_target_score", fig_type, sep = ""), plot=myplot, width=6, height=5, units="cm")


df_title = 'pattern score'

myplot <- df%>%
  ggplot(aes(x=noise_A,y=pattern_score_mu, group=method, color=method)) +
  labs(title = df_title,subtitle = df_subtitle) +
  geom_errorbar(aes(ymin=pattern_score_mu-pattern_score_sem, ymax=pattern_score_mu+pattern_score_sem), width=.05) +
  geom_line(size = 1.5, alpha = 0.8) +
  geom_point(fill = "white", size = 2, shape = 21)+
  theme_classic(base_size = 7) +
  theme(panel.grid.major.y = element_line(colour="lightgray", size=0.5)) +
  scale_x_log10(breaks =  10^(-10:10),
                minor_breaks = rep(1:9, 21) * (10 ^ rep(-10:10, each=9))) +
  labs(y = TeX("$1 - |a^T \\hat{a}|$"), x = TeX("$\\mu$")) +
  ylim(-0.01, 0.9)+
  scale_color_manual(values = color_cats, name = NULL) +
  scale_fill_manual(values = color_cats, name = NULL)  +
  geom_hline(yintercept = 1., color = "black", linetype = "dotted",
             size = 1)  
myplot
ggsave(filename=paste(data_root_dir, file_tag, "_pattern_score", fig_type, sep = ""), plot=myplot, width=6, height=5, units="cm")
