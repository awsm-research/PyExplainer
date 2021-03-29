library(ggplot2)
library(caret)
library(ggthemr)

cur.path <- 'D:/GitHub_Repo/pyExplainer/experiment/eval_result/'
plot.output.path <- paste0(cur.path, 'figures/')

openstack.result <- read.csv(paste0(cur.path,'openstack_RQ1_RQ4_eval.csv'))
openstack.result <- subset(openstack.result, select= -c(commit.id))
# df <- subset(df, select = -c(a, c))
# print(openstack.result)

qt.result <- read.csv(paste0(cur.path,'qt_RQ1_RQ4_eval.csv'))
qt.result <- subset(qt.result, select= -c(commit.id))

all.result <- rbind(openstack.result, qt.result)

# # Set theme
ggthemr('fresh')

euclidean.dist.plot = function()
{
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  g <- ggplot(data = all.result, aes(x=method, y=euc_dist_med)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Euclidean Distance') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  # export a plot as pdf
  golden.ratio <- 1.618
  plot.height <- 3.5
  ggsave(paste0(plot.output.path,'RQ1_euclidean_distance.pdf'),
         plot = g,
         width = plot.height * golden.ratio,
         height = plot.height)
}

defective.generated.instance.ratio.plot = function()
{
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  g <- ggplot(data = all.result, aes(x=method, y=defective_generated_instance_ratio)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Defective Generated Instance Ratio') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  # export a plot as pdf
  golden.ratio <- 1.618
  plot.height <- 3.5
  ggsave(paste0(plot.output.path,'RQ1_defective_generated_instance_ratio.pdf'),
         plot = g,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  print(g)
}

balance.accuracy.plot = function()
{
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  g <- ggplot(data = all.result, aes(x=method, y=bal_acc)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Balance Accuracy') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  # export a plot as pdf
  golden.ratio <- 1.618
  plot.height <- 3.5
  ggsave(paste0(plot.output.path,'RQ2_balance_accuracy.pdf'),
         plot = g,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  print(g)
}

r2_score.plot = function()
{
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  g <- ggplot(data = all.result, aes(x=method, y=r2_score)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('R2 score') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(-1, 1))
  # export a plot as pdf
  golden.ratio <- 1.618
  plot.height <- 3.5
  ggsave(paste0(plot.output.path,'RQ3_R2_score.pdf'),
         plot = g,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  print(g)
}

local.feature.ratio.plot = function()
{
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  g <- ggplot(data = all.result, aes(x=method, y=local_feature_ratio)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('% Overlapping Features') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  # export a plot as pdf
  golden.ratio <- 1.618
  plot.height <- 3.5
  ggsave(paste0(plot.output.path,'RQ4_overlapping_top_k_global_features.pdf'),
         plot = g,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  print(g)
}

rq4.old.plot = function()
{
  project = c('openstack','openstack','qt','qt')
  method = c('crossover_interpolation','LIME','crossover_interpolation','LIME')
  percent_overlap = c(0.8484,0.9797,0.7341,0.5949)
  rq4_old = data.frame(project, method, percent_overlap)
  
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  g <- ggplot(data = rq4_old, aes(x=method, y=percent_overlap)) +
    geom_bar(stat="identity") +
    ylab('') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  # export a plot as pdf
  golden.ratio <- 1.618
  plot.height <- 3.5
  ggsave(paste0(plot.output.path,'RQ4_old.pdf'),
         plot = g,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  print(g)
}

euclidean.dist.plot() # RQ1
defective.generated.instance.ratio.plot() # RQ1
balance.accuracy.plot() # RQ2
r2_score.plot() # RQ3
local.feature.ratio.plot() # RQ4

rq4.old.plot()