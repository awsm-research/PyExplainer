library(ggplot2)
library(caret)
library(ggthemr)
library(gridExtra)

cur.path <- 'D:/GitHub_Repo/pyExplainer/experiment/eval_result/'
plot.output.path <- paste0(cur.path, 'figures/')

openstack.rq1.result <- read.csv(paste0(cur.path,'RQ1_openstack.csv'))
openstack.rq2.result <- read.csv(paste0(cur.path,'RQ2_openstack.csv'))
openstack.rq3.result <- read.csv(paste0(cur.path,'RQ3_openstack.csv'))
openstack.rq4.result <- read.csv(paste0(cur.path,'RQ4_openstack.csv'))

openstack.rq1.result <- subset(openstack.rq1.result, select= -c(commit.id))
openstack.rq2.result <- subset(openstack.rq2.result, select= -c(commit.id))
openstack.rq3.result <- subset(openstack.rq3.result, select= -c(commit.id))
openstack.rq4.result <- subset(openstack.rq4.result, select= -c(commit.id))

qt.rq1.result <- read.csv(paste0(cur.path,'RQ1_qt.csv'))
qt.rq2.result <- read.csv(paste0(cur.path,'RQ2_qt.csv'))
qt.rq3.result <- read.csv(paste0(cur.path,'RQ3_qt.csv'))
qt.rq4.result <- read.csv(paste0(cur.path,'RQ4_qt.csv'))

qt.rq1.result <- subset(qt.rq1.result, select= -c(commit.id))
qt.rq2.result <- subset(qt.rq2.result, select= -c(commit.id))
qt.rq3.result <- subset(qt.rq3.result, select= -c(commit.id))
qt.rq4.result <- subset(qt.rq4.result, select= -c(commit.id))

golden.ratio <- 1.618
plot.height <- 3.5

# # Set theme
ggthemr('fresh')

rq1.plot = function()
{
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  openstack.plot <- ggplot(data = openstack.rq1.result, aes(x=method, y=euc_dist_med)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Euclidean Distance') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  qt.plot <- ggplot(data = qt.rq1.result, aes(x=method, y=euc_dist_med)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Euclidean Distance') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))

  fig = grid.arrange(openstack.plot, qt.plot, ncol=2)
  
  # print(fig)
  # export a plot as pdf

  
  ggsave(paste0(plot.output.path,'RQ1_euclidean_distance.pdf'),
         plot = fig,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  openstack.plot <- ggplot(data = openstack.rq1.result, aes(x=method, y=defective_generated_instance_ratio)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Defective Generated Instance Ratio') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.plot <- ggplot(data = qt.rq1.result, aes(x=method, y=defective_generated_instance_ratio)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Defective Generated Instance Ratio') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  fig = grid.arrange(openstack.plot, qt.plot, ncol=2)
  
  ggsave(paste0(plot.output.path,'RQ1_defective_generated_instance_ratio.pdf'),
         plot = fig,
         width = plot.height * golden.ratio,
         height = plot.height)
}

rq2.plot = function()
{
  openstack.bal.acc.plot <- ggplot(data = openstack.rq2.result, aes(x=method, y=balanced_accuracy)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Balance Accuracy') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.bal.acc.plot <- ggplot(data = qt.rq2.result, aes(x=method, y=balanced_accuracy)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('Balance Accuracy') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  openstack.auc.plot <- ggplot(data = openstack.rq2.result, aes(x=method, y=AUC)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('AUC') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.auc.plot <- ggplot(data = qt.rq2.result, aes(x=method, y=AUC)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('AUC') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  openstack.f1.plot <- ggplot(data = openstack.rq2.result, aes(x=method, y=F1)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('F-measure') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.f1.plot <- ggplot(data = qt.rq2.result, aes(x=method, y=F1)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('F-measure') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  openstack.mcc.plot <- ggplot(data = openstack.rq2.result, aes(x=method, y=MCC)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('MCC') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(-1, 1))
  
  qt.mcc.plot <- ggplot(data = qt.rq2.result, aes(x=method, y=MCC)) +
    geom_boxplot(outlier.shape = NA) +
    ylab('MCC') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(-1, 1))
  
  auc.plot = grid.arrange(openstack.auc.plot, qt.auc.plot, ncol=2)
  bal.acc.plot = grid.arrange(openstack.bal.acc.plot, qt.bal.acc.plot, ncol=2)
  f1.plot = grid.arrange(openstack.f1.plot, qt.f1.plot, ncol=2)
  mcc.plot = grid.arrange(openstack.mcc.plot, qt.mcc.plot, ncol=2)
  
  # openstack.fig = grid.arrange(openstack.auc.plot, openstack.bal.acc.plot, openstack.f1.plot, openstack.bal.acc.plot, ncol=4)
  
  print(auc.plot)
  # qt.fig = grid.arrange(qt.auc.plot, qt.bal.acc.plot, qt.f1.plot, qt.bal.acc.plot, ncol=4)
  
  
  # fig = grid.arrange(openstack.plot, qt.plot, ncol=2)
  
  ggsave(paste0(plot.output.path,'RQ2_AUC.pdf'),
         plot = auc.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  ggsave(paste0(plot.output.path,'RQ2_balance_acc.pdf'),
         plot = bal.acc.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  ggsave(paste0(plot.output.path,'RQ2_F1.pdf'),
         plot = f1.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  ggsave(paste0(plot.output.path,'RQ2_MCC.pdf'),
         plot = mcc.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
}

rq2.plot()

