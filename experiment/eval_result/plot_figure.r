
library(ggplot2)
library(caret)
library(ggthemr)
library(gridExtra)

cur.path <- 'D:/GitHub_Repo/pyExplainer/experiment/eval_result/'
plot.output.path <- paste0(cur.path, 'figures/')

openstack.rq1.result <- read.csv(paste0(cur.path,'RQ1_openstack.csv'))
openstack.rq2.result <- read.csv(paste0(cur.path,'RQ2_openstack_global_vs_local_synt_pred.csv'))
openstack.rq3.result <- read.csv(paste0(cur.path,'RQ3_openstack.csv'))
# openstack.rq4.result <- read.csv(paste0(cur.path,'RQ4_openstack_lime_decile_20_rules_new.csv'))

openstack.rq1.result <- subset(openstack.rq1.result, select= -c(commit.id))
openstack.rq2.result <- subset(openstack.rq2.result, select= -c(commit.id))
openstack.rq3.result <- subset(openstack.rq3.result, select= -c(commit.id))
# openstack.rq4.result <- subset(openstack.rq4.result, select= -c(commit.id))

qt.rq1.result <- read.csv(paste0(cur.path,'RQ1_qt.csv'))
qt.rq2.result <- read.csv(paste0(cur.path,'RQ2_qt_global_vs_local_synt_pred.csv'))
qt.rq3.result <- read.csv(paste0(cur.path,'RQ3_qt.csv'))
# qt.rq4.result <- read.csv(paste0(cur.path,'RQ4_qt_lime_decile_20_rules_new.csv'))

qt.rq1.result <- subset(qt.rq1.result, select= -c(commit.id))
qt.rq2.result <- subset(qt.rq2.result, select= -c(commit.id))
qt.rq3.result <- subset(qt.rq3.result, select= -c(commit.id))
# qt.rq4.result <- subset(qt.rq4.result, select= -c(commit.id))

levels(openstack.rq2.result$method) <- c("pyExplainer","LIME")
levels(qt.rq2.result$method) <- c("pyExplainer","LIME")

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
  openstack.euc_dist.plot <- ggplot(data = openstack.rq1.result, aes(x=reorder(method, euc_dist_med, FUN = median), y=euc_dist_med)) +
    geom_boxplot() +
    facet_grid(~project) +
    ylab('Euclidean Distance') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  qt.euc_dist.plot <- ggplot(data = qt.rq1.result, aes(x=reorder(method, euc_dist_med, FUN = median), y=euc_dist_med)) +
    geom_boxplot() +
    facet_grid(~project) +
    ylab('Euclidean Distance') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  openstack.def.ratio.plot <- ggplot(data = openstack.rq1.result, 
                                     aes(x=reorder(method, -defective_generated_instance_ratio, 
                                                   FUN = median), 
                                         y=defective_generated_instance_ratio)) +
    geom_boxplot() +
    facet_grid(~project) +
    ylab('Defective Generated Instance Ratio') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.def.ratio.plot <- ggplot(data = qt.rq1.result, 
                              aes(x=reorder(method, -defective_generated_instance_ratio, 
                                            FUN = median), 
                                  y=defective_generated_instance_ratio)) +geom_boxplot() +
    ylab('Defective Generated Instance Ratio') + xlab('') + ggtitle('') +
    facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  euc_dist_plot = grid.arrange(qt.euc_dist.plot, openstack.euc_dist.plot, 
                               ncol=2)
  
  def.ratio.plot = grid.arrange(qt.def.ratio.plot, openstack.def.ratio.plot, 
                         ncol=2)
  
  ggsave(paste0(plot.output.path,'RQ1_euc_dist.pdf'),
         plot = euc_dist_plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  ggsave(paste0(plot.output.path,'RQ1_def_ratio.pdf'),
         plot = def.ratio.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  # openstack.plot = grid.arrange(openstack.euc_dist.plot, openstack.def.ratio.plot, 
  #                               ncol=2, bottom='openstack')
  # qt.plot = grid.arrange(qt.euc_dist.plot, qt.def.ratio.plot, 
  #                        ncol=2, bottom = 'qt')

  
  # ggsave(paste0(plot.output.path,'RQ1_openstack.pdf'),
  #        plot = openstack.plot,
  #        width = plot.height * golden.ratio,
  #        height = plot.height)
  # 
  # ggsave(paste0(plot.output.path,'RQ1_qt.pdf'),
  #        plot = qt.plot,
  #        width = plot.height * golden.ratio,
  #        height = plot.height)

  
}

rq2.plot = function()
{
  openstack.bal.acc.plot <- ggplot(data = openstack.rq2.result, aes(x=reorder(method, -balanced_accuracy, 
                                                                              FUN = median), y=balanced_accuracy)) +
    geom_boxplot() +
    ylab('Balance Accuracy') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.bal.acc.plot <- ggplot(data = qt.rq2.result, aes(x=reorder(method, -balanced_accuracy, 
                                                                FUN = median), y=balanced_accuracy)) +
    geom_boxplot() +
    ylab('Balance Accuracy') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  openstack.auc.plot <- ggplot(data = openstack.rq2.result, aes(x=reorder(method, -AUC, 
                                                                          FUN = median), y=AUC)) +
    geom_boxplot() +
    ylab('AUC') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.auc.plot <- ggplot(data = qt.rq2.result, aes(x=reorder(method, -AUC, 
                                                            FUN = median), y=AUC)) +
    geom_boxplot() +
    ylab('AUC') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  openstack.f1.plot <- ggplot(data = openstack.rq2.result, aes(x=reorder(method, -F1, 
                                                                         FUN = median), y=F1)) +
    geom_boxplot() +
    ylab('F-measure') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  qt.f1.plot <- ggplot(data = qt.rq2.result, aes(x=reorder(method, -F1, 
                                                           FUN = median), y=F1)) +
    geom_boxplot() +
    ylab('F-measure') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(0, 1))
  
  openstack.mcc.plot <- ggplot(data = openstack.rq2.result, aes(x=reorder(method, -MCC, 
                                                                          FUN = median), y=MCC)) +
    geom_boxplot() +
    ylab('MCC') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(-1, 1))
  
  qt.mcc.plot <- ggplot(data = qt.rq2.result, aes(x=reorder(method, -MCC, 
                                                            FUN = median), y=MCC)) +
    geom_boxplot() +
    ylab('MCC') + xlab('') + ggtitle('') +
    # facet_grid(~project) +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          )) + 
    coord_cartesian(ylim = c(-1, 1))
  
  openstack.plot = grid.arrange(openstack.auc.plot, 
                                openstack.f1.plot, openstack.mcc.plot, ncol=3)
  qt.plot = grid.arrange(qt.auc.plot, 
                         qt.f1.plot, qt.mcc.plot, ncol=3)
  
  # print(openstack.plot)
  # openstack.fig = grid.arrange(openstack.auc.plot, openstack.bal.acc.plot, openstack.f1.plot, openstack.bal.acc.plot, ncol=4)
  
  # print(auc.plot)
  # qt.fig = grid.arrange(qt.auc.plot, qt.bal.acc.plot, qt.f1.plot, qt.bal.acc.plot, ncol=4)
  
  
  # fig = grid.arrange(openstack.plot, qt.plot, ncol=2)
  
  ggsave(paste0(plot.output.path,'RQ2_1_openstack.pdf'),
         plot = openstack.plot,
         width = 10,
         height = plot.height)
  
  ggsave(paste0(plot.output.path,'RQ2_1_qt.pdf'),
         plot = qt.plot,
         width = 10,
         height = plot.height)
  
  # ggsave(paste0(plot.output.path,'RQ2_F1.pdf'),
  #        plot = f1.plot,
  #        width = plot.height * golden.ratio,
  #        height = plot.height)
  # 
  # ggsave(paste0(plot.output.path,'RQ2_MCC.pdf'),
  #        plot = mcc.plot,
  #        width = plot.height * golden.ratio,
  #        height = plot.height)
}

rq3.plot = function()
{
  # plot.height <- 8
  ## Generate boxplots
  # X-axis = Each studied search technique
  # Y-axis = Balanced accruacy (0 to 1)
  # Facet = Each studied project
  # openstack.match.clean.commit.plot <- ggplot(data = openstack.rq3.result, 
  #                                             aes(x=reorder(method, -number_of_clean_commits_match_guidance, 
  #                                                           FUN = median), y=number_of_clean_commits_match_guidance)) +
  #   geom_boxplot() +
  #   ylab('#Clean commits that match guidance') + xlab('') + ggtitle('') +
  #   theme(text = element_text(size = 14),
  #         strip.text.x = element_text(
  #           size = 14, face = "bold.italic"
  #         ))
  # 
  # qt.match.clean.commit.plot <- ggplot(data = qt.rq3.result, 
  #                                      aes(x=reorder(method, -number_of_clean_commits_match_guidance, 
  #                                                    FUN = median), y=number_of_clean_commits_match_guidance)) +
  #   geom_boxplot() +
  #   ylab('#Clean commits that match guidance') + xlab('') + ggtitle('') +
  #   theme(text = element_text(size = 14),
  #         strip.text.x = element_text(
  #           size = 14, face = "bold.italic"
  #         ))
  
  openstack.true.positive.rate.plot <- ggplot(data = openstack.rq3.result, 
                                              aes(x=reorder(method, -true_positive_rate, 
                                                            FUN = median), y=true_positive_rate)) +
    geom_boxplot() +
    ylab('True positive rate') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  qt.true.positive.rate.plot <- ggplot(data = qt.rq3.result, 
                                       aes(x=reorder(method, -true_positive_rate, 
                                                     FUN = median), y=true_positive_rate)) +
    geom_boxplot() +
    ylab('True positive rate') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  openstack.true.negative.rate.plot <- ggplot(data = openstack.rq3.result, aes(x=method, y=true_negative_rate)) +
    geom_boxplot() +
    ylab('True negative rate') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  qt.true.negative.rate.plot <- ggplot(data = qt.rq3.result, aes(x=method, y=true_negative_rate)) +
    geom_boxplot() +
    ylab('True negative rate') + xlab('') + ggtitle('') +
    theme(text = element_text(size = 14),
          strip.text.x = element_text(
            size = 14, face = "bold.italic"
          ))
  
  openstack.plot = grid.arrange(openstack.true.positive.rate.plot, openstack.true.negative.rate.plot, 
                                ncol=2)
  qt.plot = grid.arrange(qt.true.positive.rate.plot, qt.true.negative.rate.plot, 
                         ncol=2)
  
  # print(fig)
  # export a plot as pdf
  
  
  ggsave(paste0(plot.output.path,'RQ3_openstack.pdf'),
         plot = openstack.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
  ggsave(paste0(plot.output.path,'RQ3_qt.pdf'),
         plot = qt.plot,
         width = plot.height * golden.ratio,
         height = plot.height)
  
}

rq1.plot()
# rq2.plot()
# rq3.plot()

