library(effsize)
library(reshape2)
library(dplyr)

cur.path <- 'D:/GitHub_Repo/pyExplainer/experiment/eval_result/'



rq3.df = read.csv(paste0(cur.path,'RQ3.csv'))

select.data = function(dataframe, global.model.name, project.name, method.name, metric)
{
  result = subset(dataframe, global_model==global.model.name &
                    project==project.name & 
                    method==method.name,
                  select=c(metric))
  
  return(result)
}

rq1.stat.eval = function()
{
  rq1.df = read.csv(paste0(cur.path,'RQ1.csv'))
  
  for(global.model in c('RF','LR'))
  {
    print(paste0('Global model: ',global.model))
    
    for(proj in c('openstack','qt'))
    {
      print(paste0('Project: ',proj))
      pyexp.result = select.data(rq1.df, global.model,proj,'pyExplainer','euc_dist_med')
      lime.result = select.data(rq1.df, global.model,proj,'LIME','euc_dist_med')

      # wilcox = wilcox.test(c(pyexp.result$euc_dist_med),c(lime.result$euc_dist_med))
      cliff.d = cliff.delta(c(pyexp.result$euc_dist_med),c(lime.result$euc_dist_med))
      # print(wilcox)
      print(cliff.d)

      # break
    }
    # break
  }
}

rq2.stat.eval = function()
{
  rq2.df = read.csv(paste0(cur.path,'RQ2_prediction.csv'))
  
  for(global.model in c('RF','LR'))
  {
    print(paste0('Global model: ',global.model))
    
    for(proj in c('openstack','qt'))
    {
      print(paste0('Project: ',proj))
      print('')
      print('AUC')
      pyexp.result = select.data(rq2.df, global.model,proj,'pyExplainer','AUC')
      lime.result = select.data(rq2.df, global.model,proj,'LIME','AUC')
      
      # wilcox = wilcox.test(c(pyexp.result$AUC),c(lime.result$AUC))
      cliff.d = cliff.delta(c(pyexp.result$AUC),c(lime.result$AUC))
      # print(wilcox)
      print(cliff.d)
      
      print('')
      print('F1')
      pyexp.result = select.data(rq2.df, global.model,proj,'pyExplainer','F1')
      lime.result = select.data(rq2.df, global.model,proj,'LIME','F1')
      
      # wilcox = wilcox.test(c(pyexp.result$F1),c(lime.result$F1))
      cliff.d = cliff.delta(c(pyexp.result$F1),c(lime.result$F1))
      # print(wilcox)
      print(cliff.d)
      
      # break
    }
    # break
  }
}

rq3.stat.eval = function()
{
  rq3.df = read.csv(paste0(cur.path,'RQ3.csv'))
  # print(names(rq3.df))
  for(global.model in c('RF','LR'))
  {
    print(paste0('Global model: ',global.model))

    for(proj in c('openstack','qt'))
    {
      print(paste0('Project: ',proj))
      print('')
      print('true_positive_rate')
      pyexp.result = select.data(rq3.df, global.model,proj,'pyExplainer','true_positive_rate')
      lime.result = select.data(rq3.df, global.model,proj,'LIME','true_positive_rate')

      # print(head(pyexp.result))
      # wilcox = wilcox.test(c(pyexp.result$true_positive_rate),c(lime.result$true_positive_rate))
      cliff.d = cliff.delta(c(pyexp.result$true_positive_rate),c(lime.result$true_positive_rate))
      # print(wilcox)
      print(cliff.d)

      print('')
      print('true_negative_rate')
      pyexp.result = select.data(rq3.df, global.model,proj,'pyExplainer','true_negative_rate')
      lime.result = select.data(rq3.df, global.model,proj,'LIME','true_negative_rate')

      # wilcox = wilcox.test(c(pyexp.result$true_negative_rate),c(lime.result$true_negative_rate))
      cliff.d = cliff.delta(c(pyexp.result$true_negative_rate),c(lime.result$true_negative_rate))
      # print(wilcox)
      print(cliff.d)

      # break
    }
    # break
  }
}


# rq1.stat.eval()
# rq2.stat.eval()
rq3.stat.eval()