#' Create K random folds of the data and add the fold assignment as a column to the data.table 
#' ToDo: Stratified Sampling. 
add_kfold_col <- function(dt,k, col_name = 'kfolds'){
  dt[, paste0(col_name) := sample(1:k, size = nrow(dt), replace=T, prob = rep(1/k, k))]
  return(dt)
}

#' Train the finalized GLM model on a subset of the training data and return OOF predictions. 
model_oof_predictions <- function(dt_model_train,dt_model_test, base_x_vars_model, model_type){
  
  print(paste0("Model Type is : ", model_type))
  
  if(model_type == 'glm'){
    fitted_model = h2o.glm(x = base_x_vars_model,
                         y = 'pure_premium',
                         weights_column = 'earned_exposure',
                         family = 'tweedie',
                         tweedie_link_power = 1,
                         tweedie_variance_power = 1.5,
                         remove_collinear_columns = FALSE,
                         lambda_search = FALSE,
                          alpha = 1,
                          lambda = 0.00001,
                         training_frame = as.h2o(dt_model_train[, c('pure_premium', base_x_vars_model, 'earned_exposure'), with = FALSE]))
    
    
  } else if(model_type == 'xgb'){
    
    fitted_model = h2o.gbm(x = base_x_vars_model,
                           y = 'pure_premium',
                           weights_column = 'earned_exposure',
                           distribution = 'tweedie',
                           stopping_rounds = 100,
                           max_depth = 4, 
                           sample_rate = 0.95, 
                           col_sample_rate_per_tree = 0.5,
                           tweedie_power = 1.5,
                           ntrees = 400,
                           min_split_improvement = 0.000001,
                           stopping_metric = 'deviance',
                           stopping_tolerance = 0.0001,
                           score_tree_interval = 1,
                           learn_rate = 0.1,
                           training_frame = as.h2o(dt_model_train[, c('pure_premium', base_x_vars_model, 'earned_exposure'), with = FALSE]))
    
  }
  
  dt_model_test[, paste0('oof_pred_',model_type) := as.data.frame(h2o.predict(fitted_model, as.h2o(dt_model_test)))$predict]
  
  # Add reblancing logic here 
  #rebalancing_factor = dt_model_test[,sum(pure_premium)/sum(get(paste0('oof_pred_',model_type)))]
  #dt_model_test[, paste0('oof_pred_',model_type) := get(paste0('oof_pred_',model_type))*rebalancing_factor]

  return(dt_model_test)
}


generate_oof_predictions <- function(dt_glm,
                                     dt_xgb,
                                     base_x_vars_glm,
                                     base_x_vars_xgb,
                                     k = 10,
                                     kfolds_col_name = 'kfolds'){
  
  # Create DT to collect out of fold predictions for all base (level 1) models
  oof_pred_glm = data.table()
  oof_pred_xgb = data.table()
  
  for(i in 1:k){
    
    print(paste0('Fold', i))
    
    # Create temp DTs for all base models. Train comprised of K - 1 folds, while Test is the ith fold. 
    dt_glm_train = dt_glm[get(kfolds_col_name) != i,]
    dt_glm_test = dt_glm[get(kfolds_col_name) == i,]
    dt_xgb_train = dt_xgb[get(kfolds_col_name) != i,]
    dt_xgb_test = dt_xgb[get(kfolds_col_name) == i,]
    
    # Train base models on K-1 folds of the data and get predictions for the ith Fold. Concatenate predictions for all folds
    oof_temp_pred_glm = model_oof_predictions(dt_glm_train,dt_glm_test, base_x_vars_glm, 'glm')
    oof_pred_glm = rbind(oof_pred_glm,oof_temp_pred_glm)
    
    oof_temp_pred_xgb = model_oof_predictions(dt_glm_train,dt_glm_test, base_x_vars_xgb, 'xgb')
    oof_pred_xgb = rbind(oof_pred_xgb, oof_temp_pred_xgb)
    
  }
  
  # Combine predictions from all base models into single DT. 
  meta_learner_dt = dt_xgb[order(IDs),c('IDs', 'pure_premium','earned_exposure')]
  list_dt = list(meta_learner_dt,oof_pred_glm[order(IDs),c('oof_pred_glm','IDs')],oof_pred_xgb[order(IDs),c('oof_pred_xgb','IDs')] )
  meta_learner_dt = Reduce(function(...) merge(...,by = c('IDs'), all = TRUE), list_dt)
  
  return(meta_learner_dt)
  
}