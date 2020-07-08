### Overview
#########################################
#' 0) Rework this writeup. 
#' 
#' 1) We create a meta learner atop the base models (GLM UW220, XGBoost UW320)
#' 
#' 2) We use out-of-fold predictions obtained by training the final models on CV splits
#' of the data, and predicting on the holdout splits. 
#'
#' 3) The holdout samples, along with the predictions, across CV folds are then union'd into a combined dataset 
#' that will be used to train the meta learner. 
#' 
#' 4) Original model objects are imported and used to predict on test. 
#' 
#' 5) Test predictions are fed to the level 2 model. 
######################################################
#'
#'  Open questions: 
#'  
#'  1) Rebalance within the fold or after concatanating all oof predictions? If we do out of fold predictions, the rebalancing factor 
#'  Will be a linear combination of rebalancing factors from individual fold predictions. 
#'  
#'  2) Linear model without intercept for the meta learner. Can coeffecients of linear predictor sum > 1? Do we rescale?
#'  
#'  3) Adding GBM to pipeline. 
#'  
#'  4) ToDo: Stratified Sampling for the KFolds 
#'  
##################################################### 


### Configuration
######################################################################################################
# Load Configured File Paths, Settings, Src Modules
source('configuration/configuration.R')
source('src/ensemble_functions.R')
set.seed(config_random_seed)

h2o.init()
### Read Training and Test Sets
######################################################################################################
train_cov_dt = fread(paste0(config_fp_raw_data_dir, config_fp_raw_train_file))
test_cov_dt = fread(paste0(config_fp_raw_data_dir, config_fp_raw_test_file))

### Create ID column
train_cov_dt[,IDs := 1:nrow(train_cov_dt)]
test_cov_dt[,IDs := (nrow(train_cov_dt) + 1):(nrow(train_cov_dt) + nrow(test_cov_dt))]

# Column Definitions. 
# Using Nick's recommended vars for the GLM, and all original data points + cont7_cubed for the Xgboost. 
id_vars = 'id'
categ_vars_all = colnames(train_cov_dt)[colnames(train_cov_dt) %like% 'cat']
categ_vars = c("cat57",
               "cat101",
               "cat80",
               "cat79",
               "cat103",
               "cat100",
               "cat111",
               "cat112",
               "cat81",
               "cat114",
               "cat12",
               "cat87",
               "cat108",
               "cat113",
               "cat72",
               "cat53",
               "cat44",
               "cat1",
               "cat82",
               "cat26",
               "cat91",
               "cat50")

cont_vars = colnames(train_cov_dt)[colnames(train_cov_dt) %like% 'cont']

# Dummy Earned Exposures & Pure Premium
train_cov_dt[, earned_exposure := 1]
test_cov_dt[, earned_exposure := 1]

train_cov_dt[, pure_premium := loss / earned_exposure]


# Fold assignment for Train 
#train_dt[,loss_flag := ifelse(pure_premium > 0, 1,0)]
train_cov_dt = add_kfold_col(train_cov_dt, 15)


### Run processing pipelines for XGB & GLM and define covaritates for both 
############################################################################
# We will create two functions out of the data transform pipeline for CSL 220 & 320 that take in 
# a data table object and return another DT object with all the imputations/transformations applied. 

cubed_cols = create_exponential_terms(dtable = train_cov_dt, x_cols = 'cont7', power = 3, suffix = '_cubed')
cubed_cols = create_exponential_terms(dtable = test_cov_dt, x_cols = 'cont7', power = 3, suffix = '_cubed')

cont_vars = c(cont_vars,cubed_cols )
base_x_vars_xgb = c(categ_vars_all, cont_vars)
base_x_vars_glm = c(categ_vars, c('cont2', cubed_cols))


# Create DTs for XGB (Note: GBM is used in place of XGBoost due to limitations of local system)
train_cov_dt_xgb = copy(train_cov_dt)
test_cov_dt_xgb = copy(test_cov_dt)
dtable_define_variable_classes(dtable = train_cov_dt_xgb, categ_vars = categ_vars_all, cont_vars = cont_vars)
dtable_define_variable_classes(dtable = test_cov_dt_xgb, categ_vars = categ_vars_all, cont_vars = cont_vars)


# Create DTs for GLM 
train_cov_dt_glm = copy(train_cov_dt)
test_cov_dt_glm = copy(test_cov_dt)
dtable_define_variable_classes(dtable = train_cov_dt_glm, categ_vars = categ_vars, cont_vars = cont_vars)
dtable_define_variable_classes(dtable = test_cov_dt_glm, categ_vars = categ_vars, cont_vars = cont_vars)

### Create a set of out of fold predictions from all base models
############################################################################################ 

meta_learner_dt = generate_oof_predictions(dt_glm = train_cov_dt_glm,
                                           dt_xgb = train_cov_dt_xgb,
                                           base_x_vars_glm = base_x_vars_glm,
                                           base_x_vars_xgb = base_x_vars_xgb,
                                           k = 15, # Try with 20
                                           kfolds_col_name = 'kfolds')



### Train the Meta Learner on OOF predictions 
###########################################################################
meta_x_vars = c('oof_pred_glm','oof_pred_xgb')

fitted_meta_model = h2o.glm(x = meta_x_vars,
                       y = 'pure_premium',
                       weights_column = 'earned_exposure',
                       family = 'gaussian',
                       remove_collinear_columns = FALSE,
                       standardize = T, #Helps slightly. 
                       lambda_search = F, 
                       alpha = 0,
                       lambda = 0, # Turn off regularization. 
                       intercept = F,
                       training_frame = as.h2o(meta_learner_dt[, c('pure_premium', meta_x_vars, 'earned_exposure'), with = FALSE])
                       )


# Load Saved XGB & GLM model objects and predict on test 
######################################################

# train_grid = h2o.grid(algorithm = 'glm',
#                       x = base_x_vars_glm,
#                       y = 'pure_premium',
#                       weights_column = 'earned_exposure',
#                       family = 'tweedie',
#                       tweedie_link_power = 1,
#                       tweedie_variance_power = 1.5,
#                       remove_collinear_columns = FALSE,
#                       lambda_search = FALSE,
#                       hyper_params = config_hparam_alpha_lambda_list,
#                       nfolds = 10,
#                       training_frame = as.h2o(train_cov_dt_glm[, c('pure_premium', base_x_vars_glm, 'earned_exposure'), with = FALSE]))



fitted_model_glm = h2o.glm(x = base_x_vars_glm,
                       y = 'pure_premium',
                       weights_column = 'earned_exposure',
                       family = 'tweedie',
                       tweedie_link_power = 1,
                       tweedie_variance_power = 1.5,
                       remove_collinear_columns = FALSE,
                       lambda_search = FALSE,
                       alpha = 1,
                       lambda = 0.00001,
                       training_frame = as.h2o(train_cov_dt_glm[, c('pure_premium', base_x_vars_glm, 'earned_exposure'), with = FALSE]))

test_cov_dt_glm[, loss := as.data.frame(h2o.predict(fitted_model_glm, as.h2o(test_cov_dt_glm)))$predict]
write.csv(test_cov_dt_glm[,c('id', 'loss'), with=F], 'kaggle_submit_glm.csv', row.names = FALSE)


fitted_model_xgb = h2o.gbm(x = base_x_vars_xgb,
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
                           training_frame = as.h2o(train_cov_dt_xgb[, c('pure_premium', base_x_vars_xgb, 'earned_exposure'), with = FALSE]))


test_cov_dt_xgb[, loss := as.data.frame(h2o.predict(fitted_model_xgb, as.h2o(test_cov_dt_glm)))$predict]
write.csv(test_cov_dt_xgb[,c('id', 'loss'), with=F], 'kaggle_submit_xgb.csv', row.names = FALSE)


# Combine Test predictions 
################################################ 
test_meta_learner_dt = test_cov_dt[order(IDs),c('id','earned_exposure')]
meta1 = test_cov_dt_glm[,c('id', 'loss'), with=F]
setnames(meta1, "loss", 'oof_pred_glm')
meta2 = test_cov_dt_xgb[,c('id', 'loss'), with=F]
setnames(meta2, "loss", 'oof_pred_xgb')

list_dt = list(test_meta_learner_dt,meta1,meta2)
test_meta_learner_dt = Reduce(function(...) merge(...,by = c('id'), all = TRUE), list_dt)

# test_meta_learner_dt[, loss := as.data.frame(h2o.predict(fitted_meta_model, as.h2o(test_meta_learner_dt)))$predict]
# write.csv(test_meta_learner_dt[,c('id', 'loss'), with=F], 'kaggle_submit_meta_learner.csv', row.names = FALSE)

glm_weight = fitted_meta_model@model$coefficients['oof_pred_glm']
xgb_weight = fitted_meta_model@model$coefficients['oof_pred_xgb']
test_meta_learner_dt[, loss := glm_weight*oof_pred_glm + xgb_weight*oof_pred_xgb]
write.csv(test_meta_learner_dt[,c('id', 'loss'), with=F], 'kaggle_submit_meta_learner.csv', row.names = FALSE)

