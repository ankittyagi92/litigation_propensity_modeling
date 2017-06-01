#setup environment and libararies
library(caret)
library(gbm)
library(ROCR)

getwd()
setwd('C:/Users/ankit.tyagi/Documents/CART/')

#read data
data=read.csv('Litigation_study_Model_data.csv')

#subset variables in the data, based on business relevance
data_sub=data[c('PT_NO_PAYMENTS_15',
                'PT_PAYMENT_AMT_15',
                'PT_EXPENSE_AMT_15',
                'PT_PAYMENT_AMT_wo_R',
                'PT_EXPENSE_AMT_wo_R',
                'PT_NO_PAYMENTS_15_wo_R',
                'claimant_zip_5',
                'Median_hh_inc_1999',
                'no_reserve_15',
                'no_claimant_15',
                'no_res_type_15',
                'reserve_BI_f',
                'reserve_UMBI_f',
                'reserve_UM_f',
                'damage_minor_f',
                'damage_moderate_f',
                'damage_rolled_f',
                'damage_severe_f',
                'veh_num_0_f',
                'veh_num_1_f',
                'veh_num_2_f',
                'Driveable_n_f',
                'Male_f',
                'FeMale_f',
                'Claim_role_m',
                'claim_adverse_f',
                'claim_Insured_f',
                'ACCIDENT_ROLE_m',
                'ac_role_drive_f',
                'ac_role_pass_f',
                'ac_role_other_f',
                'INJURY_Strain_f',
                'INJURY_Contusion_f',
                'INJURY_Sprain_f',
                'Policy_Loss',
                'Policy_Claim',
                'Dif_Open_Claim_rt',
                'Days_loss_report',
                'year_miss_f',
                'Claimant_age',
                'Claim_loss_401_f',
                'Claim_loss_430_f',
                'Claim_loss_419_f',
                'CNL_multi_vehicle_f',
                'CNL_coll_AV_f',
                'CNL_coll_IV_f',
                'CNL_hit_run_f',
                'loss_state_ne_claimant_f',
                'veh_state_ne_claimant_f',
                'veh_state_ne_loss_state_f',
                'total_loss_y_f',
                'total_loss_n_f',
                'policy_a_f',
                'policy_b_f',
                'policy_c_f',
                'policy_d_f',
                'policy_e_f',
                'policy_f_f',
                'Policy_Age',
                'injury_m',
                'region_g',
                'make_m',
                'division',
                'Dif_Open_Claim_rt_f',
                'veh_division',
                'loss_division',
                'Policy_Vintage_2',
                't_litigation_f',
                'no_claimant_15_d',
                'PT_NO_PAYMENTS_15_d',
                'num_res_type_15_d',
                'make_m_d',
                'division_d',
                'termprem_2',
                'vehicle_age_new')]

str(data_sub)
#remove factor type variables to get correlatio matrix

factor_vars=c('division_d','make_m_d', 'loss_division', 'veh_division', 'division', 'make_m',
              'region_g','injury_m','ACCIDENT_ROLE_m', 'Claim_role_m')
factor_vars1= names(data_sub) %in% factor_vars
correlated=cor(data_sub[!factor_vars1], method = "pearson")
write.csv(correlated,file="corr_matrix.csv")

# remove correlated vars and variables with no variation
remove_vars=c('PT_PAYMENT_AMT_wo_R', 'PT_NO_PAYMENTS_15_wo_R', 'PT_NO_PAYMENTS_15_d','PT_EXPENSE_AMT_wo_R',
              'no_claimant_15','no_res_type_15','veh_num_1_f','veh_num_0_f','claim_adverse_f','claim_Insured_f',
              'veh_state_ne_claimant_f','Policy_Age','no_claimant_15_d','num_res_type_15_d',
              'ac_role_pass_f','policy_c_f','claimant_zip_5')
remove_vars1= names(data_sub) %in% remove_vars
data_sub_2=data_sub[!remove_vars1]

#checking varimportance to subset vars, using gbm
set.seed(1701)
var_select_fit=gbm(t_litigation_f~., data=data_sub_2,
                   n.trees= 2000,
                   train.fraction = 0.8,
                   interaction.depth = 6)
var_influence=gbm.perf(var_select_fit,plot.it=TRUE,method='test')
var_influence_set=summary(var_select_fit,n.trees=var_influence)
var_influence_set[1:18,1]

#keep top 18 vars in the final model data set
keep_vars=as.character(var_influence_set[1:18,1])
keep_vars[19]='t_litigation_f'

data_sub_3=data_sub_2[,keep_vars]

#test train splitting of dataset, stratified
set.seed(1701)
index=createDataPartition(data_sub_3$t_litigation_f, p=0.7, list=FALSE)
train=data_sub_3[index,]
test=data_sub_3[-index,]


#train 1st model

fit1=gbm(t_litigation_f~., data=train,
         n.trees= 3000,
         train.fraction = 0.8,
         interaction.depth = 1,
         shrinkage = 0.005,
         cv.folds = 4)
perf1=gbm.perf(fit1,plot.it=TRUE,method='OOB')
summary(fit1,n.trees=perf1)

#plot.gbm(fit1,i.var = 2, n.trees = perf1)

#evaluate 1st model

test_frame1=data.frame(test$t_litigation_f)
test_frame1$prob= predict.gbm(fit1,test,n.trees = perf1,type = 'response')
pred=prediction( test_frame1$prob,test_frame1$test.t_litigation_f)
auc1= performance(pred,"auc")
roc1=performance(pred,'tpr','fpr')
plot(roc1)
#auc came out 68.5


#training model2
fit2=gbm(t_litigation_f~., data=train,
         n.trees= 4000,
         train.fraction = 0.8,
         interaction.depth = 6,
         shrinkage = 0.01,
         n.minobsinnode = 5,
         cv.folds=4)
perf2=gbm.perf(fit2,plot.it=TRUE,method='cv')
summary(fit2,n.trees=perf2)
#auc 0.69

test_frame2=data.frame(test$t_litigation_f)
test_frame2$prob= predict.gbm(fit2,test,n.trees = perf2,type = 'response')
pred2=prediction(test_frame2$prob,test_frame2$test.t_litigation_f)
auc2= performance(pred2,"auc")
roc2=performance(pred2,'tpr','fpr')
plot(roc2)

#finalising
#check performance on validation
test_frame2$response=0
plot(test_frame2$prob)
test_frame2$response[test_frame2$prob>0.045]=1

#confusion matrix
confusionMatrix(test_frame2$response,test_frame2$test.t_litigation_f)

#writing file for ks
write.csv(test_frame2,file='model_out_1.csv')

#check performance on development
developframe_frame2=data.frame(train$t_litigation_f)
developframe_frame2$prob= predict.gbm(fit2,train,n.trees = perf2,type = 'response')
developframe_frame2$response=0
plot(developframe_frame2$prob)
developframe_frame2$response[developframe_frame2$prob>0.045]=1

#confusion matrix
confusionMatrix(developframe_frame2$response,developframe_frame2$train.t_litigation_f)

#writing file for ks
write.csv(developframe_frame2,file='model_out_2.csv')

#Correlation of selected vars
factor_vars2=c('Dif_Open_Claim_rt',
              'make_m',
              'termprem_2',
              'Median_hh_inc_1999',
              'Policy_Vintage_2',
              'Claimant_age',
              'division',
              'vehicle_age_new',
              'Policy_Loss',
              'veh_division',
              'Days_loss_report',
              'Policy_Claim',
              'injury_m',
              'loss_division',
              'Driveable_n_f',
              'reserve_UMBI_f',
              'PT_PAYMENT_AMT_15',
              'year_miss_f'
)
factor_vars2= names(data_sub) %in% factor_vars
correlated=cor(data_sub[factor_vars2], method = "pearson")
write.csv(correlated,file="corr_matrix_final.csv")

#partial plots
plot.gbm(fit2,i.var=1)
plot.gbm(fit2,i.var=2)
plot.gbm(fit2,i.var=3)
plot.gbm(fit2,i.var=4)
plot.gbm(fit2,i.var=5)
plot.gbm(fit2,i.var=6)
plot.gbm(fit2,i.var=7)
plot.gbm(fit2,i.var=8)
plot.gbm(fit2,i.var=9)
plot.gbm(fit2,i.var=10)
plot.gbm(fit2,i.var=11)
plot.gbm(fit2,i.var=12)
plot.gbm(fit2,i.var=13)
plot.gbm(fit2,i.var=14)
plot.gbm(fit2,i.var=15)
plot.gbm(fit2,i.var=16)
plot.gbm(fit2,i.var=17)
plot.gbm(fit2,i.var=18)
