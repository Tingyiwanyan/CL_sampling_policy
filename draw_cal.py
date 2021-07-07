import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

True_logit = np.loadtxt('real_logit.out')
prob_logistic = np.loadtxt('logitstic_prob.out')
prob_rf = np.loadtxt('rf_prob.out')
prob_svm = np.loadtxt('svm_prob.out')
prob_xgb = np.loadtxt('xgb_prob.out')
prob_mlp = np.loadtxt('MLP_prob.out')
prob_lstm = np.loadtxt('LSTM_stack.out')
prob_lstm_att = np.loadtxt('LSTM_random_att.out')
prob_fl_random_time = np.loadtxt('LSTM_random_time.out')
prob_fl_self = np.loadtxt('LSTM_random_self.out')



plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#frac_of_pos_lr, mean_pred_value_lr = calibration_curve(True_logit, prob_logistic, n_bins=20)
#frac_of_pos_rf, mean_pred_value_rf = calibration_curve(True_logit, prob_rf, n_bins=20)
#frac_of_pos_svm, mean_pred_value_svm = calibration_curve(True_logit, prob_svm, n_bins=20)
#frac_of_pos_xgb, mean_pred_value_xgb = calibration_curve(True_logit, prob_xgb, n_bins=20)
#frac_of_pos_mlp, mean_pred_value_mlp = calibration_curve(True_logit, prob_mlp, n_bins=20)
frac_of_pos_lstm, mean_pred_value_lstm = calibration_curve(True_logit, prob_lstm, n_bins=20)
frac_of_pos_lstm_random, mean_pred_value_lstm_random = calibration_curve(True_logit, prob_lstm_att, n_bins=10)
frac_of_pos_fl_random_time, mean_pred_value_fl_random_time = calibration_curve(True_logit, prob_fl_random_time, n_bins=10)
frac_of_pos_fl_self, mean_pred_value_fl_self = calibration_curve(True_logit, prob_fl_self, n_bins=10)
#frac_of_pos_fl_att, mean_pred_value_fl_att = calibration_curve(True_logit, prob_fl_attributue, n_bins=10)
#plt.plot(mean_pred_value_lr, frac_of_pos_lr,  color='blue', linestyle='dashed',linewidth=1,label='LR')
#plt.plot(mean_pred_value_rf, frac_of_pos_rf,  color='green', linestyle='dashed',linewidth=1, label='RF')
#plt.plot(mean_pred_value_svm, frac_of_pos_svm,  color='violet', linestyle='dashed',linewidth=1, label='SVM')
#plt.plot(mean_pred_value_xgb, frac_of_pos_xgb,  color='red', linestyle='dashed',linewidth=1, label='XGB')
#plt.plot(mean_pred_value_mlp, frac_of_pos_mlp,  color='orange', linestyle='dashed',linewidth=1, label='MLP')
plt.plot(mean_pred_value_lstm, frac_of_pos_lstm,  color='cyan', linestyle='dashed',linewidth=1, label='LSTM_stack')
plt.plot(mean_pred_value_lstm_random, frac_of_pos_lstm_random,  color='gray', linestyle='solid',linewidth=2, label='LSTM_ATT')
plt.plot(mean_pred_value_fl_random_time, frac_of_pos_fl_random_time,  color='pink', linestyle='solid',linewidth=2, label='LSTM_RANDOM')
plt.plot(mean_pred_value_fl_self, frac_of_pos_fl_self,  color='purple', linestyle='solid',linewidth=2, label='LSTM_SELF')
plt.ylabel("Fraction of Positives")
plt.xlabel("Mean Predicted Value")
plt.ylim([-0.05, 1.05])
plt.legend(loc="lower right")
plt.title('Calibration plot')

plt.show()