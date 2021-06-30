import numpy as np
import matplotlib.pyplot as plt

step = [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17,18]

acc_mlp=[0.4875    , 0.48706667, 0.49428333, 0.49641667, 0.5537    ,
       0.58751667, 0.63375   , 0.6328    , 0.63213333, 0.54543333,
       0.7144    , 0.69293333, 0.74716667, 0.70596667, 0.72673333,
       0.67833333, 0.70496667, 0.7429    ]#, 0.74916667, 0.74406667,
       #0.72476667, 0.6898    , 0.6996    , 0.74363333]

acc_lstm = [0.4746, 0.53943333, 0.5864    , 0.6299    , 0.68003333,
       0.7111    , 0.71823333, 0.7247    , 0.7248    , 0.75033333,
       0.75043333, 0.75523333, 0.75586667, 0.75173333, 0.75023333,
       0.76256667, 0.76393333, 0.76726667]

acc_lstm_random = [0.4783    , 0.6103    , 0.69463333, 0.7011    , 0.7228,
       0.76433333, 0.73023333, 0.77703333, 0.77556667, 0.77113333,
       0.78593333, 0.7845    , 0.74146667, 0.78216667, 0.77516667,
       0.78163333, 0.7802    , 0.76843333]

acc_lstm_random_time = [0.43803333, 0.6345, 0.64536667, 0.69263333, 0.73846667,
       0.76113333, 0.75913333, 0.76016667, 0.7461    , 0.74916667,
       0.76113333, 0.74783333, 0.75216667, 0.7625    , 0.7498    ,
       0.7601    , 0.75913333, 0.7483    ]

acc_lstm_self = [0.4064    , 0.4506    , 0.49333333, 0.5338    , 0.52393333,
       0.51523333, 0.5089    , 0.51786667, 0.53286667, 0.51883333,
       0.51876667, 0.54933333, 0.54546667, 0.53396667, 0.54853333,
       0.55336667, 0.5454    , 0.59536667]

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Accuracy Curve", fontsize=14)
plt.xlim(0, 20)
plt.ylim(0.4, 0.9)

plt.plot(step,acc_mlp,"x",color='green',linestyle='dashed',linewidth=1,label='MLP')
plt.plot(step,acc_lstm,"x",color='blue',linestyle='dashed',linewidth=1,label='LSTM')
#plt.plot(fp_total_ce,tp_total_ce,color='indigo',linestyle='dashed',linewidth=2,label='CE(AUC=0.743)')
#plt.plot(recall_total_fl,precision_total_fl,color='orange',linestyle='dashed',linewidth=2,label='FL(AUPRC=0.604)')
plt.plot(step,acc_lstm_random,"x",color='red',linestyle='dashed',linewidth=1,label='LSTM_RANDOM')
plt.plot(step,acc_lstm_random_time,"x",color='orange',linestyle='dashed',linewidth=1,label='LSTM_RANDOM_TIME')
plt.plot(step,acc_lstm_self,"x",color='purple',linestyle='dashed',linewidth=1,label='LSTM_SELF')

plt.legend(loc='lower left')
plt.show()