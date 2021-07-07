import numpy as np
import matplotlib.pyplot as plt

step = [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17,18]

acc_mlp=[0.60785   , 0.6864    , 0.6985    , 0.7739    , 0.75733333,
       0.74958333, 0.7451    , 0.785     , 0.7824    , 0.77841667,
       0.77873333, 0.71878333, 0.69503333, 0.76383333, 0.80686667,
       0.79658333, 0.8037    , 0.71766667]

acc_lstm = [0.4746, 0.53943333, 0.5864    , 0.6299    , 0.68003333,
       0.7111    , 0.71823333, 0.7247    , 0.7248    , 0.75033333,
       0.75043333, 0.75523333, 0.75586667, 0.75173333, 0.75023333,
       0.76256667, 0.76393333, 0.76726667]

acc_lstm_random_time = [0.55361667, 0.70378333, 0.71606667, 0.71211667, 0.69601667,
       0.71523333, 0.72301667, 0.71655   , 0.7181    , 0.73426667,
       0.71658333, 0.73301667, 0.72771667, 0.72118333, 0.72196667,
       0.72123333, 0.72183333, 0.7229    ]

acc_lstm_random_att = [0.43803333, 0.6345, 0.64536667, 0.69263333, 0.73846667,
       0.76113333, 0.75913333, 0.76016667, 0.7461    , 0.74916667,
       0.76113333, 0.74783333, 0.75216667, 0.7625    , 0.7498    ,
       0.7601    , 0.75913333, 0.7483    ]

acc_lstm_self = [0.53121667, 0.67726667, 0.70185   , 0.6897    , 0.68223333,
       0.72156667, 0.72323333, 0.7095    , 0.73533333, 0.74053333,
       0.75213333, 0.74913333, 0.75468333, 0.74406667, 0.75083333,
       0.74596667, 0.752     , 0.74341667]

acc_bilstm = [0.7171833333333334,
 0.6808166666666667,
 0.7116,
 0.7309333333333334,
 0.7345666666666667,
 0.7751,
 0.7775333333333334,
 0.7411000000000001,
 0.7676666666666665,
 0.7756500000000001,
 0.7689499999999999,
 0.7769,
 0.7764166666666666,
 0.7622333333333333,
 0.7636333333333333,
 0.7815333333333334,
 0.7750333333333334,
 0.7800833333333334]

plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Accuracy Curve", fontsize=14)
plt.xlim(0, 20)
plt.ylim(0.4, 0.9)

#plt.plot(step,acc_mlp,"x",color='green',linestyle='dashed',linewidth=1,label='MLP')
plt.plot(step,acc_lstm,"x",color='blue',linestyle='dashed',linewidth=1,label='LSTM')
#plt.plot(fp_total_ce,tp_total_ce,color='indigo',linestyle='dashed',linewidth=2,label='CE(AUC=0.743)')
#plt.plot(recall_total_fl,precision_total_fl,color='orange',linestyle='dashed',linewidth=2,label='FL(AUPRC=0.604)')
plt.plot(step,acc_lstm_random_att,"x",color='red',linestyle='dashed',linewidth=1,label='LSTM_ATT')
plt.plot(step,acc_lstm_random_time,"x",color='orange',linestyle='dashed',linewidth=1,label='LSTM_RANDOM')
plt.plot(step,acc_lstm_self,"x",color='purple',linestyle='dashed',linewidth=1,label='LSTM_SELF')

plt.legend(loc='lower left')
plt.show()