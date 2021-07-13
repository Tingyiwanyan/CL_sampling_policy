import numpy as np
import matplotlib.pyplot as plt

step = [  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17,18]



acc_mlp=[0.4794833333333334,0.6914333333333332,0.6950999999999999,0.6825666666666667,
0.7192000000000001,0.7256666666666666,0.6604666666666666,0.6504333333333334,
0.7300000000000001,0.7278666666666667,0.7239666666666666,0.6825333333333333,0.6767333333333333,
0.6998333333333333,0.7157333333333333,0.7087666666666665,0.7039999999999998,0.7044999999999999]

acc_lstm = [0.37963333, 0.5418    , 0.68383333, 0.68295   , 0.69676667,
       0.73365   , 0.7408    , 0.74231667, 0.75543333, 0.7416    ,
       0.74628333, 0.75268333, 0.7551    , 0.75591667, 0.7482    ,
       0.7551    , 0.76115   , 0.75975   ]

acc_lstm_stack = [0.48253333, 0.72673333, 0.71083333, 0.72256667, 0.732     ,
       0.74363333, 0.7438    , 0.74903333, 0.74703333, 0.74736667,
       0.74596667, 0.7522    , 0.75073333, 0.75733333, 0.7566    ,
       0.75326667, 0.75246667, 0.75433333]

acc_lstm_random = [0.2853    , 0.6214    , 0.7087    , 0.703     , 0.7187    ,
       0.7265    , 0.72663333, 0.72813333, 0.72938333, 0.73016667,
       0.7435    , 0.74233333, 0.74306667, 0.7541    , 0.75133333,
       0.7569    , 0.7538    , 0.7507    ]

acc_lstm_random_time = [0.61026667, 0.65406667, 0.73223333, 0.7468    , 0.74633333,
       0.7454    , 0.7455    , 0.75153333, 0.7494    , 0.7479    ,
       0.7496    , 0.75213333, 0.7527    , 0.748     , 0.75346667,
       0.75546667, 0.75343333, 0.75543333]

#acc_lstm_random_att = [0.43803333, 0.6345, 0.64536667, 0.69263333, 0.73846667,
    #   0.76113333, 0.75913333, 0.76016667, 0.7461    , 0.74916667,
     #  0.76113333, 0.74783333, 0.75216667, 0.7625    , 0.7498    ,
      # 0.7601    , 0.75913333, 0.7483    ]

acc_lstm_self = [0.35556667, 0.68021667, 0.7214    , 0.71303333, 0.71063333,
       0.7218    , 0.7289    , 0.73496667, 0.74066667, 0.74193333,
       0.74143333, 0.74636667, 0.7477    , 0.74896667, 0.75373333,
       0.75673333, 0.74363333, 0.74863333]

acc_lstm_random_different = [0.41986667, 0.7055    , 0.7356    , 0.73823333, 0.7434    ,
       0.75016667, 0.75473333, 0.75343333, 0.7592    , 0.7543    ,
       0.75676667, 0.75686667, 0.75733333, 0.75493333, 0.76113333,
       0.75793333, 0.76053333, 0.75076667]


plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Accuracy Curve", fontsize=14)
plt.xlim(0, 20)
plt.ylim(0.2, 0.9)

plt.plot(step,acc_mlp,"x",color='green',linestyle='dashed',linewidth=1,label='MLP')
plt.plot(step,acc_lstm,"x",color='blue',linestyle='dashed',linewidth=1,label='LSTM')
#plt.plot(fp_total_ce,tp_total_ce,color='indigo',linestyle='dashed',linewidth=2,label='CE(AUC=0.743)')
#plt.plot(recall_total_fl,precision_total_fl,color='orange',linestyle='dashed',linewidth=2,label='FL(AUPRC=0.604)')
plt.plot(step,acc_lstm_stack,"x",color='black',linestyle='dashed',linewidth=1,label='LSTM_STACK')
plt.plot(step,acc_lstm_random,"x",color='red',linestyle='dashed',linewidth=1,label='LSTM_RANDOM')
plt.plot(step,acc_lstm_random_time,"x",color='orange',linestyle='dashed',linewidth=1,label='LSTM_RANDOM_TIME')
plt.plot(step,acc_lstm_self,"x",color='purple',linestyle='dashed',linewidth=1,label='LSTM_SELF')
plt.plot(step,acc_lstm_random_different,"x",color='pink',linestyle='dashed',linewidth=1,label='LSTM_RANDOM_DIFF')

plt.legend(loc='lower right')
plt.show()