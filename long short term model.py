import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as T
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import fn as IB

"""Time model forecast run"""
t0=T.time()

"""train/valid/test boundaries"""
train=45000
valid=50000
train_model=True

"""parameters"""
window_size=20
batch_size=25

"""save directories"""
init='C:/Users/user/anaconda3/algo/'
history_dir=init+'history/updates/'
tech_dir=init+'Tech/updates/'
trade_data_dir=init+'Trades/TradeData/'
model_data_dir=init+'Model/'
tech_data=['macd','signal','RSI','adx','aroon_osc','Stochastic_osc']

"""Objects required to instantiate instance of trading class"""
Time=IB.TimeMgmt('America/Chicago')
history=IB.DataMgmt(pd.read_csv(history_dir+'ub.csv'),indexcol=['Date'],filename='ub',savedir=history_dir,autosave=False)
techdata=IB.DataMgmt(pd.read_csv(tech_dir+'ub.csv'),indexcol=['Date'],filename='ub',savedir=tech_dir,autosave=True)
modeldata=IB.DataMgmt(pd.read_csv(model_data_dir+'ub.csv'),indexcol=['DateTime'],filename='ub',savedir=model_data_dir,autosave=True)
model_obj=IB.ModelMgmt(model_data_dir,Time)

"""select data for model. full dataset. merge with results, drop NaN. Returns in final column"""
X_train=techdata.data[tech_data].iloc[:-1].reset_index().drop(['Date'],axis=1)
Y_diff=techdata.data['Close'].diff().iloc[1:].reset_index().drop(['Date'],axis=1)
dataset_df=X_train.merge(Y_diff,left_index=True,right_index=True).dropna()
dataset_np=dataset_df.to_numpy()

"""normalize, using data up to the 'train' limit. assume valid/test data out of sample"""
scaler=StandardScaler()
scaler.fit(dataset_np[:train])

def data_windowed(series,window_size,batch_size,series_start=0,series_end=0,normaliser=None,stateful=False):
"""applies a normalization prior to converting to dataset, splitting and shuffling"""
    if normaliser!=None:
    series=normaliser.transform(series)
    dataset=tf.data.Dataset.from_tensor_slices(series[series_start:series_end])
    if stateful:
    dataset = dataset.window(window_size+1, shift=window_size+1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size+1))
    dataset=dataset.map(lambda w: (w[:,:-1], w[:,-1]))
    return dataset.batch(1).prefetch(1)
    else:
    dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size+1))
    #dataset=dataset.map(lambda w: (w[:,:-1], w[-1,-1]))
    dataset=dataset.map(lambda w: (w[:,:-1], w[-1,-1]))
    return dataset.repeat(2).shuffle(buffer_size=100000).batch(batch_size).prefetch(1)

def model_forecast(model, series, window_size, series_start=0, normaliser=None):
    if normaliser!=None:
    series=normaliser.transform(series)
    ds = tf.data.Dataset.from_tensor_slices(series[series_start-window_size:,:-1])
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(10000).prefetch(1)
    forecast = model.predict(ds)
    return forecast

def Get_Y_True(model, series, window_size, series_start=0):
ds = tf.data.Dataset.from_tensor_slices(series[series_start-window_size:])
ds = ds.window(window_size, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(window_size))
ds=ds.map(lambda w: w[-1,-1])
ds_list=[item.numpy() for item in ds]
return np.array(ds_list)

if train_model:
"""Build RNN model"""
model=keras.models.Sequential(
[#keras.layers.Conv1D(filters=20,kernel_size=10,strides=1,padding='causal',input_shape=[None,6]),
keras.layers.GRU(20,return_sequences=True,dropout=0.25,recurrent_dropout=0.25,stateful=False,input_shape=[None,6]),
keras.layers.GRU(20,dropout=0.25,recurrent_dropout=0.25,stateful=False),
keras.layers.Dense(1)
])

model.compile(loss="mse",optimizer="adam",metrics=["mse"])
print(model.summary())
"""Retrieve data for training"""
X_train=data_windowed(dataset_np,window_size,batch_size,0,train,scaler,stateful=False)
X_val=data_windowed(dataset_np,window_size,batch_size,train,valid,scaler,stateful=False)

"""Quick check of data"""
#for inputs,labels in X_train.batch(1).prefetch(1):
# print(inputs.numpy(), "=>", labels.numpy())

"""Fit model"""
callback=keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
history=model.fit(X_train,validation_data=X_val,epochs=30,verbose=1,callbacks=[callback])
model.save('rnn.h5')

else:
model=keras.models.load_model('rnn.h5')

"""Perform 'walk forward': take a sliding window, forecast, then add next data point and repeat"""
rnn_forecast = model_forecast(model, dataset_np, window_size,series_start=valid,normaliser=scaler)
trade_action=np.where(rnn_forecast>0,1,-1)
y_true=Get_Y_True(model, dataset_np, window_size, series_start=valid)
y_true=np.expand_dims(y_true,axis=-1)
print('trade_act T shape:,',trade_action.shape,'\nytrue shape:',y_true.shape)
pnl=np.multiply(trade_action,y_true)
print(pnl)
pnl_total=np.cumsum(pnl)
act_total=np.cumsum(y_true)

"""plot chart"""
plt.figure(figsize=(10, 6))
plt.plot(pnl_total, label='strategy')
plt.plot(act_total,label='actual')
plt.legend()
plt.show()


