class DataMgmt:
    """
    class for managing, updating and manipulating the historical data using pandas
    """
    def __init__(self,df=None,cols=None,indexcol=None,filename=None,savedir=None,autosave=False):
        """instantiate class.  if df is input then a copy is made.  Last entry used as observer variable"""
        """cols must be entered as a python list"""
        self._lastentry=0
        self._callbacks=[]
        self.filename=filename
        self.autosave=autosave
        self.savedir=savedir

        if df is not None:
            self._data=df.copy()
        else:
            self._data=pd.DataFrame(columns=cols)
        if indexcol:
            self._data.set_index(indexcol, inplace=True)

    def Save(self,oldvalue=None,newvalue=None):
        self._data.to_csv(self.savedir+'{}.csv'.format(self.filename))

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_value):
        self._data = new_value.copy()
        if self.autosave:
            self.Save()

    def createcopy(self):
        return self._data.copy()

    def update(self,new_index,new_lineitem):
        if new_index not in self._data.index:
            self._data.loc[new_index]=new_lineitem
            if self.autosave:
                self.Save()

    def livedataupdate(self,index,lineitem):
        self._data.loc[index]=lineitem
        self._data.drop_duplicates(keep='last')

    def update_index(self,index_col):
        self._data.set_index((index_col), inplace=True)

    def reset_index(self):
        self._data.reset_index()

    def mergedata(self,df,on_index):
        self._data=self._data.merge(df,on=on_index)

    def flushzeros(self):
        self._data=self._data.loc[~(self._data==0).all(axis=1)]

    @property
    def LastEntry(self):
        return self._lastentry

    @LastEntry.setter
    def LastEntry(self,new_value):
        old_value=self._lastentry
        self._lastentry=new_value
        if old_value!=new_value:
            self._notify_observers(new_value)

    def _notify_observers(self, new_value):
        for callback in self._callbacks:
            callback(new_value)

    def register_callback(self, callback):
        self._callbacks.append(callback)

    def MACD(self):
        """
        returns MACD for the reward asset
        """
        self._data['exp1'] = self._data['Close'].ewm(span=12, adjust=False).mean()
        self._data['exp2'] = self._data['Close'].ewm(span=26, adjust=False).mean()
        self._data['macd'] = self._data['exp1']-self._data['exp2']
        self._data['signal'] = self._data['macd'].ewm(span=9, adjust=False).mean()

    def RSI(self,period=14):
        """
        returns RSI for the reward asset
        """
        self._data['Ret']=self._data['Close'].pct_change(1)
        self._data=self._data.assign(up=lambda x: np.maximum(0,x['Ret']))
        self._data=self._data.assign(down=lambda x: -np.minimum(0,x['Ret']))
        self._data['RollDown'],self._data['RollUp'] = self._data['down'].rolling(period).mean().abs(),self._data['up'].rolling(period).mean()
        self._data['RS']=self._data['RollUp']/self._data['RollDown']
        self._data['RSI'] =100.0-(100.0/(1.0+self._data['RS']))

    def ADX(self,period):
        """
        returns ADX for reward asset.  Requires High and Low as well as closing data.
        """
        self._data['dm_plus']=np.where(((self._data['High']-self._data['High'].shift(1))>(self._data['Low'].shift(1)-self._data['Low'])),np.maximum(0,self._data['High']-self._data['High'].shift(1)),0)
        self._data['dm_minus']=np.where(((self._data['Low'].shift(1)-self._data['Low'])>(self._data['High']-self._data['High'].shift(1))),np.maximum(0,self._data['Low'].shift(1)-self._data['Low']),0)
        self._data=self._data.assign(TR=lambda x: np.maximum(np.maximum(x['High']-x['Low'],x['High']-x['Close'].shift(1)),np.maximum(x['High']-x['Close'].shift(1),x['Close'].shift(1)-x['Low'])))
        self._data['TR_smooth']=(self._data['TR'].shift(1)-self._data['TR'].shift(1)/period)+self._data['TR']
        self._data['dm_plus_smooth']=self._data['dm_plus'].ewm(com=0.5,min_periods=period).mean()
        self._data['dm_minus_smooth']=self._data['dm_minus'].ewm(com=0.5,min_periods=period).mean()
        self._data['di_plus']=(self._data['dm_plus_smooth']/self._data['TR_smooth'])*100
        self._data['di_minus']=(self._data['dm_minus_smooth']/self._data['TR_smooth'])*100
        self._data['di_diff']=np.abs((self._data['di_plus']-self._data['di_minus']))
        self._data['di_sum']=self._data['di_plus']+self._data['di_minus']
        self._data['dx']=(self._data['di_diff']/self._data['di_sum'])*100
        self._data['adx']=self._data['dx'].ewm(com=0.5,min_periods=period).mean()

    def find_loc(self,window,minmax='max'):
        search_list=window.values.tolist()
        search_list.reverse()
        if minmax=='max': search_value=max(search_list)
        else: search_value=min(search_list)
        return search_list.index(search_value)

    def Aroon(self,period):
        """
        returns Aroon indicator for reward asset.
        """
        self._data['aroon_up']=self._data['Close'].rolling(period).apply(lambda x: 100*(period-self.find_loc(x,minmax='max'))/period)
        self._data['aroon_down']=self._data['Close'].rolling(period).apply(lambda x: 100*(period-self.find_loc(x,minmax='min'))/period)
        self._data['aroon_osc']=self._data['aroon_up']-self._data['aroon_down']

    def Stochastic(self,period=14):
        """
        returns stochastic oscillator indicator for reward asset. Need High and Low data.
        """
        self._data['periodlow']=self._data['Low'].rolling(period).min()
        self._data['periodhigh']=self._data['High'].rolling(period).max()
        self._data=self._data.assign(Stochastic_osc=lambda x: 100*((self._data['Close']-self._data['periodlow'])/(self._data['periodhigh']-self._data['periodlow'])))

    def ADL(self):
        """
        calculates the accumulation/distribution indicator.
        requires high, low, close and volume data.
        """
        data_copy['mfm']=((self._data['Close']-self._data['Low'])-(self._data['High']-self._data['Close']))/(self._data['High']-self._data['Low'])
        data_copy['mfv']=self._data['Volume']*self._data['mfm']
        data_copy['adl']=0
        data_copy['adl']=self._data['adl'].shift(1)+self._data['mfv']

    def CalcTechs(self):
        self.MACD()
        self.RSI(14)
        self.ADX(14)
        self.Aroon(25)
        self.Stochastic(14)
        if self.autosave:
            self.Save()



