import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# 세운 여름 환경 데이터 불러오기
df = pd.read_csv('./seun_summer.csv', parse_dates=['timestamp', "date"])
# print(df['timestamp'].dtypes)

# 계화 겨울 환경 데이터 불러오기
df2 = pd.read_csv('./winter.csv',parse_dates=["timestamp"])

# 환경데이터 전처리
df["dt"] = (df["timestamp"] - df["timestamp"].shift(periods=1)).dt.total_seconds()/86400 # dt 계산
df = df.iloc[:,[0,1,2,12,13,7,18]]

df2[["date", "time", "area"]] = df2["timestamp"].str.split(' ', expand=True)
df2["timestamp"] = pd.to_datetime(df2["timestamp"], format="%Y-%m-%d %H:%M:%S.%f %Z")
df2["dt"] = (df2["timestamp"] - df2["timestamp"].shift(periods=1)).dt.total_seconds()/86400 # dt 계산
total_length = len(df.index)
df2 = df2.iloc[:, [0,4,5,2,12]]

# 파라미터 전처리
c_a = 0.68
c_b = 0.80

# rgr
c_gr_max = (5 * (0.1 ** 6))*60         # saturation growth rate at 20°C
c_r = 1.0                              # cr should have value in the range of 0.5-1.0. In this model 1.0
c_q10_gr = 1.6                         # Q10 factor for growth. For every temperature increase of 10°C

# respiration (f_resp)
c_resp_sht = (3.47 * (10 ** -7))*60    # shoot resp param
c_resp_rt = (1.16 * (10 ** -7))*60     # root resp param
c_q10_resp = 2                         # Q10 factor of the maintenance respiration
c_t = 0.15                             #

# photosynthetic (f_phot)
c_k = 0.9                              # extinction coefficient
c_lar = 75 * (0.1 ** 3)                # structural leaf area ratio

# canopy photosynthesis (f_phot_max)
c_w = 1.83 * (0.1 ** 3)                # density of CO2

# gamma(Γ, PPM) CO2 compensation point
c_gamma = 40                           # CO2 compensation point at 20°C
c_q10_gamma = 2                        # Q10 Γ에 따른 온도

# epsilon(ε)
c_epsilon = (17.0 * (10 ** -6))

# g_car(carboxylation)
c_car1 = (-1.32 * (0.1 ** 5))*60
c_car2 = (5.94 * (0.1 ** 4))*60
c_car3 = (-2.64 * (0.1 ** 3))*60

date = pd.to_datetime(df.iloc[:,0].values.tolist())
print(date)

# 계화 겨울
u_t = df.iloc[:,3].values.tolist()
u_co2 = df.iloc[:, 4].values.tolist()
u_par = df.iloc[:, 5].values.tolist()
dt = df.iloc[:,6].values.tolist()

# 세운 여름
# u_t = df2.iloc[:,3].values.tolist()
# u_co2 = df2.iloc[:, 4].values.tolist()
# u_par = df2.iloc[:, 5].values.tolist()
# dt = df2.iloc[:,6].values.tolist()

nsdw = np.zeros(total_length)
sdw = np.zeros(total_length)

r_gr = np.zeros(total_length)
f_resp = np.zeros(total_length)
f_phot = np.zeros(total_length)
gamma = np.zeros(total_length)
epsilon = np.zeros(total_length)
f_phot_max = np.zeros(total_length)
g_car = np.zeros(total_length)
g_co2 = np.zeros(total_length)
dw = np.zeros(total_length)
lai = np.zeros(total_length)

# 모델 수식 계산
for i in range(0, total_length):
# for i in range(0, 2):
    if i == 0:
        nsdw[0] = 0.05 * 0.25
        sdw[0] = 0.05 * 0.75
        dt[0] = 1/86400
    else:
        nsdw[i] = (((c_a * f_phot[i-1])-(r_gr[i-1]*sdw[i-1])-f_resp[i-1]-((1-c_b) * r_gr[i-1] * sdw[i-1]/c_b))) * dt[i] + nsdw[i-1]
        # print(nsdw[i-1],dt[i] ,sdw[i-1], r_gr[i-1], f_resp[i-1], f_phot[i-1])
        sdw[i] = (r_gr[i - 1] * nsdw[i - 1]) * dt[i] + sdw[i - 1]
    r_gr[i] = (c_gr_max*(nsdw[i]/(c_r*sdw[i]+nsdw[i]))*(c_q10_gr**((u_t[i]-20)/10)))*60
    f_resp[i] = (((c_resp_sht * (1-c_t) * sdw[i])+(c_resp_rt * c_t * sdw[i])) * (c_q10_resp**((u_t[i]-25)/10)))*60
    gamma[i] = (c_gamma * (c_q10_gamma**((u_t[i]-20)/10)))
    epsilon[i] = (c_epsilon * (u_co2[i] - gamma[i])/(u_co2[i]+(2*gamma[i])))*60
    g_car[i] = ((c_car1 * (u_t[i]**2))+(c_car2*u_t[i])+c_car3)
    g_co2[i] = 1/(1/(0.007*60)+1/(0.005*60)+1/g_car[i])
    f_phot_max[i] = (epsilon[i] * u_par[i] * g_co2[i] * c_w * (u_co2[i]-gamma[i]))/((epsilon[i] * u_par[i])+g_co2[i] * c_w * (u_co2[i]-gamma[i]))*60
    f_phot[i] = ((1-math.exp(-c_k * c_lar * (1-c_t) * sdw[i])) * f_phot_max[i])*60
    dw[i] = (nsdw[i]+sdw[i])/10
    lai[i] = ((1-c_t)*c_lar*sdw[i])*10
    # print(u_t[i], nsdw[i],sdw[i], r_gr[i], f_resp[i], f_phot[i], gamma[i], epsilon[i])
    # print(f_phot_max[i], g_car[i],dw[i], lai[i], g_co2[i])
# print(dw)
# print(lai)

x= np.array(date)
y = np.array(dw)
y2 = np.array(lai)

plt.figure(1)
plt.title('DW')
plt.plot(x,y)
plt.rc('axes', labelsize=5)    # fontsize of the x and y labels
plt.xticks(rotation=30)

plt.figure(2)
plt.title('LAI')
plt.plot(x,y2)
plt.rc('axes', labelsize=5)    # fontsize of the x and y labels
plt.xticks(rotation=30)
plt.show()