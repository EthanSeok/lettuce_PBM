import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import streamlit as st

# 세운 여름 환경 데이터 불러오기
df = pd.read_csv('./seun_summer.csv', parse_dates=['timestamp', "date", "time"])
# print(df['timestamp'].dtypes)

# 계화 겨울 환경 데이터 불러오기
df2 = pd.read_csv('./winter.csv',parse_dates=["timestamp"])

# 시나리오 env 파일 불러오기
df_env = pd.read_csv('env.csv', encoding='utf-8-sig')

df_raw = pd.read_csv('./env_raw.csv', encoding='utf-8-sig')

# 환경데이터 전처리
df["dt"] = (df["timestamp"] - df["timestamp"].shift(periods=1)).dt.total_seconds()/60  #dt 계산

df = df.iloc[:,[0,1,2,12,13,7,18]]

df2[["date", "time", "area"]] = df2["timestamp"].str.split(' ', expand=True)
df2["timestamp"] = pd.to_datetime(df2["timestamp"], format="%Y-%m-%d %H:%M:%S.%f %Z")

df2["dt"] = (df2["timestamp"] - df2["timestamp"].shift(periods=1)).dt.total_seconds()/60
# total_length = len(df.index)
total_length = len(df.index)

df2 = df2.iloc[:, [0,4,5,2,15]]
# print(df2.head)

date_summer = pd.to_datetime(df.iloc[:, 0].values.tolist())
date_winter = pd.to_datetime(df2.iloc[:, 0].values.tolist())


# streamlit
st.title('LETTUCE PBM MODEL')

# st.subheader('Raw data')
if st.checkbox('Show raw data'):
    st.write(df_raw)

env_t = st.select_slider("temperature", options=df_env.columns[6:20].tolist())

env_p = st.select_slider("PAR", options=df_env.columns[20:23].tolist())


# 파라미터 전처리
c_a = 0.68
c_b = 0.80

# rgr
c_gr_max = (5 * (0.1 ** 6))*60         # saturation growth rate at 20°C
c_r = 1.0                              # cr should have value in the range of 0.5-1.0. In this model 1.0
c_q10_gr = 1.6                         # Q10 factor for growth. For every temperature increase of 10°C

# respiration (f_resp)
c_resp_sht = (3.47 * (10 ** -7))    # shoot resp param
c_resp_rt = (1.16 * (10 ** -7))    # root resp param
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
c_car1 = (-1.32 * (0.1 ** 5))
c_car2 = (5.94 * (0.1 ** 4))
c_car3 = (-2.64 * (0.1 ** 3))

# 세운 여름
# u_t = df.iloc[:,3].values.tolist()
# u_co2 = df.iloc[:, 4].values.tolist()
# u_par = df.iloc[:, 5].values.tolist()
# dt = df.iloc[:,6].values.tolist()

# 계화 겨울
# u_t = df2.iloc[:,1].values.tolist()
# u_co2 = df2.iloc[:, 2].values.tolist()
# u_par = df2.iloc[:, 3].values.tolist()
# dt = df2.iloc[:, 4].values.tolist()

# env
u_t = df_env[env_t].values.tolist()
u_co2 = df_env.iloc[:, 1].values.tolist()
u_par = df_env[env_p].values.tolist()
dt = df.iloc[:, 6].values.tolist()

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
        nsdw[0]=0.72*0.25
        sdw[0] =0.72*0.75
        dt[0] = 1
    else:
        nsdw[i] = (((c_a * f_phot[i-1])-(r_gr[i-1]*sdw[i-1])-f_resp[i-1]-((1-c_b) * r_gr[i-1] * sdw[i-1]/c_b))) * dt[i] + nsdw[i-1]
        # print(nsdw[i-1],dt[i] ,sdw[i-1], r_gr[i-1], f_resp[i-1], f_phot[i-1])
        sdw[i] = (r_gr[i - 1] * sdw[i - 1]) * dt[i] + sdw[i - 1]
    r_gr[i] = (c_gr_max*(nsdw[i]/(c_r*sdw[i]+nsdw[i]))*(c_q10_gr**((u_t[i]-20)/10)))
    f_resp[i] = (((c_resp_sht * (1-c_t) * sdw[i])+(c_resp_rt * c_t * sdw[i])) * (c_q10_resp**((u_t[i]-25)/10)))
    gamma[i] = (c_gamma * (c_q10_gamma**((u_t[i]-20)/10)))
    epsilon[i] = (c_epsilon * (u_co2[i] - gamma[i])/(u_co2[i]+(2*gamma[i])))
    g_car[i] = ((c_car1 * (u_t[i]**2))+(c_car2*u_t[i])+c_car3)
    g_co2[i] = 1/(1/(0.007)+1/(0.005)+1/g_car[i])
    f_phot_max[i] = (epsilon[i] * u_par[i] * g_co2[i] * c_w * (u_co2[i]-gamma[i]))/((epsilon[i] * u_par[i])+g_co2[i] * c_w * (u_co2[i]-gamma[i]))*60
    f_phot[i] = ((1-math.exp(-c_k * c_lar * (1-c_t) * sdw[i])) * f_phot_max[i])
    dw[i] = (nsdw[i]+sdw[i])/18
    lai[i] = ((1-c_t)*c_lar*sdw[i])
    # if i<4:
    #     print(u_t[i], nsdw[i], dt[i], sdw[i], r_gr[i], f_resp[i], f_phot[i], gamma[i], epsilon[i])
    #     print(f_phot_max[i], g_car[i],dw[i], lai[i], g_co2[i])

# print(dw)
# print(lai)

x = np.array(date_summer)
y = np.array(dw)
y2 = np.array(lai)

plt.figure(1)
plt.title('DW')
plt.plot(x,y)
plt.grid(True, axis='y')
plt.xticks(rotation=30)
plt.savefig('dw_result.png')

plt.figure(2)
plt.title('LAI')
plt.plot(x,y2)
plt.grid(True, axis='y')
plt.xticks(rotation=30)
plt.show()
plt.savefig('lai_result.png')
#
# st.dataframe(df)
# df.head()

# st.dataframe(df.head())
# st.write(df.head())

# print(len(dw), dw[-5:])


# streamlit

# temp = pd.DataFrame(
#     df_env,
#     columns=['겨울 온도', '여름 온도'])
#
# st.line_chart(temp)
#
# par = pd.DataFrame(
#     df_env,
#     columns=['겨울 광량', '여름 광량'])
#
# st.line_chart(par)

# co2 = pd.DataFrame(
#     df_env,
#     columns=['겨울 CO2', '여름 CO2'])
#
# st.line_chart(co2)

# df_env_diff = pd.read_csv(r'C:\code\lettuce_PBM\lettuce_PBM\seun_temp.csv')
#
# seun_temp = pd.DataFrame(
#     df_env_diff,
#     columns=['10','11','12','13','14','15','16','17','18','19','20','21','22','23'])
#
# st.line_chart(seun_temp)




df_dw = pd.DataFrame(dw, columns=['DW'])
df_lai = pd.DataFrame(lai, columns=['LAI'])

st.subheader('DRY WEIGHT (g)')
st.line_chart(dw)

st.subheader('LAI (leaf area index,  (m2/m2)')
st.line_chart(lai)


# df_check = pd.DataFrame(data=[
# nsdw,
# sdw,
# r_gr,
# f_resp,
# gamma,
# epsilon,
# g_car,
# g_co2,
# f_phot_max,
# f_phot,
# dw,
# lai,
# ]).T
# df_check.columns = [
# "nsdw",
# "sdw",
# "r_gr",
# "f_resp",
# "gamma",
# "epsilon",
# "g_car",
# "g_co2",
# "f_phot_max",
# "f_phot",
# "dw",
# "lai",
# ]
#
# df_check.to_csv("data_check.csv", index=False)