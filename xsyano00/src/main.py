#My first python project

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import math
import scipy.io.wavfile
import wavfile
from scipy.signal import spectrogram, lfilter, freqz, tf2zpk

s, fs = sf.read('xsyano00.wav')
fig, axs = plt.subplots(3, figsize=(20,10))

t = np.arange(s.size) / fs


new_s = s


axs[0].plot(t, s)
axs[0].set_title('Zvukový signál')


print("pocet vzorku -", s.size)
print("frekvence -", fs)
print("vzorky -", s)
print("sekundy -", s.size/fs)

print("maximum -", max(s))
print("minimum -", min(s))

######################################### 2. uloha ################################

print("----------------------------------- 2. uloha --------------------------------")
stredni = sum(s) /s.size


ustr_signal = s - stredni

print("maximum -", max(ustr_signal))


normal = ustr_signal / abs(max(ustr_signal))


n_ram = math.ceil(1 + ((normal.size - 1024) / 512))
print("prekriti =", n_ram)
ramci = [0] * n_ram


for i in range(n_ram):
    ramci[i] = normal[(512*i):1024+(512*i)]



print(ramci[20])
print(len(ramci[20]))
pekny_vzor = ramci[20]
t_2 = np.arange(pekny_vzor.size) / fs
print("zacatek ve vzorcich =", 20*512)
odkud = 10240/fs
print(odkud)



axs[1].plot(t_2 + odkud, pekny_vzor)
axs[1].set_title('4. Zvukový signál')
axs[1].grid(alpha=0.5, linestyle='--')

######################################### 3. uloha ################################

print("----------------------------------- 3. uloha --------------------------------")



N = 1024
x_axis = np.arange(0, fs/2, fs/1024)
correct = np.fft.fft(pekny_vzor)
y_axis = np.abs(correct[:N//2])



axs[2].plot(x_axis, y_axis)
axs[2].set_title('vzor knihovny')
axs[2].grid(alpha=0.5, linestyle='--')



X=np.zeros(N,dtype = 'complex_')

for k in range(0,N):
    for n in range(0,N):
        X[k] += pekny_vzor[n] * (np.exp((-1j*2*np.pi*n*k)/N))

print(correct, '\n')
print(X, '\n')

print('spravny = ', correct[0])
print('what we have = ', X[0])

print('2 spravny = ', correct[1])
print('2 what we have = ', X[1])




fig, axs = plt.subplots(2, figsize=(20,10))

axs[0].plot(x_axis, y_axis)
axs[0].set_title('vzor knihovny')
axs[0].grid(alpha=0.5, linestyle='--')

N = 1024
x_axis = np.arange(0, fs/2, fs/1024)
y_axis = np.abs(X[:N//2])

axs[1].plot(x_axis, y_axis)
axs[1].set_title('muj vzor')
axs[1].grid(alpha=0.5, linestyle='--')

######################################### 4. uloha ################################

print("----------------------------------- 4. uloha --------------------------------")


spektry = [0] * n_ram
for i in range(n_ram):
    spektry[i] = np.fft.fft(ramci[i])


f, t, sgr = spectrogram(normal, fs)

sgr_log = 10 * np.log10(sgr+1e-1024/fs)

plt.figure(figsize=(16,8))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)


plt.tight_layout()


######################################### 5. uloha ################################

print("----------------------------------- 5. uloha --------------------------------")


max_2 = np.argmax(y_axis)
print(max_2)
print(correct)




idy = np.where(y_axis==y_axis[np.argmax(y_axis)])
f1 = x_axis[idy]
print("prvni frekvence", x_axis[idy])

# print(len(x_axis))

#print(max(x_axis[64:]))

x_new=x_axis[64:128]
y_new=y_axis[64:128]

idy = np.where(y_new==y_new[np.argmax(y_new)])
f2 = x_new[idy]
print("druha frekvence",x_new[idy])


x_new=x_axis[128:160]
y_new=y_axis[128:160]

idy = np.where(y_new==y_new[np.argmax(y_new)])
f3 = x_new[idy]
print("treti frekvence",x_new[idy])


x_new=x_axis[160:]
y_new=y_axis[160:]

idy = np.where(y_new==y_new[np.argmax(y_new)])
f4 = x_new[idy]
print("ctvrta frekvence",x_new[idy])

######################################### 6. uloha ################################

print("----------------------------------- 6. uloha --------------------------------")

samples=[]

for i in range(fs):
    samples.append(i / fs)


cos_1 = np.cos(2 * np.pi * f1 * np.array(samples))
cos_2 = np.cos(2 * np.pi * f2 * np.array(samples))
cos_3 = np.cos(2 * np.pi * f3 * np.array(samples))
cos_4 = np.cos(2 * np.pi * f4 * np.array(samples))


output = cos_1 + cos_2 + cos_3 + cos_4

print(output)

output2 = np.asarray(output, dtype=np.int16)


scipy.io.wavfile.write('4cos.wav', fs, output)

f, t, sgr = spectrogram(output, fs)
sgr_log = 10 * np.log10(sgr+1e-1024/fs)

plt.figure(figsize=(16,8))
plt.pcolormesh(t,f,sgr_log)
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('4 Frekvence  [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)


######################################### 7. uloha ################################

print("----------------------------------- 7. uloha --------------------------------")

print("prvni fr = ", f1)

kruhova_fr= 2 * np.pi * f1/ fs

print("kruhova = ", kruhova_fr)

plt.show()

print("It works")