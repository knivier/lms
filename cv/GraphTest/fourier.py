# ============================
# FFT between first and last peak
# # ============================
# if len(peaks) >= 2:
#     start = peaks[0]
#     end = peaks[-1]
#     segment = signal[start:end+1]

#     # Compute FFT
#     N = len(segment)
#     fft_vals = np.fft.fft(segment)
#     fft_freq = np.fft.fftfreq(N, d=1)  # d=1 assumes 1 sample per unit time; adjust if you know FPS

#     # Only take positive frequencies
#     pos_mask = fft_freq > 0
#     fft_freq = fft_freq[pos_mask]
#     fft_vals = np.abs(fft_vals[pos_mask])

#     # Plot FFT
#     plt.figure(figsize=(10,4))
#     plt.plot(fft_freq, fft_vals)
#     plt.title("FFT of segment between first and last peak")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel("Magnitude")
#     plt.show()
# else:
#     print("Not enough peaks detected for FFT.")