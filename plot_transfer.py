import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
from pathlib import Path
import numpy as np # type: ignore

project_folder = Path(__file__).resolve().parent.parent
csv_file = project_folder / "data" / "mosfet_transfer.csv"
    
data = pd.read_csv(csv_file)

VGS = data["VGS"]
ID = data["ID"]

I_OFF =ID.iloc[0]
I_ON = ID.iloc[-1]

on_off_ratio = I_ON / I_OFF
print(f"IOFF - {I_OFF:.3e} A")
print(f"ION - {I_ON:.3e} A")
print(f"ON/OFF Ratio - {on_off_ratio:.2e}")

gm = np.gradient(ID, VGS)

print("\nTransconductance values:")
for vgs_value, gm_value in zip(VGS, gm):
    print(f"VGS: {vgs_value:.2f} V, gm: {gm_value:.6f} S")


max_gm_index = np.argmax(gm)
max_gm = gm[max_gm_index]
max_gm_vgs = VGS.iloc[max_gm_index]
print(f"Maximum gm: {max_gm:.6f} S")
print(f"Occurs at VGS = {max_gm_vgs:.2f} V")

threshold_current = 10e-6  

for i in range (len(ID) - 1):
    if ID.iloc[i] <= threshold_current <= ID.iloc[i+1]:

        v1 = VGS.iloc[i]
        v2 = VGS.iloc[i+1]
        i1 = ID.iloc[i]
        i2 = ID.iloc[i+1]


        VTH = v1 + (threshold_current - i1) * (v2 - v1) / (i2 - i1)

        print("Threshold voltage (VTH) is approximately: {:.4f} V".format(VTH))
        break

plt.plot(VGS, ID, marker='o', linestyle='-', color='b')

# Mark threshold voltage
plt.axvline(VTH, linestyle="--", label=f"VTH = {VTH:.3f} V")
plt.axhline(threshold_current, linestyle="--", alpha=0.5)

plt.scatter(
    VTH,
    threshold_current,
    s=100,
    zorder=5,
    label="Threshold Point"
)

plt.xlabel("VGS (V)")
plt.ylabel("ID (A)")
plt.title("MOSFET Transfer Characteristics")
plt.grid(True)
plt.legend()
plt.show()

plt.plot(VGS, gm, marker='o', linestyle='-', color='r')

plt.scatter(
    max_gm_vgs,
    max_gm,
    s=100,
    zorder=5,
    label=f"Max gm = {max_gm:.6f} S"
)

plt.xlabel("VGS (V)")
plt.ylabel("Transconductance gm (S)")
plt.title("MOSFET Transconductance")
plt.grid(True)
plt.legend()
plt.show()

# Threshold extraction using maximum-gm linear extrapolation

# Select points around maximum gm
start = max(0, max_gm_index - 1)
end = min(len(VGS), max_gm_index + 2)

VGS_fit = VGS.iloc[start:end]
ID_fit = ID.iloc[start:end]

# Fit a straight line: ID = m*VGS + b
m, b = np.polyfit(VGS_fit, ID_fit, 1)

# Find x-intercept where ID = 0
VTH_gm = -b / m

print(f"\nVTH from constant-current method: {VTH:.4f} V")
print(f"VTH from max-gm method: {VTH_gm:.4f} V")

VGS_line = np.linspace(VTH_gm, VGS_fit.max(), 100)
ID_line = m * VGS_line + b

plt.plot(VGS, ID, marker='o', label="Measured ID")
plt.plot(VGS_line, ID_line, linestyle="--", label="Max-gm tangent")

plt.scatter(
    VTH_gm,
    0,
    s=100,
    label=f"VTH = {VTH_gm:.3f} V"
)

plt.xlabel("VGS (V)")
plt.ylabel("ID (A)")
plt.title("VTH Extraction Using Max-gm Method")
plt.grid(True)
plt.legend()
plt.show()

sqrt_ID = np.sqrt(ID)

# Use strong-inversion portion of the curve
mask = VGS >= 2.5

VGS_fit = VGS[mask]
sqrt_ID_fit = sqrt_ID[mask]

m, b = np.polyfit(VGS_fit, sqrt_ID_fit, 1)

VTH_sqrt = -b / m

print(f"VTH using sqrt(ID) method: {VTH_sqrt:.4f} V")

# Plot
VGS_line = np.linspace(VTH_sqrt, VGS_fit.max(), 100)
sqrt_ID_line = m * VGS_line + b

plt.plot(VGS, sqrt_ID, marker='o', label="sqrt(ID)")
plt.plot(VGS_line, sqrt_ID_line, linestyle="--", label="Linear fit")

plt.scatter(
    VTH_sqrt,
    0,
    s=100,
    label=f"VTH = {VTH_sqrt:.3f} V"
)

plt.xlabel("VGS (V)")
plt.ylabel("sqrt(ID)")
plt.title("VTH Extraction Using sqrt(ID) Method")
plt.grid(True)
plt.legend()
plt.show()