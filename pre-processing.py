
import pandas as pd
import numpy as np

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


def judul(teks):
    print("\n" + "=" * 70)
    print(teks)
    print("=" * 70)


# ---------------------------------------------------------------
# TAHAP 1 - Membaca data
# ---------------------------------------------------------------
judul("TAHAP 1 - Membaca data")
df_raw = pd.read_csv("Expanded_Dataset.csv")
print("Ukuran data awal (baris, kolom):", df_raw.shape)
print(df_raw.head())

# ---------------------------------------------------------------
# TAHAP 2 - Mengenal dataset (untuk Tabel 3.1)
# ---------------------------------------------------------------
judul("TAHAP 2 - Atribut dan tipe data (Tabel 3.1)")
print(df_raw.dtypes)
print("\nTahun  :", sorted(df_raw["Year"].unique()))
print("Kuartal:", sorted(df_raw["Quarter"].unique()))
print("Wilayah:", sorted(df_raw["Region"].unique()))
print("Jumlah model perangkat:", df_raw["Product Model"].nunique())
print("\nStatistik deskriptif:")
print(df_raw.describe().round(2))

# ---------------------------------------------------------------
# TAHAP 3 - Pembersihan data (3.2.1, Tabel 3.2)
# ---------------------------------------------------------------
judul("TAHAP 3a - Cek nilai kosong")
print(df_raw.isna().sum())

judul("TAHAP 3b - Cek dan hapus duplikat penuh")
jml_duplikat = df_raw.duplicated().sum()
print("Jumlah baris duplikat penuh:", jml_duplikat)
ulang = df_raw.groupby(list(df_raw.columns)).size()
print("Berapa kali tiap baris unik muncul (kali: jumlah baris unik):")
print(ulang.value_counts().sort_index().to_string())
df = df_raw.drop_duplicates(keep="first").reset_index(drop=True)
print("Ukuran setelah hapus duplikat:", df.shape)

kombinasi = df.groupby(["Year", "Quarter", "Product Model"]).size()
print(f"Kombinasi Year-Quarter-Model: {len(kombinasi)} "
      f"(= {df['Year'].nunique()} tahun x {df['Quarter'].nunique()} kuartal x "
      f"{df['Product Model'].nunique()} model), masing-masing muncul {kombinasi.max()} kali")

judul("TAHAP 3c - Cek konsistensi label 5G dengan nama model")
nama_ada_5g = df["Product Model"].str.contains("5G")
label_5g = df["5G Capability"].eq("Yes")
print("Jumlah label tidak konsisten:", (nama_ada_5g != label_5g).sum())
print(pd.crosstab(df["Product Model"], df["5G Capability"]))

judul("TAHAP 3d - Cek nilai tidak masuk akal -> diubah jadi kosong (NaN)")
aturan = {
    "Market Share (%)": lambda s: ~s.between(0, 100),
    "Regional 5G Coverage (%)": lambda s: ~s.between(0, 100),
    "5G Subscribers (millions)": lambda s: s < 0,
}
for kolom, cek in aturan.items():
    awal = cek(df_raw[kolom]).sum()
    unik = cek(df[kolom]).sum()
    print(f"{kolom:28s} | data awal: {awal:3d} record | data unik: {unik:3d} nilai -> NaN")
    df.loc[cek(df[kolom]), kolom] = np.nan

print("\nNilai kosong setelah pembersihan:")
print(df.isna().sum())

judul("TAHAP 3e - Cek outlier Units Sold dan Revenue (metode IQR, hanya ditandai)")
for kolom in ["Units Sold", "Revenue ($)"]:
    q1, q3 = df[kolom].quantile([0.25, 0.75])
    batas_bawah, batas_atas = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    jml = (~df[kolom].between(batas_bawah, batas_atas)).sum()
    print(f"{kolom:12s} | batas IQR {batas_bawah:,.0f} s.d. {batas_atas:,.0f} | outlier: {jml}")
print("Outlier tidak dihapus karena nilainya masih positif dan mungkin terjadi.")

judul("TAHAP 3f - Sebaran record per wilayah")
per_wilayah = df.groupby("Region").agg(Jumlah_record=("Units Sold", "size"),
                                        Rata_unit_per_record=("Units Sold", "mean"))
print(per_wilayah.round(0))
print("Jumlah record per wilayah tidak sama, jadi total unit per wilayah")
print("sebaiknya dibaca bersama rata-rata unit per record.")

# ---------------------------------------------------------------
# TAHAP 4 - Transformasi data (3.2.2)
# ---------------------------------------------------------------
judul("TAHAP 4 - Transformasi data")
df["Year"] = df["Year"].astype(int)
df["Quarter"] = pd.Categorical(df["Quarter"], categories=["Q1", "Q2", "Q3", "Q4"], ordered=True)
for kolom in ["Product Model", "Region", "5G Capability", "Quarter"]:
    df[kolom] = df[kolom].astype(str).str.strip()
print("Label 5G Capability:", df["5G Capability"].unique())

# ---------------------------------------------------------------
# TAHAP 5 - Penggabungan data (3.2.3)
# ---------------------------------------------------------------
judul("TAHAP 5 - Penggabungan data")
print("Tidak ada join dengan dataset lain: semua atribut sudah ada di satu file.")
print("Unit dan pendapatan diringkas sekaligus dalam satu groupby (lihat Tahap 7).")

# ---------------------------------------------------------------
# TAHAP 6 - Field tambahan (3.2.4, Tabel 3.3)
# ---------------------------------------------------------------
judul("TAHAP 6 - Membuat field tambahan")
df["Period"] = df["Year"].astype(str) + "-" + df["Quarter"]
df["Kategori 5G"] = df["5G Capability"].map({"Yes": "5G", "No": "Non 5G"})
df["Revenue (Miliar US$)"] = df["Revenue ($)"] / 1e9
df["Revenue per Unit ($)"] = df["Revenue ($)"] / df["Units Sold"]
median_pref = df["Preference for 5G (%)"].median()
df["Kategori Preferensi"] = np.where(df["Preference for 5G (%)"] >= median_pref, "Tinggi", "Rendah")
print(f"Kategori Preferensi: Tinggi jika Preference >= median ({median_pref:.2f}%)")
print(df[["Period", "Kategori 5G", "Revenue (Miliar US$)", "Revenue per Unit ($)", "Kategori Preferensi"]].head())

# ---------------------------------------------------------------
# TAHAP 7 - Ringkasan angka untuk dicocokkan dengan Tableau
# ---------------------------------------------------------------
judul("TAHAP 7 - Ringkasan tahunan (Tabel 3.4)")
tahunan = df.groupby("Year").agg(Unit=("Units Sold", "sum"), Pendapatan=("Revenue ($)", "sum"))
tahunan["Pendapatan (miliar US$)"] = (tahunan["Pendapatan"] / 1e9).round(2)
tahunan["Pertumbuhan unit (%)"] = (tahunan["Unit"].pct_change() * 100).round(2)
print(tahunan.drop(columns="Pendapatan"))

total_unit = df["Units Sold"].sum()
print(f"\nTotal unit       : {total_unit:,}")
print(f"Total pendapatan : US$ {df['Revenue ($)'].sum() / 1e9:.2f} miliar")

judul("Unit per wilayah")
print(df.groupby("Region")["Units Sold"].sum().sort_values(ascending=False))

judul("Lima model terlaris")
print(df.groupby("Product Model")["Units Sold"].sum().nlargest(5))

judul("Proporsi 5G vs Non 5G (% dari total unit)")
print((df.groupby("Kategori 5G")["Units Sold"].sum() / total_unit * 100).round(2))

judul("Kontribusi 5G per tahun (%)")
pv = df.pivot_table(index="Year", columns="Kategori 5G", values="Units Sold", aggfunc="sum")
print((pv.div(pv.sum(axis=1), axis=0) * 100).round(2))

judul("Korelasi Pearson unit terjual dengan indikator 5G")
for kolom in ["Regional 5G Coverage (%)", "Avg 5G Speed (Mbps)", "Preference for 5G (%)"]:
    d = df[[kolom, "Units Sold"]].dropna()
    print(f"{kolom:28s} r = {d[kolom].corr(d['Units Sold']):.3f}  (n = {len(d)})")

# ---------------------------------------------------------------
# TAHAP 8 - Simpan data bersih untuk Tableau
# ---------------------------------------------------------------
judul("TAHAP 8 - Simpan data bersih")
df.to_csv("Samsung_Sales_Clean.csv", index=False)
print("Tersimpan: Samsung_Sales_Clean.csv  ->", df.shape)
print("Buka file ini di Tableau (Connect > Text file).")