import os
import warnings

import matplotlib
matplotlib.use("Agg")  # khong can hien thi man hinh, chi luu file
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

#CAU HINH CHUNG

CSV_PATH = "cleaned.csv"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.unicode_minus"] = False

# Mau sac dong nhat cho toan bo bieu do
PALETTE = "viridis"


def section(title):
    """In tieu de section cho de doc ket qua tren terminal."""
    print("\n" + title)


def save_fig(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Da luu hinh: {path}")


def billion_formatter(x, pos):
    return f"{x:,.0f}"



# 1. DOC VA TONG QUAN DU LIEU
section("1. TONG QUAN DU LIEU")

df = pd.read_csv(CSV_PATH)
print(f"So dong (listing): {df.shape[0]:,}")
print(f"So cot: {df.shape[1]}")

print("\nKieu du lieu cac cot quan trong:")
key_cols = [
    "price_billion", "area_m2", "price_per_m2", "bedroom", "toilet",
    "city_norm", "district_norm", "category", "legal_status_norm",
    "house_type", "floors", "publish_date",
]
print(df[key_cols].dtypes)

print("\nTy le missing values cac cot quan trong (%):")
missing_pct = (df[key_cols].isnull().sum() / len(df) * 100).round(2)
print(missing_pct[missing_pct > 0].sort_values(ascending=False))


# 2. LAM SACH / XU LY OUTLIER CO BAN
section("2. XU LY DU LIEU TRUOC KHI PHAN TICH")

before = len(df)

# Loc cac gia tri bat hop ly ro ret (gia <=0, dien tich <=0)
df = df[(df["price_vnd"] > 0) & (df["area_m2"] > 0)].copy()

# Dung IQR de xac dinh va loai outlier cuc tri cho gia va dien tich
# (giu lai du lieu goc cho cac phan tich khac, chi tao ban "df_clean"
#  rieng de ve bieu do phan phoi gia/dien tich khong bi outlier lam meo)
def iqr_bounds(series, k=3.0):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

low_p, high_p = iqr_bounds(df["price_billion"])
low_a, high_a = iqr_bounds(df["area_m2"])

df_clean = df[
    (df["price_billion"].between(max(low_p, 0.05), high_p))
    & (df["area_m2"].between(max(low_a, 5), high_a))
].copy()

print(f"So dong ban dau: {before:,}")
print(f"So dong sau loc gia/dien tich <=0: {len(df):,}")
print(f"Nguong gia hop ly (IQR x3): {max(low_p,0.05):.2f} - {high_p:.2f} ty VND")
print(f"Nguong dien tich hop ly (IQR x3): {max(low_a,5):.0f} - {high_a:.0f} m2")
print(f"So dong sau loai outlier cuc tri (dung cho bieu do phan phoi): {len(df_clean):,}"
      f"  (loai {len(df) - len(df_clean):,} dong, {(len(df)-len(df_clean))/len(df)*100:.1f}%)")

print("\n*Ghi chu: df_clean chi dung de ve bieu do phan phoi cho de nhin.")
print(" Cac thong ke tong/group-by van dung df day du (chi loai gia/dien tich <=0).")


# 3. THONG KE MO TA GIA & DIEN TICH
section("3. THONG KE MO TA: GIA & DIEN TICH")

desc_cols = ["price_billion", "area_m2", "price_per_m2", "bedroom", "toilet"]
desc = df[desc_cols].describe().T
desc["median"] = df[desc_cols].median()
print(desc.round(2))

print("\nGia trung binh (ty VND):", round(df["price_billion"].mean(), 2))
print("Gia trung vi  (ty VND):", round(df["price_billion"].median(), 2))
print("Gia/m2 trung binh:", f"{df['price_per_m2'].mean():,.0f} VND/m2")
print("Gia/m2 trung vi  :", f"{df['price_per_m2'].median():,.0f} VND/m2")

# Bieu do 1: Phan phoi gia
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(df_clean["price_billion"], bins=50, kde=True, color="#3b6e8f", ax=axes[0])
axes[0].set_title("Phan phoi gia BDS (ty VND)")
axes[0].set_xlabel("Gia (ty VND)")
axes[0].set_ylabel("So luong tin")

sns.histplot(np.log1p(df_clean["price_billion"]), bins=50, kde=True, color="#8f3b6e", ax=axes[1])
axes[1].set_title("Phan phoi gia (thang log) - de thay ro hinh dang")
axes[1].set_xlabel("log(1 + Gia ty VND)")
axes[1].set_ylabel("So luong tin")
save_fig(fig, "01_phan_phoi_gia.png")

# Bieu do 2: Phan phoi dien tich
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_clean["area_m2"], bins=50, kde=True, color="#3b8f6e", ax=ax)
ax.set_title("Phan phoi dien tich BDS (m2)")
ax.set_xlabel("Dien tich (m2)")
ax.set_ylabel("So luong tin")
save_fig(fig, "02_phan_phoi_dien_tich.png")

# Bieu do 3: Phan phoi gia/m2
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_clean["price_per_m2"] / 1e6, bins=50, kde=True, color="#c97b2e", ax=ax)
ax.set_title("Phan phoi gia tren m2 (trieu VND/m2)")
ax.set_xlabel("Trieu VND / m2")
ax.set_ylabel("So luong tin")
save_fig(fig, "03_phan_phoi_gia_per_m2.png")


# 4. PHAN TICH THEO VI TRI (THANH PHO / QUAN HUYEN)
section("4. PHAN TICH THEO VI TRI")

city_stats = df.groupby("city_norm").agg(
    so_tin=("price_vnd", "count"),
    gia_tb_ty=("price_billion", "mean"),
    gia_trungvi_ty=("price_billion", "median"),
    gia_per_m2_tb=("price_per_m2", "mean"),
    dien_tich_tb=("area_m2", "mean"),
).round(2).sort_values("so_tin", ascending=False)
print("\nThong ke theo Thanh pho:")
print(city_stats)

top_n = 15
district_stats = df.groupby(["city_norm", "district_norm"]).agg(
    so_tin=("price_vnd", "count"),
    gia_tb_ty=("price_billion", "mean"),
    gia_per_m2_tb=("price_per_m2", "mean"),
).round(2)
district_stats = district_stats[district_stats["so_tin"] >= 30]

top_volume = district_stats.sort_values("so_tin", ascending=False).head(top_n)
print(f"\nTop {top_n} Quan/Huyen co nhieu tin rao ban nhat (>=30 tin):")
print(top_volume)

top_price = district_stats.sort_values("gia_per_m2_tb", ascending=False).head(top_n)
print(f"\nTop {top_n} Quan/Huyen co gia/m2 trung binh cao nhat (>=30 tin):")
print(top_price)

# Bieu do 4: So luong tin theo thanh pho
fig, ax = plt.subplots(figsize=(6, 5))
sns.barplot(x=city_stats.index, y=city_stats["so_tin"], hue=city_stats.index,
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("So luong tin rao ban theo Thanh pho")
ax.set_xlabel("")
ax.set_ylabel("So luong tin")
save_fig(fig, "04_so_tin_theo_thanhpho.png")

# Bieu do 5: Top quan/huyen nhieu tin nhat
fig, ax = plt.subplots(figsize=(9, 7))
plot_data = top_volume.reset_index().sort_values("so_tin")
sns.barplot(data=plot_data, x="so_tin", y="district_norm", hue="city_norm",
            dodge=False, palette="Set2", ax=ax)
ax.set_title(f"Top {top_n} Quan/Huyen co nhieu tin rao ban nhat")
ax.set_xlabel("So luong tin")
ax.set_ylabel("Quan/Huyen")
ax.legend(title="Thanh pho")
save_fig(fig, "05_top_quan_nhieu_tin.png")

# Bieu do 6: Top quan/huyen gia/m2 cao nhat
fig, ax = plt.subplots(figsize=(9, 7))
plot_data = top_price.reset_index().sort_values("gia_per_m2_tb")
plot_data["gia_per_m2_trieu"] = plot_data["gia_per_m2_tb"] / 1e6
sns.barplot(data=plot_data, x="gia_per_m2_trieu", y="district_norm", hue="city_norm",
            dodge=False, palette="Set2", ax=ax)
ax.set_title(f"Top {top_n} Quan/Huyen co gia/m2 trung binh cao nhat")
ax.set_xlabel("Gia/m2 trung binh (trieu VND/m2)")
ax.set_ylabel("Quan/Huyen")
ax.legend(title="Thanh pho")
save_fig(fig, "06_top_quan_gia_cao.png")

# Bieu do 7: So sanh gia giua 2 thanh pho (boxplot)
fig, ax = plt.subplots(figsize=(8, 5))
plot_df = df_clean[df_clean["city_norm"].isin(["Ho_Chi_Minh", "Ha_Noi"])]
sns.boxplot(data=plot_df, x="city_norm", y="price_billion", hue="city_norm",
            palette="Set2", legend=False, ax=ax)
ax.set_title("So sanh phan phoi gia giua Ha Noi va TP.HCM")
ax.set_xlabel("")
ax.set_ylabel("Gia (ty VND)")
save_fig(fig, "07_so_sanh_gia_2_thanhpho.png")


# 5. PHAN TICH THEO LOAI BAT DONG SAN (category)
section("5. PHAN TICH THEO LOAI BAT DONG SAN")

cat_stats = df.groupby("category").agg(
    so_tin=("price_vnd", "count"),
    gia_tb_ty=("price_billion", "mean"),
    gia_trungvi_ty=("price_billion", "median"),
    dien_tich_tb=("area_m2", "mean"),
    gia_per_m2_tb=("price_per_m2", "mean"),
).round(2).sort_values("so_tin", ascending=False)
print(cat_stats)

# Bieu do 8: Ty trong loai BDS
fig, ax = plt.subplots(figsize=(7, 7))
colors = sns.color_palette(PALETTE, n_colors=len(cat_stats))
ax.pie(
    cat_stats["so_tin"], labels=cat_stats.index, autopct="%1.1f%%",
    colors=colors, startangle=90, textprops={"fontsize": 9},
)
ax.set_title("Ty trong cac loai Bat dong san")
save_fig(fig, "08_ty_trong_loai_bds.png")

# Bieu do 9: Gia trung binh theo loai BDS
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = cat_stats.reset_index().sort_values("gia_tb_ty")
sns.barplot(data=plot_data, x="gia_tb_ty", y="category", hue="category",
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("Gia trung binh theo loai Bat dong san")
ax.set_xlabel("Gia trung binh (ty VND)")
ax.set_ylabel("")
save_fig(fig, "09_gia_tb_theo_loai_bds.png")

# Bieu do 10: Boxplot gia theo loai BDS
fig, ax = plt.subplots(figsize=(9, 6))
order = cat_stats.index.tolist()
sns.boxplot(data=df_clean, x="category", y="price_billion", order=order,
            hue="category", palette="Set3", legend=False, ax=ax)
ax.set_title("Phan phoi gia theo loai Bat dong san")
ax.set_xlabel("")
ax.set_ylabel("Gia (ty VND)")
ax.tick_params(axis="x", rotation=20)
save_fig(fig, "10_boxplot_gia_theo_loai.png")


# 6. PHAN TICH SO PHONG NGU / TOILET / TANG
section("6. PHAN TICH SO PHONG NGU, TOILET, SO TANG")

bedroom_stats = df.groupby("bedroom").agg(
    so_tin=("price_vnd", "count"),
    gia_tb_ty=("price_billion", "mean"),
).round(2)
print("Theo so phong ngu:")
print(bedroom_stats)

print("\nSo tang (floors) - thong ke (chi BDS dang nha/biet thu co thong tin tang):")
print(df["floors"].dropna().describe().round(2))

# Bieu do 11: Gia trung binh theo so phong ngu
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = bedroom_stats.reset_index()
plot_data = plot_data[plot_data["so_tin"] >= 10]
sns.barplot(data=plot_data, x="bedroom", y="gia_tb_ty", hue="bedroom",
            palette=PALETTE, legend=False, ax=ax)
ax.set_title("Gia trung binh theo so phong ngu")
ax.set_xlabel("So phong ngu")
ax.set_ylabel("Gia trung binh (ty VND)")
save_fig(fig, "11_gia_theo_so_phongngu.png")


# 7. MOI QUAN HE GIUA DIEN TICH VA GIA
section("7. MOI QUAN HE DIEN TICH - GIA")

corr = df[["price_billion", "area_m2", "price_per_m2", "bedroom", "toilet"]].corr()
print("Ma tran tuong quan (Pearson):")
print(corr.round(3))

corr_price_area = df["price_billion"].corr(df["area_m2"])
print(f"\n=> He so tuong quan Gia & Dien tich: {corr_price_area:.3f}")

# Bieu do 12: Scatter dien tich vs gia (mau theo thanh pho)
fig, ax = plt.subplots(figsize=(9, 6))
plot_df = df_clean[df_clean["city_norm"].isin(["Ho_Chi_Minh", "Ha_Noi"])]
sns.scatterplot(
    data=plot_df.sample(min(4000, len(plot_df)), random_state=42),
    x="area_m2", y="price_billion", hue="city_norm", alpha=0.4, s=18,
    palette="Set2", ax=ax,
)
ax.set_title("Quan he giua Dien tich va Gia BDS")
ax.set_xlabel("Dien tich (m2)")
ax.set_ylabel("Gia (ty VND)")
save_fig(fig, "12_scatter_dientich_gia.png")

# Bieu do 13: Heatmap tuong quan
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Heatmap tuong quan cac bien so")
save_fig(fig, "13_heatmap_tuongquan.png")


# 8. PHAP LY VA ANH HUONG DEN GIA
section("8. ANH HUONG CUA TINH TRANG PHAP LY DEN GIA")

legal_stats = df[df["legal_status_norm"] != "Unknown"].groupby("legal_status_norm").agg(
    so_tin=("price_vnd", "count"),
    gia_per_m2_tb=("price_per_m2", "mean"),
    gia_tb_ty=("price_billion", "mean"),
).round(2).sort_values("so_tin", ascending=False)
print("(Luu y: legal_status la ma code trong du lieu goc, khong co tu dien giai thich kem theo)")
print(legal_stats)

# Bieu do 14: Gia/m2 theo tinh trang phap ly
fig, ax = plt.subplots(figsize=(8, 5))
plot_data = legal_stats.reset_index()
plot_data["gia_per_m2_trieu"] = plot_data["gia_per_m2_tb"] / 1e6
sns.barplot(data=plot_data, x="legal_status_norm", y="gia_per_m2_trieu",
            hue="legal_status_norm", palette=PALETTE, legend=False, ax=ax)
ax.set_title("Gia/m2 trung binh theo ma tinh trang phap ly")
ax.set_xlabel("Ma tinh trang phap ly")
ax.set_ylabel("Gia/m2 (trieu VND/m2)")
save_fig(fig, "14_gia_theo_phaply.png")


# 9. XU HUONG THEO THOI GIAN DANG TIN
section("9. XU HUONG THEO THOI GIAN DANG TIN")

df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
date_range = df["publish_date"].dropna()
print(f"Khoang thoi gian dang tin: {date_range.min().date()} -> {date_range.max().date()}")

daily_counts = df.groupby(df["publish_date"].dt.date).size()
print(f"So ngay co du lieu: {daily_counts.shape[0]}")
print(f"Trung binh so tin/ngay: {daily_counts.mean():.1f}")

fig, ax = plt.subplots(figsize=(10, 5))
daily_counts.sort_index().plot(ax=ax, color="#3b6e8f", marker="o", markersize=3)
ax.set_title("So luong tin dang theo ngay")
ax.set_xlabel("Ngay dang tin")
ax.set_ylabel("So luong tin")
fig.autofmt_xdate(rotation=30)
save_fig(fig, "15_xuhuong_dangtin_theongay.png")


# 10. TONG HOP INSIGHT CHINH
section("10. TONG HOP INSIGHT CHINH (KET LUAN)")

top_city = city_stats["so_tin"].idxmax()
top_expensive_district = top_price.iloc[0]
top_volume_district = top_volume.iloc[0]
most_common_category = cat_stats["so_tin"].idxmax()
most_expensive_category = cat_stats["gia_tb_ty"].idxmax()
most_common_bedroom = bedroom_stats["so_tin"].idxmax()

insights = [
    f"1. Tong cong co {len(df):,} tin rao ban BDS, tap trung tai 2 thanh pho lon: "
    f"{city_stats.index[0]} ({city_stats['so_tin'].iloc[0]:,} tin) va "
    f"{city_stats.index[1]} ({city_stats['so_tin'].iloc[1]:,} tin).",

    f"2. Gia BDS trung vi toan thi truong la {df['price_billion'].median():.2f} ty VND, "
    f"nhung phan phoi gia bi lech phai manh (mean {df['price_billion'].mean():.2f} ty > median), "
    f"cho thay co mot so it BDS gia rat cao keo gia trung binh len.",

    f"3. Loai BDS pho bien nhat la '{most_common_category}' ({cat_stats.loc[most_common_category,'so_tin']:,} tin, "
    f"{cat_stats.loc[most_common_category,'so_tin']/len(df)*100:.1f}%), trong khi loai co gia trung binh cao nhat la "
    f"'{most_expensive_category}' ({cat_stats.loc[most_expensive_category,'gia_tb_ty']:.2f} ty VND).",

    f"4. Quan/Huyen co gia/m2 trung binh cao nhat (trong nhom >=30 tin) la "
    f"'{top_expensive_district.name[1]}' ({top_expensive_district.name[0]}) voi "
    f"{top_expensive_district['gia_per_m2_tb']/1e6:.1f} trieu VND/m2.",

    f"5. Quan/Huyen co nguon cung (so tin rao ban) lon nhat la "
    f"'{top_volume_district.name[1]}' ({top_volume_district.name[0]}) voi {top_volume_district['so_tin']:.0f} tin.",

    f"6. Tuong quan giua Dien tich va Gia la {corr_price_area:.2f} - "
    f"{'tuong quan duong vua phai, dien tich lon hon co xu huong gia cao hon nhung khong tuyet doi tuyen tinh' if 0.2 < corr_price_area < 0.7 else 'muc do tuong quan dang chu y, can xem xet them cac yeu to khac (vi tri, phap ly...)'}.",

    f"7. Loai cau hinh phong ngu pho bien nhat la {most_common_bedroom:.0f} phong ngu "
    f"({bedroom_stats.loc[most_common_bedroom,'so_tin']:,} tin).",

    f"8. Mot so cot quan trong bi thieu du lieu kha nhieu: huong nha (direction) thieu "
    f"{df['direction'].isnull().sum()/len(df)*100:.0f}%, so tang (floors) thieu "
    f"{df['floors'].isnull().sum()/len(df)*100:.0f}%, dien tich su dung (living_size) thieu "
    f"{df['living_size'].isnull().sum()/len(df)*100:.0f}% - can luu y khi phan tich sau hoac thu thap them.",
]

for line in insights:
    print(line)

# Luu insight ra file text 
with open(os.path.join(OUTPUT_DIR, "insights_summary.txt"), "w", encoding="utf-8") as f:
    f.write("TONG HOP INSIGHT - EDA DU LIEU BAT DONG SAN\n")
    f.write("=" * 60 + "\n\n")
    for line in insights:
        f.write(line + "\n\n")

print(f"\n  -> Da luu file tong hop insight: {os.path.join(OUTPUT_DIR, 'insights_summary.txt')}")

section("HOAN THANH EDA")
print(f"Tat ca hinh anh da duoc luu trong thu muc: ./{OUTPUT_DIR}/")
