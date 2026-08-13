import matplotlib

# matplotlib 解析 .ttc 字型集合檔時只認得到 "Noto Sans CJK JP" 這個名字
# (即使 fc-list 看得到 TC/SC 等變體),所以這裡指定 JP 變體,繁體字仍可正常顯示。
matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
