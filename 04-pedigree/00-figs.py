from matplotlib import font_manager

font_dirs = ["/home/hblee/.conda/envs/navi/fonts"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['arial']
plt.rcParams["mathtext.fontset"] = 'cm'

if __name__ == "__main__":
    pass
