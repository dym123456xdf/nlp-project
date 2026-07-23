"""跑 dm03_文本分析.py 里全部 5 个 todo, 把图存成 PNG, 不弹窗."""
import os, sys, warnings, logging

# 静默 jieba 的 pkg_resources deprecation 警告 (Python warnings 渠道).
warnings.filterwarnings('ignore', message=r'.*pkg_resources.*')

# 静默 matplotlib.font_manager 的 findfont 警告 (logging 渠道, 不是 warnings).
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# 1. 后端改 Agg, 不弹窗.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 2. 输出目录.
HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'output', 'dm03')
os.makedirs(OUT, exist_ok=True)

# 2b. 显式注册 SimHei 字体 (mac 上 rcParams 光写名字找不到, 必须 addfont).
SIMHEI = os.path.join(HERE, 'data', 'SimHei.ttf')
if os.path.exists(SIMHEI):
    fm.fontManager.addfont(SIMHEI)
    simhei_name = fm.FontProperties(fname=SIMHEI).get_name()
    plt.rcParams['font.sans-serif'] = [simhei_name, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print(f'[font] registered: {simhei_name}')
else:
    print('[font] WARNING: SimHei.ttf not found, 中文会乱码')

# 3. monkey-patch plt.show() -> savefig + close.
_show_idx = {'n': 0}
_orig_show = plt.show
def _patched_show(*a, **kw):
    fig = plt.gcf()
    _show_idx['n'] += 1
    p = os.path.join(OUT, f'{_show_idx["n"]:02d}.png')
    fig.savefig(p, dpi=120, bbox_inches='tight')
    print(f'  -> saved {p}')
    plt.close(fig)
plt.show = _patched_show

# 4. 切到 day02 让相对路径 (./data/...) 生效.
os.chdir(HERE)
sys.path.insert(0, HERE)

# 5. 导入原文件所有 todo 函数.
import dm03_文本分析 as dm03

print('=' * 60)
print('[1/5] dm01_label_sns_countplot  -- 标签分布')
dm03.dm01_label_sns_countplot()

print('[2/5] dm02_len_sns_distplot     -- 句子长度分布')
dm03.dm02_len_sns_distplot()

print('[3/5] dm03_sns_stripplot        -- 正负样本长度散点')
dm03.dm03_sns_stripplot()

print('[4/5] dm04_get_word_count       -- 词汇去重统计')
dm03.dm04_get_word_count()

print('[5/5] dm05_word_cloud           -- 高频形容词词云')
dm03.dm05_word_cloud()

plt.show = _orig_show
print('=' * 60)
print(f'全部完成. 输出目录: {OUT}')
