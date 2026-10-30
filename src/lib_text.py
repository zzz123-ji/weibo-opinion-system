# -*- coding: utf-8 -*-
"""中文文本清洗、分词、话题标签、省份词典。

设计原则：
- 清洗只做“去噪与匿名化”，不改写语义；
- 话题/关键词只使用公开帖文正文，不涉及账号信息；
- 省份统计是“文本提及”，不是用户定位。
"""
from __future__ import annotations

import logging
import re
import unicodedata

import jieba
import jieba.posseg as pseg

jieba.setLogLevel(logging.WARNING)

# 微博常见表情 -> 情感词（保留情绪信号，避免直接删掉）
EMOJI_MAP = {
    "泪": "难过", "允悲": "无奈", "悲伤": "悲伤", "哭": "难过", "伤心": "伤心",
    "哈哈": "开心", "笑cry": "好笑", "偷笑": "开心", "太开心": "开心", "嘻嘻": "开心",
    "怒": "生气", "愤怒": "愤怒", "费解": "困惑", "疑问": "疑问", "晕": "无语",
    "赞": "支持", "good": "支持", "心": "喜欢", "爱你": "喜欢", "抱抱": "安慰",
    "生病": "生病", "衰": "倒霉", "可怜": "可怜", "白眼": "不满", "鄙视": "不满",
    "加油": "鼓励", "拳头": "鼓励", "握手": "友好", "作揖": "感谢",
}
_EMOJI_RE = re.compile(r"\[([^\[\]]{1,8})\]")

STOPWORDS = set("""
的 了 是 在 我 有 和 就 不 人 都 一 一个 上 也 很 到 说 要 去 你 会 着 没有 看 好
自己 这 那 他 她 它 我们 你们 他们 她们 它们 这个 那个 这些 那些 这样 那样 什么 怎么
为什么 哪 哪个 哪里 谁 多少 几 很 太 更 最 非常 特别 比较 还是 或者 而且 但是 可是
因为 所以 如果 虽然 然后 已经 可以 可能 应该 需要 觉得 感觉 知道 现在 今天 昨天 明天
时候 时间 一下 一点 一些 一直 一定 一起 一样 这么 那么 只是 就是 还有 没 别 再 又
让 被 把 给 对 从 向 往 于 与 及 或 等 等等 之 其 此 该 各 每 某 另 另外 其中
啊 呀 吧 呢 哦 噢 喔 嗯 唉 哎 哈 哈哈哈 呵呵 嘿 喂 哇 嘛 啦 嘞 咯 嘞 么 呗 咯
微博 转发 评论 点赞 分享 网页 链接 展开 全文 图片 视频 原文 来源 客户端 话题 超话
显示 地图 地址 位置 全文 c 网页链接 新浪 新浪微博 微博视频 微博图片 秒拍 一直播
""".split())

# 仅保留有信息量的词性
KEEP_POS = {"n", "nr", "ns", "nt", "nz", "ng", "vn", "v", "vd", "a", "an", "ad", "i", "l"}
# 情感关键词排除人名/地名/机构名/专名，减少“明星名/地名”对情感词的干扰
KEYWORD_POS = {"n", "ng", "vn", "v", "vd", "a", "an", "ad", "i", "l"}

PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "广西", "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西",
    "甘肃", "青海", "宁夏", "新疆", "香港", "澳门", "台湾",
]

# 城市/别名 -> 省级行政区（用于“提及省份”聚合；不是用户定位）
CITY_TO_PROVINCE = {
    "武汉": "湖北", "孝感": "湖北", "黄冈": "湖北", "荆州": "湖北", "宜昌": "湖北",
    "襄阳": "湖北", "随州": "湖北", "十堰": "湖北", "恩施": "湖北",
    "广州": "广东", "深圳": "广东", "珠海": "广东", "佛山": "广东", "东莞": "广东",
    "杭州": "浙江", "温州": "浙江", "宁波": "浙江", "台州": "浙江", "金华": "浙江",
    "成都": "四川", "绵阳": "四川", "南充": "四川", "达州": "四川",
    "长沙": "湖南", "岳阳": "湖南", "邵阳": "湖南", "常德": "湖南",
    "郑州": "河南", "信阳": "河南", "南阳": "河南", "周口": "河南", "驻马店": "河南",
    "西安": "陕西", "咸阳": "陕西", "宝鸡": "陕西",
    "南京": "江苏", "苏州": "江苏", "无锡": "江苏", "常州": "江苏", "徐州": "江苏",
    "合肥": "安徽", "芜湖": "安徽", "阜阳": "安徽", "安庆": "安徽",
    "福州": "福建", "厦门": "福建", "泉州": "福建", "莆田": "福建",
    "济南": "山东", "青岛": "山东", "临沂": "山东", "潍坊": "山东", "烟台": "山东",
    "哈尔滨": "黑龙江", "齐齐哈尔": "黑龙江", "大庆": "黑龙江",
    "长春": "吉林", "吉林市": "吉林", "延边": "吉林",
    "沈阳": "辽宁", "大连": "辽宁", "鞍山": "辽宁",
    "石家庄": "河北", "唐山": "河北", "保定": "河北", "邯郸": "河北", "邢台": "河北",
    "太原": "山西", "大同": "山西", "运城": "山西",
    "南昌": "江西", "九江": "江西", "上饶": "江西",
    "昆明": "云南", "大理": "云南", "丽江": "云南",
    "贵阳": "贵州", "遵义": "贵州",
    "南宁": "广西", "桂林": "广西", "柳州": "广西",
    "海口": "海南", "三亚": "海南",
    "兰州": "甘肃", "天水": "甘肃",
    "西宁": "青海",
    "银川": "宁夏",
    "乌鲁木齐": "新疆", "喀什": "新疆",
    "拉萨": "西藏",
    "呼和浩特": "内蒙古", "包头": "内蒙古",
    "天津": "天津", "北京": "北京", "上海": "上海", "重庆": "重庆",
    "香港": "香港", "澳门": "澳门", "台北": "台湾", "台湾": "台湾",
}

PROVINCE_ALIASES = {}
for _p in PROVINCES:
    PROVINCE_ALIASES[_p] = _p
    PROVINCE_ALIASES[_p + "市"] = _p
    PROVINCE_ALIASES[_p + "省"] = _p
    PROVINCE_ALIASES[_p + "自治区"] = _p
PROVINCE_ALIASES.update({
    # 只保留低歧义别名；单字简称（如“新”“川”“吉”）误报率高，不用于统计
    "内蒙": "内蒙古",
})

# 允许“被截断的 URL”（如结尾只有 http:// 或 www.），一并清除
_URL_RE = re.compile(r"(?:https?://|www\.)[^\s]*", re.IGNORECASE)
# 用户名可能含中点/下划线/点/连字符，如 @·Expecto_Patronum·；不能只匹配中文/字母
_MENTION_RE = re.compile(r"@[A-Za-z0-9_.\-·•\u4e00-\u9fff]{1,40}")
_HASHTAG_RE = re.compile(r"#([^#\s]{1,40})#")
_PRIVATE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ue000-\uf8ff\ufffd]")
_MULTI_SPACE_RE = re.compile(r"[ \t\u3000]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")
_BRACKET_NOISE = {"图片", "视频", "转发", "话筒", "位置", "地图", "原文", "来源", "展开全文"}


def clean_text(text) -> str:
    """去 URL/@/私用区字符/微博尾巴，保留话题与情绪表情。"""
    if text is None:
        return ""
    s = str(text)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("展开全文c", " ").replace("展开全文", " ")
    s = _URL_RE.sub(" ", s)
    s = s.replace("网页链接", " ")
    s = _MENTION_RE.sub(" ", s)
    s = s.replace("@", " ")  # 清除 @..@、单独 @ 等残留
    s = _PRIVATE_RE.sub("", s)

    def _emoji(m):
        name = m.group(1)
        if name in _BRACKET_NOISE:
            return " "
        return " " + EMOJI_MAP.get(name, name) + " "

    s = _EMOJI_RE.sub(_emoji, s)
    s = s.replace("#", " ")
    s = _MULTI_SPACE_RE.sub(" ", s)
    s = _MULTI_NL_RE.sub("\n", s)
    return s.strip()


def normalize_for_match(text) -> str:
    """用于跨表回填日期的严格归一化（与原始表文本精确匹配）。"""
    if text is None:
        return ""
    s = str(text)
    s = re.sub(r"\s+", "", s)
    s = s.replace("展开全文c", "").replace("展开全文", "")
    return s


def valid_text(text, min_len: int = 4) -> bool:
    s = clean_text(text)
    # 至少包含一个中文或字母
    return len(s) >= min_len and bool(re.search(r"[\u4e00-\u9fffA-Za-z]", s))


def tokenize(text, keep_pos: bool = True) -> list[str]:
    """jieba 分词；保留名词/动词/形容词，过滤停用词、数字、单字。"""
    s = clean_text(text)
    if not s:
        return []
    out = []
    if keep_pos:
        for word, flag in pseg.cut(s):
            w = word.strip()
            if len(w) < 2 or w in STOPWORDS:
                continue
            if flag not in KEEP_POS:
                continue
            if re.fullmatch(r"[\d\.\-%]+", w):
                continue
            out.append(w)
    else:
        for w in jieba.lcut(s):
            w = w.strip()
            if len(w) >= 2 and w not in STOPWORDS and not re.fullmatch(r"[\d\.\-%]+", w):
                out.append(w)
    return out


def tokenize_keywords(text) -> list[str]:
    """情感关键词专用分词：排除人名/地名/机构名/专名，降低专名干扰。"""
    s = clean_text(text)
    if not s:
        return []
    out = []
    for word, flag in pseg.cut(s):
        w = word.strip()
        if len(w) < 2 or w in STOPWORDS:
            continue
        if flag not in KEYWORD_POS:
            continue
        if re.fullmatch(r"[\d\.\-%]+", w):
            continue
        out.append(w)
    return out


def extract_hashtags(text) -> list[str]:
    if text is None:
        return []
    tags = []
    for t in _HASHTAG_RE.findall(str(text)):
        t = t.strip()
        if 1 <= len(t) <= 30 and t not in tags:
            tags.append(t)
    return tags


def extract_provinces(text) -> list[str]:
    """返回文本中提及的省级行政区（去重，按首次出现排序）。"""
    if text is None:
        return []
    s = str(text)
    hits = []
    # 先匹配长别名，避免“北京市”只匹配到“北京”后重复
    for alias in sorted(PROVINCE_ALIASES, key=len, reverse=True):
        if alias in s:
            p = PROVINCE_ALIASES[alias]
            if p not in hits:
                hits.append(p)
    for city, p in CITY_TO_PROVINCE.items():
        if city in s and p not in hits:
            hits.append(p)
    return hits


def text_stats(text) -> dict:
    s = clean_text(text)
    return {
        "text_len": len(s),
        "has_url": bool(_URL_RE.search(str(text or ""))),
        "has_mention": bool(_MENTION_RE.search(str(text or ""))),
        "has_hashtag": bool(_HASHTAG_RE.search(str(text or ""))),
        "n_hashtags": len(extract_hashtags(text)),
    }
