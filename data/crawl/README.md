# 1,000 条采集样本（user_id + time + context）

## 这是什么
本目录是“公开数据集采集落盘”演示样本，共 1,000 条，来自 ScienceDB 公开数据集
《Weibo and Twitter datasets》微博图文子集（DOI: 10.57760/sciencedb.j00133.00366，CC BY 4.0）。
不是实时爬取微博站内接口：微博接口/用户页需要登录且返回 403，项目按合规红线不绕验证码、
不携带登录态。图片从官方 dataset.zip 精确抽取。

## 字段
| 字段 | 说明 |
|---|---|
| post_id | 帖文不可逆哈希，跨表关联用 |
| user_id | 发布账号的 sha256 前 16 位，稳定但不可逆；原始账号名默认不落盘 |
| user_id_type | 固定为 sha256_pseudonym |
| time | 发布时间（精确到分钟，2020 年） |
| date | 日期（只到天，供趋势分析） |
| text | 清洗后正文（去 URL/@/微博尾巴） |
| label / label_name | -1 负面 / 0 中性 / 1 正面 |
| image_file | 本地图片相对路径 |
| context | JSON：`{"text": 清洗后正文, "image": 本地图片相对路径}` |
| image_sha256 / image_width / image_height / image_bytes | 图片完整性信息 |
| source / dataset_doi | 公开数据来源与 DOI |

## 隐私与合规
- 原始账号名只在内存中生成不可逆 user_id，默认不落盘；
- 输出表不含原始微博 ID、图片 URL、账号名；
- `_internal_provenance.csv` 含原始帖文 ID 与图片 URL，仅供内部溯源，不得提交；
- 如需内部核对账号，运行 `--keep-raw-account` 生成 `_internal_account_map.csv`，同样不得提交。

## 复现
```bash
python src/01c_export_crawl_1000.py --n 1000
```

## 统计
- 条数：1,000
- 唯一 user_id：992
- 类别：{-1: 334, 0: 333, 1: 333}
- 日期范围：2020-01-01 ~ 2020-02-18
- 图片：1,000 张，合计 44.2 MB
