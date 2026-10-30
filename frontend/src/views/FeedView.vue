<template>
  <div class="page">
    <h1 class="page-title">舆情列表</h1>
    <p class="page-sub">按关键词、来源、情感和时间筛选帖子；点击卡片查看评论、转发与互动曲线。</p>

    <div class="toolbar panel panel-pad">
      <el-input v-model="filters.q" placeholder="搜索正文" clearable style="width: 190px" @keyup.enter="search" />
      <el-select v-model="filters.keyword" placeholder="关键词" clearable style="width: 140px">
        <el-option v-for="k in keywords" :key="k.id" :label="k.word" :value="k.word" />
      </el-select>
      <el-select v-model="filters.source" placeholder="来源" clearable style="width: 120px">
        <el-option label="关键词" value="keyword" />
        <el-option label="监控账号" value="account" />
        <el-option label="热搜" value="hotsearch" />
      </el-select>
      <el-select v-model="filters.sentiment" placeholder="情感" clearable style="width: 110px">
        <el-option label="负面" value="neg" />
        <el-option label="中性" value="neu" />
        <el-option label="正面" value="pos" />
        <el-option label="未打分" value="unscored" />
      </el-select>
      <el-button type="primary" @click="search">查询</el-button>
      <el-button @click="reset">重置</el-button>
      <div class="spacer" />
      <span class="muted">共 {{ total }} 条</span>
    </div>

    <div v-loading="loading" class="feed">
      <div v-for="item in posts" :key="item.mid" class="post panel" @click="openDetail(item.mid)">
        <div class="post-head">
          <div class="author">
            <div class="avatar">{{ (item.author_nickname || '匿')[0] }}</div>
            <div>
              <div class="name">{{ item.author_nickname || '匿名用户' }}</div>
              <div class="meta">{{ fmtDate(item.created_at) }} · {{ sourceText(item.sources) }}</div>
            </div>
          </div>
          <div class="post-right" @click.stop>
            <el-tag :type="tagType(item.sentiment)" effect="light">
              {{ labelText(item.sentiment) }}
            </el-tag>
            <el-select
              :model-value="item.sentiment"
              size="small"
              placeholder="标注"
              style="width: 84px"
              @change="(v) => submitLabel(item, v)"
            >
              <el-option label="负面" value="neg" />
              <el-option label="中性" value="neu" />
              <el-option label="正面" value="pos" />
            </el-select>
          </div>
        </div>
        <div class="post-text">{{ item.text }}</div>
        <div class="post-foot">
          <span>转发 {{ item.reposts }}</span>
          <span>评论 {{ item.comments }}</span>
          <span>赞 {{ item.likes }}</span>
          <span class="inter">互动 {{ item.interactions }}</span>
          <span class="spacer" />
          <span class="muted">{{ item.sentiment_source === 'manual' ? '人工最终' : item.sentiment_source === 'auto' ? '自动打分' : '未打分' }}</span>
        </div>
      </div>
      <div v-if="!loading && !posts.length" class="empty">没有符合条件的帖子</div>
    </div>

    <div class="pager">
      <el-pagination
        v-model:current-page="filters.page"
        :page-size="filters.page_size"
        :total="total"
        layout="prev, pager, next"
        @current-change="loadPosts"
      />
    </div>

    <el-drawer v-model="drawer" size="620px" :title="detail ? '帖子详情' : '加载中'">
      <template v-if="detail">
        <div class="detail-head">
          <div class="name">{{ detail.author_nickname || '匿名用户' }}</div>
          <div class="meta">{{ fmtDate(detail.created_at) }} · 互动 {{ detail.interactions }}</div>
          <el-tag :type="tagType(detail.sentiment)" effect="light">{{ labelText(detail.sentiment) }}</el-tag>
        </div>
        <p class="detail-text">{{ detail.text }}</p>
        <el-divider content-position="left">互动趋势</el-divider>
        <div ref="metricChart" class="chart" style="height: 210px"></div>
        <el-divider content-position="left">评论 {{ detail.detail_comments.length }}</el-divider>
        <div v-for="c in detail.detail_comments" :key="c.id" class="reply">
          <div class="reply-head">{{ c.author_nickname || '匿名' }} · {{ fmtDate(c.created_at) }}</div>
          <div>{{ c.text }}</div>
        </div>
        <div v-if="!detail.detail_comments.length" class="muted">暂无评论数据</div>
        <el-divider content-position="left">转发 {{ detail.detail_reposts.length }}</el-divider>
        <div v-for="r in detail.detail_reposts" :key="r.mid" class="reply">
          <div class="reply-head">{{ r.author_nickname || '匿名' }} · {{ fmtDate(r.created_at) }}</div>
          <div>{{ r.text }}</div>
        </div>
        <div v-if="!detail.detail_reposts.length" class="muted">暂无转发数据</div>
      </template>
    </el-drawer>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import api, { errorMessage } from '../api'

const keywords = ref([])
const posts = ref([])
const total = ref(0)
const loading = ref(false)
const drawer = ref(false)
const detail = ref(null)
const metricChart = ref(null)
let chart = null

const filters = reactive({
  q: '',
  keyword: '',
  source: '',
  sentiment: '',
  page: 1,
  page_size: 12
})

const labelMap = { neg: '负面', neu: '中性', pos: '正面' }
const labelText = (v) => labelMap[v] || '未打分'
const tagType = (v) => (v === 'neg' ? 'danger' : v === 'pos' ? 'success' : 'info')
const sourceText = (sources) => {
  if (!sources || !sources.length) return '未知来源'
  return [...new Set(sources.map((s) => s.source_value))].slice(0, 2).join(' / ')
}
const fmtDate = (v) => (v ? String(v).replace('T', ' ').slice(0, 16) : '')

async function loadKeywords() {
  try {
    const { data } = await api.get('/config/keywords', { params: { enabled: true } })
    keywords.value = data
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadPosts() {
  loading.value = true
  try {
    const params = { ...filters }
    Object.keys(params).forEach((k) => {
      if (params[k] === '' || params[k] === null) delete params[k]
    })
    const { data } = await api.get('/posts', { params })
    posts.value = data.items
    total.value = data.total
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

function search() {
  filters.page = 1
  loadPosts()
}

function reset() {
  Object.assign(filters, { q: '', keyword: '', source: '', sentiment: '', page: 1 })
  loadPosts()
}

async function submitLabel(item, label) {
  if (!label) return
  try {
    const { data } = await api.put(`/label/post/${item.mid}`, { label })
    item.sentiment = data.final_label
    item.sentiment_source = data.final_source
    ElMessage.success('已保存人工标注')
  } catch (error) {
    ElMessage.error(errorMessage(error))
    loadPosts()
  }
}

async function openDetail(mid) {
  drawer.value = true
  detail.value = null
  try {
    const { data } = await api.get(`/posts/${mid}`)
    detail.value = data
    await nextTick()
    drawMetric()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function drawMetric() {
  if (!metricChart.value || !detail.value) return
  if (chart) chart.dispose()
  chart = echarts.init(metricChart.value)
  const rows = detail.value.metrics || []
  chart.setOption({
    grid: { left: 38, right: 16, top: 20, bottom: 28 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: rows.map((x) => String(x.ts).slice(5, 16)), axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value' },
    series: [
      { name: '转发', type: 'line', smooth: true, data: rows.map((x) => x.reposts) },
      { name: '评论', type: 'line', smooth: true, data: rows.map((x) => x.comments) },
      { name: '赞', type: 'line', smooth: true, data: rows.map((x) => x.likes) }
    ]
  })
}

function onResize() { if (chart) chart.resize() }

onMounted(() => {
  loadKeywords()
  loadPosts()
  window.addEventListener('resize', onResize)
})
onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  if (chart) chart.dispose()
})
</script>

<style scoped>
.feed { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
.post { padding: 16px 18px; cursor: pointer; transition: transform .15s, box-shadow .15s; }
.post:hover { transform: translateY(-2px); box-shadow: 0 10px 28px rgba(23, 32, 51, .08); }
.post-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
.author { display: flex; gap: 10px; align-items: center; }
.avatar {
  width: 34px; height: 34px; border-radius: 50%;
  background: linear-gradient(135deg, #dbe6ff, #ece4ff);
  color: #3153a8; display: grid; place-items: center; font-weight: 700;
}
.name { font-weight: 700; font-size: 13px; }
.meta { color: var(--muted); font-size: 11px; margin-top: 2px; }
.post-right { display: flex; gap: 8px; align-items: center; }
.post-text { margin: 12px 0; line-height: 1.8; white-space: pre-wrap; word-break: break-word; }
.post-foot { display: flex; gap: 14px; color: var(--muted); font-size: 12px; border-top: 1px dashed var(--line); padding-top: 10px; }
.inter { color: var(--brand); font-weight: 700; }
.pager { display: flex; justify-content: center; margin-top: 22px; }
.detail-head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.detail-text { line-height: 1.9; white-space: pre-wrap; }
.reply { padding: 10px 0; border-bottom: 1px dashed var(--line); line-height: 1.7; }
.reply-head { color: var(--muted); font-size: 12px; margin-bottom: 3px; }
@media (max-width: 1100px) { .feed { grid-template-columns: 1fr; } }
</style>
