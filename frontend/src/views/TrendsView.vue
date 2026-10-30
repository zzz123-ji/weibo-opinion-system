<template>
  <div class="page">
    <h1 class="page-title">趋势看板</h1>
    <p class="page-sub">声量、情感占比、负面突增与热搜命中；默认展示数据窗口内的全部日期。</p>

    <div class="toolbar panel panel-pad">
      <el-date-picker
        v-model="range"
        type="daterange"
        value-format="YYYY-MM-DD"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        style="width: 240px"
      />
      <el-select v-model="granularity" style="width: 100px">
        <el-option label="按天" value="day" />
        <el-option label="按小时" value="hour" />
      </el-select>
      <el-select v-model="keywords" multiple collapse-tags placeholder="全部关键词" style="width: 210px">
        <el-option v-for="k in keywordOptions" :key="k.id" :label="k.word" :value="k.word" />
      </el-select>
      <el-checkbox v-model="split">按关键词拆线</el-checkbox>
      <el-button type="primary" @click="loadAll">刷新</el-button>
      <div class="spacer" />
      <span class="muted">{{ rangeText }}</span>
    </div>

    <div class="metric-grid">
      <div class="metric">
        <div class="label">窗口声量</div>
        <div class="value">{{ kpi.volume }}</div>
      </div>
      <div class="metric">
        <div class="label">负面占比</div>
        <div class="value" style="color: var(--neg)">{{ kpi.negRatio }}%</div>
      </div>
      <div class="metric">
        <div class="label">负面突增时段</div>
        <div class="value">{{ alerts.length }}</div>
      </div>
      <div class="metric">
        <div class="label">热搜命中</div>
        <div class="value">{{ hotsearch.filter((x) => x.matched_keyword_id).length }}</div>
      </div>
    </div>

    <div class="panel panel-pad" style="margin-bottom: 16px">
      <h3 class="card-title">声量趋势</h3>
      <div ref="volumeChart" class="chart"></div>
    </div>

    <div class="two-col" style="margin-bottom: 16px">
      <div class="panel panel-pad">
        <h3 class="card-title">情感占比</h3>
        <div ref="sentimentChart" class="chart"></div>
      </div>
      <div class="panel panel-pad">
        <h3 class="card-title">负面突增</h3>
        <div v-if="!alerts.length" class="empty">当前窗口未触发突增规则</div>
        <div v-for="a in alerts" :key="a.bucket" class="alert-row">
          <div>
            <div class="mono">{{ a.bucket }}</div>
            <div class="muted">负面 {{ a.neg_count }} 条 · 基线 {{ a.baseline }} 条/小时</div>
          </div>
          <el-tag type="danger">{{ a.ratio }}×</el-tag>
        </div>
        <div class="rule muted">规则：负面条数 ≥ 前 7 天同时段均值 × 3，且 ≥ 10 条</div>
      </div>
    </div>

    <div class="two-col">
      <div class="panel panel-pad">
        <h3 class="card-title">高互动负面帖 Top 10</h3>
        <el-table :data="topNegative" size="small" style="width: 100%">
          <el-table-column prop="author_nickname" label="作者" width="90" />
          <el-table-column prop="text" label="正文" show-overflow-tooltip />
          <el-table-column prop="interactions" label="互动" width="70" align="right" />
        </el-table>
      </div>
      <div class="panel panel-pad">
        <h3 class="card-title">热搜命中</h3>
        <el-table :data="hotsearch" size="small" height="300">
          <el-table-column prop="ts" label="时间" width="130">
            <template #default="{ row }">{{ String(row.ts).replace('T', ' ').slice(5, 16) }}</template>
          </el-table-column>
          <el-table-column prop="word" label="词条" />
          <el-table-column prop="heat" label="热度" width="80" align="right" />
          <el-table-column label="命中" width="70" align="center">
            <template #default="{ row }">
              <el-tag v-if="row.matched_keyword_id" type="success" size="small">是</el-tag>
              <span v-else class="muted">—</span>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import api, { errorMessage } from '../api'

const range = ref([])
const granularity = ref('day')
const keywords = ref([])
const keywordOptions = ref([])
const split = ref(false)
const volume = ref({ points: [], series: {} })
const sentiment = ref({ points: [] })
const alerts = ref([])
const hotsearch = ref([])
const topNegative = ref([])
const volumeChart = ref(null)
const sentimentChart = ref(null)
let volumeInstance = null
let sentimentInstance = null

const rangeText = computed(() => {
  if (!range.value || range.value.length !== 2) return '全部日期'
  return `${range.value[0]} ~ ${range.value[1]}`
})

const kpi = computed(() => {
  const volumeCount = (volume.value.points || []).reduce((s, x) => s + x.count, 0)
  const neg = (sentiment.value.points || []).reduce((s, x) => s + x.neg, 0)
  const total = (sentiment.value.points || []).reduce(
    (s, x) => s + x.pos + x.neu + x.neg + x.unscored, 0
  )
  return {
    volume: volumeCount,
    negRatio: total ? ((neg / total) * 100).toFixed(1) : '0.0'
  }
})

function params() {
  const p = { granularity: granularity.value }
  if (range.value && range.value.length === 2) {
    p.time_from = `${range.value[0]}T00:00:00`
    p.time_to = `${range.value[1]}T23:59:59`
  }
  if (keywords.value.length) p.keywords = keywords.value
  p.split_by_keyword = split.value
  return p
}

async function initRange() {
  if (range.value.length) return
  try {
    const { data } = await api.get('/trend/volume', { params: { granularity: 'day' } })
    const points = data.points || []
    if (points.length) range.value = [points[0].bucket, points[points.length - 1].bucket]
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadAll() {
  try {
    const p = params()
    const [v, s, a, h, t] = await Promise.all([
      api.get('/trend/volume', { params: p }),
      api.get('/trend/sentiment', { params: p }),
      api.get('/trend/alerts', { params: { time_from: p.time_from, time_to: p.time_to } }),
      api.get('/hotsearch', { params: { limit: 60 } }),
      api.get('/trend/top-negative', {
        params: {
          limit: 10,
          time_from: p.time_from,
          time_to: p.time_to,
          ...(keywords.value.length ? { keywords: keywords.value } : {})
        }
      })
    ])
    volume.value = v.data
    sentiment.value = s.data
    alerts.value = a.data
    hotsearch.value = h.data
    topNegative.value = t.data
    await nextTick()
    drawVolume()
    drawSentiment()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function drawVolume() {
  if (!volumeChart.value) return
  if (volumeInstance) volumeInstance.dispose()
  volumeInstance = echarts.init(volumeChart.value)
  const points = volume.value.points || []
  const series = volume.value.series || {}
  const seriesList = Object.keys(series).length
    ? Object.entries(series).map(([name, rows]) => ({
        name,
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: rows.map((x) => [x.bucket, x.count])
      }))
    : [{
        name: '声量',
        type: 'line',
        smooth: true,
        areaStyle: { opacity: .10 },
        showSymbol: false,
        data: points.map((x) => [x.bucket, x.count])
      }]
  volumeInstance.setOption({
    tooltip: { trigger: 'axis' },
    legend: { top: 0, type: 'scroll' },
    grid: { left: 44, right: 20, top: 40, bottom: 34 },
    xAxis: { type: 'category', axisLabel: { fontSize: 10 } },
    yAxis: { type: 'value', name: '条' },
    series: seriesList
  })
}

function drawSentiment() {
  if (!sentimentChart.value) return
  if (sentimentInstance) sentimentInstance.dispose()
  sentimentInstance = echarts.init(sentimentChart.value)
  const rows = sentiment.value.points || []
  const buckets = rows.map((x) => x.bucket)
  sentimentInstance.setOption({
    tooltip: { trigger: 'axis' },
    legend: { top: 0, data: ['负面', '中性', '正面', '未打分', '负面占比'] },
    grid: { left: 44, right: 50, top: 40, bottom: 34 },
    xAxis: { type: 'category', data: buckets, axisLabel: { fontSize: 10 } },
    yAxis: [
      { type: 'value', name: '条' },
      { type: 'value', name: '占比', axisLabel: { formatter: '{value}%' }, max: 100 }
    ],
    series: [
      { name: '负面', type: 'line', stack: 'total', areaStyle: {}, data: rows.map((x) => x.neg), color: '#e5484d' },
      { name: '中性', type: 'line', stack: 'total', areaStyle: {}, data: rows.map((x) => x.neu), color: '#98a2b3' },
      { name: '正面', type: 'line', stack: 'total', areaStyle: {}, data: rows.map((x) => x.pos), color: '#12a594' },
      { name: '未打分', type: 'line', stack: 'total', areaStyle: {}, data: rows.map((x) => x.unscored), color: '#d0d5dd' },
      {
        name: '负面占比',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        symbol: 'none',
        lineStyle: { width: 2, type: 'dashed' },
        data: rows.map((x) => +(x.neg_ratio * 100).toFixed(1)),
        color: '#7a271a'
      }
    ]
  })
}

function onResize() {
  if (volumeInstance) volumeInstance.resize()
  if (sentimentInstance) sentimentInstance.resize()
}

onMounted(async () => {
  try {
    const { data } = await api.get('/config/keywords', { params: { enabled: true } })
    keywordOptions.value = data
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
  await initRange()
  await loadAll()
  window.addEventListener('resize', onResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  if (volumeInstance) volumeInstance.dispose()
  if (sentimentInstance) sentimentInstance.dispose()
})
</script>

<style scoped>
.alert-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 10px 0; border-bottom: 1px dashed var(--line);
}
.rule { font-size: 11px; margin-top: 10px; }
</style>
