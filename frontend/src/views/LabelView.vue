<template>
  <div class="page">
    <h1 class="page-title">标注工作台</h1>
    <p class="page-sub">每次一条文本；快捷键 1 正面、2 中性、3 负面、S 跳过、← 返回上一条。</p>

    <div class="toolbar panel panel-pad">
      <el-select v-model="strategy" style="width: 180px" @change="loadNext">
        <el-option label="随机抽样" value="random" />
        <el-option label="低置信度优先" value="low_confidence" />
        <el-option label="双引擎不一致优先" value="disagreement" />
      </el-select>
      <el-select v-model="textType" clearable placeholder="全部文本类型" style="width: 150px" @change="loadNext">
        <el-option label="帖子" value="post" />
        <el-option label="评论" value="comment" />
        <el-option label="转发" value="repost" />
      </el-select>
      <el-input v-model="keyword" placeholder="限定关键词" clearable style="width: 150px" @keyup.enter="loadNext" />
      <el-button type="primary" @click="loadNext">下一条</el-button>
      <div class="spacer" />
      <span class="muted">今日 {{ stats.my_today }} · 累计 {{ stats.my_total }} · 一致率 {{ agreementText }}</span>
    </div>

    <div v-loading="loading" class="label-layout">
      <div class="panel panel-pad sample">
        <template v-if="sample">
          <div class="sample-head">
            <el-tag size="small" effect="plain">{{ typeText(sample.text_type) }}</el-tag>
            <span class="mono muted">{{ sample.text_id }}</span>
            <div class="spacer" />
            <el-tag v-if="sample.final_source === 'manual'" type="success" size="small">已有人工标注</el-tag>
            <el-tag v-else-if="sample.final_source === 'auto'" type="info" size="small">当前为自动结果</el-tag>
          </div>
          <div class="sample-text">{{ sample.text }}</div>

          <div v-if="sample.post_context" class="context">
            <div class="context-title">所属帖子</div>
            <div>{{ sample.post_context.text }}</div>
          </div>

          <div class="auto">
            <div class="auto-title">自动打分</div>
            <div v-if="!sample.auto_labels.length" class="muted">暂无自动结果</div>
            <div v-for="a in sample.auto_labels" :key="a.engine + a.engine_version" class="auto-row">
              <span class="mono">{{ a.engine }} / {{ a.engine_version }}</span>
              <el-tag :type="tagType(a.label)" size="small">{{ labelText(a.label) }}</el-tag>
              <span class="muted">置信度 {{ (a.score * 100).toFixed(1) }}%</span>
            </div>
          </div>

          <div class="actions">
            <el-button type="success" size="large" @click="submit('pos')">1 正面</el-button>
            <el-button size="large" @click="submit('neu')">2 中性</el-button>
            <el-button type="danger" size="large" @click="submit('neg')">3 负面</el-button>
            <el-button size="large" @click="skip">S 跳过</el-button>
            <el-button size="large" text @click="back">← 上一条</el-button>
          </div>
        </template>
        <div v-else class="empty">
          <div>当前没有待标注样本</div>
          <div class="muted" style="margin-top: 8px">可以切换抽样策略，或先在设置中导入数据/采集。</div>
        </div>
      </div>

      <div class="side">
        <div class="panel panel-pad">
          <h3 class="card-title">今日进度</h3>
          <div class="progress-row"><span>我的标注</span><b>{{ stats.my_total }}</b></div>
          <div class="progress-row"><span>今日新增</span><b>{{ stats.my_today }}</b></div>
          <div class="progress-row"><span>整体已标注</span><b>{{ stats.labeled_texts }} / {{ stats.total_texts }}</b></div>
          <div class="progress-row"><span>参与标注员</span><b>{{ stats.annotators }}</b></div>
          <div class="progress-row"><span>两两一致率</span><b>{{ agreementText }}</b></div>
        </div>
        <div class="panel panel-pad" style="margin-top: 16px">
          <h3 class="card-title">快捷键</h3>
          <div class="key-row"><kbd>1</kbd> 正面</div>
          <div class="key-row"><kbd>2</kbd> 中性</div>
          <div class="key-row"><kbd>3</kbd> 负面</div>
          <div class="key-row"><kbd>S</kbd> 跳过</div>
          <div class="key-row"><kbd>←</kbd> 上一条</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import api, { errorMessage } from '../api'

const strategy = ref('random')
const textType = ref('')
const keyword = ref('')
const sample = ref(null)
const loading = ref(false)
const history = ref([])
const stats = reactive({
  total_texts: 0,
  labeled_texts: 0,
  my_total: 0,
  my_today: 0,
  annotators: 0,
  pairwise_agreement: null
})

const labelMap = { neg: '负面', neu: '中性', pos: '正面' }
const labelText = (v) => labelMap[v] || '未打分'
const tagType = (v) => (v === 'neg' ? 'danger' : v === 'pos' ? 'success' : 'info')
const typeText = (v) => ({ post: '帖子', comment: '评论', repost: '转发' }[v] || v)
const agreementText = computed(() =>
  stats.pairwise_agreement == null ? '—' : `${(stats.pairwise_agreement * 100).toFixed(1)}%`
)

async function loadStats() {
  try {
    const { data } = await api.get('/label/stats')
    Object.assign(stats, data)
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function loadNext() {
  loading.value = true
  try {
    const params = { strategy: strategy.value }
    if (textType.value) params.text_type = textType.value
    if (keyword.value) params.keyword = keyword.value
    const { data } = await api.get('/label/next', { params })
    sample.value = data
    await loadStats()
  } catch (error) {
    if (error?.response?.status === 404) {
      sample.value = null
      await loadStats()
    } else {
      ElMessage.error(errorMessage(error))
    }
  } finally {
    loading.value = false
  }
}

async function submit(label) {
  if (!sample.value) return
  const current = sample.value
  try {
    await api.put(`/label/${current.text_type}/${current.text_id}`, { label })
    ElMessage.success(`已标注为${labelText(label)}`)
    history.value.push(current)
    if (history.value.length > 20) history.value.shift()
    await loadNext()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

function skip() {
  if (!sample.value) return
  history.value.push(sample.value)
  if (history.value.length > 20) history.value.shift()
  loadNext()
}

function back() {
  const previous = history.value.pop()
  if (!previous) {
    ElMessage.info('没有可返回的上一条')
    return
  }
  sample.value = previous
}

function onKey(event) {
  if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return
  if (event.key === '1') submit('pos')
  else if (event.key === '2') submit('neu')
  else if (event.key === '3') submit('neg')
  else if (event.key.toLowerCase() === 's') skip()
  else if (event.key === 'ArrowLeft') back()
}

onMounted(() => {
  loadNext()
  window.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
</script>

<style scoped>
.label-layout { display: grid; grid-template-columns: 1.4fr .6fr; gap: 16px; }
.sample-head { display: flex; align-items: center; gap: 10px; }
.sample-text {
  margin: 20px 0; font-size: 20px; line-height: 2; white-space: pre-wrap;
  word-break: break-word; min-height: 120px;
}
.context {
  background: #f7f9fd; border: 1px solid var(--line); border-radius: 10px;
  padding: 12px 14px; line-height: 1.8; color: #475467;
}
.context-title { font-size: 12px; color: var(--muted); margin-bottom: 5px; }
.auto { margin-top: 18px; padding-top: 14px; border-top: 1px dashed var(--line); }
.auto-title { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
.auto-row { display: flex; align-items: center; gap: 10px; padding: 5px 0; }
.actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 26px; }
.side { align-self: start; }
.progress-row { display: flex; justify-content: space-between; padding: 9px 0; border-bottom: 1px dashed var(--line); }
.key-row { padding: 7px 0; color: var(--muted); }
kbd {
  display: inline-block; min-width: 24px; text-align: center;
  border: 1px solid var(--line); border-bottom-width: 2px;
  border-radius: 5px; padding: 1px 5px; margin-right: 8px; color: var(--ink);
  background: #fafbfe; font-family: inherit;
}
@media (max-width: 1100px) { .label-layout { grid-template-columns: 1fr; } }
</style>
