<template>
  <div class="page">
    <h1 class="page-title">系统设置</h1>
    <p class="page-sub">关键词、监控账号、微博账号池、情感引擎、评估、用户与采集日志。仅管理员可见。</p>

    <el-tabs v-model="active" class="panel panel-pad" @tab-change="loadTab">
      <el-tab-pane label="关键词" name="keywords">
        <div class="toolbar">
          <el-button type="primary" @click="keywordDialog = true">新增关键词</el-button>
          <span class="muted">临时关键词由热搜命中自动创建，24 小时过期。</span>
        </div>
        <el-table :data="keywords" size="small">
          <el-table-column prop="id" label="ID" width="60" />
          <el-table-column prop="word" label="关键词" />
          <el-table-column label="启用" width="90">
            <template #default="{ row }">
              <el-switch v-model="row.enabled" @change="updateKeyword(row)" />
            </template>
          </el-table-column>
          <el-table-column label="类型" width="90">
            <template #default="{ row }">
              <el-tag :type="row.is_temp ? 'warning' : 'info'" size="small">{{ row.is_temp ? '临时' : '常驻' }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="created_at" label="创建时间" width="170">
            <template #default="{ row }">{{ fmt(row.created_at) }}</template>
          </el-table-column>
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button text type="danger" @click="deleteKeyword(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="监控账号" name="accounts">
        <div class="toolbar">
          <el-button type="primary" @click="accountDialog = true">新增监控账号</el-button>
        </div>
        <el-table :data="accounts" size="small">
          <el-table-column prop="uid" label="UID" width="160" />
          <el-table-column prop="nickname" label="昵称" />
          <el-table-column label="启用" width="80">
            <template #default="{ row }">
              <el-tag :type="row.enabled ? 'success' : 'info'" size="small">{{ row.enabled ? '启用' : '停用' }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button text type="danger" @click="deleteAccount(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="微博账号池" name="cookies">
        <div class="toolbar">
          <el-button type="primary" @click="cookieDialog = true">手动导入 Cookie</el-button>
          <el-button @click="qrcodeLogin">扫码登录</el-button>
          <span class="muted">扫码需要 Playwright + 人工扫码；未启用时请手动导入。</span>
        </div>
        <el-table :data="cookies" size="small">
          <el-table-column prop="account_name" label="账号" />
          <el-table-column label="状态" width="110">
            <template #default="{ row }">
              <el-tag :type="cookieType(row.status)" size="small">{{ cookieText(row.status) }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="fail_count" label="失败次数" width="90" />
          <el-table-column prop="cooling_until" label="冷却截止" width="170">
            <template #default="{ row }">{{ fmt(row.cooling_until) || '—' }}</template>
          </el-table-column>
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button text type="danger" @click="deleteCookie(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="情感引擎" name="engine">
        <div class="two-col">
          <div>
            <h3 class="card-title">当前引擎</h3>
            <el-form label-width="100px">
              <el-form-item label="引擎">
                <el-select v-model="engineForm.name" style="width: 220px">
                  <el-option v-for="e in engine.available" :key="e" :label="e" :value="e" />
                </el-select>
              </el-form-item>
              <el-form-item label="版本">
                <el-input v-model="engineForm.version" placeholder="可选，例如 tfidf-lr-v1" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="switchEngine">切换引擎</el-button>
              </el-form-item>
            </el-form>
            <el-alert
              type="info"
              :closable="false"
              title="切换只影响后续打分；历史重跑请使用下方“入队重跑”。"
            />
          </div>
          <div>
            <h3 class="card-title">历史重跑</h3>
            <el-form label-width="90px">
              <el-form-item label="文本类型">
                <el-select v-model="rerunForm.text_type" clearable placeholder="全部" style="width: 160px">
                  <el-option label="帖子" value="post" />
                  <el-option label="评论" value="comment" />
                  <el-option label="转发" value="repost" />
                </el-select>
              </el-form-item>
              <el-form-item label="上限">
                <el-input-number v-model="rerunForm.limit" :min="1" :max="100000" />
              </el-form-item>
              <el-form-item>
                <el-button @click="rerun">入队重跑</el-button>
                <el-button type="primary" @click="runOnce">立即消费一批</el-button>
              </el-form-item>
            </el-form>
            <div class="muted">队列：{{ queueText }}</div>
          </div>
        </div>
      </el-tab-pane>

      <el-tab-pane label="评估" name="eval">
        <div class="toolbar">
          <el-button @click="loadEval">刷新</el-button>
          <span class="muted">以人工标注为基准；当前人工样本 {{ evalData.n_manual_texts }} 条。</span>
        </div>
        <el-table :data="evalData.engines" size="small">
          <el-table-column prop="engine" label="引擎" width="100" />
          <el-table-column prop="engine_version" label="版本" width="160" />
          <el-table-column prop="n" label="样本" width="70" align="right" />
          <el-table-column prop="accuracy" label="准确率" width="90" align="right">
            <template #default="{ row }">{{ (row.accuracy * 100).toFixed(1) }}%</template>
          </el-table-column>
          <el-table-column prop="macro_f1" label="Macro-F1" width="100" align="right">
            <template #default="{ row }">{{ row.macro_f1.toFixed(4) }}</template>
          </el-table-column>
          <el-table-column label="负面 F1" width="90" align="right">
            <template #default="{ row }">{{ (row.per_class.neg?.f1 ?? 0).toFixed(4) }}</template>
          </el-table-column>
          <el-table-column label="中性 F1" width="90" align="right">
            <template #default="{ row }">{{ (row.per_class.neu?.f1 ?? 0).toFixed(4) }}</template>
          </el-table-column>
          <el-table-column label="正面 F1" width="90" align="right">
            <template #default="{ row }">{{ (row.per_class.pos?.f1 ?? 0).toFixed(4) }}</template>
          </el-table-column>
        </el-table>
        <div v-if="!evalData.engines.length" class="empty">还没有人工标注，先去标注工作台标几条。</div>
      </el-tab-pane>

      <el-tab-pane label="用户" name="users">
        <div class="toolbar">
          <el-button type="primary" @click="userDialog = true">新增用户</el-button>
        </div>
        <el-table :data="users" size="small">
          <el-table-column prop="id" label="ID" width="60" />
          <el-table-column prop="username" label="用户名" />
          <el-table-column label="角色" width="100">
            <template #default="{ row }">
              <el-tag :type="row.role === 'admin' ? 'danger' : 'info'" size="small">{{ row.role === 'admin' ? '管理员' : '标注员' }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="80">
            <template #default="{ row }">
              <el-tag :type="row.enabled ? 'success' : 'info'" size="small">{{ row.enabled ? '启用' : '禁用' }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="80">
            <template #default="{ row }">
              <el-button text type="danger" @click="deleteUser(row)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-tab-pane>

      <el-tab-pane label="采集日志" name="logs">
        <div class="toolbar">
          <el-button @click="loadLogs">刷新</el-button>
        </div>
        <el-table :data="logs" size="small">
          <el-table-column prop="started_at" label="开始" width="170">
            <template #default="{ row }">{{ fmt(row.started_at) }}</template>
          </el-table-column>
          <el-table-column prop="task_type" label="任务" width="100" />
          <el-table-column prop="target" label="目标" show-overflow-tooltip />
          <el-table-column prop="status" label="状态" width="90">
            <template #default="{ row }">
              <el-tag :type="row.status === 'success' ? 'success' : row.status === 'failed' ? 'danger' : 'info'" size="small">{{ row.status }}</el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="items" label="条数" width="70" align="right" />
          <el-table-column prop="error_msg" label="错误" show-overflow-tooltip />
        </el-table>
      </el-tab-pane>
    </el-tabs>

    <el-dialog v-model="keywordDialog" title="新增关键词" width="420px">
      <el-input v-model="newKeyword" placeholder="例如：武汉" @keyup.enter="createKeyword" />
      <template #footer>
        <el-button @click="keywordDialog = false">取消</el-button>
        <el-button type="primary" @click="createKeyword">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="accountDialog" title="新增监控账号" width="420px">
      <el-form label-width="70px">
        <el-form-item label="UID"><el-input v-model="newAccount.uid" /></el-form-item>
        <el-form-item label="昵称"><el-input v-model="newAccount.nickname" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="accountDialog = false">取消</el-button>
        <el-button type="primary" @click="createAccount">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="cookieDialog" title="手动导入 Cookie" width="520px">
      <el-alert type="warning" :closable="false" title="Cookie 会加密存储；请只导入你有权使用的账号，不要导入他人 Cookie。" style="margin-bottom: 14px" />
      <el-form label-width="90px">
        <el-form-item label="账号名"><el-input v-model="newCookie.account_name" /></el-form-item>
        <el-form-item label="Cookie"><el-input v-model="newCookie.cookie" type="textarea" :rows="5" /></el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="cookieDialog = false">取消</el-button>
        <el-button type="primary" @click="createCookie">保存</el-button>
      </template>
    </el-dialog>

    <el-dialog v-model="userDialog" title="新增用户" width="420px">
      <el-form label-width="80px">
        <el-form-item label="用户名"><el-input v-model="newUser.username" /></el-form-item>
        <el-form-item label="密码"><el-input v-model="newUser.password" type="password" /></el-form-item>
        <el-form-item label="角色">
          <el-select v-model="newUser.role" style="width: 160px">
            <el-option label="标注员" value="annotator" />
            <el-option label="管理员" value="admin" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="userDialog = false">取消</el-button>
        <el-button type="primary" @click="createUser">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import api, { errorMessage } from '../api'

const active = ref('keywords')
const keywords = ref([])
const accounts = ref([])
const cookies = ref([])
const users = ref([])
const logs = ref([])
const engine = reactive({ active_engine: 'local', active_version: '', available: [] })
const engineForm = reactive({ name: 'local', version: '' })
const rerunForm = reactive({ text_type: '', limit: 1000 })
const evalData = reactive({ n_manual_texts: 0, engines: [] })
const queue = reactive({})
const keywordDialog = ref(false)
const accountDialog = ref(false)
const cookieDialog = ref(false)
const userDialog = ref(false)
const newKeyword = ref('')
const newAccount = reactive({ uid: '', nickname: '' })
const newCookie = reactive({ account_name: '', cookie: '' })
const newUser = reactive({ username: '', password: '', role: 'annotator' })

const fmt = (v) => (v ? String(v).replace('T', ' ').slice(0, 16) : '')
const queueText = ref('—')
const cookieText = (s) => ({ ok: '正常', cooling: '冷却中', need_login: '需重新登录', abnormal: '异常' }[s] || s)
const cookieType = (s) => (s === 'ok' ? 'success' : s === 'need_login' ? 'warning' : 'danger')

async function get(url, config) {
  const { data } = await api.get(url, config)
  return data
}

async function loadKeywords() { keywords.value = await get('/config/keywords') }
async function loadAccounts() { accounts.value = await get('/config/accounts') }
async function loadCookies() { cookies.value = await get('/config/cookies') }
async function loadUsers() { users.value = await get('/config/users') }
async function loadLogs() { logs.value = await get('/crawl/logs', { params: { limit: 100 } }) }
async function loadEngine() {
  const data = await get('/config/engine')
  Object.assign(engine, data)
  engineForm.name = data.active_engine
  engineForm.version = data.active_version
}
async function loadEval() {
  const data = await get('/eval')
  Object.assign(evalData, data)
}
async function loadQueue() {
  try {
    const data = await get('/sentiment/queue')
    Object.assign(queue, data)
    queueText.value = Object.entries(data).map(([k, v]) => `${k} ${v}`).join(' · ') || '空'
  } catch {
    queueText.value = '—'
  }
}

async function loadTab(name) {
  try {
    if (name === 'keywords') await loadKeywords()
    else if (name === 'accounts') await loadAccounts()
    else if (name === 'cookies') await loadCookies()
    else if (name === 'engine') { await loadEngine(); await loadQueue() }
    else if (name === 'eval') await loadEval()
    else if (name === 'users') await loadUsers()
    else if (name === 'logs') await loadLogs()
  } catch (error) {
    ElMessage.error(errorMessage(error))
  }
}

async function createKeyword() {
  if (!newKeyword.value.trim()) return
  try {
    await api.post('/config/keywords', { word: newKeyword.value.trim() })
    newKeyword.value = ''
    keywordDialog.value = false
    await loadKeywords()
    ElMessage.success('已新增')
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function updateKeyword(row) {
  try {
    await api.put(`/config/keywords/${row.id}`, { enabled: row.enabled })
  } catch (error) { ElMessage.error(errorMessage(error)); loadKeywords() }
}
async function deleteKeyword(row) {
  try {
    await ElMessageBox.confirm(`删除关键词「${row.word}」？`, '确认')
    await api.delete(`/config/keywords/${row.id}`)
    await loadKeywords()
  } catch (error) { if (error !== 'cancel') ElMessage.error(errorMessage(error)) }
}
async function createAccount() {
  try {
    await api.post('/config/accounts', newAccount)
    Object.assign(newAccount, { uid: '', nickname: '' })
    accountDialog.value = false
    await loadAccounts()
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function deleteAccount(row) {
  try {
    await ElMessageBox.confirm(`删除监控账号「${row.uid}」？`, '确认')
    await api.delete(`/config/accounts/${row.id}`)
    await loadAccounts()
  } catch (error) { if (error !== 'cancel') ElMessage.error(errorMessage(error)) }
}
async function createCookie() {
  try {
    await api.post('/config/cookies', newCookie)
    Object.assign(newCookie, { account_name: '', cookie: '' })
    cookieDialog.value = false
    await loadCookies()
    ElMessage.success('Cookie 已加密保存')
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function deleteCookie(row) {
  try {
    await ElMessageBox.confirm(`删除账号「${row.account_name}」的 Cookie？`, '确认')
    await api.delete(`/config/cookies/${row.id}`)
    await loadCookies()
  } catch (error) { if (error !== 'cancel') ElMessage.error(errorMessage(error)) }
}
async function qrcodeLogin() {
  try {
    await api.post('/config/cookies/qrcode')
  } catch (error) {
    ElMessage.warning(errorMessage(error))
  }
}
async function switchEngine() {
  try {
    const { data } = await api.post('/config/engine', engineForm)
    Object.assign(engine, data)
    ElMessage.success('已切换')
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function rerun() {
  try {
    const { data } = await api.post('/sentiment/rerun', { ...rerunForm, engine: engineForm.name, version: engineForm.version || null })
    ElMessage.success(data.message)
    await loadQueue()
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function runOnce() {
  try {
    const { data } = await api.post('/sentiment/run-once', null, { params: { batch_size: 64, max_batches: 10 } })
    ElMessage.success(data.message)
    await loadQueue()
    await loadEval()
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function createUser() {
  try {
    await api.post('/config/users', newUser)
    Object.assign(newUser, { username: '', password: '', role: 'annotator' })
    userDialog.value = false
    await loadUsers()
  } catch (error) { ElMessage.error(errorMessage(error)) }
}
async function deleteUser(row) {
  try {
    await ElMessageBox.confirm(`删除用户「${row.username}」？`, '确认')
    await api.delete(`/config/users/${row.id}`)
    await loadUsers()
  } catch (error) { if (error !== 'cancel') ElMessage.error(errorMessage(error)) }
}

onMounted(async () => {
  await loadKeywords()
  await loadEngine()
  await loadQueue()
})
</script>
