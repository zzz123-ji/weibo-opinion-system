<template>
  <div class="login-page">
    <div class="login-card">
      <div class="brand">
        <div class="mark">舆</div>
        <div>
          <h1>微博舆情管理系统</h1>
          <p>采集 · 趋势 · 标注 · 引擎评估</p>
        </div>
      </div>
      <el-form :model="form" label-position="top" @submit.prevent="submit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" size="large" placeholder="admin" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码">
          <el-input
            v-model="form.password"
            size="large"
            type="password"
            show-password
            placeholder="请输入密码"
            autocomplete="current-password"
            @keyup.enter="submit"
          />
        </el-form-item>
        <el-button type="primary" size="large" class="submit" :loading="loading" @click="submit">
          登录
        </el-button>
      </el-form>
      <div class="hint">
        <div>演示管理员：<span class="mono">admin / admin123</span></div>
        <div>演示标注员：<span class="mono">annotator / annotator123</span></div>
        <div class="small">首次启动若未设置 ADMIN_PASSWORD，请查看后端日志中的随机密码。</div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import api, { errorMessage } from '../api'

const router = useRouter()
const loading = ref(false)
const form = reactive({ username: 'admin', password: 'admin123' })

async function submit() {
  if (!form.username || !form.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    const { data } = await api.post('/auth/login', form)
    localStorage.setItem('token', data.access_token)
    localStorage.setItem('user', JSON.stringify(data.user))
    ElMessage.success('登录成功')
    router.replace('/feed')
  } catch (error) {
    ElMessage.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background:
    radial-gradient(900px 500px at 10% 10%, #dce8ff 0, transparent 60%),
    radial-gradient(700px 500px at 90% 90%, #e7dcff 0, transparent 55%),
    #f5f7fb;
}
.login-card {
  width: 420px;
  background: #fff;
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 32px 34px 26px;
  box-shadow: 0 20px 60px rgba(23, 32, 51, .10);
}
.brand { display: flex; gap: 14px; align-items: center; margin-bottom: 26px; }
.mark {
  width: 48px; height: 48px; border-radius: 14px;
  display: grid; place-items: center;
  background: linear-gradient(135deg, #1f5eff, #7b5cff);
  color: #fff; font-size: 24px; font-weight: 700;
}
h1 { font-size: 20px; margin: 0 0 4px; }
p { margin: 0; color: var(--muted); font-size: 12px; }
.submit { width: 100%; margin-top: 6px; }
.hint {
  margin-top: 20px; padding-top: 16px; border-top: 1px dashed var(--line);
  color: var(--muted); font-size: 12px; line-height: 1.9;
}
.small { color: #98a2b3; font-size: 11px; margin-top: 4px; }
</style>
