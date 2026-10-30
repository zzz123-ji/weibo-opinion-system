<template>
  <el-container class="shell">
    <el-aside width="220px" class="aside">
      <div class="brand">
        <div class="brand-mark">舆</div>
        <div>
          <div class="brand-name">微博舆情</div>
          <div class="brand-sub">内部监测系统</div>
        </div>
      </div>
      <el-menu :default-active="$route.path" router class="menu">
        <el-menu-item index="/feed">舆情列表</el-menu-item>
        <el-menu-item index="/trends">趋势看板</el-menu-item>
        <el-menu-item index="/label">标注工作台</el-menu-item>
        <el-menu-item v-if="isAdmin" index="/settings">设置</el-menu-item>
      </el-menu>
      <div class="aside-foot">
        <div>单机内网部署</div>
        <div class="mono">{{ version }}</div>
      </div>
    </el-aside>

    <el-container>
      <el-header class="header">
        <div>
          <span class="crumb">{{ currentTitle }}</span>
          <span class="env">内网</span>
        </div>
        <div class="user">
          <span>{{ user.username || '未登录' }}</span>
          <el-tag size="small" :type="isAdmin ? 'danger' : 'info'">
            {{ isAdmin ? '管理员' : '标注员' }}
          </el-tag>
          <el-button text @click="logout">退出</el-button>
        </div>
      </el-header>
      <el-main class="main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const route = useRoute()
const router = useRouter()
const user = JSON.parse(localStorage.getItem('user') || '{}')
const isAdmin = computed(() => user.role === 'admin')
const version = 'v1.0.0'

const titles = {
  '/feed': '舆情列表',
  '/trends': '趋势看板',
  '/label': '标注工作台',
  '/settings': '系统设置'
}
const currentTitle = computed(() => titles[route.path] || '微博舆情')

function logout() {
  localStorage.removeItem('token')
  localStorage.removeItem('user')
  router.replace('/login')
}
</script>

<style scoped>
.shell { min-height: 100vh; }
.aside {
  background: #0f1b35;
  color: #fff;
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}
.brand { display: flex; gap: 12px; align-items: center; padding: 22px 20px 18px; }
.brand-mark {
  width: 38px; height: 38px; border-radius: 11px;
  display: grid; place-items: center;
  background: linear-gradient(135deg, #3b7cff, #7b5cff);
  font-size: 20px; font-weight: 700;
}
.brand-name { font-size: 16px; font-weight: 700; letter-spacing: .5px; }
.brand-sub { font-size: 11px; color: #93a2c2; margin-top: 3px; }
.menu { border-right: 0; background: transparent; flex: 1; }
:deep(.el-menu-item) { color: #b9c5de; height: 46px; margin: 4px 12px; border-radius: 9px; }
:deep(.el-menu-item:hover) { background: #1a2a4d; color: #fff; }
:deep(.el-menu-item.is-active) { background: #1f5eff; color: #fff; }
.aside-foot { padding: 16px 20px; color: #8190ad; font-size: 11px; line-height: 1.8; }
.header {
  background: #fff; border-bottom: 1px solid var(--line);
  display: flex; align-items: center; justify-content: space-between;
  height: 58px;
}
.crumb { font-weight: 700; font-size: 15px; }
.env {
  margin-left: 10px; font-size: 11px; color: #12a594;
  background: #eafbf7; border: 1px solid #c8f1e7;
  border-radius: 999px; padding: 2px 8px;
}
.user { display: flex; align-items: center; gap: 10px; color: var(--muted); }
.main { padding: 0; background: var(--bg); }
</style>
