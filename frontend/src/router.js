import { createRouter, createWebHashHistory } from 'vue-router'
import MainLayout from './layout/MainLayout.vue'
import LoginView from './views/LoginView.vue'
import FeedView from './views/FeedView.vue'
import TrendsView from './views/TrendsView.vue'
import LabelView from './views/LabelView.vue'
import SettingsView from './views/SettingsView.vue'

const routes = [
  { path: '/login', component: LoginView, meta: { public: true } },
  {
    path: '/',
    component: MainLayout,
    redirect: '/feed',
    children: [
      { path: 'feed', component: FeedView },
      { path: 'trends', component: TrendsView },
      { path: 'label', component: LabelView },
      { path: 'settings', component: SettingsView, meta: { admin: true } }
    ]
  }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

router.beforeEach((to) => {
  const token = localStorage.getItem('token')
  if (!to.meta.public && !token) return '/login'
  if (to.path === '/login' && token) return '/feed'
  if (to.meta.admin) {
    const user = JSON.parse(localStorage.getItem('user') || '{}')
    if (user.role !== 'admin') return '/feed'
  }
  return true
})

export default router
