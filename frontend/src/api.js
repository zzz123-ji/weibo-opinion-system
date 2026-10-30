import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (resp) => resp,
  (error) => {
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      if (!location.hash.includes('/login')) location.hash = '#/login'
    }
    return Promise.reject(error)
  }
)

export function errorMessage(error) {
  const detail = error?.response?.data?.detail
  if (Array.isArray(detail)) return detail.map((x) => x.msg).join('；')
  return detail || error?.message || '请求失败'
}

export default api
