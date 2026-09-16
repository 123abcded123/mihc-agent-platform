// 后端 API 封装（对齐后端 app.py 的 /api/v1 接口契约）

const BASE = import.meta.env.VITE_API_BASE || ''

const TOKEN_KEY = 'mihc_token'
const USER_KEY = 'mihc_user'

export const getToken = () => localStorage.getItem(TOKEN_KEY) || ''
export const setToken = (token) => localStorage.setItem(TOKEN_KEY, token)
export const clearToken = () => {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(USER_KEY)
}
export const getStoredUser = () => {
  try { return JSON.parse(localStorage.getItem(USER_KEY) || 'null') } catch { return null }
}

async function request(url, options = {}) {
  const token = getToken()
  const resp = await fetch(`${BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
    ...options,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    const msg = body?.error?.message || body?.detail?.error?.message || `HTTP ${resp.status}`
    const error = new Error(msg)
    error.status = resp.status
    throw error
  }
  return resp.json()
}

export const login = (username, password) =>
  request('/api/v1/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) })

export const register = (username, password, displayName = '') =>
  request('/api/v1/auth/register', { method: 'POST', body: JSON.stringify({ username, password, display_name: displayName }) })

export const chat = (query, sessionId, projectId = null) =>
  request('/api/v1/chat', {
    method: 'POST',
    body: JSON.stringify({ query, session_id: sessionId || null, project_id: projectId || null }),
  })

export const listProjects = () => request('/api/v1/projects')

export const createProject = (projectName, description = '') =>
  request('/api/v1/projects', { method: 'POST', body: JSON.stringify({ project_name: projectName, description }) })

export const getProject = (projectId) => request(`/api/v1/projects/${projectId}`)

export const addSample = (projectId, { sample_id, group, batch = '', sample_type = '' }) =>
  request(`/api/v1/projects/${projectId}/samples`, {
    method: 'POST',
    body: JSON.stringify({ sample_id, group, batch, sample_type }),
  })

export const addMarker = (projectId, { name, channel = '', antibody = '', threshold = null, unit = '' }) =>
  request(`/api/v1/projects/${projectId}/markers`, {
    method: 'POST',
    body: JSON.stringify({ name, channel, antibody, threshold, unit }),
  })

export const uploadProjectFile = async (projectId, file, kind = 'unknown') => {
  const form = new FormData()
  form.append('file', file)
  form.append('kind', kind)
  const resp = await fetch(`${BASE}/api/v1/projects/${projectId}/files`, {
    method: 'POST',
    headers: getToken() ? { Authorization: `Bearer ${getToken()}` } : {},
    body: form,
  })
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body?.error?.message || body?.detail?.error?.message || `HTTP ${resp.status}`)
  }
  return resp.json()
}

export const analyzeTable = (projectId, { question, file_ids }) =>
  request(`/api/v1/projects/${projectId}/analysis/table`, {
    method: 'POST',
    body: JSON.stringify({ project_id: projectId, question, file_ids }),
  })

export const getAnalysisRun = (runId) => request(`/api/v1/analysis-runs/${runId}`)

export const ingestLiterature = (query, maxResults = 10, maxDownload = 5) =>
  request('/api/v1/literature/ingest', {
    method: 'POST',
    body: JSON.stringify({ query, max_results: maxResults, max_download: maxDownload }),
  })

export const listDocuments = () => request('/api/v1/documents')

export const ingestDocument = async (file, fields = {}) => {
  const form = new FormData()
  form.append('file', file)
  if (fields.version) form.append('version', fields.version)
  if (fields.source) form.append('source', fields.source)
  if (fields.permission) form.append('permission', fields.permission)
  const resp = await fetch(`${BASE}/api/v1/documents/ingest`, {
    method: 'POST',
    headers: getToken() ? { Authorization: `Bearer ${getToken()}` } : {},
    body: form,
  })
  if (!resp.ok) throw new Error((await resp.json().catch(() => ({})))?.error?.message || `HTTP ${resp.status}`)
  return resp.json()
}
