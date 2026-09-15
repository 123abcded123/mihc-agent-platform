<template>
  <el-container class="layout">
    <!-- 侧边栏：会话列表 -->
    <el-aside width="264px" class="sidebar">
      <div class="brand">
        <span class="logo">⚕</span>
        <div class="brand-text">
          <div class="brand-title">MIHC Agent</div>
          <div class="brand-sub">多组学 mIHC 科研平台</div>
        </div>
      </div>

      <el-button class="new-chat-btn" type="primary" @click="newConversation">
        <el-icon style="margin-right:6px"><Plus /></el-icon>新对话
      </el-button>

      <div class="conv-label">会话记录</div>
      <div class="conv-list">
        <div v-for="c in conversations" :key="c.id"
          class="conv-item" :class="{ active: c.id === activeId }" @click="switchConversation(c.id)">
          <div class="conv-main">
            <div class="conv-title">{{ c.title || '新对话' }}</div>
            <div class="conv-meta">{{ formatTime(c.updatedAt) }} · {{ c.messages.length }} 条</div>
          </div>
          <el-icon class="conv-del" @click.stop="removeConversation(c.id)"><Close /></el-icon>
        </div>
        <div v-if="!conversations.length" class="conv-empty">暂无会话，点击上方「新对话」开始</div>
      </div>

      <div class="sidebar-footer">
        <div v-if="user" class="user-box">
          <div class="user-avatar">{{ user.display_name.slice(0, 1) }}</div>
          <div class="user-info">
            <div class="user-name">{{ user.display_name }}</div>
            <el-link type="primary" :underline="false" @click="onUserCommand('logout')">退出登录</el-link>
          </div>
        </div>
        <el-button v-else class="login-btn" type="primary" plain @click="authVisible = true">登录 / 注册</el-button>
      </div>
    </el-aside>

    <!-- 主区 -->
    <el-container class="right">
      <el-header class="chat-header">
        <div class="header-left">
          <span class="header-title">{{ activeConv ? activeConv.title : '新对话' }}</span>
          <el-tag v-if="selectedProject" size="small" type="success" effect="light" class="project-tag">
            项目：{{ selectedProject.project_name }}
          </el-tag>
        </div>
        <div class="header-actions">
          <el-button size="small" text @click="openProjects"><el-icon style="margin-right:4px"><Folder /></el-icon>项目工作台</el-button>
          <el-button size="small" text @click="showDocs = true"><el-icon style="margin-right:4px"><Collection /></el-icon>知识库管理</el-button>
        </div>
      </el-header>

      <el-main class="chat-main">
        <div class="chat-list" ref="chatList">
          <div v-if="!messages.length" class="empty">
            <div class="empty-logo">⚕</div>
            <h2>向 MIHC 科研 Agent 提问</h2>
            <p>产品咨询 · 实验设计 · 表格分析 · 文献推荐 · 报告生成</p>
            <div class="example-chips">
              <span v-for="ex in examples" :key="ex" class="chip" @click="useExample(ex)">{{ ex }}</span>
            </div>
          </div>

          <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
            <div class="avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
            <div class="bubble">
              <div class="content markdown" v-html="render(m.content)"></div>

              <div v-if="m.citations && m.citations.length" class="meta">
                <el-collapse>
                  <el-collapse-item :title="`引用来源（${m.citations.length}）`">
                    <div v-for="(c, j) in m.citations" :key="j" class="citation">
                      <b>[{{ j + 1 }}]</b> {{ c.title }} — {{ c.source }}
                      <div class="snippet">{{ c.text_snippet }}</div>
                    </div>
                  </el-collapse-item>
                </el-collapse>
              </div>

              <div v-if="m.risks && m.risks.length" class="risks">
                <el-alert v-for="(r, j) in m.risks" :key="j"
                  :title="r.message" :type="r.level === 'critical' ? 'error' : r.level === 'warning' ? 'warning' : 'info'"
                  :closable="false" show-icon size="small" />
              </div>

              <div v-if="m.agent_chain && m.agent_chain.length" class="chain">
                <el-tag v-for="a in m.agent_chain" :key="a" size="small" effect="plain">{{ a }}</el-tag>
                <span class="trace">trace: {{ m.trace_id }}</span>
              </div>
            </div>
          </div>

          <div v-if="loading" class="msg assistant">
            <div class="avatar">AI</div>
            <div class="bubble typing"><span class="dot"></span><span class="dot"></span><span class="dot"></span> 检索 + 多 Agent 编排中…</div>
          </div>
        </div>

        <div class="input-area">
          <div class="input-card">
            <el-input v-model="query" type="textarea" :rows="2" resize="none" class="query-input"
              placeholder="输入科研问题，回车发送，Shift+回车换行"
              @keydown.enter.exact.prevent="send" />
            <div class="input-actions">
              <div class="input-left">
                <el-select v-model="selectedProjectId" size="small" placeholder="关联项目（可选）"
                  clearable style="width:210px" @change="onProjectChange">
                  <el-option v-for="p in projects" :key="p.project_id" :label="`${p.project_name}（${p.project_id}）`" :value="p.project_id" />
                </el-select>
                <span class="hint" v-if="!user">登录后可关联项目编号</span>
              </div>
              <el-button type="primary" class="send-btn" :loading="loading" @click="send">
                发送<el-icon style="margin-left:4px"><Promotion /></el-icon>
              </el-button>
            </div>
          </div>
          <div class="disclaimer">回答仅供科研参考，不构成诊疗建议</div>
        </div>
      </el-main>
    </el-container>

    <!-- 登录/注册 -->
    <el-dialog v-model="authVisible" title="登录 MIHC 平台" width="400px" class="auth-dialog">
      <el-tabs v-model="authMode">
        <el-tab-pane label="登录" name="login">
          <el-form label-position="top" @submit.prevent>
            <el-form-item label="用户名"><el-input v-model="authForm.username" size="large" /></el-form-item>
            <el-form-item label="密码"><el-input v-model="authForm.password" type="password" size="large" show-password @keydown.enter="doLogin" /></el-form-item>
            <el-button type="primary" size="large" style="width:100%" :loading="authLoading" @click="doLogin">登录</el-button>
          </el-form>
        </el-tab-pane>
        <el-tab-pane label="注册" name="register">
          <el-form label-position="top" @submit.prevent>
            <el-form-item label="用户名"><el-input v-model="authForm.username" size="large" /></el-form-item>
            <el-form-item label="密码（至少 8 位）"><el-input v-model="authForm.password" type="password" size="large" show-password /></el-form-item>
            <el-button type="primary" size="large" style="width:100%" :loading="authLoading" @click="doRegister">注册</el-button>
          </el-form>
        </el-tab-pane>
      </el-tabs>
    </el-dialog>

    <!-- 项目工作台 -->
    <el-drawer v-model="showProjects" title="客户项目工作台" size="46%">
      <div v-if="!user" class="tip-box">
        <el-alert title="请先登录后再管理项目、上传数据和分析" type="warning" :closable="false" />
        <el-button type="primary" style="margin-top:12px" @click="authVisible = true">去登录</el-button>
      </div>
      <template v-else>
        <div class="toolbar">
          <el-input v-model="newProjectName" placeholder="新项目名称，如：肿瘤免疫微环境研究" style="flex:1" />
          <el-button type="primary" @click="doCreateProject">创建项目</el-button>
          <el-button @click="loadProjects">刷新</el-button>
        </div>
        <el-divider>我的项目</el-divider>
        <el-table :data="projects" size="small" highlight-current-row @current-change="selectProject">
          <el-table-column prop="project_name" label="项目名" min-width="150" />
          <el-table-column prop="project_id" label="项目 ID" width="150" />
          <el-table-column prop="status" label="状态" width="70" />
        </el-table>

        <template v-if="currentProject">
          <el-divider>{{ currentProject.project_name }} — 项目明细</el-divider>
          <el-tabs>
            <el-tab-pane label="样本 / Marker">
              <div class="toolbar">
                <el-input v-model="sampleForm.sample_id" placeholder="样本 ID，如 S1" style="width:120px" />
                <el-input v-model="sampleForm.group" placeholder="分组，如 treatment" style="width:140px" />
                <el-button size="small" type="primary" @click="doAddSample">添加样本</el-button>
              </div>
              <div class="toolbar" style="margin-top:6px">
                <el-input v-model="markerForm.name" placeholder="Marker，如 CD8" style="width:120px" />
                <el-input v-model="markerForm.threshold" placeholder="阳性阈值，如 0.5" style="width:140px" />
                <el-button size="small" type="primary" @click="doAddMarker">添加 Marker</el-button>
              </div>
              <el-table :data="projectDetail.samples" size="small" style="margin-top:8px">
                <el-table-column prop="sample_id" label="样本" width="100" />
                <el-table-column prop="group" label="分组" width="120" />
                <el-table-column prop="sample_type" label="类型" />
              </el-table>
              <el-table :data="projectDetail.markers" size="small" style="margin-top:8px">
                <el-table-column prop="name" label="Marker" width="100" />
                <el-table-column prop="channel" label="通道" width="90" />
                <el-table-column prop="threshold" label="阈值" width="90" />
                <el-table-column prop="unit" label="单位" />
              </el-table>
            </el-tab-pane>

            <el-tab-pane label="数据文件">
              <el-select v-model="uploadKind" size="small" style="width:200px;margin-bottom:8px">
                <el-option label="细胞表格 cell_table" value="cell_table" />
                <el-option label="组学表格 omics_table" value="omics_table" />
                <el-option label="原始图像 image" value="image" />
                <el-option label="其他" value="unknown" />
              </el-select>
              <el-upload drag :show-file-list="false" :http-request="doUploadFile" :disabled="uploading">
                <el-icon><UploadFilled /></el-icon>
                <div class="el-upload__text">拖入或<em>点击上传</em>（CSV/Excel/图像）</div>
              </el-upload>
              <el-table :data="projectDetail.files" size="small" style="margin-top:8px">
                <el-table-column prop="file_name" label="文件名" min-width="160" />
                <el-table-column prop="kind" label="类型" width="110" />
                <el-table-column prop="status" label="状态" width="90" />
              </el-table>
            </el-tab-pane>

            <el-tab-pane label="表格分析">
              <div class="tip-box">
                <el-alert title="统计由确定性工具计算：样本对齐 → 字段/阈值校验 → 阳性率/组间比较，模型只负责解释" type="info" :closable="false" />
              </div>
              <el-input v-model="analysisForm.question" type="textarea" :rows="2"
                placeholder="如：比较治疗组和对照组 CD8 阳性率" style="margin:8px 0" />
              <div class="toolbar">
                <el-checkbox-group v-model="analysisForm.file_ids">
                  <el-checkbox v-for="f in projectDetail.files.filter(x => x.kind === 'cell_table' || x.kind === 'omics_table')" :key="f.file_id" :label="f.file_id">{{ f.file_name }}</el-checkbox>
                </el-checkbox-group>
              </div>
              <el-button type="primary" size="small" :loading="analyzing" @click="doAnalyze" style="margin-top:8px">开始分析</el-button>
              <el-alert v-if="analysisResult" :type="analysisResult.status === 'completed' ? 'success' : 'warning'"
                :title="`分析状态：${analysisResult.status}（run: ${analysisResult.analysis_run_id}）`" :closable="false" style="margin-top:8px" />
              <pre v-if="analysisResult" class="result-json">{{ pretty(analysisResult.data) }}</pre>
            </el-tab-pane>
          </el-tabs>
        </template>
      </template>
    </el-drawer>

    <!-- 知识库抽屉（含文献自动下载） -->
    <el-drawer v-model="showDocs" title="知识库管理" size="40%">
      <el-divider>mIHC 文献自动下载入库</el-divider>
      <div class="lit-box">
        <el-input v-model="litQuery" placeholder="PubMed 检索词，如：mIHC tumor immune microenvironment" />
        <div class="toolbar" style="margin-top:8px">
          <el-button type="primary" :loading="litLoading" @click="doLiterature">检索并下载入库</el-button>
          <span class="hint">自动检索 PubMed → 下载开放获取 PDF → 解析入库</span>
        </div>
        <el-alert v-if="litResult" :type="litResult.ingested ? 'success' : 'info'" :closable="false" style="margin-top:8px"
          :title="`已下载 ${litResult.downloaded} 篇，成功入库 ${litResult.ingested} 篇`" />
        <pre v-if="litResult" class="result-json">{{ pretty(litResult.details) }}</pre>
      </div>

      <el-divider>手动上传文档</el-divider>
      <el-upload drag :show-file-list="false" :http-request="doIngest" accept=".pdf,.docx,.md,.txt">
        <el-icon><UploadFilled /></el-icon>
        <div class="el-upload__text">拖入文件或<em>点击上传</em>（PDF/DOCX/MD/TXT）</div>
        <template #tip><div class="el-upload__tip">入库流程：解析 → 清洗 → 切分 → 嵌入 → Milvus + 关键词库 + PostgreSQL 元数据</div></template>
      </el-upload>

      <el-divider>已入库文档</el-divider>
      <el-table :data="documents" size="small" stripe>
        <el-table-column prop="file_name" label="文件名" min-width="160" />
        <el-table-column prop="chunk_count" label="片段数" width="80" />
        <el-table-column prop="version" label="版本" width="100" />
        <el-table-column prop="permission" label="权限" width="110" />
      </el-table>
    </el-drawer>
  </el-container>
</template>

<script setup>
import { ref, nextTick, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Loading, UploadFilled, Plus, Close, Promotion, Folder, Collection } from '@element-plus/icons-vue'
import { marked } from 'marked'
import {
  chat, login, register, setToken, clearToken, getStoredUser,
  listProjects, createProject, getProject, addSample, addMarker,
  uploadProjectFile, analyzeTable, ingestDocument, listDocuments, ingestLiterature,
} from './api'

const CONV_KEY = 'mihc_conversations'

const query = ref('')
const loading = ref(false)
const chatList = ref(null)
const showDocs = ref(false)
const documents = ref([])

const examples = [
  'HyperView 适合什么样本？',
  '有没有适合肿瘤免疫微环境的成熟 Panel？',
  '我想研究 CD8、FOXP3、PD-L1，应该如何做预实验？',
  '推荐几篇 mIHC 空间分析和免疫浸润相关文献',
]

// ---- 会话侧边栏（localStorage 持久化）----
const conversations = ref([])
const activeId = ref(null)
const activeConv = computed(() => conversations.value.find(c => c.id === activeId.value) || null)
const messages = computed(() => activeConv.value ? activeConv.value.messages : [])

const loadConvs = () => {
  try { conversations.value = JSON.parse(localStorage.getItem(CONV_KEY)) || [] } catch { conversations.value = [] }
  if (conversations.value.length) activeId.value = conversations.value[0].id
}
const saveConvs = () => localStorage.setItem(CONV_KEY, JSON.stringify(conversations.value))

const newConversation = () => {
  const conv = {
    id: 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    title: '新对话',
    serverSessionId: null,
    messages: [],
    updatedAt: Date.now(),
  }
  conversations.value.unshift(conv)
  activeId.value = conv.id
  saveConvs()
  query.value = ''
}

const switchConversation = (id) => {
  activeId.value = id
  nextTick(scrollBottom)
}

const removeConversation = async (id) => {
  const conv = conversations.value.find(c => c.id === id)
  if (!conv) return
  try { await ElMessageBox.confirm(`删除会话「${conv.title}」？`, '提示', { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' }) } catch { return }
  conversations.value = conversations.value.filter(c => c.id !== id)
  if (activeId.value === id) activeId.value = conversations.value.length ? conversations.value[0].id : null
  saveConvs()
  ElMessage.success('会话已删除')
}

const formatTime = (ts) => {
  if (!ts) return ''
  const d = new Date(ts)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  return sameDay ? `今天 ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
    : `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
}

// ---- 鉴权 ----
const user = ref(getStoredUser())
const authVisible = ref(false)
const authMode = ref('login')
const authLoading = ref(false)
const authForm = ref({ username: '', password: '' })

const applyAuth = (data) => {
  setToken(data.token)
  user.value = data.user
  authVisible.value = false
  ElMessage.success(`欢迎，${data.user.display_name}`)
  loadProjects()
}

const doLogin = async () => {
  authLoading.value = true
  try { applyAuth(await login(authForm.value.username, authForm.value.password)) }
  catch (e) { ElMessage.error(e.message) } finally { authLoading.value = false }
}

const doRegister = async () => {
  authLoading.value = true
  try { applyAuth(await register(authForm.value.username, authForm.value.password)) }
  catch (e) { ElMessage.error(e.message) } finally { authLoading.value = false }
}

const onUserCommand = (command) => {
  if (command === 'logout') {
    clearToken()
    user.value = null
    projects.value = []
    currentProject.value = null
    selectedProjectId.value = null
    ElMessage.success('已退出登录')
  }
}

// ---- 项目 ----
const showProjects = ref(false)
const projects = ref([])
const currentProject = ref(null)
const projectDetail = ref({ samples: [], markers: [], files: [] })
const newProjectName = ref('')
const sampleForm = ref({ sample_id: '', group: '', batch: '', sample_type: '' })
const markerForm = ref({ name: '', channel: '', antibody: '', threshold: '', unit: '' })
const uploadKind = ref('cell_table')
const uploading = ref(false)
const analyzing = ref(false)
const analysisForm = ref({ question: '', file_ids: [] })
const analysisResult = ref(null)
const selectedProjectId = ref(null)
const selectedProject = computed(() => projects.value.find(p => p.project_id === selectedProjectId.value) || null)

const onProjectChange = () => { ElMessage.success(selectedProjectId.value ? `已关联项目：${selectedProjectId.value}` : '已取消项目关联') }

const openProjects = () => {
  showProjects.value = true
  if (user.value) loadProjects()
}

const loadProjects = async () => {
  try { projects.value = (await listProjects()).projects } catch (e) { ElMessage.error(e.message) }
}

const doCreateProject = async () => {
  const name = newProjectName.value.trim()
  if (!name) return ElMessage.warning('请输入项目名称')
  try {
    const created = await createProject(name)
    newProjectName.value = ''
    ElMessage.success(`项目已创建：${created.project_id}`)
    await loadProjects()
  } catch (e) { ElMessage.error(e.message) }
}

const selectProject = async (row) => {
  if (!row) return
  try {
    const detail = await getProject(row.project_id)
    currentProject.value = detail.project
    projectDetail.value = detail
    analysisResult.value = null
  } catch (e) { ElMessage.error(e.message) }
}

const doAddSample = async () => {
  if (!sampleForm.value.sample_id) return ElMessage.warning('请输入样本 ID')
  try {
    await addSample(currentProject.value.project_id, sampleForm.value)
    sampleForm.value = { sample_id: '', group: '', batch: '', sample_type: '' }
    ElMessage.success('样本已添加')
    selectProject(currentProject.value)
  } catch (e) { ElMessage.error(e.message) }
}

const doAddMarker = async () => {
  if (!markerForm.value.name) return ElMessage.warning('请输入 Marker 名称')
  try {
    const threshold = markerForm.value.threshold === '' ? null : Number(markerForm.value.threshold)
    await addMarker(currentProject.value.project_id, { ...markerForm.value, threshold })
    markerForm.value = { name: '', channel: '', antibody: '', threshold: '', unit: '' }
    ElMessage.success('Marker 已添加')
    selectProject(currentProject.value)
  } catch (e) { ElMessage.error(e.message) }
}

const doUploadFile = async ({ file }) => {
  if (!currentProject.value) return
  uploading.value = true
  try {
    await uploadProjectFile(currentProject.value.project_id, file, uploadKind.value)
    ElMessage.success(`文件已上传：${file.name}`)
    selectProject(currentProject.value)
  } catch (e) { ElMessage.error(e.message) } finally { uploading.value = false }
}

const doAnalyze = async () => {
  if (!analysisForm.value.question.trim()) return ElMessage.warning('请输入分析问题')
  if (!analysisForm.value.file_ids.length) return ElMessage.warning('请选择至少一个表格文件')
  analyzing.value = true
  try {
    analysisResult.value = await analyzeTable(currentProject.value.project_id, {
      question: analysisForm.value.question,
      file_ids: analysisForm.value.file_ids,
    })
  } catch (e) { ElMessage.error(e.message) } finally { analyzing.value = false }
}

// ---- 知识库 + 文献 ----
const litQuery = ref('mIHC tumor immune microenvironment')
const litLoading = ref(false)
const litResult = ref(null)

const doLiterature = async () => {
  if (!litQuery.value.trim()) return ElMessage.warning('请输入检索词')
  litLoading.value = true
  litResult.value = null
  try {
    litResult.value = await ingestLiterature(litQuery.value.trim(), 10, 5)
    if (!litResult.value.ingested) ElMessage.info(litResult.value.message || '未检索到可下载文献')
  } catch (e) { ElMessage.error(e.message) } finally { litLoading.value = false }
}

const doIngest = async ({ file }) => {
  try {
    const result = await ingestDocument(file)
    ElMessage.success(`入库成功：${result.file_name}，${result.chunks} 个片段`)
    documents.value = (await listDocuments()).documents
  } catch (e) { ElMessage.error(`入库失败：${e.message}`) }
}

// ---- 对话 ----
marked.setOptions({ breaks: true })
const render = (text) => marked.parse(text || '')
const pretty = (value) => JSON.stringify(value ?? {}, null, 2)

const scrollBottom = async () => {
  await nextTick()
  if (chatList.value) chatList.value.scrollTop = chatList.value.scrollHeight
}

const useExample = (text) => { query.value = text }

const send = async () => {
  const q = query.value.trim()
  if (!q || loading.value) return
  if (!activeConv.value) newConversation()
  const conv = activeConv.value
  query.value = ''
  if (!conv.messages.length) conv.title = q.slice(0, 18)
  conv.messages.push({ role: 'user', content: q })
  conv.updatedAt = Date.now()
  loading.value = true
  saveConvs()
  await scrollBottom()
  try {
    const data = await chat(q, conv.serverSessionId, selectedProjectId.value || null)
    conv.serverSessionId = data.session_id || conv.serverSessionId
    conv.messages.push({
      role: 'assistant',
      content: data.answer,
      citations: data.citations,
      risks: data.risks,
      agent_chain: data.agent_chain,
      trace_id: data.trace_id,
      branch: data.branch,
    })
  } catch (e) {
    conv.messages.push({ role: 'assistant', content: `请求失败：${e.message}` })
  } finally {
    conv.updatedAt = Date.now()
    loading.value = false
    saveConvs()
    await scrollBottom()
  }
}

onMounted(() => {
  loadConvs()
  if (user.value) loadProjects()
  try { listDocuments().then(r => documents.value = r.documents) } catch { /* 未登录 */ }
})
</script>

<style>
:root {
  --el-color-primary: #0d9488;
  --el-color-primary-light-3: #2dd4bf;
  --el-color-primary-light-5: #5eead4;
  --el-color-primary-light-7: #99f6e4;
  --el-color-primary-light-8: #ccfbf1;
  --el-color-primary-light-9: #f0fdfa;
  --el-color-primary-dark-2: #0f766e;
  --el-border-radius-base: 8px;
}

body { margin: 0; font-family: "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif; background: #f1f5f9; }
.layout { height: 100vh; }

/* ---- 侧边栏 ---- */
.sidebar {
  background: #ffffff;
  border-right: 1px solid #e2e8f0;
  display: flex;
  flex-direction: column;
  padding: 16px 12px;
  box-sizing: border-box;
}
.brand { display: flex; align-items: center; gap: 10px; padding: 4px 6px 14px; }
.logo {
  width: 40px; height: 40px; border-radius: 12px; flex-shrink: 0;
  background: linear-gradient(135deg, #0d9488, #06b6d4);
  color: #fff; font-size: 22px; display: flex; align-items: center; justify-content: center;
  box-shadow: 0 4px 10px rgba(13,148,136,.28);
}
.brand-title { font-weight: 700; font-size: 16px; color: #0f172a; }
.brand-sub { font-size: 11px; color: #94a3b8; }

.new-chat-btn { width: 100%; margin-bottom: 14px; border-radius: 10px; font-weight: 600; }

.conv-label { font-size: 11px; color: #94a3b8; padding: 0 6px 6px; letter-spacing: .5px; }
.conv-list { flex: 1; overflow-y: auto; margin: 0 -4px; }
.conv-item {
  display: flex; align-items: center; gap: 6px; padding: 9px 10px; border-radius: 10px;
  cursor: pointer; margin-bottom: 4px; transition: background .15s;
}
.conv-item:hover { background: #f1f5f9; }
.conv-item.active { background: #f0fdfa; border: 1px solid #99f6e4; }
.conv-main { flex: 1; min-width: 0; }
.conv-title { font-size: 13px; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
.conv-item.active .conv-title { color: #0f766e; font-weight: 600; }
.conv-meta { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.conv-del { color: #cbd5e1; font-size: 14px; visibility: hidden; }
.conv-item:hover .conv-del { visibility: visible; }
.conv-del:hover { color: #ef4444; }
.conv-empty { font-size: 12px; color: #cbd5e1; text-align: center; padding: 20px 0; }

.sidebar-footer { border-top: 1px solid #e2e8f0; padding-top: 12px; }
.user-box { display: flex; align-items: center; gap: 10px; }
.user-avatar {
  width: 34px; height: 34px; border-radius: 50%; background: linear-gradient(135deg, #0d9488, #06b6d4);
  color: #fff; display: flex; align-items: center; justify-content: center; font-weight: 700;
}
.user-info { display: flex; flex-direction: column; }
.user-name { font-size: 13px; color: #334155; font-weight: 600; }
.login-btn { width: 100%; border-radius: 10px; }

/* ---- 主区头部 ---- */
.right { background: #f1f5f9; }
.chat-header {
  display: flex; align-items: center; justify-content: space-between;
  background: #fff; border-bottom: 1px solid #e2e8f0; padding: 0 24px; height: 56px;
}
.header-left { display: flex; align-items: center; gap: 10px; min-width: 0; }
.header-title { font-weight: 600; font-size: 15px; color: #0f172a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 400px; }
.project-tag { flex-shrink: 0; }
.header-actions { display: flex; gap: 4px; }

/* ---- 聊天区 ---- */
.chat-main { display: flex; flex-direction: column; padding: 0; }
.chat-list { flex: 1; overflow-y: auto; padding: 24px 20% 8px; }
.empty { text-align: center; color: #64748b; margin-top: 10%; }
.empty-logo {
  width: 64px; height: 64px; margin: 0 auto 16px; border-radius: 20px; font-size: 34px;
  background: linear-gradient(135deg, #0d9488, #06b6d4); color: #fff;
  display: flex; align-items: center; justify-content: center; box-shadow: 0 8px 24px rgba(13,148,136,.3);
}
.empty h2 { color: #0f172a; margin: 0 0 8px; }
.empty p { margin: 0 0 20px; font-size: 13px; }
.example-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; }
.chip {
  font-size: 12px; padding: 8px 14px; border-radius: 999px; cursor: pointer;
  background: #fff; border: 1px solid #e2e8f0; color: #475569; transition: all .15s;
}
.chip:hover { border-color: #0d9488; color: #0d9488; background: #f0fdfa; }

.msg { display: flex; gap: 12px; margin: 18px 0; }
.msg.user { flex-direction: row-reverse; }
.avatar {
  width: 38px; height: 38px; border-radius: 12px; flex-shrink: 0;
  background: linear-gradient(135deg, #0d9488, #06b6d4); color: #fff;
  display: flex; align-items: center; justify-content: center; font-size: 14px; font-weight: 600;
  box-shadow: 0 3px 8px rgba(13,148,136,.22);
}
.msg.user .avatar { background: linear-gradient(135deg, #6366f1, #8b5cf6); box-shadow: 0 3px 8px rgba(99,102,241,.22); }
.bubble {
  max-width: 78%; background: #fff; border-radius: 4px 14px 14px 14px; padding: 12px 16px;
  box-shadow: 0 1px 3px rgba(15,23,42,.06); border: 1px solid #f1f5f9; line-height: 1.7;
}
.msg.user .bubble { background: linear-gradient(135deg, #f0fdfa, #ecfeff); border-radius: 14px 4px 14px 14px; border-color: #ccfbf1; }
.typing { display: flex; align-items: center; gap: 5px; color: #94a3b8; }
.dot { width: 6px; height: 6px; border-radius: 50%; background: #0d9488; animation: blink 1.2s infinite; }
.dot:nth-child(2) { animation-delay: .2s; }
.dot:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%, 100% { opacity: .2; } 50% { opacity: 1; } }

.markdown p { margin: 4px 0; }
.markdown pre { background: #f8fafc; padding: 12px; border-radius: 8px; overflow-x: auto; border: 1px solid #e2e8f0; }
.markdown code { background: #f1f5f9; padding: 2px 5px; border-radius: 4px; font-size: 13px; }
.markdown pre code { background: none; padding: 0; }
.markdown h1, .markdown h2, .markdown h3 { margin: 10px 0 4px; }

.meta { margin-top: 10px; }
.citation { font-size: 12px; color: #475569; padding: 6px 0; border-bottom: 1px dashed #e2e8f0; }
.snippet { color: #94a3b8; margin-top: 2px; }
.risks { margin-top: 10px; display: flex; flex-direction: column; gap: 4px; }
.chain { margin-top: 10px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.trace { font-size: 11px; color: #cbd5e1; margin-left: auto; }

/* ---- 输入区 ---- */
.input-area { padding: 8px 20% 16px; }
.input-card { background: #fff; border-radius: 14px; padding: 12px 14px 10px; box-shadow: 0 4px 16px rgba(15,23,42,.08); border: 1px solid #e2e8f0; }
.query-input :deep(.el-textarea__inner) { box-shadow: none !important; border: none; font-size: 14px; }
.input-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 6px; }
.input-left { display: flex; align-items: center; gap: 10px; }
.send-btn { border-radius: 10px; font-weight: 600; padding: 0 20px; }
.hint { font-size: 11px; color: #cbd5e1; }
.disclaimer { text-align: center; font-size: 11px; color: #cbd5e1; margin-top: 8px; }

/* ---- 抽屉/弹窗通用 ---- */
.toolbar { display: flex; gap: 8px; align-items: center; }
.tip-box { padding: 8px 0; }
.result-json { background: #f8fafc; padding: 10px; border-radius: 8px; font-size: 12px; max-height: 300px; overflow: auto; white-space: pre-wrap; word-break: break-all; border: 1px solid #e2e8f0; }
.lit-box { background: #f0fdfa; border: 1px solid #ccfbf1; border-radius: 10px; padding: 12px; }
.auth-dialog { border-radius: 14px; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
