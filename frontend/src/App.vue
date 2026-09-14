<template>
  <el-container class="layout">
    <!-- 顶栏 -->
    <el-header class="header">
      <div class="brand">
        <span class="logo">⚕</span>
        <span class="title">MIHC Agent · 医疗科研智能平台</span>
      </div>
      <div class="header-actions">
        <el-tag size="small" type="info">产品咨询 · 实验设计 · 表格分析 · 文献推荐 · 报告</el-tag>
        <el-button size="small" text @click="newSession">新会话</el-button>
        <el-button size="small" text @click="openProjects">项目工作台</el-button>
        <el-button size="small" text @click="showDocs = true">知识库管理</el-button>
        <el-dropdown v-if="user" @command="onUserCommand">
          <el-button size="small" type="primary" text>{{ user.display_name }}</el-button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
        <el-button v-else size="small" type="primary" @click="authVisible = true">登录 / 注册</el-button>
      </div>
    </el-header>

    <el-main class="main">
      <!-- 对话区 -->
      <div class="chat-list" ref="chatList">
        <div v-if="messages.length === 0" class="empty">
          <h2>向 MIHC 科研 Agent 提问</h2>
          <p>示例：HyperView 适合什么样本？CD8、FOXP3、PD-L1 应该如何做预实验？</p>
        </div>
        <div v-for="(m, i) in messages" :key="i" class="msg" :class="m.role">
          <div class="avatar">{{ m.role === 'user' ? '我' : 'AI' }}</div>
          <div class="bubble">
            <div class="content markdown" v-html="render(m.content)"></div>

            <!-- 引用 -->
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

            <!-- 风险提示 -->
            <div v-if="m.risks && m.risks.length" class="risks">
              <el-alert v-for="(r, j) in m.risks" :key="j"
                :title="r.message" :type="r.level === 'critical' ? 'error' : r.level === 'warning' ? 'warning' : 'info'"
                :closable="false" show-icon size="small" />
            </div>

            <!-- 链路信息 -->
            <div v-if="m.agent_chain && m.agent_chain.length" class="chain">
              <el-tag v-for="a in m.agent_chain" :key="a" size="small" effect="plain">{{ a }}</el-tag>
              <span class="trace">trace: {{ m.trace_id }}</span>
            </div>
          </div>
        </div>

        <div v-if="loading" class="msg assistant">
          <div class="avatar">AI</div>
          <div class="bubble"><el-icon class="is-loading"><Loading /></el-icon> 处理中（检索 + 多 Agent 编排）…</div>
        </div>
      </div>

      <!-- 输入区 -->
      <div class="input-area">
        <el-input v-model="query" type="textarea" :rows="3" resize="none"
          placeholder="输入科研问题，回车发送，Shift+回车换行"
          @keydown.enter.exact.prevent="send" />
        <div class="input-actions">
          <span class="hint">会话 {{ sessionId ? sessionId.slice(0, 8) : '—' }} · 回答仅供科研参考</span>
          <el-button type="primary" :loading="loading" @click="send">发送</el-button>
        </div>
      </div>
    </el-main>

    <!-- 登录/注册 -->
    <el-dialog v-model="authVisible" title="登录 MIHC 平台" width="380px">
      <el-tabs v-model="authMode">
        <el-tab-pane label="登录" name="login">
          <el-form label-position="top" @submit.prevent>
            <el-form-item label="用户名"><el-input v-model="authForm.username" /></el-form-item>
            <el-form-item label="密码"><el-input v-model="authForm.password" type="password" show-password /></el-form-item>
            <el-button type="primary" style="width:100%" :loading="authLoading" @click="doLogin">登录</el-button>
          </el-form>
        </el-tab-pane>
        <el-tab-pane label="注册" name="register">
          <el-form label-position="top" @submit.prevent>
            <el-form-item label="用户名"><el-input v-model="authForm.username" /></el-form-item>
            <el-form-item label="密码（至少 8 位）"><el-input v-model="authForm.password" type="password" show-password /></el-form-item>
            <el-button type="primary" style="width:100%" :loading="authLoading" @click="doRegister">注册</el-button>
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
          <el-table-column prop="project_name" label="项目名" min-width="160" />
          <el-table-column prop="project_id" label="项目 ID" width="150" />
          <el-table-column prop="status" label="状态" width="80" />
        </el-table>

        <template v-if="currentProject">
          <el-divider>{{ currentProject.project_name }} — 项目明细</el-divider>
          <el-tabs>
            <!-- 样本与 marker -->
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

            <!-- 文件 -->
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

            <!-- 表格分析 -->
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

    <!-- 知识库抽屉 -->
    <el-drawer v-model="showDocs" title="知识库管理" size="40%">
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
import { ref, nextTick, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Loading, UploadFilled } from '@element-plus/icons-vue'
import { marked } from 'marked'
import {
  chat, login, register, setToken, clearToken, getStoredUser,
  listProjects, createProject, getProject, addSample, addMarker,
  uploadProjectFile, analyzeTable, ingestDocument, listDocuments,
} from './api'

const query = ref('')
const messages = ref([])
const sessionId = ref(null)
const loading = ref(false)
const showDocs = ref(false)
const documents = ref([])
const chatList = ref(null)

// ---- 鉴权 ----
const user = ref(getStoredUser())
const authVisible = ref(false)
const authMode = ref('login')
const authLoading = ref(false)
const authForm = ref({ username: '', password: '' })

// ---- 项目工作台 ----
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

marked.setOptions({ breaks: true })

const render = (text) => marked.parse(text || '')
const pretty = (value) => JSON.stringify(value ?? {}, null, 2)

const scrollBottom = async () => {
  await nextTick()
  if (chatList.value) chatList.value.scrollTop = chatList.value.scrollHeight
}

const send = async () => {
  const q = query.value.trim()
  if (!q || loading.value) return
  query.value = ''
  messages.value.push({ role: 'user', content: q })
  loading.value = true
  await scrollBottom()
  try {
    const data = await chat(q, sessionId.value)
    sessionId.value = data.session_id || sessionId.value
    messages.value.push({
      role: 'assistant',
      content: data.answer,
      citations: data.citations,
      risks: data.risks,
      agent_chain: data.agent_chain,
      trace_id: data.trace_id,
    })
  } catch (e) {
    messages.value.push({ role: 'assistant', content: `请求失败：${e.message}` })
  } finally {
    loading.value = false
    await scrollBottom()
  }
}

const newSession = () => {
  sessionId.value = null
  messages.value = []
}

// ---- 鉴权操作 ----
const applyAuth = (data) => {
  setToken(data.token)
  user.value = data.user
  authVisible.value = false
  ElMessage.success(`欢迎，${data.user.display_name}`)
  loadProjects()
}

const doLogin = async () => {
  authLoading.value = true
  try {
    applyAuth(await login(authForm.value.username, authForm.value.password))
  } catch (e) { ElMessage.error(e.message) } finally { authLoading.value = false }
}

const doRegister = async () => {
  authLoading.value = true
  try {
    applyAuth(await register(authForm.value.username, authForm.value.password))
  } catch (e) { ElMessage.error(e.message) } finally { authLoading.value = false }
}

const onUserCommand = (command) => {
  if (command === 'logout') {
    clearToken()
    user.value = null
    projects.value = []
    currentProject.value = null
    ElMessage.success('已退出登录')
  }
}

// ---- 项目操作 ----
const openProjects = () => {
  showProjects.value = true
  if (user.value) loadProjects()
}

const loadProjects = async () => {
  try {
    projects.value = (await listProjects()).projects
  } catch (e) { ElMessage.error(e.message) }
}

const doCreateProject = async () => {
  const name = newProjectName.value.trim()
  if (!name) return ElMessage.warning('请输入项目名称')
  try {
    await createProject(name)
    newProjectName.value = ''
    ElMessage.success('项目已创建')
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

// ---- 知识库 ----
const doIngest = async ({ file }) => {
  try {
    const result = await ingestDocument(file)
    ElMessage.success(`入库成功：${result.file_name}，${result.chunks} 个片段`)
    documents.value = (await listDocuments()).documents
  } catch (e) {
    ElMessage.error(`入库失败：${e.message}`)
  }
}

onMounted(async () => {
  try {
    documents.value = (await listDocuments()).documents
  } catch (e) { /* 未登录或知识库暂不可用 */ }
})
</script>

<style>
body { margin: 0; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
.layout { height: 100vh; }
.header { display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #e4e7ed; background: #fff; }
.brand { display: flex; align-items: center; gap: 8px; font-weight: 600; }
.logo { font-size: 22px; color: #409eff; }
.header-actions { display: flex; align-items: center; gap: 8px; }
.main { display: flex; flex-direction: column; padding: 0 12%; background: #f5f7fa; }
.chat-list { flex: 1; overflow-y: auto; padding: 16px 0; }
.empty { text-align: center; color: #909399; margin-top: 15%; }
.msg { display: flex; gap: 10px; margin: 12px 0; }
.msg.user { flex-direction: row-reverse; }
.avatar { width: 34px; height: 34px; border-radius: 8px; background: #409eff; color: #fff; display: flex; align-items: center; justify-content: center; flex-shrink: 0; font-size: 13px; }
.msg.user .avatar { background: #67c23a; }
.bubble { max-width: 75%; background: #fff; border-radius: 10px; padding: 12px 14px; box-shadow: 0 1px 4px rgba(0,0,0,.06); }
.msg.user .bubble { background: #e8f3ff; }
.markdown p { margin: 4px 0; }
.markdown pre { background: #f5f7fa; padding: 10px; border-radius: 6px; overflow-x: auto; }
.markdown code { background: #f5f7fa; padding: 2px 4px; border-radius: 4px; }
.meta { margin-top: 8px; }
.citation { font-size: 12px; color: #606266; padding: 4px 0; border-bottom: 1px dashed #ebeef5; }
.snippet { color: #909399; margin-top: 2px; }
.risks { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
.chain { margin-top: 8px; display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.trace { font-size: 11px; color: #c0c4cc; margin-left: auto; }
.input-area { padding: 12px 0 20px; }
.input-actions { display: flex; justify-content: space-between; align-items: center; margin-top: 8px; }
.hint { font-size: 12px; color: #c0c4cc; }
.toolbar { display: flex; gap: 8px; align-items: center; }
.tip-box { padding: 8px 0; }
.result-json { background: #f5f7fa; padding: 10px; border-radius: 6px; font-size: 12px; max-height: 300px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
</style>
