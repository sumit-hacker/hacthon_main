export type RagMode = 'local' | 'online' | 'hybrid'

export type ApplyConfigRequest = {
  rag_mode: RagMode
  openai_api_key?: string
  azure_openai_api_key?: string
  azure_openai_endpoint?: string
  azure_openai_deployment?: string
}

export type ApplyConfigResponse = {
  ok: boolean
  mode: {
    mode: string
    llm: string
    embeddings: string
    vectorstore: string
    database: string
    is_fully_online: boolean
    is_fully_local: boolean
    is_hybrid: boolean
  }
}

export type WarmupResponse = {
  ok: boolean
  llm: Record<string, unknown>
}

export type BuildIndexResponse = {
  ok: boolean
  indexed_chunks: number
  index_path: string
}

export type ChatRequest = {
  message: string
  use_rag?: boolean
  top_k?: number
}

export type ChatResponse = {
  answer: string
  sources?: Array<{ source_file?: string; chunk_id?: string; score?: number }>
  used_rag?: boolean
  timings_ms?: Record<string, number>
}

export type StatusResponse = {
  mode: {
    mode: string
    llm: string
    embeddings: string
    vectorstore: string
    database: string
  }
  components: {
    llm: string
    embeddings: string
    vectorstore: string
    database: string
  }
  faiss: {
    loaded: boolean
    index_path: string | null
    vector_count: number
  }
}

const DEFAULT_TIMEOUT_MS = 120_000

async function requestJson<T>(path: string, init?: RequestInit, timeoutMs = DEFAULT_TIMEOUT_MS): Promise<T> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)

  try {
    const resp = await fetch(`/api${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
      signal: controller.signal,
    })

    const text = await resp.text()
    const data = text ? JSON.parse(text) : {}

    if (!resp.ok) {
      const msg = (data && (data.detail || data.error)) || `HTTP ${resp.status}`
      throw new Error(msg)
    }

    return data as T
  } finally {
    clearTimeout(timeout)
  }
}

export const api = {
  health: () => requestJson<{ ok: boolean }>(`/health`, { method: 'GET' }, 15_000),
  status: () => requestJson<StatusResponse>(`/status`, { method: 'GET' }, 15_000),
  applyConfig: (body: ApplyConfigRequest) =>
    requestJson<ApplyConfigResponse>(`/config/apply`, { method: 'POST', body: JSON.stringify(body) }, 30_000),
  warmup: () => requestJson<WarmupResponse>(`/warmup`, { method: 'POST' }, 120_000),
  buildIndex: () => requestJson<BuildIndexResponse>(`/rag/index/build`, { method: 'POST' }, 120_000),
  chat: (body: ChatRequest) => requestJson<ChatResponse>(`/chat`, { method: 'POST', body: JSON.stringify(body) }, 120_000),
}
