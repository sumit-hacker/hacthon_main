import { useEffect, useMemo, useState } from 'react'
import './App.css'
import { api, type RagMode } from './lib/api'

type ChatTurn = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Array<{ source_file?: string; chunk_id?: string; score?: number }>
  pending?: boolean
  used_rag?: boolean
  timings_ms?: Record<string, number>
}

function newId() {
  // crypto.randomUUID is not available in all environments
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (crypto as any).randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function App() {
  const [backendOk, setBackendOk] = useState<boolean | null>(null)
  const [statusText, setStatusText] = useState<string>('')
  const [errorText, setErrorText] = useState<string>('')

  const [ragMode, setRagMode] = useState<RagMode>('local')
  const [openaiApiKey, setOpenaiApiKey] = useState('')
  const [azureApiKey, setAzureApiKey] = useState('')
  const [azureEndpoint, setAzureEndpoint] = useState('')
  const [azureDeployment, setAzureDeployment] = useState('')

  const [ragPolicy, setRagPolicy] = useState<'auto' | 'on' | 'off'>('auto')
  const [topK, setTopK] = useState(5)
  const [autoWarmup, setAutoWarmup] = useState(true)
  const [message, setMessage] = useState('')
  const [chat, setChat] = useState<ChatTurn[]>([])

  const [busy, setBusy] = useState<string | null>(null)
  const canUseAzure = ragMode === 'online' || ragMode === 'hybrid'
  const canUseOpenAI = ragMode === 'online'

  const title = useMemo(() => {
    return 'Team SARAS RAG – Control Panel'
  }, [])

  async function refreshStatus() {
    try {
      setErrorText('')
      await api.health()
      setBackendOk(true)
      const status = await api.status()
      const parts = [
        status.mode?.mode ? `mode=${status.mode.mode}` : null,
        status.mode?.llm ? `llm=${status.mode.llm}` : null,
        status.mode?.embeddings ? `emb=${status.mode.embeddings}` : null,
        status.mode?.vectorstore ? `vs=${status.mode.vectorstore}` : null,
        typeof status.faiss?.loaded === 'boolean' ? `index=${status.faiss.loaded ? 'loaded' : 'not-loaded'}` : null,
        typeof status.faiss?.vector_count === 'number' ? `vectors=${status.faiss.vector_count}` : null,
      ].filter(Boolean)
      setStatusText(parts.join(' · '))
    } catch (e) {
      setBackendOk(false)
      const msg = e instanceof Error ? e.message : 'Backend unreachable'
      setStatusText(msg)
      setErrorText(msg)
    }
  }

  // Load persisted UI state
  useEffect(() => {
    try {
      const raw = localStorage.getItem('chitti.ui')
      if (!raw) return
      const s = JSON.parse(raw) as Partial<{
        ragMode: RagMode
        openaiApiKey: string
        azureApiKey: string
        azureEndpoint: string
        azureDeployment: string
        ragPolicy: 'auto' | 'on' | 'off'
        topK: number
        autoWarmup: boolean
      }>
      if (s.ragMode) setRagMode(s.ragMode)
      if (typeof s.openaiApiKey === 'string') setOpenaiApiKey(s.openaiApiKey)
      if (typeof s.azureApiKey === 'string') setAzureApiKey(s.azureApiKey)
      if (typeof s.azureEndpoint === 'string') setAzureEndpoint(s.azureEndpoint)
      if (typeof s.azureDeployment === 'string') setAzureDeployment(s.azureDeployment)
      if (s.ragPolicy === 'auto' || s.ragPolicy === 'on' || s.ragPolicy === 'off') setRagPolicy(s.ragPolicy)
      if (typeof s.topK === 'number' && Number.isFinite(s.topK)) setTopK(s.topK)
      if (typeof s.autoWarmup === 'boolean') setAutoWarmup(s.autoWarmup)
    } catch {
      // ignore
    }
  }, [])

  // Persist UI state
  useEffect(() => {
    const payload = {
      ragMode,
      openaiApiKey,
      azureApiKey,
      azureEndpoint,
      azureDeployment,
      ragPolicy,
      topK,
      autoWarmup,
    }
    try {
      localStorage.setItem('chitti.ui', JSON.stringify(payload))
    } catch {
      // ignore
    }
  }, [ragMode, openaiApiKey, azureApiKey, azureEndpoint, azureDeployment, ragPolicy, topK, autoWarmup])

  // Initial status check (+ optional warmup)
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      await refreshStatus()
      if (cancelled) return
      if (autoWarmup) {
        try {
          setBusy('Warming up model…')
          await api.warmup()
        } catch (e) {
          setErrorText(e instanceof Error ? e.message : 'Warmup failed')
        } finally {
          setBusy(null)
        }
      }
    })()
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function onApplyConfig() {
    setBusy('Applying config…')
    try {
      setErrorText('')
      await api.applyConfig({
        rag_mode: ragMode,
        openai_api_key: openaiApiKey || undefined,
        azure_openai_api_key: azureApiKey || undefined,
        azure_openai_endpoint: azureEndpoint || undefined,
        azure_openai_deployment: azureDeployment || undefined,
      })
      await refreshStatus()
    } catch (e) {
      setErrorText(e instanceof Error ? e.message : 'Apply failed')
    } finally {
      setBusy(null)
    }
  }

  async function onWarmup() {
    setBusy('Warming up model…')
    try {
      setErrorText('')
      await api.warmup()
      await refreshStatus()
    } catch (e) {
      setErrorText(e instanceof Error ? e.message : 'Warmup failed')
    } finally {
      setBusy(null)
    }
  }

  async function onBuildIndex() {
    setBusy('Building index…')
    try {
      setErrorText('')
      await api.buildIndex()
      await refreshStatus()
    } catch (e) {
      setErrorText(e instanceof Error ? e.message : 'Index build failed')
    } finally {
      setBusy(null)
    }
  }

  async function onSend() {
    const trimmed = message.trim()
    if (!trimmed) return

    setMessage('')
    const userId = newId()
    const pendingId = newId()
    setChat((prev) => [
      ...prev,
      { id: userId, role: 'user', content: trimmed },
      { id: pendingId, role: 'assistant', content: '', pending: true },
    ])
    setBusy('Thinking…')

    try {
      setErrorText('')
      const resp = await api.chat({
        message: trimmed,
        use_rag: ragPolicy === 'auto' ? undefined : ragPolicy === 'on',
        top_k: topK,
      })
      setChat((prev) =>
        prev.map((t) =>
          t.id === pendingId
            ? {
                ...t,
                pending: false,
                content: resp.answer,
                sources: resp.sources,
                used_rag: resp.used_rag,
                timings_ms: resp.timings_ms,
              }
            : t,
        ),
      )
    } catch (e) {
      setErrorText(e instanceof Error ? e.message : 'Chat failed')
      setChat((prev) => prev.filter((t) => !t.pending))
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div>
          <h1>{title}</h1>
          <div className="subtle">
            Backend: {backendOk === null ? 'checking…' : backendOk ? 'connected' : 'disconnected'}
            {statusText ? ` · ${statusText}` : ''}
          </div>
        </div>
        <div className="row gap">
          <button className="secondary" onClick={() => void refreshStatus()} disabled={!!busy}>
            Refresh
          </button>
          <button onClick={() => void onWarmup()} disabled={!!busy || backendOk === false}>
            Warmup
          </button>
        </div>
      </header>

      <main className="grid">
        <section className="card">
          <h2>Settings</h2>
          <div className="row gap wrap">
            <label className="field">
              <div className="label">RAG Mode</div>
              <select value={ragMode} onChange={(e) => setRagMode(e.target.value as RagMode)} disabled={!!busy}>
                <option value="local">local (TinyLlama + FAISS)</option>
                <option value="online">online (cloud LLM)</option>
                <option value="hybrid">hybrid</option>
              </select>
            </label>
            <label className="field">
              <div className="label">Top K</div>
              <input
                type="number"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                disabled={!!busy}
              />
            </label>
            <label className="field">
              <div className="label">RAG</div>
              <select value={ragPolicy} onChange={(e) => setRagPolicy(e.target.value as 'auto' | 'on' | 'off')} disabled={!!busy}>
                <option value="auto">auto (Saras decides)</option>
                <option value="on">on (force docs)</option>
                <option value="off">off (no docs)</option>
              </select>
            </label>
            <label className="field">
              <div className="label">Auto warmup</div>
              <input type="checkbox" checked={autoWarmup} onChange={(e) => setAutoWarmup(e.target.checked)} disabled={!!busy} />
            </label>
          </div>

          <div className="row gap wrap">
            <label className="field grow">
              <div className="label">OpenAI API Key</div>
              <input
                placeholder={canUseOpenAI ? 'sk-…' : 'Not needed for local mode'}
                value={openaiApiKey}
                onChange={(e) => setOpenaiApiKey(e.target.value)}
                disabled={!!busy || !canUseOpenAI}
                type="password"
              />
            </label>
          </div>

          <div className="row gap wrap">
            <label className="field grow">
              <div className="label">Azure OpenAI Key</div>
              <input
                placeholder={canUseAzure ? '…' : 'Not needed for local mode'}
                value={azureApiKey}
                onChange={(e) => setAzureApiKey(e.target.value)}
                disabled={!!busy || !canUseAzure}
                type="password"
              />
            </label>
            <label className="field grow">
              <div className="label">Azure Endpoint</div>
              <input
                placeholder={canUseAzure ? 'https://…openai.azure.com' : 'Not needed for local mode'}
                value={azureEndpoint}
                onChange={(e) => setAzureEndpoint(e.target.value)}
                disabled={!!busy || !canUseAzure}
              />
            </label>
            <label className="field grow">
              <div className="label">Azure Deployment</div>
              <input
                placeholder={canUseAzure ? 'deployment-name' : 'Not needed for local mode'}
                value={azureDeployment}
                onChange={(e) => setAzureDeployment(e.target.value)}
                disabled={!!busy || !canUseAzure}
              />
            </label>
          </div>

          <div className="row gap">
            <button onClick={() => void onApplyConfig()} disabled={!!busy || backendOk === false}>
              Apply
            </button>
            <button className="secondary" onClick={() => void onBuildIndex()} disabled={!!busy || backendOk === false}>
              Build/Load Index
            </button>
            <div className="subtle">{busy ?? ''}</div>
          </div>
          {errorText ? <div className="error">{errorText}</div> : null}
        </section>

        <section className="card">
          <h2>Chat</h2>
          <div className="chat">
            {chat.length === 0 ? (
              <div className="subtle">Send a message to begin. Set RAG to Auto/On/Off to compare outputs.</div>
            ) : (
              chat.map((t, idx) => (
                <div key={t.id || idx} className={`turn ${t.role}`}>
                  <div className="role">{t.role}</div>
                  <div className="content">
                    {t.pending ? (
                      <span className="typing">
                        Saras is typing<span className="dots"><span>.</span><span>.</span><span>.</span></span>
                      </span>
                    ) : (
                      t.content
                    )}
                  </div>
                  {t.role === 'assistant' && !t.pending ? (
                    <div className="subtle">
                      {typeof t.used_rag === 'boolean' ? (t.used_rag ? 'Used documents (RAG)' : 'No documents used') : ''}
                      {t.timings_ms?.total ? ` · ${Math.round(t.timings_ms.total)}ms` : ''}
                    </div>
                  ) : null}
                  {t.role === 'assistant' && t.sources && t.sources.length > 0 ? (
                    <div className="sources">
                      <div className="label">Sources</div>
                      <ul>
                        {t.sources.slice(0, 6).map((s, sidx) => (
                          <li key={sidx}>
                            {s.source_file || 'unknown'}
                            {s.chunk_id ? ` · ${s.chunk_id}` : ''}
                            {typeof s.score === 'number' ? ` · score=${s.score.toFixed(3)}` : ''}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              ))
            )}
          </div>
          <div className="row gap">
            <input
              className="grow"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder={backendOk === false ? 'Start backend to chat…' : 'Ask a question…'}
              disabled={!!busy || backendOk === false}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void onSend()
              }}
            />
            <button onClick={() => void onSend()} disabled={!!busy || backendOk === false}>
              Send
            </button>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
