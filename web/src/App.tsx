import { useState } from 'react'
import type { FormEvent } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ask, MAX_QUESTION_CHARS } from './api'
import type { Answer } from './api'
import { fromHref, label, markCitations } from './citations'

const EXAMPLES = [
  'Does UHC cover bilateral cochlear implants for adults?',
  'What does HCPCS code A0431 mean?',
  'Is a taxi ride to the emergency room covered as ambulance transport?',
]

type State =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'done'; result: Answer }
  | { status: 'error'; message: string }

export default function App() {
  const [question, setQuestion] = useState('')
  const [state, setState] = useState<State>({ status: 'idle' })
  const loading = state.status === 'loading'

  async function submit(text: string) {
    const trimmed = text.trim()
    if (!trimmed || loading) return
    setState({ status: 'loading' })
    try {
      setState({ status: 'done', result: await ask(trimmed) })
    } catch (e) {
      setState({ status: 'error', message: e instanceof Error ? e.message : String(e) })
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault()
    void submit(question)
  }

  return (
    <main>
      <h1>fineprint</h1>
      <p className="lede">
        Ask what a UnitedHealthcare medical policy says. Every answer cites the page it came from.
      </p>

      <form onSubmit={onSubmit}>
        <label htmlFor="question">Your question</label>
        <textarea
          id="question"
          rows={3}
          maxLength={MAX_QUESTION_CHARS}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter makes a new line. While an input method is still
            // composing a character, Enter belongs to it and must not send.
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) onSubmit(e)
          }}
          placeholder="Is bariatric surgery covered for a BMI of 36 with type 2 diabetes?"
        />
        <div className="row">
          <button type="submit" disabled={loading || !question.trim()}>
            {loading ? 'Reading the policies…' : 'Ask'}
          </button>
          <span className="count">
            {question.length}/{MAX_QUESTION_CHARS}
          </span>
        </div>
      </form>

      <ul className="examples" aria-label="Example questions">
        {EXAMPLES.map((example) => (
          <li key={example}>
            <button
              type="button"
              disabled={loading}
              onClick={() => {
                setQuestion(example)
                void submit(example)
              }}
            >
              {example}
            </button>
          </li>
        ))}
      </ul>

      <section aria-live="polite">
        {loading && <p className="status">Reading the policies…</p>}
        {state.status === 'error' && <p className="error">{state.message}</p>}
        {state.status === 'done' && <Result result={state.result} />}
      </section>

      <footer>
        Answers are drawn from five UnitedHealthcare commercial medical policies. This is a
        demo, not coverage advice: a member's own plan document decides what is covered.
      </footer>
    </main>
  )
}

function Result({ result }: { result: Answer }) {
  const confirmed = new Set(result.citations.map((c) => `${c.file}:${c.page}`))
  return (
    <article>
      <Markdown
        remarkPlugins={[remarkGfm]} // tables, which the policies' code lists can come back as
        disallowedElements={['img']} // an image would load from whatever address the model wrote
        components={{
          // A citation arrives as a link (see markCitations). Draw it as a tag, but only
          // if the API confirmed that page. Any other link the model wrote is shown as
          // plain text, never as a live link.
          a({ href, children }) {
            const cited = fromHref(href)
            return cited && confirmed.has(`${cited.file}:${cited.page}`) ? (
              <span className="cite">{children}</span>
            ) : (
              <>{children}</>
            )
          },
        }}
      >
        {markCitations(result.answer, result.citations)}
      </Markdown>

      {result.citations.length > 0 && (
        <>
          <h2>Pages cited</h2>
          <ul className="sources">
            {result.citations.map((c) => (
              <li key={`${c.file}:${c.page}`}>
                <span className="cite">{label(c)}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <p className="usage">
        {result.model_calls} model {result.model_calls === 1 ? 'call' : 'calls'} · ${result.cost_usd.toFixed(4)} · {result.seconds} s
      </p>
    </article>
  )
}
