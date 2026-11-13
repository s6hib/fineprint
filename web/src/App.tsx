import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, RefObject } from 'react'
import { createPortal } from 'react-dom'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { ApiError, ask, decide, MAX_QUESTION_CHARS } from './api'
import type { Answer, Citation, Draft } from './api'
import { fromHref, label, markCitations } from './citations'

const EXAMPLES = [
  'Does UHC cover bilateral cochlear implants for adults?',
  'What does HCPCS code A0431 mean?',
  'Is a taxi ride to the emergency room covered as ambulance transport?',
  'Draft a prior-authorization request for cochlear implantation, CPT 69930.',
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

// The panel is 30rem wide at most (index.css). Up to that width it fills the window.
const PANEL_COVERS_PAGE = '(max-width: 30rem)'

type Show = (citation: Citation, from: HTMLElement) => void

/** A citation drawn as a small button that opens its page. */
function Tag({ citation, onShow }: { citation: Citation; onShow: Show }) {
  return (
    <button
      type="button"
      className="cite"
      aria-haspopup="dialog"
      onClick={(e) => onShow(citation, e.currentTarget)}
    >
      {label(citation)}
    </button>
  )
}

function Result({ result }: { result: Answer }) {
  const [open, setOpen] = useState<Citation | null>(null)
  const opener = useRef<HTMLElement | null>(null) // the tag that was pressed, to hand focus back
  const panel = useRef<HTMLElement>(null)
  const giveFocusBack = useRef(false)

  const show = useCallback<Show>((citation, from) => {
    opener.current = from
    setOpen(citation)
    panel.current?.focus() // already open on this page: opening it again changes nothing else
  }, [])

  const close = useCallback(() => {
    // Hand focus back to the tag only if it was in the panel. Someone who pressed Escape
    // while typing a new question keeps their place.
    const active = document.activeElement
    giveFocusBack.current = active === document.body || !!panel.current?.contains(active)
    setOpen(null)
  }, [])

  // Runs after the panel is gone, when the page behind it can take focus again.
  useEffect(() => {
    if (!open && giveFocusBack.current) {
      giveFocusBack.current = false
      opener.current?.focus()
    }
  }, [open])

  // Rendered once per answer. Opening the panel draws this component again, and a fresh
  // render of the Markdown would replace every tag, including the one focus returns to.
  const answer = useMemo(() => {
    const confirmed = new Map(result.citations.map((c) => [`${c.file}:${c.page}`, c]))
    return (
      <Markdown
        remarkPlugins={[remarkGfm]} // tables, which the policies' code lists can come back as
        disallowedElements={['img']} // an image would load from whatever address the model wrote
        components={{
          // A citation arrives as a link (see markCitations). Draw it as a tag that opens
          // the page, but only if the API confirmed that page. Any other link the model
          // wrote is shown as plain text, never as a live link.
          a({ href, children }) {
            const ref = fromHref(href)
            const cited = ref && confirmed.get(`${ref.file}:${ref.page}`)
            return cited ? <Tag citation={cited} onShow={show} /> : <>{children}</>
          },
        }}
      >
        {markCitations(result.answer, result.citations)}
      </Markdown>
    )
  }, [result, show])

  return (
    <article>
      {answer}

      {result.citations.length > 0 && (
        <>
          <h2>Pages cited</h2>
          <ul className="sources">
            {result.citations.map((c) => (
              <li key={`${c.file}:${c.page}`}>
                <Tag citation={c} onShow={show} />
              </li>
            ))}
          </ul>
        </>
      )}

      {result.draft && <DraftCard draft={result.draft} />}

      <p className="usage">
        {result.model_calls} model {result.model_calls === 1 ? 'call' : 'calls'} · ${result.cost_usd.toFixed(4)} · {result.seconds} s
      </p>

      {/* Drawn at the end of <body>: inside this live region a screen reader would read
          the whole page out, and the page behind has to be switched off separately. */}
      {open &&
        createPortal(<Source citation={open} panel={panel} onClose={close} />, document.body)}
    </article>
  )
}

type Verdict =
  | { status: 'waiting'; problem?: string }
  | { status: 'sending' }
  | { status: 'approved'; file: string }
  | { status: 'declined' }
  | { status: 'gone'; problem: string } // the server no longer holds this draft

/** A request the agent drafted, with the two buttons that decide whether it is saved. */
function DraftCard({ draft }: { draft: Draft }) {
  const [verdict, setVerdict] = useState<Verdict>({ status: 'waiting' })
  const outcome = useRef<HTMLParagraphElement>(null)

  async function send(approve: boolean) {
    setVerdict({ status: 'sending' })
    try {
      const file = await decide(draft, approve)
      setVerdict(file === null ? { status: 'declined' } : { status: 'approved', file })
    } catch (e) {
      const problem = e instanceof Error ? e.message : String(e)
      // A 404 means the draft was used or waited too long, so the buttons can do no more.
      setVerdict(
        e instanceof ApiError && e.status === 404
          ? { status: 'gone', problem }
          : { status: 'waiting', problem },
      )
    }
  }

  const open = verdict.status === 'waiting' || verdict.status === 'sending'

  // The buttons leave the page once the draft is decided, and focus would be dropped
  // with them. Put it on the line that says what happened.
  useEffect(() => {
    if (!open) outcome.current?.focus()
  }, [open])
  return (
    <section className="draft" aria-labelledby="draft-title">
      <h2 id="draft-title">Drafted request</h2>
      <pre>{draft.text}</pre>
      {open && (
        <div className="row">
          <p>Nothing is saved unless you approve it.</p>
          <div className="choices">
            <button type="button" disabled={verdict.status === 'sending'} onClick={() => void send(false)}>
              Decline
            </button>
            <button
              type="button"
              className="primary"
              disabled={verdict.status === 'sending'}
              onClick={() => void send(true)}
            >
              Approve and save
            </button>
          </div>
        </div>
      )}
      {/* Read out as it changes: the answer sits in the page's live region. */}
      <p className={'problem' in verdict ? 'verdict problem' : 'verdict'} ref={outcome} tabIndex={-1}>
        {verdict.status === 'approved' && `Approved. Saved as ${verdict.file}.`}
        {verdict.status === 'declined' && 'Declined. Nothing was saved.'}
        {'problem' in verdict && verdict.problem}
      </p>
    </section>
  )
}

/** A panel at the side with the text of one cited page and a link to the publisher's copy. */
function Source({
  citation,
  panel,
  onClose,
}: {
  citation: Citation
  panel: RefObject<HTMLElement | null>
  onClose: () => void
}) {
  // Move focus into the panel when it opens or switches page, so a keyboard or screen
  // reader user lands on the text they asked for.
  useEffect(() => {
    panel.current?.focus()
    panel.current?.scrollTo(0, 0)
  }, [citation, panel])

  // On a narrow screen the panel covers everything, so the page behind it is switched
  // off until the panel closes. Otherwise Tab would reach controls nobody can see.
  useEffect(() => {
    const main = document.querySelector('main')
    const covers = window.matchMedia(PANEL_COVERS_PAGE)
    const sync = () => {
      if (main) main.inert = covers.matches
    }
    sync()
    covers.addEventListener('change', sync)
    return () => {
      covers.removeEventListener('change', sync)
      if (main) main.inert = false
    }
  }, [])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <aside className="source" ref={panel} tabIndex={-1} role="dialog" aria-labelledby="source-title">
      <header>
        <div>
          <h2 id="source-title">
            {citation.title}, page {citation.page}
          </h2>
          <p className="meta">
            UnitedHealthcare policy, effective {citation.effective_date}
          </p>
        </div>
        <button type="button" className="close" onClick={onClose}>
          Close
        </button>
      </header>

      {citation.sections.map((section, i) => (
        <section key={i}>
          <h3>{section.heading}</h3>
          <p className="page-text">{section.text}</p>
        </section>
      ))}

      <p className="origin">
        This is the page's text as fineprint read it.{' '}
        {citation.url ? (
          <>
            <a href={citation.url} target="_blank" rel="noopener noreferrer">
              Open the policy on uhcprovider.com
            </a>{' '}
            to see the original. UnitedHealthcare may have revised it since, so the page
            number there can differ.
          </>
        ) : (
          <>This copy was read from a scan, so there is no link to the original.</>
        )}
      </p>
    </aside>
  )
}
