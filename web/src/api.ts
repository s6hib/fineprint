export const MAX_QUESTION_CHARS = 500 // the API turns away anything longer

export type Section = { heading: string; text: string }

/** One page the answer cites, with that page's text as the index holds it. */
export type Citation = {
  file: string
  page: number
  title: string
  effective_date: string
  url: string | null // the publisher's copy, or null when there is no link
  sections: Section[]
}

/** A request the agent drafted. Nothing is saved until it is approved. */
export type Draft = { id: string; text: string }

export type Answer = {
  answer: string
  citations: Citation[]
  draft: Draft | null
  model_calls: number
  cost_usd: number
  seconds: number
}

/** A reply the API sent with an error status. */
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

/** POST to the API and return the parsed reply. Throws an Error whose message can be shown. */
async function post(path: string, body?: unknown): Promise<unknown> {
  let response: Response
  try {
    response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body ?? {}), // never empty: the API wants a stated length
    })
  } catch {
    throw new Error('Could not reach the server. Check your connection and try again.')
  }
  // Errors from the API carry a sentence in `detail`. Anything else (a proxy's error
  // page, for example) has no such field, so fall back to the status code.
  const reply: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = (reply as { detail?: unknown } | null)?.detail
    throw new ApiError(
      typeof detail === 'string' ? detail : `Something went wrong (${response.status}).`,
      response.status,
    )
  }
  return reply
}

const UNREADABLE = 'The server sent a reply this page could not read. Try again.'

/** Ask the API one question. Throws an Error whose message can be shown to the user. */
export async function ask(question: string): Promise<Answer> {
  // A 200 that isn't the API's own reply, such as a Wi-Fi sign-in page
  const answer = (await post('/ask', { question })) as Partial<Answer> | null
  if (typeof answer?.answer !== 'string' || !Array.isArray(answer.citations)) {
    throw new Error(UNREADABLE)
  }
  return answer as Answer
}

/**
 * Approve or decline a draft. Returns the name of the saved file, or null after a decline.
 * The id comes from the API and is checked here, because it becomes part of the address.
 */
export async function decide(draft: Draft, approve: boolean): Promise<string | null> {
  if (!/^[\w-]+$/.test(draft.id)) throw new Error(UNREADABLE)
  const reply = (await post(`/drafts/${draft.id}/${approve ? 'approve' : 'decline'}`)) as {
    saved?: unknown
  } | null
  const saved = reply?.saved
  if (approve ? typeof saved !== 'string' : saved !== null) throw new Error(UNREADABLE)
  return saved as string | null
}
