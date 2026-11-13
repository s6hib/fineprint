export const MAX_QUESTION_CHARS = 500 // the API turns away anything longer

export type Citation = { file: string; page: number }

export type Answer = {
  answer: string
  citations: Citation[]
  model_calls: number
  cost_usd: number
  seconds: number
}

/** Ask the API one question. Throws an Error whose message can be shown to the user. */
export async function ask(question: string): Promise<Answer> {
  let response: Response
  try {
    response = await fetch('/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
  } catch {
    throw new Error('Could not reach the server. Check your connection and try again.')
  }
  // Errors from the API carry a sentence in `detail`. Anything else (a proxy's error
  // page, for example) has no such field, so fall back to the status code.
  const body: unknown = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail
    throw new Error(typeof detail === 'string' ? detail : `Something went wrong (${response.status}).`)
  }
  // A 200 that isn't the API's own reply, such as a Wi-Fi sign-in page
  const answer = body as Partial<Answer> | null
  if (typeof answer?.answer !== 'string' || !Array.isArray(answer.citations)) {
    throw new Error('The server sent a reply this page could not read. Try again.')
  }
  return answer as Answer
}
