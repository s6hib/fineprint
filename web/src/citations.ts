import type { Citation } from './api'

/** The two parts that name a page. A tag in the answer carries only these. */
export type PageRef = Pick<Citation, 'file' | 'page'>

// The same shape the API reads: [name.pdf p.12], one page per tag.
const TAG = /\[([\w.-]+\.pdf) p\.([1-9][0-9]{0,3})\]/g

const HREF = '#cite:'

/** A short label for a cited page, without the .pdf ending. */
export function label(c: PageRef): string {
  return `${c.file.replace(/\.pdf$/, '')} p.${c.page}`
}

/**
 * Rewrite each citation tag as a Markdown link, so the renderer hands it to us as a link
 * and we can draw it our own way. Tags the API did not confirm are left as plain text.
 */
export function markCitations(answer: string, confirmed: PageRef[]): string {
  const known = new Set(confirmed.map((c) => `${c.file}:${c.page}`))
  return answer.replace(TAG, (tag, file: string, page: string) =>
    known.has(`${file}:${page}`)
      ? `[${label({ file, page: Number(page) })}](${HREF}${file}:${page})`
      : tag,
  )
}

/** The citation a link made by markCitations points to, or null for any other link. */
export function fromHref(href: string | undefined): PageRef | null {
  if (!href?.startsWith(HREF)) return null
  const [file, page] = href.slice(HREF.length).split(':')
  return { file, page: Number(page) }
}
