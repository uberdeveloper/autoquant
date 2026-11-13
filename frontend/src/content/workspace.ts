/** Typed view of workspace.json (scripts/web_snapshot.py). Same
 * degrade-to-empty contract as the landing loader. Imported only by the
 * lazily-loaded workspace screens, so the landing does not ship it (#39). */
import generated from "./workspace.json";

export interface Article { slug: string | null; title: string | null; source: string | null; url: string | null; posted: string | null; stage: string | null; blurb: string; triageScore: number | null; }
export interface TriageRow { slug: string | null; title: string | null; url: string | null; claim: string; assetClass: string | null; score: number | null; redFlags: string[]; missing: string[]; reason: string; }
export interface SpecSummary { slug: string; title: string | null; claim: string | null; universe: string[]; bar: string | null; signal: string | null; entry: string | null; exit: string | null; direction: string | null; status: string | null; hasCode: boolean; raw: string; rawTruncated?: boolean; }
export interface RunRow { slug: string | null; lane: "article" | "external"; status: string | null; netSharpe: number | null; beatsNull: boolean | null; date: string; runAt?: string; repo?: string; artifactPath?: string; artifactKey?: string; flags: string[]; error: string | null; }
export interface Report { metrics: { label: string; strategy: string; benchmark: string }[]; text: string; }
export interface FailedRow { slug: string; error: string; }
export interface CatalogEntry { id: string; title: string | null; provider: string; tags: string[]; frequency: string | null; coverageStart: string | null; license: string | null; notes: string | null; fallback: string | null; }
export interface Workspace {
  articles: Article[]; triage: TriageRow[]; specs: SpecSummary[];
  runs: RunRow[]; reports: Record<string, Report>; failed: FailedRow[];
  catalog: CatalogEntry[]; catalogCount: number;
}

const empty: Workspace = {
  articles: [], triage: [], specs: [], runs: [], reports: {},
  failed: [], catalog: [], catalogCount: 0,
};

/** Full timestamps order same-day runs; later appended rows win timestamp ties.
 * Older snapshots only carry a date, so retain that as a fallback. */
export function newestRuns(runs: RunRow[]): RunRow[] {
  const timestamp = (run: RunRow) => Date.parse(run.runAt || run.date) || 0;
  return [...runs].reverse().sort((a, b) => timestamp(b) - timestamp(a));
}

export function loadWorkspace(raw: unknown): Workspace {
  if (!raw || typeof raw !== "object") return empty;
  const w = raw as Partial<Workspace>;
  const specs = (Array.isArray(w.specs) ? w.specs : [])
    .filter((s) => s && typeof s === "object" && typeof s.slug === "string" && s.slug.length > 0)
    .map((s) => ({
      ...s,
      universe: Array.isArray(s.universe)
        ? s.universe
        : s.universe
          ? [String(s.universe)]
          : [],
    }));
  return {
    articles: Array.isArray(w.articles) ? w.articles : [],
    triage: Array.isArray(w.triage) ? w.triage : [],
    specs,
    runs: Array.isArray(w.runs) ? w.runs : [],
    reports: w.reports && typeof w.reports === "object" ? w.reports : {},
    failed: Array.isArray(w.failed) ? w.failed : [],
    catalog: Array.isArray(w.catalog) ? w.catalog : [],
    catalogCount: w.catalogCount ?? 0,
  };
}

export const workspace: Workspace = loadWorkspace(generated);
