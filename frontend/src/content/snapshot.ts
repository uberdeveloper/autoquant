/** Typed view of scripts/web_snapshot.py's landing snapshot. The loader
 * never throws: a missing or malformed snapshot degrades to a valid empty
 * one so the landing always renders (empty states, never fake content).
 * Workspace data lives in workspace.json and is imported lazily by the
 * workspace screens (#39). */

export interface Counts { articles: number; triaged: number; specs: number; backtests: number; external: number; }
export interface Stage { index: string; name: string; description: string; artifact: string; doc: string; count?: number | null; }
export interface ChainStage { kind: "article" | "spec" | "code" | "report"; title: string; body: string; repoPath: string; }
export interface Chain { slug: string; title: string; stages: ChainStage[]; }
export interface ArchiveRow { slug: string; source: string; lane: "article" | "external"; status: string | null; netSharpe: number | null; beatsNull: boolean | null; date: string; runAt?: string; }
export interface Headline { slug: string; metrics: { label: string; strategy: string; benchmark: string }[]; }
export interface CatalogInfo { series: number; providers: string[]; }
export interface Snapshot { generatedAt: string; counts: Counts; stages: Stage[]; chains: Chain[]; archive: ArchiveRow[]; headline: Headline | null; catalog: CatalogInfo; }

export const emptySnapshot = (): Snapshot => ({
  generatedAt: "",
  counts: { articles: 0, triaged: 0, specs: 0, backtests: 0, external: 0 },
  stages: [],
  chains: [],
  archive: [],
  headline: null,
  catalog: { series: 0, providers: [] },
});

export function loadSnapshot(raw: unknown): Snapshot {
  if (!raw || typeof raw !== "object") return emptySnapshot();
  const s = raw as Partial<Snapshot>;
  return {
    generatedAt: s.generatedAt ?? "",
    counts: { ...emptySnapshot().counts, ...s.counts },
    stages: Array.isArray(s.stages) ? s.stages : [],
    chains: Array.isArray(s.chains) ? s.chains : [],
    archive: Array.isArray(s.archive) ? s.archive : [],
    headline: s.headline ?? null,
    catalog: s.catalog ?? { series: 0, providers: [] },
  };
}

/** Vite static import of the generated JSON. Run scripts/web_snapshot.py first. */
import generated from "./snapshot.json";
export const snapshot: Snapshot = loadSnapshot(generated);
