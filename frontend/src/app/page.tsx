"use client";

import { useState, useCallback, useRef } from "react";
import type { ForensicReport, Verdict } from "@/lib/api";
import { analyzeImage, imageB64ToDataURL } from "@/lib/api";
import { ForensicDashboard } from "@/components/ForensicDashboard";


/* ------------------------------------------------------------------ */
/*  Verdict styling map                                                */
/* ------------------------------------------------------------------ */

const VERDICT_STYLE: Record<
  Verdict,
  { bg: string; ring: string; text: string; label: string }
> = {
  AI_GENERATED: {
    bg: "bg-red-500/20",
    ring: "ring-red-500/50",
    text: "text-red-400",
    label: "AI GENERATED",
  },
  MANIPULATED: {
    bg: "bg-amber-500/20",
    ring: "ring-amber-500/50",
    text: "text-amber-400",
    label: "MANIPULATED",
  },
  AUTHENTIC: {
    bg: "bg-emerald-500/20",
    ring: "ring-emerald-500/50",
    text: "text-emerald-400",
    label: "AUTHENTIC",
  },
};

/* ------------------------------------------------------------------ */
/*  Page component                                                     */
/* ------------------------------------------------------------------ */

export default function Home() {
  const [report, setReport] = useState<ForensicReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  /* ---- shared analysis handler ----------------------------------- */
  const handleFile = useCallback(async (file: File) => {
    const allowed = ["image/jpeg", "image/png", "image/webp"];
    if (!allowed.includes(file.type)) {
      setError("Unsupported file type. Please upload a JPEG, PNG, or WebP image.");
      return;
    }

    setPreviewUrl(URL.createObjectURL(file));
    setError(null);
    setReport(null);
    setLoading(true);

    try {
      const result = await analyzeImage(file);
      setReport(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  }, []);

  /* ---- drag-and-drop --------------------------------------------- */
  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const onDragLeave = useCallback(() => setDragOver(false), []);

  /* ---- browse button --------------------------------------------- */
  const onBrowse = useCallback(() => fileInputRef.current?.click(), []);
  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  /* ---- reset ----------------------------------------------------- */
  const reset = useCallback(() => {
    setReport(null);
    setError(null);
    setLoading(false);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  /* ================================================================ */
  /*  Render                                                           */
  /* ================================================================ */
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 font-[family-name:var(--font-geist-sans)]">
      {/* ---- Header ------------------------------------------------ */}
      <header className="border-b border-zinc-800/60 px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center gap-3">
          {/* icon */}
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-violet-600 to-cyan-500">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-5 w-5 text-white"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.3-4.3" />
            </svg>
          </div>
          <h1 className="text-lg font-semibold tracking-tight">
            Forge<span className="text-cyan-400">Lens</span>
          </h1>
          <span className="ml-2 rounded-full bg-zinc-800 px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wider text-zinc-400">
            Forensics
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-12">
        {/* ---- Error banner --------------------------------------- */}
        {error && (
          <div className="mb-8 flex items-center gap-3 rounded-xl border border-red-500/30 bg-red-500/10 px-5 py-4 text-sm text-red-300">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 20 20"
              fill="currentColor"
              className="h-5 w-5 shrink-0 text-red-400"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z"
                clipRule="evenodd"
              />
            </svg>
            <p>{error}</p>
          </div>
        )}

        {/* ======================================================== */}
        {/*  STATE 1 — Upload / Loading                                */}
        {/* ======================================================== */}
        {!report && (
          <div className="flex flex-col items-center gap-10 pt-12">
            <div className="text-center">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                Image Forensic Analysis
              </h2>
              <p className="mt-3 text-zinc-400">
                Multi-branch neural network detection for AI-generated &amp;
                manipulated imagery
              </p>
            </div>

            {/* Drop zone */}
            <div
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className={`group relative flex w-full max-w-xl cursor-pointer flex-col items-center justify-center gap-4 rounded-2xl border-2 border-dashed px-8 py-20 text-center transition-all duration-300 ${
                dragOver
                  ? "border-cyan-400 bg-cyan-400/5"
                  : "border-zinc-700 bg-zinc-900/50 hover:border-zinc-500 hover:bg-zinc-900"
              }`}
            >
              {loading ? (
                /* ---- spinner ------------------------------------ */
                <div className="flex flex-col items-center gap-4">
                  <svg
                    className="h-10 w-10 animate-spin text-cyan-400"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  <p className="text-sm font-medium text-cyan-300">
                    Analyzing forensic signals…
                  </p>
                </div>
              ) : (
                /* ---- idle upload prompt ------------------------- */
                <>
                  <div className="flex h-14 w-14 items-center justify-center rounded-full bg-zinc-800 text-zinc-400 transition-colors group-hover:bg-zinc-700 group-hover:text-zinc-200">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={1.5}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-7 w-7"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" x2="12" y1="3" y2="15" />
                    </svg>
                  </div>

                  <div>
                    <p className="text-base font-medium text-zinc-200">
                      Drop an image to analyze
                    </p>
                    <p className="mt-1 text-sm text-zinc-500">
                      JPEG, PNG, or WebP — up to 15 MB
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={onBrowse}
                    className="rounded-lg bg-zinc-800 px-5 py-2 text-sm font-medium text-zinc-200 transition-colors hover:bg-zinc-700"
                  >
                    Browse files
                  </button>
                </>
              )}

              {/* hidden file input */}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={onFileChange}
              />
            </div>
          </div>
        )}

        {/* ======================================================== */}
        {/*  STATE 2 — Results                                         */}
        {/* ======================================================== */}
        {report && (
          <div className="space-y-10">
            {/* ---- Verdict + scores ------------------------------ */}
            <section className="flex flex-col items-center gap-6 text-center">
              {/* verdict pill */}
              {(() => {
                const v = VERDICT_STYLE[report.verdict];
                return (
                  <span
                    className={`inline-flex items-center rounded-full px-6 py-2 text-lg font-bold tracking-wide ring-2 ${v.bg} ${v.ring} ${v.text}`}
                  >
                    {v.label}
                  </span>
                );
              })()}

              {/* confidence */}
              <p className="text-4xl font-extrabold tabular-nums">
                {(report.confidence * 100).toFixed(1)}%
                <span className="ml-2 text-base font-normal text-zinc-500">
                  confidence
                </span>
              </p>

              {/* suspicion bar */}
              <div className="w-full max-w-md">
                <div className="mb-1 flex justify-between text-xs text-zinc-400">
                  <span>Suspicion Score</span>
                  <span className="tabular-nums">
                    {(report.suspicionScore * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-3 w-full overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-red-600 to-red-400 transition-all duration-700"
                    style={{ width: `${report.suspicionScore * 100}%` }}
                  />
                </div>
              </div>

              {/* inference time */}
              <p className="text-xs text-zinc-500">
                Inference completed in{" "}
                <span className="font-medium text-zinc-400">
                  {report.inferenceTimeMs.toFixed(0)} ms
                </span>
              </p>

              {/* reset button */}
              <button
                type="button"
                onClick={reset}
                className="mt-2 rounded-lg border border-zinc-700 bg-zinc-900 px-5 py-2.5 text-sm font-medium text-zinc-300 transition-colors hover:border-zinc-500 hover:bg-zinc-800"
              >
                Analyze another image
              </button>
            </section>
            {/* ---- Forensic Dashboard ---------------------------- */}
            <ForensicDashboard report={report} previewUrl={previewUrl} />

            {/* ---- Detailed Signal Channels ---------------------- */}
            <section>
              <h3 className="mb-4 text-xs font-bold uppercase tracking-wider text-zinc-400">
                Detailed Forensic Signal Channels
              </h3>
              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                {/* ELA */}
                <ImagePanel
                  label="Error Level Analysis (ELA)"
                  src={imageB64ToDataURL(report.elaImageB64)}
                />
                {/* FFT */}
                <ImagePanel
                  label="FFT Spectrum"
                  src={imageB64ToDataURL(report.fftImageB64)}
                />
                {/* Noise Residual */}
                <ImagePanel
                  label="Noise Residual"
                  src={imageB64ToDataURL(report.noiseImageB64)}
                />
              </div>
            </section>
          </div>
        )}
      </main>

      {/* ---- Footer ------------------------------------------------ */}
      <footer className="border-t border-zinc-800/60 py-6 text-center text-xs text-zinc-600">
        ForgeLens — Multi-branch neural image forensics
      </footer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Image panel sub-component                                          */
/* ------------------------------------------------------------------ */

function ImagePanel({ label, src }: { label: string; src: string }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900/60">
      <div className="border-b border-zinc-800 px-4 py-2.5">
        <span className="text-xs font-medium uppercase tracking-wider text-zinc-400">
          {label}
        </span>
      </div>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={label}
        className="h-auto w-full object-contain"
      />
    </div>
  );
}
