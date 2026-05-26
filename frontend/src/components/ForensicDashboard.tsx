"use client";

import React, { useState } from "react";
import { type ForensicReport, imageB64ToDataURL } from "@/lib/api";

interface ForensicDashboardProps {
  report: ForensicReport;
  previewUrl: string | null;
}

export function ForensicDashboard({ report, previewUrl }: ForensicDashboardProps) {
  const [showOverlay, setShowOverlay] = useState(true);

  // Extract anomaly scores
  const elaAnomaly = report.elaAnomalyScore ?? 0;
  const fftAnomaly = report.fftAnomalyScore ?? 0;
  const noiseAnomaly = report.noiseAnomalyScore ?? 0;
  const metadataAnomaly = report.metadataAnomalyScore ?? 0;

  // Helper to determine anomaly dot color
  const getDotColor = (score: number) => {
    if (score < 0.3) {
      return "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.8)]";
    }
    if (score <= 0.6) {
      return "bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]";
    }
    return "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.8)]";
  };

  // Helper to format metadata
  const metadata = report.metadataReport || {};
  const isSuspiciousSoftware = !!(metadata.suspiciousSoftware ?? metadata.suspicious_software);

  const getRawMetaVal = (camelKey: string, snakeKey: string) => {
    return metadata[camelKey] !== undefined ? metadata[camelKey] : metadata[snakeKey];
  };

  const formatMetaVal = (key: string, val: unknown) => {
    if (val === null || val === undefined || val === "") return "—";
    if (key === "gps") {
      return val ? "Yes" : "No";
    }
    if (key === "suspicious") {
      return val ? "Yes (Suspicious)" : "No";
    }
    if (key === "delta") {
      return typeof val === "number" ? `${val.toFixed(1)} hours` : `${String(val)} hours`;
    }
    return String(val);
  };

  const metadataRows = [
    {
      label: "Software",
      value: formatMetaVal("software", getRawMetaVal("software", "software")),
      highlight: false,
    },
    {
      label: "Camera make",
      value: formatMetaVal("make", getRawMetaVal("make", "make")),
      highlight: false,
    },
    {
      label: "Camera model",
      value: formatMetaVal("model", getRawMetaVal("model", "model")),
      highlight: false,
    },
    {
      label: "GPS present",
      value: formatMetaVal("gps", getRawMetaVal("gpsPresent", "gps_present")),
      highlight: false,
    },
    {
      label: "Datetime delta",
      value: formatMetaVal("delta", getRawMetaVal("datetimeDeltaHours", "datetime_delta_hours")),
      highlight: false,
    },
    {
      label: "Suspicious software flag",
      value: formatMetaVal("suspicious", getRawMetaVal("suspiciousSoftware", "suspicious_software")),
      highlight: isSuspiciousSoftware,
    },
  ];

  // Extract branch contributions
  const rgbContribution = report.branchContributions?.rgb ?? report.branchContributions?.RGB ?? 0;
  const elaContribution = report.branchContributions?.ela ?? report.branchContributions?.ELA ?? 0;
  const fftContribution = report.branchContributions?.fft ?? report.branchContributions?.FFT ?? 0;

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------------------- */}
      {/* Section 4 — Per-signal scores                                 */}
      {/* ------------------------------------------------------------- */}
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
        {[
          { label: "ELA Anomaly", score: elaAnomaly },
          { label: "FFT Anomaly", score: fftAnomaly },
          { label: "Noise Anomaly", score: noiseAnomaly },
          { label: "Metadata Anomaly", score: metadataAnomaly },
        ].map((sig, idx) => (
          <div
            key={idx}
            className="flex flex-col justify-between rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-4 transition-all duration-300 hover:border-zinc-700/60 hover:bg-zinc-900/60"
          >
            <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
              {sig.label}
            </span>
            <div className="mt-2 flex items-baseline justify-between">
              <span className="text-2xl font-black text-zinc-100 tabular-nums">
                {(sig.score * 100).toFixed(1)}%
              </span>
              <span className={`h-2.5 w-2.5 rounded-full ${getDotColor(sig.score)}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Main dashboard sections grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ------------------------------------------------------------- */}
        {/* Section 1 — Heatmap overlay                                   */}
        {/* ------------------------------------------------------------- */}
        <div className="flex flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">
              Grad-CAM Heatmap Overlay
            </h3>
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className={`rounded-lg px-3 py-1.5 text-xs font-semibold uppercase tracking-wider transition-all duration-300 ${
                showOverlay
                  ? "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"
                  : "bg-zinc-800 text-zinc-400 border border-zinc-700/60"
              }`}
            >
              {showOverlay ? "Show Original" : "Show Heatmap"}
            </button>
          </div>

          <div className="relative aspect-[4/3] w-full overflow-hidden rounded-xl border border-zinc-850 bg-zinc-950/80 flex items-center justify-center">
            {previewUrl && (
              <img
                src={previewUrl}
                alt="Original"
                className={`absolute max-h-full max-w-full object-contain transition-opacity duration-300 ${
                  showOverlay ? "opacity-0" : "opacity-100"
                }`}
              />
            )}
            <img
              src={imageB64ToDataURL(report.heatmapB64)}
              alt="Grad-CAM Heatmap Overlay"
              className={`absolute max-h-full max-w-full object-contain transition-opacity duration-300 ${
                showOverlay ? "opacity-100" : "opacity-0"
              }`}
            />
          </div>

          <span className="mt-3 text-xs font-medium text-zinc-500 text-center">
            Tamper localization — suspicious regions shown in orange/white
          </span>
        </div>

        {/* Right side: Findings & Metadata */}
        <div className="space-y-6">
          {/* ------------------------------------------------------------- */}
          {/* Section 3 — Forensic findings                                 */}
          {/* ------------------------------------------------------------- */}
          <div className="flex flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h3 className="mb-4 text-xs font-bold uppercase tracking-wider text-zinc-400">
              Forensic Findings &amp; Contributions
            </h3>
            
            {report.forensicFindings.length > 0 ? (
              <ul className="mb-6 space-y-2.5">
                {report.forensicFindings.map((finding, idx) => (
                  <li key={idx} className="flex items-start gap-2.5 text-sm text-zinc-300">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="mt-0.5 h-4 w-4 shrink-0 text-amber-500"
                    >
                      <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
                      <line x1="12" y1="9" x2="12" y2="13" />
                      <line x1="12" y1="17" x2="12.01" y2="17" />
                    </svg>
                    <span>{finding}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="mb-6 text-sm text-zinc-500 italic">No forensic findings generated.</p>
            )}

            {/* Horizontal Bar Chart (Pure CSS) */}
            <div className="space-y-3.5 border-t border-zinc-800/80 pt-4">
              <span className="text-[10px] font-bold uppercase tracking-wider text-zinc-500">
                Model Branch Weights
              </span>

              {/* RGB Bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-semibold text-zinc-400">
                  <span>RGB Analysis</span>
                  <span className="tabular-nums text-zinc-300">{(rgbContribution * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-zinc-850 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-violet-600 to-indigo-500 transition-all duration-500"
                    style={{ width: `${rgbContribution * 100}%` }}
                  />
                </div>
              </div>

              {/* ELA Bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-semibold text-zinc-400">
                  <span>Error Level Analysis (ELA)</span>
                  <span className="tabular-nums text-zinc-300">{(elaContribution * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-zinc-850 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-pink-500 to-rose-400 transition-all duration-500"
                    style={{ width: `${elaContribution * 100}%` }}
                  />
                </div>
              </div>

              {/* FFT+Noise Bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-semibold text-zinc-400">
                  <span>FFT+Noise Residuals</span>
                  <span className="tabular-nums text-zinc-300">{(fftContribution * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 w-full rounded-full bg-zinc-850 overflow-hidden">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-500"
                    style={{ width: `${fftContribution * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* ------------------------------------------------------------- */}
          {/* Section 2 — Metadata panel                                    */}
          {/* ------------------------------------------------------------- */}
          <div className="flex flex-col rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5">
            <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-zinc-400">
              Metadata Analysis
            </h3>
            <div className="overflow-hidden rounded-xl border border-zinc-850">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-zinc-850 bg-zinc-900/80 text-xs font-bold uppercase tracking-wider text-zinc-500">
                    <th className="px-4 py-3">Field</th>
                    <th className="px-4 py-3">Value</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-850/60 text-sm">
                  {metadataRows.map((row, idx) => (
                    <tr
                      key={idx}
                      className={`transition-colors ${
                        row.highlight
                          ? "bg-amber-500/10 text-amber-300 border-l-2 border-amber-500"
                          : "hover:bg-zinc-900/20 text-zinc-300"
                      }`}
                    >
                      <td className="px-4 py-2.5 font-medium">{row.label}</td>
                      <td className="px-4 py-2.5 tabular-nums font-semibold">{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
