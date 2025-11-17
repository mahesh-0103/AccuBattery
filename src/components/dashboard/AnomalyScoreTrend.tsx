import React from "react";
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js-dist-min';
import { motion } from "framer-motion";
import { Activity } from "lucide-react";
(Plot as any).plotly = Plotly;

export default function AnomalyScoreTrend({ df }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.4 }}
      className="relative group"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-emerald-500/10 rounded-2xl blur-xl group-hover:blur-2xl transition-all" />

      <div className="relative p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">
        
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 rounded-xl bg-blue-500/20">
            <Activity className="h-6 w-6 text-blue-400" />
          </div>
          <h2 className="text-xl font-semibold text-white">Anomaly Score Trend</h2>
        </div>

        <Plot
          data={[
            {
              y: df.map(r => r.anomaly_score),
              type: "scatter",
              mode: "lines",
              line: { color: "#34d399", width: 3 },
            },
          ]}
          layout={{
            paper_bgcolor: "rgba(0,0,0,0)",
            plot_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#fff" },
            margin: { l: 40, r: 20, t: 40, b: 30 },
          }}
          config={{ displayModeBar: false }}
          className="w-full h-80"
        />
      </div>
    </motion.div>
  );
}
