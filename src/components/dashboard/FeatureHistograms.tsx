import React from "react";
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js-dist-min';
import { motion } from "framer-motion";
(Plot as any).plotly = Plotly;

export default function FeatureHistograms({ df }) {
  const keys = Object.keys(df[0]).slice(0, 6); 

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {keys.map((key, index) => (
        <motion.div
          key={key}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          whileHover={{ y: -4 }}
          transition={{ delay: index * 0.05 }}
          className="relative group"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/10 to-emerald-500/10 rounded-2xl blur-xl" />

          <div className="relative p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">
            <h2 className="text-lg font-semibold text-white mb-4">{key} Distribution</h2>

            <Plot
              data={[
                {
                  x: df.map(r => r[key]),
                  type: "histogram",
                  marker: { color: "#34d399" },
                },
              ]}
              layout={{
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                font: { color: "#fff" },
                margin: { l: 40, r: 20, t: 20, b: 30 },
              }}
              config={{ displayModeBar: false }}
              className="w-full h-64"
            />
          </div>
        </motion.div>
      ))}
    </div>
  );
}
