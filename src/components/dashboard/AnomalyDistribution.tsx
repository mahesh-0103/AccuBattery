import React from "react";
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js-dist-min';
import { motion } from "framer-motion";
import { PieChart } from "lucide-react";
(Plot as any).plotly = Plotly;


export default function AnomalyDistribution({ df }) {
  const normal = df.filter(r => r.anomaly_flag === "Normal").length;
  const anomaly = df.filter(r => r.anomaly_flag === "Anomaly").length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.4 }}
      className="relative group"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-emerald-500/10 to-red-500/10 rounded-2xl blur-xl group-hover:blur-2xl transition-all" />

      <div className="relative p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">

        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 rounded-xl bg-emerald-500/20">
            <PieChart className="h-6 w-6 text-emerald-400" />
          </div>
          <h2 className="text-xl text-white font-semibold">Anomaly Distribution</h2>
        </div>

        <Plot
          data={[
            {
              values: [normal, anomaly],
              labels: ["Normal", "Anomaly"],
              type: "pie",
              marker: {
                colors: ["#10b981", "#ef4444"],
              },
              textinfo: "label+percent",
            },
          ]}
          layout={{
            paper_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#fff" },
            margin: { t: 30 },
          }}
          config={{ displayModeBar: false }}
          className="w-full h-80"
        />
      </div>
    </motion.div>
  );
}
