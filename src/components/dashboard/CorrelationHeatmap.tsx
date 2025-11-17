import React from "react";
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js-dist-min';
import { motion } from "framer-motion";
import { Grid3x3 } from "lucide-react";
(Plot as any).plotly = Plotly;


export default function CorrelationHeatmap({ df }) {
  const keys = Object.keys(df[0]).filter(k => typeof df[0][k] === "number");
  
  const matrix = keys.map(k1 =>
    keys.map(k2 => {
      const A = df.map(r => r[k1]);
      const B = df.map(r => r[k2]);
      const mA = A.reduce((a,b)=>a+b)/A.length;
      const mB = B.reduce((a,b)=>a+b)/B.length;
      const cov = A.map((v,i)=>(v-mA)*(B[i]-mB)).reduce((a,b)=>a+b);
      const sA = Math.sqrt(A.map(v=>(v-mA)**2).reduce((a,b)=>a+b));
      const sB = Math.sqrt(B.map(v=>(v-mB)**2).reduce((a,b)=>a+b));
      return cov / (sA*sB + 1e-9);
    })
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ y: -4 }}
      transition={{ duration: 0.4 }}
      className="relative group"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-2xl blur-xl" />

      <div className="relative p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm">

        <div className="flex items-center space-x-3 mb-4">
          <div className="p-3 rounded-xl bg-purple-500/20">
            <Grid3x3 className="h-6 w-6 text-purple-400" />
          </div>
          <h2 className="text-xl text-white font-semibold">Correlation Heatmap</h2>
        </div>

        <Plot
          data={[
            {
              z: matrix,
              x: keys,
              y: keys,
              type: "heatmap",
              colorscale: "Viridis",
            },
          ]}
          layout={{
            paper_bgcolor: "rgba(0,0,0,0)",
            font: { color: "#fff" },
            margin: { l: 50, r: 20, t: 40, b: 30 },
          }}
          config={{ displayModeBar: false }}
          className="w-full h-96"
        />
      </div>
    </motion.div>
  );
}
