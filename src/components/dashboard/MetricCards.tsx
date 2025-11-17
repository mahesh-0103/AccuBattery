import { motion } from "framer-motion";
import { Battery, AlertTriangle, TrendingUp, Activity } from "lucide-react";
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js-dist-min';

(Plot as any).plotly = Plotly;

export default function MetricCards({ df, threshold }) {
  
  // ----- REAL METRICS -----
  const anomalyCount = df.filter(r => r.anomaly_flag === "Anomaly").length;
  const anomalyPct = (anomalyCount / df.length * 100).toFixed(2);

  const avgVoltage = (
    df.reduce((a, b) => a + Number(b.volt || b.voltage || 0), 0) / df.length
  ).toFixed(2);

  const avgScore = (
    df.reduce((a, b) => a + Number(b.anomaly_score || 0), 0) / df.length
  ) / df.length;

  const batteryHealth = (100 - avgScore * 100).toFixed(1) + "%";

  const metrics = [
    {
      icon: Battery,
      title: "Battery Health",
      value: batteryHealth,
      change: "+ Stable",
      positive: true,
      color: "emerald",
    },
    {
      icon: AlertTriangle,
      title: "Anomalies Detected",
      value: anomalyCount,
      change: `${anomalyPct}% flagged`,
      positive: anomalyCount < df.length * threshold,
      color: "amber",
    },
    {
      icon: TrendingUp,
      title: "Anomaly Score (Avg)",
      value: avgScore.toFixed(3),
      change: avgScore < threshold ? "Normal" : "High Risk",
      positive: avgScore < threshold,
      color: "blue",
    },
    {
      icon: Activity,
      title: "Average Voltage",
      value: `${avgVoltage} V`,
      change: "Stable",
      positive: true,
      color: "emerald",
    },
  ];

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, index) => (
        <motion.div
          key={metric.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          whileHover={{ y: -4, transition: { duration: 0.2 } }}
          className="relative group"
        >
          <div
            className={`absolute inset-0 bg-gradient-to-br from-${metric.color}-500/10 to-${metric.color}-600/10 rounded-2xl blur-xl group-hover:blur-2xl transition-all`}
          />
          <div className="relative p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm hover:border-slate-700/50 transition-all">
            <div className="flex items-start justify-between mb 4">
              <div
                className={`p-3 rounded-xl bg-${metric.color}-500/20 group-hover:bg-${metric.color}-500/30 transition-all`}
              >
                <metric.icon className={`h-6 w-6 text-${metric.color}-400`} />
              </div>
            </div>

            <div className="space-y-1">
              <p className="text-slate-400 text-sm font-medium">{metric.title}</p>
              <p className="text-2xl font-bold text-slate-100">{metric.value}</p>
              <p
                className={`text-xs font-medium ${
                  metric.positive ? "text-emerald-400" : "text-red-400"
                }`}
              >
                {metric.change}
              </p>
            </div>
          </div>
        </motion.div>
      ))}
    </div>
  );
}
