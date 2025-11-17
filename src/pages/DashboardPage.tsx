import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Download, FileDown } from "lucide-react";

import MetricCards from "../components/dashboard/MetricCards";
import AnomalyScoreTrend from "../components/dashboard/AnomalyScoreTrend";
import TemperatureTrend from "../components/dashboard/TemperatureTrend";
import AnomalyDistribution from "../components/dashboard/AnomalyDistribution";
import CorrelationHeatmap from "../components/dashboard/CorrelationHeatmap";
import FeatureHistograms from "../components/dashboard/FeatureHistograms";
import UnitToggle from "../components/dashboard/UnitToggle";

export default function DashboardPage() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("accubattery_result");
    if (stored) {
      setData(JSON.parse(stored));
    }
  }, []);

  if (!data) {
    return (
      <div className="text-center py-24 text-slate-300 text-lg">
        No data available. Please upload a CSV first.
      </div>
    );
  }

  const df = data.data;        
  const rows = data.rows;      
  const threshold = data.threshold;

  // ============================
  // CSV EXPORT
  // ============================
  const handleExportCSV = () => {
    const csv = [
      Object.keys(df[0]).join(","),
      ...df.map((row: any) => Object.values(row).join(",")),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "accubattery_output.csv";
    link.click();
  };

  // ============================
  // PDF EXPORT
  // ============================
  const handleDownloadPDF = () => {
    fetch("http://127.0.0.1:8000/export_pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: df }),
    })
      .then((res) => res.blob())
      .then((blob) => {
        const link = document.createElement("a");
        link.href = window.URL.createObjectURL(blob);
        link.download = "AccuBattery_Report.pdf";
        link.click();
      })
      .catch(() => alert("PDF creation failed"));
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        {/* HEADER */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-8 gap-4">
          <div>
            <h1 className="text-3xl font-bold mb-2 bg-gradient-to-r from-emerald-400 to-blue-400 bg-clip-text text-transparent">
              Battery Analytics Dashboard
            </h1>
            <p className="text-slate-400">AI-powered insights & anomaly detection</p>
          </div>

          <div className="flex items-center space-x-3">
            <UnitToggle />

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleExportCSV}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-300 font-medium transition-all flex items-center space-x-2 border border-slate-700"
            >
              <FileDown className="h-4 w-4" />
              <span>Export CSV</span>
            </motion.button>

            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleDownloadPDF}
              className="px-4 py-2 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-lg text-white font-medium shadow-lg shadow-emerald-500/20 transition-all flex items-center space-x-2"
            >
              <Download className="h-4 w-4" />
              <span>Download PDF</span>
            </motion.button>
          </div>
        </div>

        {/* METRICS */}
        <MetricCards df={df} rows={rows} threshold={threshold} />

        {/* CHARTS */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <AnomalyScoreTrend df={df} />
          <TemperatureTrend df={df} />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
          <AnomalyDistribution df={df} />
          <CorrelationHeatmap df={df} />
        </div>

        <FeatureHistograms df={df} />
      </motion.div>
    </div>
  );
}
