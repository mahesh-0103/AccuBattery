import { motion } from "framer-motion";
import { useSettings } from "../../context/SettingsContext";
import Plot from 'react-plotly.js';
import * as Plotly from 'plotly.js-dist-min';
(Plot as any).plotly = Plotly;


export default function UnitToggle() {
  const { settings, updateSettings } = useSettings();

  return (
    <div className="flex items-center space-x-2 p-1 bg-slate-800/50 rounded-lg border border-slate-700 backdrop-blur-md shadow-md shadow-black/20">
      {["metric", "imperial"].map(mode => (
        <button
          key={mode}
          onClick={() => updateSettings({ unitSystem: mode })}
          className={`relative px-3 py-1.5 text-sm font-medium rounded transition-all ${
            settings.unitSystem === mode
              ? "text-white"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          {settings.unitSystem === mode && (
            <motion.div
              layoutId="unit-toggle"
              className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-blue-500 rounded"
              transition={{ type: "spring", stiffness: 450, damping: 28 }}
            />
          )}
          <span className="relative z-10 capitalize">{mode}</span>
        </button>
      ))}
    </div>
  );
}
