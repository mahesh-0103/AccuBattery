import { useState } from 'react';
import { motion } from 'framer-motion';
import { Settings as SettingsIcon, Moon, Sun, Save, Gauge } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { useSettings } from '../context/SettingsContext';

export default function SettingsPage() {
  const { theme, toggleTheme } = useTheme();
  const { settings, updateSettings } = useSettings();
  const [threshold, setThreshold] = useState(settings.anomalyThreshold);
  const [showSaved, setShowSaved] = useState(false);

  const handleSave = () => {
    updateSettings({ anomalyThreshold: threshold });
    setShowSaved(true);
    setTimeout(() => setShowSaved(false), 2000);
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="mb-8">
          <div className="flex items-center space-x-3 mb-2">
            <SettingsIcon className="h-8 w-8 text-emerald-400" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-blue-400 bg-clip-text text-transparent">
              Settings
            </h1>
          </div>
          <p className="text-slate-400">Customize your dashboard preferences</p>
        </div>

        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm"
          >
            <h2 className="text-xl font-semibold text-slate-200 mb-4">Unit System</h2>
            <div className="flex items-center space-x-4">
              <button
                onClick={() => updateSettings({ unitSystem: 'metric' })}
                className={`flex-1 py-3 px-6 rounded-xl font-medium transition-all ${
                  settings.unitSystem === 'metric'
                    ? 'bg-gradient-to-r from-emerald-500 to-blue-500 text-white shadow-lg shadow-emerald-500/30'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                Metric (°C, kW)
              </button>
              <button
                onClick={() => updateSettings({ unitSystem: 'imperial' })}
                className={`flex-1 py-3 px-6 rounded-xl font-medium transition-all ${
                  settings.unitSystem === 'imperial'
                    ? 'bg-gradient-to-r from-emerald-500 to-blue-500 text-white shadow-lg shadow-emerald-500/30'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                }`}
              >
                Imperial (°F, kW)
              </button>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm"
          >
            <div className="flex items-center space-x-3 mb-4">
              <Gauge className="h-5 w-5 text-emerald-400" />
              <h2 className="text-xl font-semibold text-slate-200">Anomaly Threshold</h2>
            </div>
            <p className="text-slate-400 text-sm mb-4">
              Set the sensitivity level for anomaly detection (0 = very sensitive, 1 = less sensitive)
            </p>
            <div className="space-y-3">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">Very Sensitive</span>
                <span className="text-emerald-400 font-semibold text-lg">
                  {threshold.toFixed(2)}
                </span>
                <span className="text-slate-400">Less Sensitive</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="p-6 rounded-2xl bg-slate-900/50 border border-slate-800/50 backdrop-blur-sm"
          >
            <h2 className="text-xl font-semibold text-slate-200 mb-4">Theme</h2>
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                {theme === 'dark' ? (
                  <>
                    <Moon className="h-5 w-5 text-slate-400" />
                    <span className="text-slate-300">Dark Mode</span>
                  </>
                ) : (
                  <>
                    <Sun className="h-5 w-5 text-amber-400" />
                    <span className="text-slate-300">Light Mode</span>
                  </>
                )}
              </div>
              <button
                onClick={toggleTheme}
                className={`relative w-16 h-8 rounded-full transition-colors ${
                  theme === 'dark' ? 'bg-emerald-500' : 'bg-slate-600'
                }`}
              >
                <motion.div
                  animate={{ x: theme === 'dark' ? 32 : 4 }}
                  transition={{ type: 'spring', stiffness: 500, damping: 30 }}
                  className="absolute top-1 left-1 w-6 h-6 bg-white rounded-full shadow-lg"
                />
              </button>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="flex items-center justify-end space-x-4"
          >
            {showSaved && (
              <motion.span
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className="text-emerald-400 font-medium"
              >
                Settings saved!
              </motion.span>
            )}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSave}
              className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-xl text-white font-semibold shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all flex items-center space-x-2"
            >
              <Save className="h-5 w-5" />
              <span>Save Preferences</span>
            </motion.button>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}
