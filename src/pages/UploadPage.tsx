import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, CheckCircle, AlertCircle, ArrowRight } from 'lucide-react';

export default function UploadPage() {
  const navigate = useNavigate();
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadComplete, setUploadComplete] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  // =============================
  // DRAG & DROP HANDLERS
  // =============================
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (!droppedFile?.name.endsWith('.csv')) {
      setErrorMsg("Only CSV files are allowed.");
      return;
    }

    setFile(droppedFile);
    setUploadComplete(false);
    setErrorMsg("");
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];

    if (!selected?.name.endsWith('.csv')) {
      setErrorMsg("Only CSV files are allowed.");
      return;
    }

    setFile(selected);
    setUploadComplete(false);
    setErrorMsg("");
  };

  // =============================
  //    REAL BACKEND INTEGRATION
  // =============================
  const handleAnalyze = async () => {
    if (!file) return;

    setIsUploading(true);
    setUploadProgress(10);
    setErrorMsg("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const backendURL = "http://127.0.0.1:8000/predict_csv";

      const res = await fetch(backendURL, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Backend error. Check server console.");
      }

      setUploadProgress(60);

      const output = await res.json();

      if (output.error) {
        throw new Error(output.error);
      }

      // Save to dashboard
      sessionStorage.setItem("accubattery_result", JSON.stringify(output));

      setUploadProgress(100);
      setUploadComplete(true);
    } catch (err: any) {
      setErrorMsg(err.message || "Unexpected error");
    }

    setIsUploading(false);
  };

  const navigateToDashboard = () => {
    navigate("/dashboard");
  };

  // =============================
  // JSX UI
  // =============================
  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-emerald-400 to-blue-400 bg-clip-text text-transparent">
            Upload Battery Data
          </h1>
          <p className="text-slate-400 text-lg">
            Upload your CSV file to begin advanced battery analytics
          </p>
        </div>

        {/* Error Message */}
        {errorMsg && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400"
          >
            {errorMsg}
          </motion.div>
        )}

        <div className="relative">
          <motion.div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            animate={{
              scale: isDragging ? 1.02 : 1,
              borderColor: isDragging ? "rgb(52, 211, 153)" : "rgb(51, 65, 85)"
            }}
            className="relative p-12 rounded-3xl border-2 border-dashed transition-all duration-300 bg-slate-900/30 backdrop-blur-sm"
          >
            {/* File not selected */}
            {!file ? (
              <div className="text-center">
                <motion.div
                  animate={{
                    y: isDragging ? -10 : 0,
                    scale: isDragging ? 1.1 : 1
                  }}
                  className="mb-6"
                >
                  <Upload className="h-16 w-16 mx-auto text-emerald-400" />
                </motion.div>
                <h3 className="text-xl font-semibold text-slate-200 mb-2">
                  Drag and drop your CSV file here
                </h3>
                <p className="text-slate-400 mb-6">or</p>

                <label>
                  <input type="file" accept=".csv" onChange={handleFileInput} className="hidden" />
                  <motion.span
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="px-6 py-3 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-xl text-white font-medium cursor-pointer inline-block shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/40 transition-all"
                  >
                    Browse Files
                  </motion.span>
                </label>

                <p className="text-slate-500 text-sm mt-4">Supported format: CSV</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* File card */}
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="flex items-center space-x-4 p-6 rounded-2xl bg-slate-800/50 border border-slate-700/50"
                >
                  <div className="p-3 rounded-xl bg-emerald-500/20">
                    <FileText className="h-6 w-6 text-emerald-400" />
                  </div>

                  <div className="flex-1">
                    <h4 className="font-semibold text-slate-200">{file.name}</h4>
                    <p className="text-sm text-slate-400">
                      {(file.size / 1024).toFixed(2)} KB
                    </p>
                  </div>

                  {uploadComplete && (
                    <CheckCircle className="h-6 w-6 text-emerald-400" />
                  )}
                </motion.div>

                {/* Upload progress */}
                <AnimatePresence>
                  {isUploading && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="space-y-2"
                    >
                      <div className="flex items-center justify-between text-sm text-slate-400">
                        <span>Processing...</span>
                        <span>{uploadProgress}%</span>
                      </div>

                      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${uploadProgress}%` }}
                          className="h-full bg-gradient-to-r from-emerald-500 to-blue-500"
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Completed message */}
                {uploadComplete && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center space-x-2 p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30"
                  >
                    <CheckCircle className="h-5 w-5 text-emerald-400" />
                    <span className="text-emerald-400 font-medium">
                      Analysis complete! Ready to view dashboard.
                    </span>
                  </motion.div>
                )}

                {/* Buttons */}
                <div className="flex items-center space-x-4">
                  {!uploadComplete ? (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleAnalyze}
                      disabled={isUploading}
                      className="flex-1 py-3 px-6 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-xl text-white font-semibold shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isUploading ? "Analyzing..." : "Analyze Battery Data"}
                    </motion.button>
                  ) : (
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={navigateToDashboard}
                      className="flex-1 py-3 px-6 bg-gradient-to-r from-emerald-500 to-blue-500 rounded-xl text-white font-semibold shadow-lg shadow-emerald-500/30 hover:shadow-emerald-500/50 transition-all flex items-center justify-center space-x-2"
                    >
                      <span>View Dashboard</span>
                      <ArrowRight className="h-5 w-5" />
                    </motion.button>
                  )}

                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      setFile(null);
                      setUploadProgress(0);
                      setUploadComplete(false);
                      setErrorMsg("");
                    }}
                    className="px-6 py-3 bg-slate-800 hover:bg-slate-700 rounded-xl text-slate-300 font-medium transition-all"
                  >
                    Clear
                  </motion.button>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}

