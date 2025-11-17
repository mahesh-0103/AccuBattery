import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

import Layout from './components/Layout';
import UploadPage from './pages/UploadPage';
import DashboardPage from './pages/DashboardPage';
import SettingsPage from './pages/SettingsPage';

import { ThemeProvider } from './context/ThemeContext';
import { SettingsProvider } from './context/SettingsContext';

function App() {
  return (
    <ThemeProvider>
      <SettingsProvider>
        <Router>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<UploadPage />} />
              <Route path="upload" element={<UploadPage />} />
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="settings" element={<SettingsPage />} />
              {/* Fallback route */}
              <Route path="*" element={<Navigate to="/" replace />} />
            </Route>
          </Routes>
        </Router>
      </SettingsProvider>
    </ThemeProvider>
  );
}

export default App;
