import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Battery, Upload, LayoutDashboard, Settings, Moon, Sun } from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

export default function Navbar() {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  const links = [
    { path: '/upload', label: 'Upload', icon: Upload },
    { path: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <nav className="relative z-50 border-b border-slate-800/50 backdrop-blur-xl bg-slate-950/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2 group">
            <div className="relative">
              <Battery className="h-8 w-8 text-emerald-400 group-hover:text-emerald-300 transition-colors" />
              <div className="absolute inset-0 bg-emerald-400/20 blur-xl group-hover:bg-emerald-400/30 transition-all" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-emerald-400 to-blue-400 bg-clip-text text-transparent">
              AccuBattery
            </span>
          </Link>

          <div className="hidden md:flex items-center space-x-1">
            {links.map(({ path, label, icon: Icon }) => {
              const isActive = location.pathname === path;
              return (
                <Link key={path} to={path} className="relative px-4 py-2 group">
                  <div className="flex items-center space-x-2">
                    <Icon className={`h-4 w-4 transition-colors ${isActive ? 'text-emerald-400' : 'text-slate-400 group-hover:text-slate-200'}`} />
                    <span className={`text-sm font-medium transition-colors ${isActive ? 'text-emerald-400' : 'text-slate-400 group-hover:text-slate-200'}`}>
                      {label}
                    </span>
                  </div>
                  {isActive && (
                    <motion.div
                      layoutId="navbar-indicator"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-emerald-400 to-blue-400"
                      transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                    />
                  )}
                </Link>
              );
            })}
          </div>

          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors border border-slate-700/50"
          >
            {theme === 'dark' ? (
              <Sun className="h-5 w-5 text-amber-400" />
            ) : (
              <Moon className="h-5 w-5 text-slate-400" />
            )}
          </button>
        </div>
      </div>
    </nav>
  );
}
