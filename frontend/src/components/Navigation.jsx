import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShieldCheck, Stethoscope, BarChart2, Zap, Sun, Moon } from 'lucide-react';
import DisclaimerBar from './DisclaimerBar';
import { useTheme } from '../context/ThemeContext';

const Navigation = () => {
  const location = useLocation();
  const { theme, toggleTheme } = useTheme();

  const isActive = (path) => location.pathname === path;

  const navItems = [
    { path: '/predict', label: 'Clinical Prediction', icon: Stethoscope },
    { path: '/attacks', label: 'Adversarial Benchmarks', icon: Zap },
    { path: '/metrics', label: 'Scientific Metrics', icon: BarChart2 },
  ];

  return (
    <div className="sticky top-0 z-40">
      {/* Global Medical Disclaimer Bar */}
      <DisclaimerBar />

      {/* Sticky Healthcare Navbar */}
      <header className="bg-[#FFFDF9]/95 dark:bg-[#1C1814]/95 backdrop-blur-md border-b border-[#E7DDD2] dark:border-[#2E2721] shadow-xs transition-all">
        <div className="max-w-[1200px] mx-auto px-6 h-20 flex items-center justify-between">
          
          {/* Brand Logo */}
          <Link to="/" className="flex items-center space-x-3 group text-decoration-none">
            <div className="w-11 h-11 rounded-2xl bg-[#F4EFE6] dark:bg-[#2E2721] border border-[#E7DDD2] dark:border-[#3D332A] shadow-md shadow-[#8B6B4A]/20 flex items-center justify-center p-1.5 group-hover:scale-105 transition-all">
              <img 
                src="/logo.png" 
                alt="DermShield AI Logo" 
                className="w-full h-full object-contain" 
              />
            </div>
            <div>
              <div className="font-extrabold text-xl text-[#3B2F2F] dark:text-[#F5EFEB] tracking-tight flex items-center space-x-2">
                <span>DermShield AI</span>
                <span className="text-xs px-2.5 py-0.5 rounded-full font-bold bg-[#F4EFE6] dark:bg-[#2E2721] text-[#8B6B4A] dark:text-[#D4AF37] uppercase tracking-wider border border-[#E7DDD2] dark:border-[#3D332A]">
                  RASC-Net Defense
                </span>
              </div>
              <div className="text-[11px] text-[#7A624A] dark:text-[#B8A594] font-medium">Clinical Dermoscopy Decision Support System</div>
            </div>
          </Link>

          {/* Desktop Navigation Links & Dark Mode Toggle */}
          <div className="flex items-center space-x-3">
            <nav className="hidden md:flex items-center space-x-2 bg-[#F8F5F0] dark:bg-[#12100E] p-1.5 rounded-2xl border border-[#E7DDD2] dark:border-[#2E2721]">
              {navItems.map((item) => {
                const Icon = item.icon;
                const active = isActive(item.path);
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
                      active
                        ? 'bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] text-white shadow-md shadow-[#8B6B4A]/20 font-bold'
                        : 'text-[#7A624A] dark:text-[#B8A594] hover:text-[#3B2F2F] dark:hover:text-[#F5EFEB] hover:bg-[#FFFDF9] dark:hover:bg-[#1C1814]'
                    }`}
                  >
                    <Icon className={`w-4 h-4 ${active ? 'text-white' : 'text-[#8B6B4A] dark:text-[#D4AF37]'}`} />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </nav>

            {/* Dark Mode Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              aria-label="Toggle Dark Mode"
              className="p-2.5 rounded-2xl bg-[#F8F5F0] dark:bg-[#12100E] border border-[#E7DDD2] dark:border-[#2E2721] text-[#8B6B4A] dark:text-[#D4AF37] hover:scale-105 active:scale-95 transition-all shadow-xs cursor-pointer"
            >
              {theme === 'dark' ? (
                <Sun className="w-4 h-4 text-[#D4AF37]" />
              ) : (
                <Moon className="w-4 h-4 text-[#8B6B4A]" />
              )}
            </button>
          </div>

        </div>
      </header>
    </div>
  );
};

export default Navigation;
