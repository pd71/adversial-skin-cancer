import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ShieldCheck, Stethoscope, BarChart2, Zap, Eye } from 'lucide-react';
import DisclaimerBar from './DisclaimerBar';

const Navigation = () => {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const navItems = [
    { path: '/predict', label: 'Clinical Prediction', icon: Stethoscope },
    { path: '/gradcam', label: 'Grad-CAM', icon: Eye },
    { path: '/attacks', label: 'Adversarial Benchmarks', icon: Zap },
    { path: '/metrics', label: 'Scientific Metrics', icon: BarChart2 },
  ];

  return (
    <div className="sticky top-0 z-40">
      {/* Global Medical Disclaimer Bar */}
      <DisclaimerBar />

      {/* Sticky Healthcare Navbar */}
      <header className="bg-[#FFFDF9]/95 backdrop-blur-md border-b border-[#E7DDD2] shadow-xs transition-all">
        <div className="max-w-[1200px] mx-auto px-6 h-20 flex items-center justify-between">
          
          {/* Brand Logo */}
          <Link to="/" className="flex items-center space-x-3 group text-decoration-none">
            <div className="w-11 h-11 rounded-2xl bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] flex items-center justify-center text-white shadow-md shadow-[#8B6B4A]/25 group-hover:scale-105 transition-all">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <div className="font-extrabold text-xl text-[#3B2F2F] tracking-tight flex items-center space-x-2">
                <span>RASC-Net</span>
                <span className="text-xs px-2.5 py-0.5 rounded-full font-bold bg-[#F4EFE6] text-[#8B6B4A] uppercase tracking-wider border border-[#E7DDD2]">
                  Clinical AI
                </span>
              </div>
              <div className="text-[11px] text-[#7A624A] font-medium">Skin Cancer Decision Support System</div>
            </div>
          </Link>

          {/* Navigation Links */}
          <nav className="flex items-center space-x-1 sm:space-x-2">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.path);
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl font-semibold text-xs sm:text-sm transition-all duration-200 ${
                    active
                      ? 'bg-[#8B6B4A] text-white shadow-md shadow-[#8B6B4A]/20'
                      : 'text-[#5C4A38] hover:text-[#3B2F2F] hover:bg-[#F4EFE6]/80'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${active ? 'text-white' : 'text-[#8B6B4A]'}`} />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

        </div>
      </header>
    </div>
  );
};

export default Navigation;
