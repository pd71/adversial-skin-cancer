import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Stethoscope, Eye, Zap, BarChart2, ShieldCheck, ArrowRight, Activity, 
  Sparkles, CheckCircle2, ChevronRight, Layers, HeartPulse, RefreshCw
} from 'lucide-react';

const Home = () => {
  const [started, setStarted] = useState(false);
  const navigate = useNavigate();

  const modules = [
    {
      id: 'predict',
      path: '/predict',
      title: 'Clinical Prediction & CDS',
      category: 'Parallel Dual Inference',
      description: 'Automated real-time classification using RASC-Net Proposed and Soft Voting Ensemble paired with rule-based clinical risk assessment.',
      icon: Stethoscope,
      badge: 'Dual Model Consensus',
      gradient: 'from-[#8B6B4A] to-[#6E5338]',
    },
    {
      id: 'attacks',
      path: '/attacks',
      title: 'Adversarial Benchmarks',
      category: 'Robustness Evaluation',
      description: 'On-the-fly FGSM, PGD, and CW gradient perturbation generation paired with defensive recovery rate benchmarking.',
      icon: Zap,
      badge: 'FGSM / PGD / CW',
      gradient: 'from-[#A67C52] to-[#8B6B4A]',
    },
    {
      id: 'gradcam',
      path: '/gradcam',
      title: 'Grad-CAM Explainability',
      category: 'Visual Explainability',
      description: 'Gradient-weighted class activation mapping targeting RASC-Net conv2d_20 layer for transparent spatial heatmap overlays.',
      icon: Eye,
      badge: 'Spatial Heatmaps',
      gradient: 'from-[#C8A97E] to-[#A67C52]',
    },
    {
      id: 'metrics',
      path: '/metrics',
      title: 'Scientific Metrics',
      category: 'Empirical Results',
      description: 'Master benchmark suite comparing 4 model candidates across clean accuracy, ECE calibration, robustness, and hardware throughput.',
      icon: BarChart2,
      badge: 'Full Evaluation',
      gradient: 'from-[#6E5338] to-[#3B2F2F]',
    },
  ];

  return (
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] font-sans relative overflow-hidden flex flex-col justify-center">
      
      {/* Background Ambient Glowing Blobs */}
      <div className="absolute top-10 left-1/4 w-[550px] h-[550px] bg-[#E7DDD2]/50 rounded-full blur-3xl pointer-events-none animate-pulse-glow" />
      <div className="absolute bottom-10 right-1/4 w-[450px] h-[450px] bg-[#C8A97E]/30 rounded-full blur-3xl pointer-events-none animate-float" />

      <AnimatePresence mode="wait">
        {!started ? (
          /* ================================================================ */
          /* SCREEN 1 — FULL SCREEN HERO LANDING PAGE                          */
          /* ================================================================ */
          <motion.div
            key="landing-hero"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, y: -40, filter: 'blur(10px)' }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="max-w-[1200px] mx-auto px-6 py-12 flex flex-col items-center justify-center text-center space-y-10 relative z-10 my-auto"
          >
            {/* Top Subtitle Badge */}
            <motion.div 
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center space-x-2.5 px-4 py-2 rounded-full bg-[#FFFDF9] border border-[#E7DDD2] shadow-xs text-xs font-semibold text-[#8B6B4A]"
            >
              <Sparkles className="w-4 h-4 text-[#8B6B4A]" />
              <span className="uppercase tracking-wider font-bold">Clinical AI Decision Support Platform</span>
            </motion.div>

            {/* Medical Skin Scanner & Neural SVG Graphic */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3, duration: 0.7 }}
              className="relative w-44 h-44 mx-auto"
            >
              {/* Pulse & Radar Scan Animations */}
              <div className="absolute inset-0 rounded-full border-2 border-[#C8A97E]/40 animate-pulse-glow" />
              <div className="absolute -inset-4 rounded-full border border-dashed border-[#8B6B4A]/40 animate-spin" style={{ animationDuration: '30s' }} />
              
              <div className="w-full h-full rounded-full bg-[#FFFDF9] border-2 border-[#8B6B4A]/40 flex items-center justify-center shadow-lg relative overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-b from-[#8B6B4A]/15 via-transparent to-transparent animate-scan" />
                <ShieldCheck className="w-20 h-20 text-[#8B6B4A] stroke-[1.5]" />
              </div>
            </motion.div>

            {/* Main Hero Header & Subtitle */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-4 max-w-3xl"
            >
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-[#3B2F2F] tracking-tight leading-tight">
                RASC-Net Clinical AI
              </h1>
              <h2 className="text-lg sm:text-2xl font-bold text-[#8B6B4A]">
                AI Assisted Skin Cancer Analysis and Adversarial Defense Platform
              </h2>
              <p className="text-sm sm:text-base text-[#7A624A] max-w-xl mx-auto leading-relaxed">
                Early detection through Artificial Intelligence, Explainable AI (Grad-CAM), and Adversarial Robustness Benchmarking.
              </p>
            </motion.div>

            {/* 3 Short Feature Bullets */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="flex flex-wrap items-center justify-center gap-3 sm:gap-6 pt-2"
            >
              {[
                { label: 'Clinical Prediction', icon: Stethoscope },
                { label: 'Adversarial Robustness', icon: Zap },
                { label: 'Explainable AI', icon: Eye },
              ].map((feat, i) => {
                const Icon = feat.icon;
                return (
                  <div key={i} className="flex items-center space-x-2 px-4 py-2 bg-[#FFFDF9] rounded-xl border border-[#E7DDD2] text-xs font-semibold text-[#4B3B2A] shadow-xs">
                    <Icon className="w-4 h-4 text-[#8B6B4A]" />
                    <span>{feat.label}</span>
                  </div>
                );
              })}
            </motion.div>

            {/* Large Start Button */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="pt-4"
            >
              <button
                onClick={() => setStarted(true)}
                className="px-10 py-4 bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#6E5338] hover:to-[#4B3B2A] text-white rounded-2xl font-bold text-lg shadow-xl shadow-[#8B6B4A]/30 flex items-center space-x-3 mx-auto transition-all transform hover:scale-105 active:scale-95 cursor-pointer"
              >
                <span>Start Clinical Assessment</span>
                <ArrowRight className="w-5 h-5" />
              </button>
            </motion.div>
          </motion.div>
        ) : (
          /* ================================================================ */
          /* SCREEN 2 — 2x2 DASHBOARD MODULE SELECTION GRID                   */
          /* ================================================================ */
          <motion.div
            key="dashboard-grid"
            initial={{ opacity: 0, y: 40, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="max-w-[1200px] mx-auto px-6 py-12 space-y-10 relative z-10 w-full my-auto"
          >
            {/* Dashboard Header */}
            <div className="flex flex-col sm:flex-row items-center justify-between border-b border-[#E7DDD2] pb-6 gap-4">
              <div>
                <span className="text-xs font-bold text-[#8B6B4A] uppercase tracking-widest bg-[#FFFDF9] px-3.5 py-1 rounded-full border border-[#E7DDD2]">
                  Clinical Dashboard
                </span>
                <h2 className="text-3xl font-extrabold text-[#3B2F2F] mt-2">
                  Select Intelligence Module
                </h2>
                <p className="text-sm text-[#7A624A] mt-1">
                  Choose a module below to launch the clinical decision support workflow.
                </p>
              </div>

              <button
                onClick={() => setStarted(false)}
                className="px-4 py-2 bg-[#FFFDF9] hover:bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2] rounded-xl text-xs font-bold transition-all shadow-xs flex items-center space-x-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Return to Landing</span>
              </button>
            </div>

            {/* ISSUE 2: 2x2 Responsive Grid Layout */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {modules.map((mod, idx) => {
                const Icon = mod.icon;
                return (
                  <motion.div
                    key={mod.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: idx * 0.1 }}
                    whileHover={{ y: -6 }}
                    onClick={() => navigate(mod.path)}
                    className="bg-[#FFFDF9] rounded-2xl p-7 border border-[#E7DDD2] shadow-md hover:shadow-xl transition-all flex flex-col justify-between cursor-pointer space-y-6 group relative overflow-hidden h-full"
                  >
                    {/* Top Gradient Stripe */}
                    <div className={`absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r ${mod.gradient}`} />

                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="w-13 h-13 rounded-2xl bg-[#F4EFE6] border border-[#E7DDD2] flex items-center justify-center text-[#8B6B4A] group-hover:bg-[#8B6B4A] group-hover:text-white transition-all shadow-xs">
                          <Icon className="w-6 h-6" />
                        </div>
                        <span className="text-xs font-bold px-3 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                          {mod.badge}
                        </span>
                      </div>

                      <div className="space-y-1">
                        <div className="text-xs font-semibold text-[#8B6B4A] uppercase tracking-wider">{mod.category}</div>
                        <h3 className="text-xl font-bold text-[#3B2F2F] group-hover:text-[#8B6B4A] transition-colors">
                          {mod.title}
                        </h3>
                      </div>

                      <p className="text-sm text-[#7A624A] leading-relaxed">
                        {mod.description}
                      </p>
                    </div>

                    <div className="pt-4 border-t border-[#F4EFE6] flex items-center justify-between text-sm font-bold text-[#8B6B4A] group-hover:text-[#6E5338]">
                      <span>Open Module</span>
                      <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  );
};

export default Home;
