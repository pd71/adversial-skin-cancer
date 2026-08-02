import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  ShieldCheck, Stethoscope, Zap, Eye, ArrowRight, Sparkles
} from 'lucide-react';

const Landing = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] font-sans relative overflow-hidden flex flex-col justify-center items-center px-6 py-12">
      
      {/* Background Soft Ambient Blobs */}
      <div className="absolute top-10 left-1/4 w-[550px] h-[550px] bg-[#E7DDD2]/50 rounded-full blur-3xl pointer-events-none animate-pulse-glow" />
      <div className="absolute bottom-10 right-1/4 w-[450px] h-[450px] bg-[#C8A97E]/30 rounded-full blur-3xl pointer-events-none animate-float" />

      {/* Full Screen Hero Section (No 4 Cards Here) */}
      <motion.div 
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: 'easeOut' }}
        className="max-w-[1000px] mx-auto text-center space-y-10 relative z-10 my-auto"
      >
        {/* Top Header Badge */}
        <motion.div 
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="inline-flex items-center space-x-2.5 px-4.5 py-2 rounded-full bg-[#FFFDF9] border border-[#E7DDD2] shadow-xs text-xs font-semibold text-[#8B6B4A]"
        >
          <Sparkles className="w-4 h-4 text-[#8B6B4A]" />
          <span className="uppercase tracking-widest font-bold">Clinical AI Decision Support Platform</span>
        </motion.div>

        {/* Animated Medical Scan SVG Graphic */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3, duration: 0.7 }}
          className="relative w-44 h-44 mx-auto"
        >
          {/* Animated Glowing & Radar Rings */}
          <div className="absolute inset-0 rounded-full border-2 border-[#C8A97E]/40 animate-pulse-glow" />
          <div className="absolute -inset-4 rounded-full border border-dashed border-[#8B6B4A]/40 animate-spin" style={{ animationDuration: '30s' }} />
          
          <div className="w-full h-full rounded-full bg-[#FFFDF9] border-2 border-[#8B6B4A]/40 flex items-center justify-center shadow-xl relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-b from-[#8B6B4A]/15 via-transparent to-transparent animate-scan" />
            <ShieldCheck className="w-20 h-20 text-[#8B6B4A] stroke-[1.5]" />
          </div>
        </motion.div>

        {/* Main Title & Subtitle */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="space-y-4"
        >
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-[#3B2F2F] tracking-tight leading-tight">
            RASC-Net Clinical AI
          </h1>
          <h2 className="text-lg sm:text-2xl font-bold text-[#8B6B4A] max-w-2xl mx-auto">
            AI Assisted Skin Cancer Analysis and Adversarial Defense Platform
          </h2>
          <p className="text-sm sm:text-base text-[#7A624A] max-w-xl mx-auto leading-relaxed">
            Early detection through Artificial Intelligence, Explainable AI (Grad-CAM), and Adversarial Robustness Benchmarking.
          </p>
        </motion.div>

        {/* Three Feature Badges */}
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
              <div key={i} className="flex items-center space-x-2 px-4.5 py-2.5 bg-[#FFFDF9] rounded-xl border border-[#E7DDD2] text-xs font-semibold text-[#3B2F2F] shadow-xs">
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
            onClick={() => navigate('/dashboard')}
            className="px-10 py-4.5 bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#6E5338] hover:to-[#3B2F2F] text-white rounded-2xl font-bold text-lg shadow-xl shadow-[#8B6B4A]/30 flex items-center space-x-3 mx-auto transition-all transform hover:scale-105 active:scale-95 cursor-pointer"
          >
            <span>Start Clinical Assessment</span>
            <ArrowRight className="w-5 h-5" />
          </button>
        </motion.div>
      </motion.div>

    </div>
  );
};

export default Landing;
