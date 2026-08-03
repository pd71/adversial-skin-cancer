import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Stethoscope, Zap, BarChart2, ChevronRight, ShieldCheck
} from 'lucide-react';

const Dashboard = () => {
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
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] pb-16 font-sans relative overflow-hidden">
      
      {/* Background Soft Blobs */}
      <div className="absolute top-10 left-1/4 w-[500px] h-[500px] bg-[#E7DDD2]/40 rounded-full blur-3xl pointer-events-none animate-pulse-glow" />
      <div className="absolute bottom-10 right-1/4 w-[400px] h-[400px] bg-[#C8A97E]/25 rounded-full blur-3xl pointer-events-none animate-float" />

      <div className="max-w-[1200px] mx-auto px-6 pt-10 space-y-10 relative z-10">
        
        {/* Header Hero Section */}
        <div className="bg-[#FFFDF9] rounded-3xl p-8 md:p-10 shadow-md border border-[#E7DDD2] flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="space-y-3 max-w-2xl">
            <div className="inline-flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-bold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
              <ShieldCheck className="w-4 h-4 text-[#8B6B4A]" />
              <span>IPD Capstone • Production Medical AI Suite</span>
            </div>
            <h1 className="text-3xl md:text-4xl font-extrabold text-[#3B2F2F] tracking-tight">
              RASC-Net Clinical Decision Support Platform
            </h1>
            <p className="text-sm md:text-base text-[#7A624A] leading-relaxed">
              Residual Attention Skin Cancer Network with adversarial robustness benchmarking, clinical decision engine, and scientific evaluations.
            </p>
          </div>

          <button
            onClick={() => navigate('/predict')}
            className="px-6 py-3.5 bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#7A5B3D] hover:to-[#5E442B] text-white font-bold rounded-2xl shadow-lg shadow-[#8B6B4A]/25 flex items-center space-x-2 transition-all group shrink-0"
          >
            <span>Launch Clinical Predictor</span>
            <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>

        {/* Modules Grid */}
        <div>
          <h2 className="text-2xl font-bold text-[#3B2F2F] mb-6 tracking-tight">
            System Modules & Clinical Workflows
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {modules.map((mod) => {
              const Icon = mod.icon;
              return (
                <motion.div
                  key={mod.id}
                  whileHover={{ y: -4 }}
                  onClick={() => navigate(mod.path)}
                  className="bg-[#FFFDF9] rounded-3xl p-7 shadow-sm hover:shadow-md border border-[#E7DDD2] flex flex-col justify-between cursor-pointer group transition-all"
                >
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className={`p-3.5 rounded-2xl text-white bg-gradient-to-r ${mod.gradient} shadow-md`}>
                        <Icon className="w-6 h-6" />
                      </div>
                      <span className="text-[11px] font-bold px-3 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                        {mod.badge}
                      </span>
                    </div>

                    <div>
                      <div className="text-xs font-semibold text-[#8B6B4A] uppercase tracking-wider mb-1">
                        {mod.category}
                      </div>
                      <h3 className="text-xl font-bold text-[#3B2F2F] group-hover:text-[#8B6B4A] transition-colors">
                        {mod.title}
                      </h3>
                      <p className="text-xs text-[#7A624A] mt-2 leading-relaxed">
                        {mod.description}
                      </p>
                    </div>
                  </div>

                  <div className="pt-6 border-t border-[#F4EFE6] mt-6 flex items-center justify-between text-xs font-bold text-[#8B6B4A]">
                    <span>Open Module</span>
                    <ChevronRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;
