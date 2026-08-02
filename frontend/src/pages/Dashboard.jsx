import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Stethoscope, Eye, Zap, BarChart2, ChevronRight, RefreshCw, ShieldCheck
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
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] pb-16 font-sans relative overflow-hidden">
      
      {/* Background Soft Blobs */}
      <div className="absolute top-10 left-1/4 w-[500px] h-[500px] bg-[#E7DDD2]/40 rounded-full blur-3xl pointer-events-none animate-pulse-glow" />
      <div className="absolute bottom-10 right-1/4 w-[400px] h-[400px] bg-[#C8A97E]/25 rounded-full blur-3xl pointer-events-none animate-float" />

      <div className="max-w-[1200px] mx-auto px-6 py-10 space-y-10 relative z-10">
        
        {/* Dashboard Header */}
        <div className="flex flex-col sm:flex-row items-center justify-between border-b border-[#E7DDD2] pb-6 gap-4">
          <div>
            <span className="text-xs font-bold text-[#8B6B4A] uppercase tracking-widest bg-[#FFFDF9] px-3.5 py-1 rounded-full border border-[#E7DDD2]">
              Clinical Intelligence Dashboard
            </span>
            <h2 className="text-3xl font-extrabold text-[#3B2F2F] mt-2">
              Select AI Intelligence Module
            </h2>
            <p className="text-sm text-[#7A624A] mt-1">
              Choose a module below to launch the clinical decision support workflow.
            </p>
          </div>

          <button
            onClick={() => navigate('/')}
            className="px-4 py-2 bg-[#FFFDF9] hover:bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2] rounded-xl text-xs font-bold transition-all shadow-xs flex items-center space-x-1.5 cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Return to Landing</span>
          </button>
        </div>

        {/* 2 × 2 Responsive Cards Grid */}
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

      </div>
    </div>
  );
};

export default Dashboard;
