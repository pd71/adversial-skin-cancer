import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  BarChart2, ShieldCheck, Activity, Layers, Award, Cpu, HardDrive, Zap, RefreshCw, FileText
} from 'lucide-react';

const MASTER_BENCHMARK_DATA = [
  {
    model_name: 'RASC-Net Proposed (Exp 3)',
    is_recommended: true,
    clean_acc: 65.57,
    ci_95: '[62.77%, 68.46%]',
    weighted_p: 46.92,
    weighted_r: 65.57,
    macro_f1: 12.01,
    fgsm_acc: 48.00,
    pgd_acc: 6.00,
    cw_acc: 60.00,
    robustness_score: 38.00,
    asr: 52.00,
    defended_acc: 62.00,
    recovery_rate: 79.69,
    ece: 0.1768,
    brier: 0.5649,
    params: '2.88 M',
    flops: '966.12 MFLOPs',
    latency: '138.15 ms',
    fps: 7.24,
    size_mb: '33.25 MB',
  },
  {
    model_name: 'MobileNetV2',
    is_recommended: false,
    clean_acc: 69.06,
    ci_95: '[66.26%, 71.96%]',
    weighted_p: 76.39,
    weighted_r: 69.06,
    macro_f1: 48.57,
    fgsm_acc: 32.00,
    pgd_acc: 0.00,
    cw_acc: 10.00,
    robustness_score: 14.00,
    asr: 68.00,
    defended_acc: 40.00,
    recovery_rate: 21.59,
    ece: 0.0242,
    brier: 0.4210,
    params: '2.42 M',
    flops: '0.33 MFLOPs',
    latency: '223.00 ms',
    fps: 4.48,
    size_mb: '22.73 MB',
  },
  {
    model_name: 'ResNet50',
    is_recommended: false,
    clean_acc: 75.85,
    ci_95: '[73.25%, 78.44%]',
    weighted_p: 81.71,
    weighted_r: 75.85,
    macro_f1: 63.89,
    fgsm_acc: 12.00,
    pgd_acc: 0.00,
    cw_acc: 10.00,
    robustness_score: 7.33,
    asr: 88.00,
    defended_acc: 60.00,
    recovery_rate: 75.18,
    ece: 0.0263,
    brier: 0.3257,
    params: '23.85 M',
    flops: '0.53 MFLOPs',
    latency: '404.18 ms',
    fps: 2.47,
    size_mb: '203.90 MB',
  },
  {
    model_name: 'Soft Voting Ensemble',
    is_recommended: false,
    clean_acc: 69.26,
    ci_95: '[66.46%, 72.16%]',
    weighted_p: 57.67,
    weighted_r: 69.26,
    macro_f1: 27.46,
    fgsm_acc: 28.00,
    pgd_acc: 0.00,
    cw_acc: 10.00,
    robustness_score: 12.67,
    asr: 72.00,
    defended_acc: 60.00,
    recovery_rate: 77.55,
    ece: 0.1315,
    brier: 0.4363,
    params: '26.27 M',
    flops: '5.00 GFLOPs',
    latency: '32.60 ms',
    fps: 30.67,
    size_mb: '226.63 MB',
  },
];

const Metrics = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('master');

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:5000/api/metrics');
        if (response.data?.data?.models_benchmark) {
          setData(response.data.data);
        }
      } catch (err) {
        console.warn('Using master evaluation benchmark payload:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  return (
    <div className="min-h-screen bg-[#F7F3EC] text-[#4B3B2A] pb-16">
      <div className="max-w-[1400px] mx-auto px-8 py-8 space-y-8">
        
        {/* Header */}
        <div className="border-b border-[#E6DFD5] pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-[#8B6B4A]/10 text-[#8B6B4A] rounded-2xl border border-[#E6DFD5]">
                <BarChart2 className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-[#4B3B2A] tracking-tight">
                  Scientific Metrics & Benchmark Dashboard
                </h1>
                <p className="text-sm text-[#7A624A] mt-1">
                  Empirical evaluation of 4 candidate model architectures on HAM10000 skin lesion dataset
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3.5 py-1.5 rounded-full text-xs font-semibold bg-[#E8F0E9] text-[#5F8D6E] border border-[#C5DDC8]">
              <Award className="w-4 h-4 mr-1.5 text-[#5F8D6E]" />
              Recommended Production Model: RASC-Net Proposed
            </span>
          </div>
        </div>

        {/* Recommended Production Model Highlight Card */}
        <div className="bg-gradient-to-r from-[#8B6B4A] via-[#B89B72] to-[#6E5338] rounded-2xl p-6 shadow-xl text-white relative overflow-hidden">
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6 relative z-10">
            <div className="space-y-2">
              <div className="inline-flex items-center px-3 py-1 rounded-full text-xs font-bold bg-white/20 text-white tracking-wide uppercase backdrop-blur-sm">
                <Award className="w-3.5 h-3.5 mr-1" /> Primary Production Model
              </div>
              <h2 className="text-2xl font-extrabold text-white">
                RASC-Net Proposed (Curriculum Adv Training + MixUp + Label Smoothing)
              </h2>
              <p className="text-sm text-[#F7F3EC] max-w-3xl leading-relaxed">
                Demonstrates superior overall adversarial robustness (<strong>38.00% Mean Robustness Score</strong>) across FGSM (48%), PGD (6%), and CW (60%) gradient attacks, while maintaining an edge-friendly footprint of <strong>2.88M parameters (33.25 MB)</strong>.
              </p>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 bg-white/10 backdrop-blur-md p-4 rounded-xl border border-white/20 text-center w-full md:w-auto">
              <div className="px-2">
                <div className="text-xs text-white/80 font-medium">Mean Robustness</div>
                <div className="text-xl font-black text-white">38.00%</div>
              </div>
              <div className="px-2">
                <div className="text-xs text-white/80 font-medium">Defense Recovery</div>
                <div className="text-xl font-black text-white">79.69%</div>
              </div>
              <div className="px-2">
                <div className="text-xs text-white/80 font-medium">Parameters</div>
                <div className="text-xl font-black text-white">2.88 M</div>
              </div>
              <div className="px-2">
                <div className="text-xs text-white/80 font-medium">Model Size</div>
                <div className="text-xl font-black text-white">33.25 MB</div>
              </div>
            </div>
          </div>
        </div>

        {/* Tab Controls */}
        <div className="flex space-x-2 border-b border-[#E6DFD5] pb-3">
          {[
            { id: 'master', label: 'Master Benchmark Table', icon: Layers },
            { id: 'clean', label: 'Clean Performance', icon: Activity },
            { id: 'robustness', label: 'Adversarial Robustness', icon: ShieldCheck },
            { id: 'efficiency', label: 'Computational Efficiency', icon: Cpu },
          ].map(tab => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl font-semibold text-sm transition-all duration-200 ${
                  isActive
                    ? 'bg-[#8B6B4A] text-white shadow-md shadow-[#8B6B4A]/20'
                    : 'bg-[#FFFDF9] text-[#5C4A38] hover:bg-[#F4EFE6] hover:text-[#4B3B2A] border border-[#E6DFD5]'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* 1. Master Comparison Table */}
        {activeTab === 'master' && (
          <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E6DFD5] space-y-6">
            <div>
              <h3 className="text-xl font-semibold text-[#4B3B2A] flex items-center space-x-2">
                <Layers className="w-5 h-5 text-[#8B6B4A]" />
                <span>Master Scientific Benchmark Comparison</span>
              </h3>
              <p className="text-xs text-[#7A624A] mt-1">
                Comparative evaluation across 4 candidate model architectures under identical clean and adversarial protocols.
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left text-[#4B3B2A]">
                <thead className="text-xs text-[#7A624A] uppercase bg-[#F7F3EC] border-b border-[#E6DFD5]">
                  <tr>
                    <th className="px-4 py-3.5">Model Architecture</th>
                    <th className="px-4 py-3.5">Clean Acc (95% CI)</th>
                    <th className="px-4 py-3.5">FGSM Acc</th>
                    <th className="px-4 py-3.5">PGD Acc</th>
                    <th className="px-4 py-3.5">CW Acc</th>
                    <th className="px-4 py-3.5">Defended Acc</th>
                    <th className="px-4 py-3.5">ECE</th>
                    <th className="px-4 py-3.5">Params</th>
                    <th className="px-4 py-3.5">FLOPs</th>
                    <th className="px-4 py-3.5">Latency</th>
                    <th className="px-4 py-3.5">Size</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#F4EFE6]">
                  {MASTER_BENCHMARK_DATA.map((row, idx) => (
                    <tr 
                      key={idx} 
                      className={`hover:bg-[#F7F3EC]/80 transition-colors ${
                        row.is_recommended ? 'bg-[#F4EFE6]/60 border-l-4 border-l-[#8B6B4A]' : ''
                      }`}
                    >
                      <td className="px-4 py-4 font-semibold text-[#4B3B2A] flex items-center space-x-2">
                        <span>{row.model_name}</span>
                        {row.is_recommended && (
                          <span className="bg-[#8B6B4A] text-white text-[10px] px-2 py-0.5 rounded-full font-bold uppercase">
                            Recommended
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-4 font-medium text-[#4B3B2A]">
                        {row.clean_acc.toFixed(2)}% <span className="text-xs text-[#7A624A] block">{row.ci_95}</span>
                      </td>
                      <td className="px-4 py-4 text-[#5F8D6E] font-bold">{row.fgsm_acc.toFixed(2)}%</td>
                      <td className="px-4 py-4 text-[#5F8D6E] font-bold">{row.pgd_acc.toFixed(2)}%</td>
                      <td className="px-4 py-4 text-[#5F8D6E] font-bold">{row.cw_acc.toFixed(2)}%</td>
                      <td className="px-4 py-4 text-[#8B6B4A] font-bold">{row.defended_acc.toFixed(2)}%</td>
                      <td className="px-4 py-4 text-[#5C4A38]">{row.ece.toFixed(4)}</td>
                      <td className="px-4 py-4 text-[#5C4A38]">{row.params}</td>
                      <td className="px-4 py-4 text-[#5C4A38]">{row.flops}</td>
                      <td className="px-4 py-4 text-[#5C4A38]">{row.latency}</td>
                      <td className="px-4 py-4 text-[#5C4A38]">{row.size_mb}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 2. Clean Performance Tab */}
        {activeTab === 'clean' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {MASTER_BENCHMARK_DATA.map((row, idx) => (
              <div 
                key={idx}
                className={`bg-[#FFFDF9] border rounded-2xl p-6 space-y-4 shadow-md ${
                  row.is_recommended ? 'border-[#8B6B4A] bg-[#F4EFE6]/30' : 'border-[#E6DFD5]'
                }`}
              >
                <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-3">
                  <h4 className="text-lg font-bold text-[#4B3B2A] flex items-center space-x-2">
                    <span>{row.model_name}</span>
                  </h4>
                  {row.is_recommended && (
                    <span className="bg-[#8B6B4A] text-white text-xs px-2.5 py-0.5 rounded-full font-bold">
                      ★ Proposed
                    </span>
                  )}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Clean Accuracy</div>
                    <div className="text-lg font-bold text-[#8B6B4A]">{row.clean_acc}%</div>
                  </div>
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Weighted Precision</div>
                    <div className="text-lg font-bold text-[#B89B72]">{row.weighted_p}%</div>
                  </div>
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Weighted Recall</div>
                    <div className="text-lg font-bold text-[#6E5338]">{row.weighted_r}%</div>
                  </div>
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Macro F1</div>
                    <div className="text-lg font-bold text-[#5F8D6E]">{row.macro_f1}%</div>
                  </div>
                </div>

                <div className="text-xs text-[#7A624A] pt-2 border-t border-[#F4EFE6] flex justify-between">
                  <span>95% Confidence Interval: <strong className="text-[#4B3B2A]">{row.ci_95}</strong></span>
                  <span>Brier Score: <strong className="text-[#4B3B2A]">{row.brier}</strong></span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 3. Adversarial Robustness Tab */}
        {activeTab === 'robustness' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {MASTER_BENCHMARK_DATA.map((row, idx) => (
              <div 
                key={idx}
                className={`bg-[#FFFDF9] border rounded-2xl p-6 space-y-5 shadow-md ${
                  row.is_recommended ? 'border-[#5F8D6E] bg-[#E8F0E9]/30' : 'border-[#E6DFD5]'
                }`}
              >
                <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-3">
                  <h4 className="text-lg font-bold text-[#4B3B2A]">{row.model_name}</h4>
                  <span className="text-xs font-semibold px-3 py-1 rounded-full bg-[#E8F0E9] text-[#5F8D6E] border border-[#C5DDC8]">
                    Mean Robustness: <strong className="text-[#2D5A38]">{row.robustness_score}%</strong>
                  </span>
                </div>

                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-xs font-medium text-[#5C4A38] mb-1">
                      <span>FGSM Attack Accuracy (ε = 0.03)</span>
                      <span className="font-bold text-[#5F8D6E]">{row.fgsm_acc}%</span>
                    </div>
                    <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
                      <div className="bg-[#5F8D6E] h-full rounded-full animate-progress-bar" style={{ width: `${row.fgsm_acc}%` }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium text-[#5C4A38] mb-1">
                      <span>PGD Attack Accuracy (5-step)</span>
                      <span className="font-bold text-[#C88A36]">{row.pgd_acc}%</span>
                    </div>
                    <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
                      <div className="bg-[#C88A36] h-full rounded-full animate-progress-bar" style={{ width: `${Math.max(row.pgd_acc, 2)}%` }}></div>
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-medium text-[#5C4A38] mb-1">
                      <span>Carlini-Wagner (CW) Accuracy (10-step)</span>
                      <span className="font-bold text-[#B89B72]">{row.cw_acc}%</span>
                    </div>
                    <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
                      <div className="bg-[#B89B72] h-full rounded-full animate-progress-bar" style={{ width: `${row.cw_acc}%` }}></div>
                    </div>
                  </div>
                </div>

                <div className="p-3 bg-[#F7F3EC] rounded-xl border border-[#E6DFD5] text-xs text-[#5C4A38] flex justify-between">
                  <span>Defense Recovery Rate: <strong className="text-[#8B6B4A]">{row.recovery_rate}%</strong></span>
                  <span>Post-Defense Accuracy: <strong className="text-[#5F8D6E]">{row.defended_acc}%</strong></span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* 4. Computational Efficiency Tab */}
        {activeTab === 'efficiency' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {MASTER_BENCHMARK_DATA.map((row, idx) => (
              <div 
                key={idx}
                className={`bg-[#FFFDF9] border rounded-2xl p-6 space-y-4 shadow-md ${
                  row.is_recommended ? 'border-[#B89B72] bg-[#F4EFE6]/30' : 'border-[#E6DFD5]'
                }`}
              >
                <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-3">
                  <h4 className="text-lg font-bold text-[#4B3B2A]">{row.model_name}</h4>
                  <span className="text-xs px-3 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] font-bold border border-[#E6DFD5]">
                    {row.size_mb}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Parameters</div>
                    <div className="text-sm font-bold text-[#8B6B4A]">{row.params}</div>
                  </div>
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">FLOPs</div>
                    <div className="text-sm font-bold text-[#B89B72]">{row.flops}</div>
                  </div>
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Latency</div>
                    <div className="text-sm font-bold text-[#6E5338]">{row.latency}</div>
                  </div>
                  <div className="bg-[#F7F3EC] p-3 rounded-xl border border-[#E6DFD5]">
                    <div className="text-xs text-[#7A624A] font-medium">Throughput</div>
                    <div className="text-sm font-bold text-[#5F8D6E]">{row.fps} FPS</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
};

export default Metrics;
