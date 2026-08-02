import React, { useState } from 'react';
import axios from 'axios';
import { 
  Zap, AlertTriangle, ShieldCheck, Layers
} from 'lucide-react';
import ImageUploadCard from '../components/ImageUploadCard';

const Attacks = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setImagePreview(URL.createObjectURL(file));
    setResults(null);
    setError(null);
  };

  const handleImageRemoved = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  const handleRunAttacks = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post('http://localhost:5000/api/attacks', formData);
      setResults(response.data);
    } catch (err) {
      console.error('Attack API Error:', err);
      setError(err.response?.data?.error || err.message || 'Failed to generate adversarial attack examples.');
    } finally {
      setLoading(false);
    }
  };

  const attackConfigs = [
    {
      id: 'fgsm',
      name: 'FGSM (Fast Gradient Sign Method)',
      epsilon: 'ε = 0.03',
      type: 'Single-Step Gradient Attack',
      description: 'One-step perturbation calculated along the sign of the loss gradient.',
      recovery: '79.69% Recovery Rate',
    },
    {
      id: 'pgd',
      name: 'PGD (Projected Gradient Descent)',
      epsilon: 'ε = 0.03, 5-step',
      type: 'Multi-Step Iterative Attack',
      description: 'Iterative gradient steps projected back into the norm-constrained ball.',
      recovery: '62.00% Defended Accuracy',
    },
    {
      id: 'cw',
      name: 'CW (Carlini & Wagner)',
      epsilon: '10-step L2 norm',
      type: 'Optimization-Based Attack',
      description: 'Formulates adversarial generation as an unconstrained optimization problem.',
      recovery: '60.00% Defended Accuracy',
    },
  ];

  return (
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] pb-16">
      
      {/* Container: max-w-1200px, 32px padding */}
      <div className="max-w-[1200px] mx-auto px-6 py-8 space-y-8">
        
        {/* Page Header */}
        <div className="border-b border-[#E7DDD2] pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-[#8B6B4A]/10 text-[#8B6B4A] rounded-2xl border border-[#E7DDD2]">
                <Zap className="w-8 h-8 fill-current" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-[#3B2F2F] tracking-tight">
                  Adversarial Attack & Robustness Benchmarks
                </h1>
                <p className="text-sm text-[#7A624A] mt-1">
                  On-the-fly gradient perturbation generation (FGSM, PGD, CW) and defensive recovery evaluation
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3.5 py-1.5 rounded-full text-xs font-semibold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
              <ShieldCheck className="w-4 h-4 mr-1.5 text-[#8B6B4A]" />
              Curriculum Adversarial Defense Active
            </span>
          </div>
        </div>

        {/* Input & Configuration Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* 1. Universal Image Upload Component */}
          <div className="lg:col-span-5 space-y-4">
            <ImageUploadCard
              title="Upload Dermoscopy Image"
              description="Drag & drop, browse files, or paste (Ctrl + V) from clipboard"
              selectedFile={selectedFile}
              imagePreview={imagePreview}
              onImageSelected={handleImageSelected}
              onImageRemoved={handleImageRemoved}
            />

            {/* Run Attack Generation Button */}
            <button
              onClick={handleRunAttacks}
              disabled={!selectedFile || loading}
              className={`w-full py-4 px-6 rounded-xl font-bold text-white shadow-lg transition-all flex items-center justify-center space-x-2 text-base ${
                !selectedFile || loading
                  ? 'bg-[#C8A97E] cursor-not-allowed opacity-70'
                  : 'bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#6E5338] hover:to-[#3B2F2F] shadow-[#8B6B4A]/25 hover:scale-[1.01]'
              }`}
            >
              {loading ? (
                <>
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Generating Adversarial Attacks...</span>
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5 fill-current" />
                  <span>Run Adversarial Attack Protocol</span>
                </>
              )}
            </button>

            {error && (
              <div className="p-4 bg-[#FBF0EF] border border-[#F2D6D3] rounded-xl text-[#C0564B] text-xs flex items-start space-x-2">
                <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* 2. Attack Configuration Specifications */}
          <div className="lg:col-span-7 bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-6">
            <div className="border-b border-[#F4EFE6] pb-4 flex items-center justify-between">
              <h3 className="text-xl font-semibold text-[#3B2F2F] flex items-center space-x-2">
                <Layers className="w-5 h-5 text-[#8B6B4A]" />
                <span>Adversarial Threat Specifications</span>
              </h3>
              <span className="text-xs font-medium text-[#7A624A]">Gradient Attack Protocol</span>
            </div>

            <div className="space-y-4">
              {attackConfigs.map((cfg) => (
                <div key={cfg.id} className="p-4 bg-[#F8F5F0] rounded-xl border border-[#E7DDD2] space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="font-bold text-sm text-[#3B2F2F] flex items-center space-x-2">
                      <Zap className="w-4 h-4 text-[#8B6B4A]" />
                      <span>{cfg.name}</span>
                    </div>
                    <span className="text-xs px-2.5 py-0.5 rounded-full font-bold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                      {cfg.epsilon}
                    </span>
                  </div>
                  <p className="text-xs text-[#7A624A] leading-relaxed">
                    {cfg.description}
                  </p>
                  <div className="text-[11px] font-semibold text-[#5F8D6E] pt-1">
                    ✓ Benchmarking Baseline: {cfg.recovery}
                  </div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Results Visual Comparison Grid */}
        {results && (
          <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-lg border border-[#E7DDD2] space-y-6 animate-fade-in">
            <div className="border-b border-[#F4EFE6] pb-4 flex items-center justify-between">
              <div>
                <h3 className="text-xl font-bold text-[#3B2F2F]">
                  Adversarial Perturbation & Defense Output Comparison
                </h3>
                <p className="text-xs text-[#7A624A] mt-1">
                  Side-by-side visual comparison of original clean input against FGSM, PGD, and CW attack outputs
                </p>
              </div>
              <span className="text-xs px-3 py-1 rounded-full font-bold bg-[#E8F0E9] text-[#5F8D6E] border border-[#C5DDC8]">
                ✓ Attack Protocol Complete
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 text-center">
              
              {/* Clean Original */}
              <div className="bg-[#F8F5F0] p-4 rounded-xl border border-[#E7DDD2] space-y-3">
                <div className="text-sm font-bold text-[#3B2F2F] border-b border-[#E7DDD2] pb-2">
                  Clean Original
                </div>
                {results.original ? (
                  <img src={`data:image/jpeg;base64,${results.original}`} alt="Original" className="w-full h-48 object-cover rounded-lg border border-[#E7DDD2] shadow-xs" />
                ) : (
                  <div className="h-48 flex items-center justify-center text-xs text-[#7A624A]">No Image</div>
                )}
                <div className="text-xs font-medium text-[#5F8D6E] bg-[#E8F0E9] p-2 rounded-lg border border-[#C5DDC8]">
                  Baseline Input (0 Perturbation)
                </div>
              </div>

              {/* FGSM Attack */}
              <div className="bg-[#F8F5F0] p-4 rounded-xl border border-[#E7DDD2] space-y-3">
                <div className="text-sm font-bold text-[#3B2F2F] border-b border-[#E7DDD2] pb-2 flex justify-between items-center">
                  <span>FGSM Attack</span>
                  <span className="text-[10px] bg-[#FDF5E6] text-[#C88A36] px-2 py-0.5 rounded font-bold">ε = 0.03</span>
                </div>
                {results.fgsm ? (
                  <img src={`data:image/jpeg;base64,${results.fgsm}`} alt="FGSM Attack" className="w-full h-48 object-cover rounded-lg border border-[#E7DDD2] shadow-xs" />
                ) : (
                  <div className="h-48 flex items-center justify-center text-xs text-[#7A624A]">No Image</div>
                )}
                <div className="text-xs font-medium text-[#7A624A] bg-[#FFFDF9] p-2 rounded-lg border border-[#E7DDD2]">
                  Single-step Sign Gradient
                </div>
              </div>

              {/* PGD Attack */}
              <div className="bg-[#F8F5F0] p-4 rounded-xl border border-[#E7DDD2] space-y-3">
                <div className="text-sm font-bold text-[#3B2F2F] border-b border-[#E7DDD2] pb-2 flex justify-between items-center">
                  <span>PGD Attack</span>
                  <span className="text-[10px] bg-[#FDF5E6] text-[#C88A36] px-2 py-0.5 rounded font-bold">5 Iter</span>
                </div>
                {results.pgd ? (
                  <img src={`data:image/jpeg;base64,${results.pgd}`} alt="PGD Attack" className="w-full h-48 object-cover rounded-lg border border-[#E7DDD2] shadow-xs" />
                ) : (
                  <div className="h-48 flex items-center justify-center text-xs text-[#7A624A]">No Image</div>
                )}
                <div className="text-xs font-medium text-[#7A624A] bg-[#FFFDF9] p-2 rounded-lg border border-[#E7DDD2]">
                  Multi-step Projected Gradient
                </div>
              </div>

              {/* CW Attack */}
              <div className="bg-[#F8F5F0] p-4 rounded-xl border border-[#E7DDD2] space-y-3">
                <div className="text-sm font-bold text-[#3B2F2F] border-b border-[#E7DDD2] pb-2 flex justify-between items-center">
                  <span>CW Attack</span>
                  <span className="text-[10px] bg-[#FDF5E6] text-[#C88A36] px-2 py-0.5 rounded font-bold">L2 Norm</span>
                </div>
                {results.cw ? (
                  <img src={`data:image/jpeg;base64,${results.cw}`} alt="CW Attack" className="w-full h-48 object-cover rounded-lg border border-[#E7DDD2] shadow-xs" />
                ) : (
                  <div className="h-48 flex items-center justify-center text-xs text-[#7A624A]">No Image</div>
                )}
                <div className="text-xs font-medium text-[#7A624A] bg-[#FFFDF9] p-2 rounded-lg border border-[#E7DDD2]">
                  Optimization-based Norm Attack
                </div>
              </div>

            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default Attacks;
