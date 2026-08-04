import React, { useState } from 'react';
import axios from 'axios';
import { 
  CheckCircle2, AlertTriangle, X, HeartPulse, 
  Stethoscope, FileText, Printer, Zap, Sparkles, Eye, RefreshCw
} from 'lucide-react';
import ImageUploadCard from '../components/ImageUploadCard';
import SearchableCombobox from '../components/SearchableCombobox';
import medicalOptions from '../data/medicalOptions';

const rawApiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';
const API_BASE = rawApiBase.replace(/\/+$/, '');

const Predict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  
  // Patient Metadata Form State (12 HAM10000 Fields)
  const [metadata, setMetadata] = useState({
    age_approx: '60',
    sex: 'male',
    anatom_site_1: 'torso',
    anatom_site_2: 'cheek',
    anatom_site_3: 'right side',
    melanocytic: 'false',
    concomitant_biopsy: 'false',
    diagnosis_1: 'none',
    diagnosis_2: 'none',
    diagnosis_3: 'none',
    diagnosis_4: 'none',
    diagnosis_5: 'none',
  });

  // State for Dual Predictions
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictionData, setPredictionData] = useState(null);

  // State for Grad-CAM
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState(null);
  const [gradcamData, setGradcamData] = useState(null);
  const [activeGradcamTab, setActiveGradcamTab] = useState('overlay'); // 'overlay' | 'heatmap'

  // Modal State
  const [isReportModalOpen, setIsReportModalOpen] = useState(false);

  const handleMetadataChange = (e) => {
    const { name, value } = e.target;
    setMetadata(prev => ({ ...prev, [name]: value }));
  };

  const handleImageSelected = (file) => {
    setSelectedFile(file);
    setImagePreview(URL.createObjectURL(file));
    setError(null);
    setPredictionData(null);
    setGradcamData(null);
    setGradcamError(null);
  };

  const handleImageRemoved = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setPredictionData(null);
    setError(null);
    setGradcamData(null);
    setGradcamError(null);
  };

  const fetchGradcam = async (fileToUse) => {
    const file = fileToUse || selectedFile;
    if (!file) return;

    setGradcamLoading(true);
    setGradcamError(null);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await axios.post(`${API_BASE}/api/gradcam`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (res.data && res.data.success) {
        setGradcamData(res.data);
      } else {
        setGradcamError(res.data?.error || 'Failed to generate Grad-CAM visualization.');
      }
    } catch (err) {
      console.error('Grad-CAM API error:', err);
      const serverErr = err.response?.data?.error || err.response?.data?.message;
      setGradcamError(serverErr ? `Grad-CAM failed: ${serverErr}` : `Grad-CAM request failed: ${err.message}`);
    } finally {
      setGradcamLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please upload a dermoscopy image before running prediction.');
      return;
    }

    setLoading(true);
    setError(null);
    setPredictionData(null);
    setGradcamData(null);
    setGradcamError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('metadata', JSON.stringify(metadata));

    try {
      const res = await axios.post(`${API_BASE}/api/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPredictionData(res.data);
      setLoading(false);

    } catch (err) {
      const serverErr = err.response?.data?.error || err.response?.data?.message;
      setError(serverErr ? `Prediction failed: ${serverErr}` : `Prediction failed: ${err.message || 'Error executing dual model prediction'}`);
      setLoading(false);
    }
  };

  const rascData = predictionData?.rasc_net_proposed;
  const ensembleData = predictionData?.soft_voting_ensemble;
  const riskData = predictionData?.clinical_assessment;
  const modelsAgree = predictionData?.models_agree;

  const handlePrintReport = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] pb-16 font-sans">
      
      <div className="max-w-[1200px] mx-auto px-6 pt-8 space-y-8">
        
        {/* Page Title Header */}
        <div className="border-b border-[#E7DDD2] pb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-[#8B6B4A]/10 text-[#8B6B4A] rounded-2xl border border-[#E7DDD2]">
                <Stethoscope className="w-7 h-7" />
              </div>
              <div>
                <h1 className="text-3xl font-extrabold text-[#3B2F2F] tracking-tight">
                  Clinical Dual Prediction Engine
                </h1>
                <p className="text-xs text-[#7A624A] mt-0.5">
                  Consensus diagnostic classification combining Proposed RASC-Net with Soft Voting Ensemble
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-semibold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
              <Sparkles className="w-3.5 h-3.5 mr-1.5" />
              HAM10000 7-Class Dermoscopy
            </span>
          </div>
        </div>

        {/* Universal Image Upload Card Component */}
        <ImageUploadCard
          title="Step 1: Upload Dermoscopy Lesion Image"
          description="Drag & drop, browse files, or paste (Ctrl + V) from clipboard"
          selectedFile={selectedFile}
          imagePreview={imagePreview}
          onImageSelected={handleImageSelected}
          onImageRemoved={handleImageRemoved}
        />

        {/* Patient Metadata Form Section */}
        <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-6">
          <div className="border-b border-[#F4EFE6] pb-4 flex items-center justify-between">
            <div>
              <h3 className="text-lg font-bold text-[#3B2F2F] flex items-center space-x-2">
                <FileText className="w-5 h-5 text-[#8B6B4A]" />
                <span>Step 2: Patient Clinical Metadata (HAM10000 Features)</span>
              </h3>
              <p className="text-xs text-[#7A624A] mt-0.5">
                Configure patient demographic and lesion attributes to feed the Rule-Based Risk Engine
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 text-xs">
            
            {/* Age Approx */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F]">Patient Age (Approx)</label>
              <input
                type="number"
                name="age_approx"
                value={metadata.age_approx}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] bg-[#F8F5F0] text-[#3B2F2F] font-semibold focus:outline-none focus:border-[#8B6B4A]"
              />
            </div>

            {/* Sex */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F]">Patient Sex</label>
              <select
                name="sex"
                value={metadata.sex}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] bg-[#F8F5F0] text-[#3B2F2F] font-semibold focus:outline-none focus:border-[#8B6B4A]"
              >
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="unknown">Unknown</option>
              </select>
            </div>

            {/* Anatomical Site 1 */}
            <div className="space-y-1.5">
              <SearchableCombobox
                label="Anatomical Region (Site 1)"
                value={metadata.anatom_site_1}
                options={medicalOptions.anatom_site_1}
                onChange={(val) => setMetadata(prev => ({ ...prev, anatom_site_1: val }))}
              />
            </div>

            {/* Anatomical Site 2 */}
            <div className="space-y-1.5">
              <SearchableCombobox
                label="Specific Location (Site 2)"
                value={metadata.anatom_site_2}
                options={medicalOptions.anatom_site_2}
                onChange={(val) => setMetadata(prev => ({ ...prev, anatom_site_2: val }))}
              />
            </div>

            {/* Anatomical Site 3 */}
            <div className="space-y-1.5">
              <SearchableCombobox
                label="Sub-Location / Side (Site 3)"
                value={metadata.anatom_site_3}
                options={medicalOptions.anatom_site_3}
                onChange={(val) => setMetadata(prev => ({ ...prev, anatom_site_3: val }))}
              />
            </div>

            {/* Melanocytic Status */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F]">Melanocytic Lesion</label>
              <select
                name="melanocytic"
                value={metadata.melanocytic}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] bg-[#F8F5F0] text-[#3B2F2F] font-semibold focus:outline-none focus:border-[#8B6B4A]"
              >
                <option value="false">False (Non-melanocytic)</option>
                <option value="true">True (Melanocytic)</option>
              </select>
            </div>

            {/* Concomitant Biopsy */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F]">Concomitant Biopsy Performed</label>
              <select
                name="concomitant_biopsy"
                value={metadata.concomitant_biopsy}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] bg-[#F8F5F0] text-[#3B2F2F] font-semibold focus:outline-none focus:border-[#8B6B4A]"
              >
                <option value="false">False</option>
                <option value="true">True</option>
              </select>
            </div>

            {/* Primary Diagnosis */}
            <div className="space-y-1.5 md:col-span-2">
              <SearchableCombobox
                label="Primary Clinical Impression (Diagnosis 1)"
                value={metadata.diagnosis_1}
                options={medicalOptions.diagnosis_1}
                onChange={(val) => setMetadata(prev => ({ ...prev, diagnosis_1: val }))}
              />
            </div>

          </div>
        </div>

        {/* Action Button Section */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-[#7A624A]">
            Click below to execute parallel Dual Model Consensus inference & Clinical Risk Evaluation.
          </div>

          <button
            onClick={handlePredict}
            disabled={loading || !selectedFile}
            className={`px-8 py-3.5 rounded-2xl font-extrabold text-white text-sm shadow-lg transition-all flex items-center space-x-2 ${
              loading || !selectedFile
                ? 'bg-[#A69585] cursor-not-allowed opacity-60'
                : 'bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#7A5B3D] hover:to-[#5E442B] shadow-[#8B6B4A]/25'
            }`}
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Executing Dual Inference...</span>
              </>
            ) : (
              <>
                <Stethoscope className="w-5 h-5" />
                <span>Execute Dual Model Prediction</span>
              </>
            )}
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="p-4 bg-[#FBF0EF] border border-[#F2D6D3] rounded-2xl text-[#C0564B] text-xs flex items-center space-x-3 animate-fade-in shadow-xs">
            <AlertTriangle className="w-5 h-5 shrink-0 text-[#C0564B]" />
            <div className="font-semibold">{error}</div>
          </div>
        )}

        {/* Results Section */}
        {predictionData && (
          <div className="space-y-8 animate-fade-in">
            
            {/* Dual Consensus Banner */}
            <div className={`p-4 rounded-2xl border flex items-center justify-between text-xs font-bold ${
              modelsAgree 
                ? 'bg-[#E8F0E9] text-[#5F8D6E] border-[#C5DDC8]' 
                : 'bg-[#FDF5E6] text-[#C88A36] border-[#F5E2C4]'
            }`}>
              <div className="flex items-center space-x-2">
                <CheckCircle2 className="w-4 h-4" />
                <span>{predictionData.agreement_message || (modelsAgree ? "Consensus Reached: Both models agree on diagnosis." : "Divergent Predictions: Models suggest different diagnostic possibilities.")}</span>
              </div>
              <span className="uppercase text-[10px] px-2.5 py-0.5 rounded-full font-bold bg-white/70">
                {modelsAgree ? 'AGREEMENT' : 'DIVERGENCE'}
              </span>
            </div>

            {/* Model Comparison Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              
              {/* Proposed RASC-Net Card */}
              <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-3 mb-4">
                    <div>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#8B6B4A]/10 text-[#8B6B4A] uppercase tracking-wider">
                        Primary Architecture
                      </span>
                      <h3 className="text-lg font-bold text-[#3B2F2F] mt-1">Proposed RASC-Net</h3>
                    </div>
                    <span className="text-xs font-extrabold px-3 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                      {rascData.confidence_pct}% ({rascData.confidence_level})
                    </span>
                  </div>

                  <div className="space-y-3">
                    <div className="text-2xl font-extrabold text-[#3B2F2F]">
                      {rascData.lesion_name}
                      <span className="text-sm font-semibold text-[#7A624A] ml-2">({rascData.class_code?.toUpperCase()})</span>
                    </div>

                    {/* Top 3 Predictions */}
                    <div className="space-y-2 pt-2">
                      <div className="text-[11px] font-bold text-[#7A624A] uppercase tracking-wider">Top Probability Candidates</div>
                      {rascData.top_3_predictions?.map((cand, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs font-semibold text-[#3B2F2F]">
                            <span>{cand.lesion_name} ({cand.class_code?.toUpperCase()})</span>
                            <span className="text-[#8B6B4A]">{cand.confidence_pct}%</span>
                          </div>
                          <div className="w-full bg-[#F4EFE6] h-2 rounded-full overflow-hidden">
                            <div 
                              className="bg-[#8B6B4A] h-full rounded-full transition-all duration-700 animate-progress-bar" 
                              style={{ width: `${cand.confidence_pct}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Soft Voting Ensemble Card */}
              <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-5 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-3 mb-4">
                    <div>
                      <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#C8A97E]/20 text-[#6E5338] uppercase tracking-wider">
                        Baseline Ensemble
                      </span>
                      <h3 className="text-lg font-bold text-[#3B2F2F] mt-1">Soft Voting Ensemble</h3>
                    </div>
                    <span className="text-xs font-extrabold px-3 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                      {ensembleData.confidence_pct}% ({ensembleData.confidence_level})
                    </span>
                  </div>

                  <div className="space-y-3">
                    <div className="text-2xl font-extrabold text-[#3B2F2F]">
                      {ensembleData.lesion_name}
                      <span className="text-sm font-semibold text-[#7A624A] ml-2">({ensembleData.class_code?.toUpperCase()})</span>
                    </div>

                    {/* Top 3 Predictions */}
                    <div className="space-y-2 pt-2">
                      <div className="text-[11px] font-bold text-[#7A624A] uppercase tracking-wider">Top Probability Candidates</div>
                      {ensembleData.top_3_predictions?.map((cand, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs font-semibold text-[#3B2F2F]">
                            <span>{cand.lesion_name} ({cand.class_code?.toUpperCase()})</span>
                            <span className="text-[#8B6B4A]">{cand.confidence_pct}%</span>
                          </div>
                          <div className="w-full bg-[#F4EFE6] h-2 rounded-full overflow-hidden">
                            <div 
                              className="bg-[#C8A97E] h-full rounded-full transition-all duration-700 animate-progress-bar" 
                              style={{ width: `${cand.confidence_pct}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

            </div>

            {/* Grad-CAM Card Section */}
            <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-4">
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#F4EFE6] pb-4">
                <div>
                  <h3 className="text-xl font-semibold text-[#3B2F2F] flex items-center space-x-2">
                    <Eye className="w-5 h-5 text-[#8B6B4A]" />
                    <span>Grad-CAM Visual Explainability</span>
                  </h3>
                  <p className="text-xs text-[#7A624A] mt-0.5">
                    Spatial class activation map targeting RASC-Net Proposed architecture
                  </p>
                </div>

                <button
                  onClick={() => fetchGradcam(selectedFile)}
                  disabled={gradcamLoading}
                  className="px-4 py-2.5 bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#7A5B3D] hover:to-[#5E442B] text-white rounded-xl text-xs font-bold shadow-md shadow-[#8B6B4A]/20 flex items-center space-x-2 transition-all disabled:opacity-60"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${gradcamLoading ? 'animate-spin' : ''}`} />
                  <span>{gradcamLoading ? 'Generating Heatmap...' : gradcamData ? 'Regenerate Grad-CAM' : 'Generate Grad-CAM'}</span>
                </button>
              </div>

              {gradcamLoading && (
                <div className="h-56 bg-[#F8F5F0] rounded-2xl border border-[#E7DDD2] flex items-center justify-center text-xs text-[#7A624A] space-x-3">
                  <div className="w-6 h-6 border-2 border-[#8B6B4A] border-t-transparent rounded-full animate-spin"></div>
                  <span>Computing RASC-Net Gradient Activation Map...</span>
                </div>
              )}

              {gradcamError && (
                <div className="p-4 bg-[#FBF0EF] border border-[#F2D6D3] rounded-xl text-[#C0564B] text-xs flex items-center justify-between">
                  <span>{gradcamError}</span>
                  <button onClick={() => fetchGradcam(selectedFile)} className="px-3 py-1 bg-[#C0564B] text-white rounded-lg text-xs font-bold shadow-xs">Retry</button>
                </div>
              )}

              {gradcamData && !gradcamLoading && (
                <div className="bg-[#F8F5F0] p-6 rounded-2xl border border-[#E7DDD2] space-y-4">
                  
                  {/* Tab Switcher */}
                  <div className="flex items-center justify-center space-x-2 border-b border-[#E7DDD2] pb-3">
                    <button
                      onClick={() => setActiveGradcamTab('overlay')}
                      className={`px-4 py-1.5 rounded-xl text-xs font-bold transition-all ${
                        activeGradcamTab === 'overlay'
                          ? 'bg-[#8B6B4A] text-white shadow-xs'
                          : 'bg-[#FFFDF9] text-[#7A624A] hover:text-[#3B2F2F]'
                      }`}
                    >
                      Heatmap Overlay
                    </button>
                    <button
                      onClick={() => setActiveGradcamTab('heatmap')}
                      className={`px-4 py-1.5 rounded-xl text-xs font-bold transition-all ${
                        activeGradcamTab === 'heatmap'
                          ? 'bg-[#8B6B4A] text-white shadow-xs'
                          : 'bg-[#FFFDF9] text-[#7A624A] hover:text-[#3B2F2F]'
                      }`}
                    >
                      Raw Spatial Activation
                    </button>
                  </div>

                  {/* Image Display */}
                  <div className="text-center space-y-3">
                    <img 
                      src={activeGradcamTab === 'overlay' ? gradcamData.overlay_image : gradcamData.gradcam_image} 
                      alt="RASC-Net Grad-CAM Visualization" 
                      className="max-w-[420px] w-full mx-auto rounded-2xl border border-[#E7DDD2] shadow-md object-cover hover:scale-[1.02] transition-transform duration-300" 
                    />
                    <p className="text-xs text-[#7A624A] max-w-lg mx-auto leading-relaxed">
                      Highlighted red and warm spatial activations pinpoint exact anatomical structures driving RASC-Net Proposed's diagnostic decision ({gradcamData.predicted_class?.toUpperCase()} - {gradcamData.confidence}% confidence).
                    </p>
                  </div>

                </div>
              )}
            </div>

            {/* Clinical Assessment Card */}
            {riskData && (
              <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-5">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#F4EFE6] pb-4">
                  <div>
                    <h3 className="text-xl font-semibold text-[#3B2F2F] flex items-center space-x-2">
                      <HeartPulse className="w-5 h-5 text-[#C0564B]" />
                      <span>Rule-Based Clinical Risk Assessment</span>
                    </h3>
                    <p className="text-xs text-[#7A624A] mt-0.5">
                      Integrated Decision Support Engine (RASC-Net Proposed + HAM10000 Metadata)
                    </p>
                  </div>

                  <div className="flex items-center space-x-3">
                    <span className={`px-3.5 py-1.5 rounded-full text-xs font-bold uppercase border ${
                      riskData.risk_level === 'HIGH RISK' 
                        ? 'bg-[#FBF0EF] text-[#C0564B] border-[#F2D6D3]' 
                        : riskData.risk_level === 'MODERATE RISK' 
                        ? 'bg-[#FDF5E6] text-[#C88A36] border-[#F5E2C4]' 
                        : 'bg-[#E8F0E9] text-[#5F8D6E] border-[#C5DDC8]'
                    }`}>
                      {riskData.risk_level} ({riskData.score} PTS)
                    </span>

                    <button
                      onClick={() => setIsReportModalOpen(true)}
                      className="px-4 py-2.5 bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] text-white rounded-xl font-semibold text-xs shadow-md shadow-[#8B6B4A]/20 flex items-center space-x-2 transition-all"
                    >
                      <Printer className="w-4 h-4" />
                      <span>Generate Hospital PDF Report</span>
                    </button>
                  </div>
                </div>

                <div className="p-4 bg-[#F8F5F0] rounded-xl border border-[#E7DDD2] space-y-2 text-sm text-[#3B2F2F]">
                  <div><strong className="text-[#3B2F2F]">Clinical Recommendation:</strong> {riskData.recommendation}</div>
                  {riskData.explanation && riskData.explanation.length > 0 && (
                    <div className="text-xs text-[#7A624A]">
                      <strong className="text-[#3B2F2F]">Risk Factor Breakdown:</strong> {riskData.explanation.join(' | ')}
                    </div>
                  )}
                </div>
              </div>
            )}

          </div>
        )}

      </div>

      {/* Hospital Assessment PDF Report Modal */}
      {isReportModalOpen && rascData && (
        <div className="fixed inset-0 bg-[#2A2118]/70 backdrop-blur-sm z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white text-slate-900 rounded-2xl max-w-3xl w-full p-8 shadow-2xl space-y-6 relative max-h-[90vh] overflow-y-auto">
            
            <div className="flex justify-between items-center border-b pb-4 no-print">
              <h3 className="text-lg font-bold text-[#3B2F2F]">Hospital Clinical Assessment Report</h3>
              <div className="flex items-center space-x-2">
                <button onClick={handlePrintReport} className="px-3.5 py-1.5 bg-[#F4EFE6] hover:bg-[#E7DDD2] text-[#3B2F2F] text-xs font-semibold rounded-xl border border-[#E7DDD2] flex items-center space-x-1">
                  <Printer className="w-3.5 h-3.5" /> <span>Print</span>
                </button>
                <button onClick={() => setIsReportModalOpen(false)} className="p-1.5 hover:bg-slate-100 rounded-xl">
                  <X className="w-5 h-5 text-slate-500" />
                </button>
              </div>
            </div>

            {/* Header */}
            <div className="text-center border-b-2 border-slate-900 pb-4">
              <h2 className="text-xl font-black text-slate-900 tracking-tight">GENERAL HOSPITAL DERMATOLOGY CENTER</h2>
              <h4 className="text-sm font-bold text-[#8B6B4A]">AI SKIN CANCER CLINICAL ASSESSMENT REPORT</h4>
              <p className="text-xs text-slate-500 mt-1">Generated via RASC-Net Proposed Architecture</p>
            </div>

            {/* Patient Metadata */}
            <div className="space-y-2 text-xs">
              <h4 className="font-bold border-b pb-1">1. Patient Clinical Information</h4>
              <div className="grid grid-cols-2 gap-2 text-slate-700">
                <div><strong>Age Approx:</strong> {metadata.age_approx} yrs</div>
                <div><strong>Sex:</strong> {metadata.sex}</div>
                <div><strong>Anatomical Site:</strong> {metadata.anatom_site_1} ({metadata.anatom_site_2})</div>
                <div><strong>Melanocytic:</strong> {metadata.melanocytic}</div>
                <div><strong>Concomitant Biopsy:</strong> {metadata.concomitant_biopsy}</div>
                <div><strong>Diagnosis History:</strong> {metadata.diagnosis_1}</div>
              </div>
            </div>

            {/* Prediction */}
            <div className="space-y-2 text-xs bg-slate-50 p-4 rounded-xl border">
              <h4 className="font-bold border-b pb-1 text-slate-900">2. RASC-Net Proposed Visual Classification</h4>
              <div className="flex justify-between items-center text-sm font-bold">
                <span>{rascData.lesion_name} ({rascData.class_code?.toUpperCase() || ''})</span>
                <span className="text-[#8B6B4A]">{rascData.confidence_pct}% ({rascData.confidence_level})</span>
              </div>
            </div>

            {/* Grad-CAM Section in Report */}
            {gradcamData && gradcamData.overlay_image && (
              <div className="space-y-2 text-xs border-t pt-3">
                <h4 className="font-bold border-b pb-1">3. Grad-CAM Spatial Visual Explainability Heatmap</h4>
                <div className="flex items-center space-x-4 bg-slate-50 p-3 rounded-xl border">
                  <img src={gradcamData.overlay_image} alt="Grad-CAM Overlay" className="w-28 h-28 object-cover rounded-lg border" />
                  <p className="text-slate-600 leading-relaxed">
                    Red and warm spatial activations highlight exact anatomical structures driving RASC-Net Proposed diagnosis.
                  </p>
                </div>
              </div>
            )}

            {/* Risk Assessment */}
            {riskData && (
              <div className="space-y-2 text-xs bg-emerald-50/50 p-4 rounded-xl border border-emerald-200">
                <h4 className="font-bold border-b border-emerald-200 pb-1 text-emerald-900">{gradcamData ? '4.' : '3.'} Rule-Based Clinical Risk Assessment</h4>
                <div><strong>Risk Level:</strong> <span className="font-extrabold text-emerald-800">{riskData.risk_level} ({riskData.score} PTS)</span></div>
                <div><strong>Recommendation:</strong> {riskData.recommendation}</div>
              </div>
            )}

          </div>
        </div>
      )}

    </div>
  );
};

export default Predict;
