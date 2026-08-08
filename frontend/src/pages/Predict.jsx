import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import {
  AlertTriangle, X, HeartPulse,
  Stethoscope, FileText, Printer, Sparkles, Eye, RefreshCw, ShieldCheck, Loader2, Clock, Cpu
} from 'lucide-react';
import ImageUploadCard from '../components/ImageUploadCard';
import MedicalCellularBackground from '../components/MedicalCellularBackground';
import { getApiBaseUrl } from '../config';

const API_BASE = getApiBaseUrl();

const Predict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);

  // Ref for smooth scrolling to prediction results
  const resultsRef = useRef(null);

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

  // State for Prediction
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

  // Smooth scroll to results section after prediction renders
  useEffect(() => {
    if (predictionData && resultsRef.current) {
      const timer = setTimeout(() => {
        resultsRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
      return () => clearTimeout(timer);
    }
  }, [predictionData]);

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

      const resImg = res.data?.overlay_image || res.data?.gradcam_image || res.data?.gradcam || res.data?.gradcam_image_base64;
      if (res.data && (res.data.success || res.data.status === 'success' || resImg)) {
        const overlay = res.data.overlay_image || res.data.gradcam || res.data.gradcam_image_base64 || res.data.gradcam_image;
        const heatmap = res.data.gradcam_image || res.data.gradcam || res.data.gradcam_image_base64 || res.data.overlay_image;
        const predClass = res.data.predicted_class || 'nv';
        const conf = res.data.confidence || res.data.confidence_pct || 0;

        setGradcamData({
          overlay_image: overlay,
          gradcam_image: heatmap,
          predicted_class: predClass,
          confidence: conf
        });
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
    } catch (err) {
      const serverErr = err.response?.data?.error || err.response?.data?.message;
      setError(serverErr ? `Prediction failed: ${serverErr}` : `Prediction failed: ${err.message || 'Error executing model prediction'}`);
    } finally {
      setLoading(false);
    }
  };

  const ensembleData = predictionData?.soft_voting_ensemble;
  const riskData = predictionData?.clinical_assessment;

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
                  Clinical Prediction Engine
                </h1>
                <p className="text-xs text-[#7A624A] mt-0.5">
                  Automated diagnostic classification utilizing Soft Voting Ensemble Architecture
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
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Patient Age (Approx)</label>
              <input
                type="number"
                name="age_approx"
                value={metadata.age_approx}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              />
            </div>

            {/* Sex */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Patient Sex</label>
              <select
                name="sex"
                value={metadata.sex}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              >
                <option value="male" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">Male</option>
                <option value="female" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">Female</option>
                <option value="unknown" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">Unknown</option>
              </select>
            </div>

            {/* Anatomical Site 1 */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Anatomical Region (Site 1)</label>
              <input
                type="text"
                name="anatom_site_1"
                value={metadata.anatom_site_1}
                onChange={handleMetadataChange}
                placeholder="e.g. torso, head/neck"
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              />
            </div>

            {/* Anatomical Site 2 */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Specific Location (Site 2)</label>
              <input
                type="text"
                name="anatom_site_2"
                value={metadata.anatom_site_2}
                onChange={handleMetadataChange}
                placeholder="e.g. cheek, chest"
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              />
            </div>

            {/* Anatomical Site 3 */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Sub-Location / Side (Site 3)</label>
              <input
                type="text"
                name="anatom_site_3"
                value={metadata.anatom_site_3}
                onChange={handleMetadataChange}
                placeholder="e.g. right side, anterior"
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              />
            </div>

            {/* Melanocytic Status */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Melanocytic Lesion</label>
              <select
                name="melanocytic"
                value={metadata.melanocytic}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              >
                <option value="false" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">False (Non-melanocytic)</option>
                <option value="true" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">True (Melanocytic)</option>
              </select>
            </div>

            {/* Concomitant Biopsy */}
            <div className="space-y-1.5">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Concomitant Biopsy Performed</label>
              <select
                name="concomitant_biopsy"
                value={metadata.concomitant_biopsy}
                onChange={handleMetadataChange}
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              >
                <option value="false" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">False</option>
                <option value="true" className="bg-[#FFFDF9] dark:bg-[#1C1814] text-[#3B2F2F] dark:text-[#F5EFEB]">True</option>
              </select>
            </div>

            {/* Primary Diagnosis */}
            <div className="space-y-1.5 md:col-span-2">
              <label className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB]">Primary Clinical Impression (Diagnosis 1)</label>
              <input
                type="text"
                name="diagnosis_1"
                value={metadata.diagnosis_1}
                onChange={handleMetadataChange}
                placeholder="e.g. none, melanoma"
                className="w-full px-3.5 py-2.5 rounded-xl border border-[#E7DDD2] dark:border-[#3D332B] bg-[#F8F5F0] dark:bg-[#26201B] text-[#3B2F2F] dark:text-[#F5EFEB] font-semibold focus:outline-none focus:border-[#8B6B4A] dark:focus:border-[#D4AF37] transition-colors"
              />
            </div>

          </div>
        </div>

        {/* Action Button Section */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-xs text-[#7A624A]">
            Click below to execute Soft Voting Ensemble classification & Clinical Risk Evaluation.
          </div>

          <button
            onClick={handlePredict}
            disabled={loading || !selectedFile}
            className={`px-8 py-3.5 rounded-2xl font-extrabold text-white text-sm shadow-lg transition-all flex items-center space-x-2 ${loading || !selectedFile
                ? 'bg-[#A69585] cursor-not-allowed opacity-60'
                : 'bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#7A5B3D] hover:to-[#5E442B] shadow-[#8B6B4A]/25'
              }`}
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Executing Ensemble Inference...</span>
              </>
            ) : (
              <>
                <Stethoscope className="w-5 h-5" />
                <span>Execute Ensemble Prediction</span>
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
        {predictionData && ensembleData && (
          <div ref={resultsRef} className="space-y-8 animate-fade-in scroll-mt-24">

            {/* 2-Column Side-by-Side Model Comparison Dashboard (50/50 Layout) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">

              {/* LEFT COLUMN (50%): Soft Voting Ensemble Prediction Card */}
              <div className="bg-[#FFFDF9] rounded-2xl p-6 md:p-8 shadow-md border border-[#E7DDD2] flex flex-col justify-between space-y-6">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b border-[#F4EFE6] pb-4 gap-3">
                  <div>
                    <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-[#8B6B4A]/10 text-[#8B6B4A] uppercase tracking-wider">
                      Ensemble Classification Architecture
                    </span>
                    <h3 className="text-xl font-bold text-[#3B2F2F] mt-1">Soft Voting Ensemble Prediction</h3>
                  </div>
                  <span className="text-sm font-extrabold px-4 py-1.5 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                    {ensembleData.confidence_pct}% ({ensembleData.confidence_level})
                  </span>
                </div>

                <div className="space-y-5 flex-1 flex flex-col justify-between">
                  <div className="text-3xl font-extrabold text-[#3B2F2F]">
                    {ensembleData.lesion_name}
                    <span className="text-base font-semibold text-[#7A624A] ml-3">({ensembleData.class_code?.toUpperCase()})</span>
                  </div>

                  {/* Top Probability Candidates */}
                  <div className="space-y-3 pt-2">
                    <div className="text-xs font-bold text-[#7A624A] uppercase tracking-wider">Top Probability Candidates</div>
                    {ensembleData.top_3_predictions?.map((cand, idx) => (
                      <div key={idx} className="space-y-1.5">
                        <div className="flex justify-between text-xs font-semibold text-[#3B2F2F]">
                          <span>{cand.lesion_name} ({cand.class_code?.toUpperCase()})</span>
                          <span className="text-[#8B6B4A] font-bold">{cand.confidence_pct}%</span>
                        </div>
                        <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
                          <div
                            className="bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] h-full rounded-full transition-all duration-700 animate-progress-bar"
                            style={{ width: `${cand.confidence_pct}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* RIGHT COLUMN (50%): RASC-Net Prediction Card */}
              {(() => {
                const rascData = predictionData?.rasc_net_proposed;
                return (
                  <div className="bg-[#FFFDF9] rounded-2xl p-6 md:p-8 shadow-md border border-[#E7DDD2] flex flex-col justify-between space-y-6 relative overflow-hidden">
                    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b border-[#F4EFE6] pb-4 gap-3">
                      <div>
                        <span className="text-[10px] font-bold px-2.5 py-1 rounded-full bg-[#8B6B4A]/10 text-[#8B6B4A] uppercase tracking-wider border border-[#E7DDD2]">
                          PROPOSED RASC-NET
                        </span>
                        <h3 className="text-xl font-bold text-[#3B2F2F] mt-1">RASC-Net Prediction</h3>
                      </div>
                      <span className="text-xs font-extrabold px-3.5 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2] flex items-center space-x-1.5">
                        {rascData ? (
                          <span>{rascData.confidence_pct}% ({rascData.confidence_level})</span>
                        ) : (
                          <>
                            <Clock className="w-3.5 h-3.5" />
                            <span>Under Optimization</span>
                          </>
                        )}
                      </span>
                    </div>

                    {rascData ? (
                      <div className="space-y-5 flex-1 flex flex-col justify-between">
                        <div className="text-3xl font-extrabold text-[#3B2F2F]">
                          {rascData.lesion_name}
                          <span className="text-base font-semibold text-[#7A624A] ml-3">({rascData.class_code?.toUpperCase()})</span>
                        </div>

                        {/* Top Probability Candidates */}
                        <div className="space-y-3 pt-2">
                          <div className="text-xs font-bold text-[#7A624A] uppercase tracking-wider">Top Probability Candidates</div>
                          {rascData.top_3_predictions?.map((cand, idx) => (
                            <div key={idx} className="space-y-1.5">
                              <div className="flex justify-between text-xs font-semibold text-[#3B2F2F]">
                                <span>{cand.lesion_name} ({cand.class_code?.toUpperCase()})</span>
                                <span className="text-[#8B6B4A] font-bold">{cand.confidence_pct}%</span>
                              </div>
                              <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
                                <div
                                  className="bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] h-full rounded-full transition-all duration-700 animate-progress-bar"
                                  style={{ width: `${cand.confidence_pct}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      /* Polished Coming Soon Center State */
                      <div className="flex-1 flex flex-col items-center justify-center text-center p-6 space-y-4 my-auto">
                        <div className="w-16 h-16 rounded-2xl bg-[#F4EFE6] border border-[#E7DDD2] flex items-center justify-center text-[#8B6B4A] shadow-xs relative">
                          <Cpu className="w-8 h-8 text-[#8B6B4A]" />
                          <span className="absolute -top-1 -right-1 w-3 h-3 bg-[#8B6B4A] rounded-full border-2 border-[#FFFDF9] animate-ping" />
                        </div>

                        <div className="space-y-1.5 max-w-sm">
                          <h4 className="font-extrabold text-base text-[#3B2F2F]">
                            RASC-Net Model Evaluation in Progress
                          </h4>
                          <p className="text-xs text-[#7A624A] leading-relaxed">
                            RASC-Net model evaluation is being finalized. Standalone prediction results will be available here soon.
                          </p>
                        </div>

                        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-[#F8F5F0] border border-[#E7DDD2] text-[11px] font-semibold text-[#8B6B4A]">
                          <Sparkles className="w-3.5 h-3.5 text-[#8B6B4A]" />
                          <span>Curriculum Adversarial Optimization Active</span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}

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
                  <span>Computing Gradient Activation Map...</span>
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
                      className={`px-4 py-1.5 rounded-xl text-xs font-bold transition-all ${activeGradcamTab === 'overlay'
                          ? 'bg-[#8B6B4A] text-white shadow-xs'
                          : 'bg-[#FFFDF9] text-[#7A624A] hover:text-[#3B2F2F]'
                        }`}
                    >
                      Heatmap Overlay
                    </button>
                    <button
                      onClick={() => setActiveGradcamTab('heatmap')}
                      className={`px-4 py-1.5 rounded-xl text-xs font-bold transition-all ${activeGradcamTab === 'heatmap'
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
                      alt="Grad-CAM Visualization"
                      className="max-w-[420px] w-full mx-auto rounded-2xl border border-[#E7DDD2] shadow-md object-cover hover:scale-[1.02] transition-transform duration-300"
                    />
                    <p className="text-xs text-[#7A624A] max-w-lg mx-auto leading-relaxed">
                      Highlighted red and warm spatial activations pinpoint exact anatomical structures driving diagnostic decision ({gradcamData.predicted_class?.toUpperCase()} - {gradcamData.confidence}% confidence).
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
                      Integrated Decision Support Engine (Soft Voting Ensemble + HAM10000 Metadata)
                    </p>
                  </div>

                  <div className="flex items-center space-x-3">
                    <span className={`px-3.5 py-1.5 rounded-full text-xs font-bold uppercase border ${riskData.risk_level === 'HIGH RISK'
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
      {isReportModalOpen && ensembleData && (
        <div className="fixed inset-0 bg-[#2A2118]/80 backdrop-blur-md z-50 flex items-center justify-center p-4 overflow-y-auto">
          <div className="bg-white text-slate-900 rounded-2xl max-w-3xl w-full p-8 shadow-2xl space-y-6 relative max-h-[90vh] overflow-y-auto printable-modal">

            {/* Modal Control Bar */}
            <div className="flex justify-between items-center border-b border-slate-200 pb-4 no-print">
              <h3 className="text-lg font-bold flex items-center space-x-2" style={{ color: '#0F172A' }}>
                <FileText className="w-5 h-5" style={{ color: '#8B6B4A' }} />
                <span style={{ color: '#0F172A' }}>Hospital Clinical Assessment Report</span>
              </h3>
              <div className="flex items-center space-x-2">
                <button 
                  onClick={handlePrintReport} 
                  className="px-4 py-2 bg-[#8B6B4A] hover:bg-[#6E5338] text-white text-xs font-bold rounded-xl shadow-xs flex items-center space-x-1.5 transition-all cursor-pointer"
                >
                  <Printer className="w-4 h-4" /> <span>Print Report</span>
                </button>
                <button 
                  onClick={() => setIsReportModalOpen(false)} 
                  className="p-2 hover:bg-slate-100 rounded-xl text-slate-500 hover:text-slate-800 transition-colors cursor-pointer"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Document Header */}
            <div className="text-center border-b-2 border-slate-900 pb-4">
              <h2 className="text-xl font-black tracking-tight" style={{ color: '#0F172A' }}>GENERAL HOSPITAL DERMATOLOGY CENTER</h2>
              <h4 className="text-sm font-bold uppercase tracking-wide" style={{ color: '#8B6B4A' }}>AI SKIN CANCER CLINICAL ASSESSMENT REPORT</h4>
              <p className="text-xs mt-1" style={{ color: '#64748B' }}>Generated via Soft Voting Ensemble Architecture</p>
            </div>

            {/* Section 1: Patient Clinical Information */}
            <div className="space-y-2 text-xs">
              <h4 className="font-bold border-b border-slate-300 pb-1 text-slate-900 uppercase tracking-wider">1. Patient Clinical Information</h4>
              <div className="grid grid-cols-2 gap-3 text-slate-800 p-3.5 bg-slate-50 rounded-xl border border-slate-200">
                <div><span className="text-slate-500 font-semibold">Age Approx:</span> <strong className="text-slate-900">{metadata.age_approx || '60'} yrs</strong></div>
                <div><span className="text-slate-500 font-semibold">Sex:</span> <strong className="text-slate-900">{metadata.sex || 'male'}</strong></div>
                <div><span className="text-slate-500 font-semibold">Anatomical Site:</span> <strong className="text-slate-900">{metadata.anatom_site_1 || 'torso'} ({metadata.anatom_site_2 || 'cheek'})</strong></div>
                <div><span className="text-slate-500 font-semibold">Melanocytic:</span> <strong className="text-slate-900">{metadata.melanocytic || 'false'}</strong></div>
                <div><span className="text-slate-500 font-semibold">Concomitant Biopsy:</span> <strong className="text-slate-900">{metadata.concomitant_biopsy || 'false'}</strong></div>
                <div><span className="text-slate-500 font-semibold">Diagnosis History:</span> <strong className="text-slate-900">{metadata.diagnosis_1 || 'none'}</strong></div>
              </div>
            </div>

            {/* Section 2: Visual Classification */}
            <div className="space-y-2 text-xs bg-slate-50 p-4 rounded-xl border border-slate-200">
              <h4 className="font-bold border-b border-slate-200 pb-1 text-slate-900 uppercase tracking-wider">2. Soft Voting Ensemble Visual Classification</h4>
              <div className="flex justify-between items-center text-sm font-extrabold text-slate-900 pt-1">
                <span className="text-slate-900">{ensembleData.lesion_name} ({ensembleData.class_code?.toUpperCase() || ''})</span>
                <span className="text-[#8B6B4A] font-extrabold">{ensembleData.confidence_pct}% ({ensembleData.confidence_level})</span>
              </div>
            </div>

            {/* Section 3: Grad-CAM Heatmap */}
            {gradcamData && gradcamData.overlay_image && (
              <div className="space-y-2 text-xs border-t border-slate-200 pt-3">
                <h4 className="font-bold border-b border-slate-200 pb-1 text-slate-900 uppercase tracking-wider">3. Grad-CAM Spatial Visual Explainability Heatmap</h4>
                <div className="flex items-center space-x-4 bg-slate-50 p-3 rounded-xl border border-slate-200">
                  <img src={gradcamData.overlay_image} alt="Grad-CAM Overlay" className="w-28 h-28 object-cover rounded-lg border border-slate-300 shadow-xs" />
                  <p className="text-slate-700 leading-relaxed text-xs">
                    Red and warm spatial activations highlight exact anatomical structures driving diagnostic decision.
                  </p>
                </div>
              </div>
            )}

            {/* Section 4: Risk Assessment */}
            {riskData && (
              <div className="space-y-2 text-xs bg-emerald-50/70 p-4 rounded-xl border border-emerald-200 text-emerald-950">
                <h4 className="font-bold border-b border-emerald-200 pb-1 text-emerald-900 uppercase tracking-wider">
                  {gradcamData ? '4.' : '3.'} Rule-Based Clinical Risk Assessment
                </h4>
                <div className="pt-1">
                  <strong className="text-emerald-950">Risk Level:</strong> <span className="font-extrabold text-emerald-800">{riskData.risk_level} ({riskData.score} PTS)</span>
                </div>
                <div>
                  <strong className="text-emerald-950">Recommendation:</strong> <span className="text-emerald-900">{riskData.recommendation}</span>
                </div>
              </div>
            )}

          </div>
        </div>
      )}

      {/* Full-Screen Glassmorphism Processing Overlay */}
      {loading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#3B2F2F]/40 backdrop-blur-md transition-all duration-300 animate-fade-in pointer-events-auto">
          <div className="bg-[#FFFDF9]/95 backdrop-blur-xl border border-[#E7DDD2] shadow-2xl rounded-3xl p-8 max-w-md w-full text-center space-y-6 transform scale-100 transition-all">

            {/* Clinical Animated Scanner Ring */}
            <div className="relative w-20 h-20 mx-auto flex items-center justify-center">
              <div className="absolute inset-0 rounded-full border-4 border-[#8B6B4A]/20 border-t-[#8B6B4A] animate-spin" />
              <div className="absolute -inset-2 rounded-full border border-dashed border-[#8B6B4A]/30 animate-spin" style={{ animationDuration: '12s' }} />
              <ShieldCheck className="w-9 h-9 text-[#8B6B4A]" />
            </div>

            {/* Status & Medical Subtitles */}
            <div className="space-y-2">
              <h3 className="text-xl font-extrabold text-[#3B2F2F] tracking-tight">
                Analyzing Image...
              </h3>
              <p className="text-xs font-bold text-[#8B6B4A] uppercase tracking-wider">
                Running Ensemble Prediction
              </p>
              <p className="text-xs text-[#7A624A] pt-1">
                Evaluating multi-model probabilities & clinical risk indicators...
              </p>
            </div>

            {/* Progress Accent Bar */}
            <div className="w-full bg-[#F4EFE6] h-1.5 rounded-full overflow-hidden">
              <div className="bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] h-full rounded-full animate-progress-bar w-full" />
            </div>

          </div>
        </div>
      )}

    </div>
  );
};

export default Predict;
