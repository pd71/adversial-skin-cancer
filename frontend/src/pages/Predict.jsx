import React, { useState } from 'react';
import axios from 'axios';
import { 
  CheckCircle2, AlertTriangle, X, HeartPulse, 
  Stethoscope, FileText, Printer, Zap, Eye, RefreshCw, Sparkles
} from 'lucide-react';
import ImageUploadCard from '../components/ImageUploadCard';
import SearchableCombobox from '../components/SearchableCombobox';
import medicalOptions from '../data/medicalOptions';

// React Error Boundary to prevent page crashes
class GradcamErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("GradCAM Component Render Crash:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-4 bg-[#FBF0EF] border border-[#F2D6D3] rounded-xl text-[#C0564B] text-xs space-y-1">
          <div className="font-bold">Grad-CAM Visualization Component Error:</div>
          <div>{this.state.error?.message || 'React rendering error occurred'}</div>
        </div>
      );
    }
    return this.props.children;
  }
}

const Predict = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  
  // Patient Metadata Form State (12 HAM10000 Fields)
  const [metadata, setMetadata] = useState({
    age_approx: 45,
    sex: 'male',
    anatom_site_1: 'head/neck',
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

  // State for Dual Predictions & Grad-CAM
  const [loading, setLoading] = useState(false);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gradcamError, setGradcamError] = useState(null);
  
  const [predictionData, setPredictionData] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);

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
    setGradcamError(null);
    setPredictionData(null);
    setGradcamImage(null);
  };

  const handleImageRemoved = () => {
    setSelectedFile(null);
    setImagePreview(null);
    setPredictionData(null);
    setGradcamImage(null);
    setError(null);
  };

  const fetchGradcam = async (fileToUse) => {
    const file = fileToUse || selectedFile;
    if (!file) return;

    setGradcamLoading(true);
    setGradcamError(null);

    const gradcamFormData = new FormData();
    gradcamFormData.append('image', file);

    try {
      const gradcamRes = await axios.post('http://localhost:5000/api/gradcam', gradcamFormData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const base64Img = gradcamRes.data?.gradcam || gradcamRes.data?.gradcam_image_base64;
      if (typeof base64Img === 'string' && base64Img.startsWith('data:image/png;base64,')) {
        setGradcamImage(base64Img);
      } else {
        setGradcamError('Grad-CAM failed: Invalid or missing Base64 image payload from backend.');
      }
    } catch (gErr) {
      console.error('Grad-CAM generation error:', gErr);
      const serverErr = gErr.response?.data?.error || gErr.response?.data?.message;
      const networkErr = gErr.message || 'Unknown network error';
      setGradcamError(`Grad-CAM failed: ${serverErr || networkErr}`);
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
    setGradcamError(null);
    setPredictionData(null);
    setGradcamImage(null);

    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('metadata', JSON.stringify(metadata));

    try {
      const res = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setPredictionData(res.data);
      setLoading(false);

      fetchGradcam(selectedFile);

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

  const handlePrintReport = () => window.print();

  return (
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] pb-16">
      
      {/* Centered Container (max-w-1200px) */}
      <div className="max-w-[1200px] mx-auto px-6 py-8 space-y-8">
        
        {/* Page Title Header */}
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-[#E7DDD2] pb-6">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-[#8B6B4A]/10 text-[#8B6B4A] rounded-2xl border border-[#E7DDD2]">
                <Stethoscope className="w-7 h-7" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-[#3B2F2F] tracking-tight">
                  Clinical Decision Support & Dual Inference System
                </h1>
                <p className="text-sm text-[#7A624A] mt-1">
                  Automated parallel classification via RASC-Net Proposed & Soft Voting Ensemble
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3.5 py-1.5 rounded-full text-xs font-semibold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
              <Sparkles className="w-4 h-4 mr-1.5 text-[#8B6B4A]" />
              Searchable Medical Comboboxes Active
            </span>
          </div>
        </div>

        {/* Universal Upload Component Integration */}
        <div className="space-y-4">
          <ImageUploadCard
            title="Upload Dermoscopy Lesion Scan"
            description="Drag & drop image, browse local files, or paste (Ctrl + V) from clipboard"
            selectedFile={selectedFile}
            imagePreview={imagePreview}
            onImageSelected={handleImageSelected}
            onImageRemoved={handleImageRemoved}
          />

          {/* Execute Prediction Action Bar */}
          {selectedFile && (
            <div className="bg-[#FFFDF9] rounded-2xl p-4 shadow-sm border border-[#E7DDD2] flex flex-col sm:flex-row items-center justify-between gap-4 animate-fade-in">
              <div className="text-xs text-[#7A624A]">
                Image ready for dual model prediction. Complete patient clinical information below.
              </div>
              <button
                onClick={handlePredict}
                disabled={loading}
                className={`py-3 px-8 rounded-xl font-bold text-white shadow-md transition-all flex items-center justify-center space-x-2 text-sm whitespace-nowrap ${
                  loading
                    ? 'bg-[#C8A97E] cursor-not-allowed opacity-70'
                    : 'bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#6E5338] hover:to-[#3B2F2F] shadow-[#8B6B4A]/25'
                }`}
              >
                {loading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    <span>Running Inference...</span>
                  </>
                ) : (
                  <>
                    <Zap className="w-4.5 h-4.5 fill-current" />
                    <span>Execute Dual Prediction</span>
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="p-3 bg-[#FBF0EF] border border-[#F2D6D3] rounded-xl text-[#C0564B] text-xs flex items-start space-x-2">
              <AlertTriangle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Intelligent Searchable Patient Clinical Information Form */}
        <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-6">
          <div className="border-b border-[#F4EFE6] pb-4 flex items-center justify-between">
            <div>
              <h3 className="text-xl font-semibold text-[#3B2F2F] flex items-center space-x-2">
                <FileText className="w-5 h-5 text-[#8B6B4A]" />
                <span>Patient Clinical Information</span>
              </h3>
              <p className="text-xs text-[#7A624A] mt-0.5">
                Type to search or enter custom medical values (HAM10000 / ISIC terminology)
              </p>
            </div>
            <span className="text-xs font-bold px-3 py-1 rounded-full bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
              Search & Free Typing
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {/* Age Numeric Field */}
            <div>
              <label className="block text-sm font-medium text-[#3B2F2F] mb-1.5">
                Age Approx (Years)
              </label>
              <input 
                type="number" 
                name="age_approx"
                value={metadata.age_approx} 
                onChange={handleMetadataChange}
                className="w-full bg-[#F8F5F0] border border-[#E7DDD2] rounded-xl px-3.5 py-2.5 text-sm text-[#3B2F2F] focus:outline-none focus:ring-2 focus:ring-[#8B6B4A] focus:bg-white transition-all shadow-xs" 
              />
            </div>

            {/* Biological Sex Combobox */}
            <SearchableCombobox
              label="Biological Sex"
              name="sex"
              value={metadata.sex}
              onChange={handleMetadataChange}
              options={medicalOptions.SEX_OPTIONS}
              placeholder="Search sex or type custom value..."
            />

            {/* Anatomical Site 1 Combobox */}
            <SearchableCombobox
              label="Anatomical Site 1"
              name="anatom_site_1"
              value={metadata.anatom_site_1}
              onChange={handleMetadataChange}
              options={medicalOptions.ANATOM_SITE_1_OPTIONS}
              placeholder="e.g. Head / Neck, Face, Forehead..."
            />

            {/* Anatomical Site 2 (Subsite) Combobox */}
            <SearchableCombobox
              label="Anatomical Site 2 (Subsite)"
              name="anatom_site_2"
              value={metadata.anatom_site_2}
              onChange={handleMetadataChange}
              options={medicalOptions.ANATOM_SITE_2_OPTIONS}
              placeholder="e.g. Left Cheek, Right Forearm, Lateral..."
            />

            {/* Anatomical Site 3 (Region) Combobox */}
            <SearchableCombobox
              label="Anatomical Site 3 (Region)"
              name="anatom_site_3"
              value={metadata.anatom_site_3}
              onChange={handleMetadataChange}
              options={medicalOptions.ANATOM_SITE_3_OPTIONS}
              placeholder="e.g. Supraclavicular, Temporal, Dorsum..."
            />

            {/* Melanocytic Status Combobox */}
            <SearchableCombobox
              label="Melanocytic Status"
              name="melanocytic"
              value={metadata.melanocytic}
              onChange={handleMetadataChange}
              options={medicalOptions.MELANOCYTIC_OPTIONS}
              placeholder="Select or type status..."
            />

            {/* Concomitant Biopsy Combobox */}
            <SearchableCombobox
              label="Concomitant Biopsy Indicated"
              name="concomitant_biopsy"
              value={metadata.concomitant_biopsy}
              onChange={handleMetadataChange}
              options={medicalOptions.CONCOMITANT_BIOPSY_OPTIONS}
              placeholder="Select True, False, or type status..."
            />

            {/* Diagnosis 1 (Primary) Combobox */}
            <SearchableCombobox
              label="Diagnosis 1 (Primary)"
              name="diagnosis_1"
              value={metadata.diagnosis_1}
              onChange={handleMetadataChange}
              options={medicalOptions.DIAGNOSIS_OPTIONS}
              placeholder="e.g. Melanocytic Nevus, Melanoma, BCC..."
            />

            {/* Diagnosis 2 Combobox */}
            <SearchableCombobox
              label="Diagnosis 2"
              name="diagnosis_2"
              value={metadata.diagnosis_2}
              onChange={handleMetadataChange}
              options={medicalOptions.DIAGNOSIS_OPTIONS}
              placeholder="Search or enter secondary diagnosis..."
            />

            {/* Diagnosis 3 Combobox */}
            <SearchableCombobox
              label="Diagnosis 3"
              name="diagnosis_3"
              value={metadata.diagnosis_3}
              onChange={handleMetadataChange}
              options={medicalOptions.DIAGNOSIS_OPTIONS}
              placeholder="Search or enter tertiary diagnosis..."
            />

            {/* Diagnosis 4 Combobox */}
            <SearchableCombobox
              label="Diagnosis 4"
              name="diagnosis_4"
              value={metadata.diagnosis_4}
              onChange={handleMetadataChange}
              options={medicalOptions.DIAGNOSIS_OPTIONS}
              placeholder="Search or enter diagnosis 4..."
            />

            {/* Diagnosis 5 Combobox */}
            <SearchableCombobox
              label="Diagnosis 5"
              name="diagnosis_5"
              value={metadata.diagnosis_5}
              onChange={handleMetadataChange}
              options={medicalOptions.DIAGNOSIS_OPTIONS}
              placeholder="Search or enter diagnosis 5..."
            />
          </div>
        </div>

        {/* Prediction Results & Visualizations */}
        {predictionData && rascData && ensembleData && (
          <div className="space-y-8 animate-fade-in pt-2">
            
            {/* Consensus Banner */}
            <div className={`p-5 rounded-2xl border shadow-sm flex items-center justify-between ${
              modelsAgree 
                ? 'bg-[#E8F0E9] border-[#C5DDC8] text-[#2D5A38]' 
                : 'bg-[#FDF5E6] border-[#F5E2C4] text-[#8C5D1E]'
            }`}>
              <div className="flex items-center space-x-3.5">
                <div className={`p-2.5 rounded-xl ${modelsAgree ? 'bg-[#5F8D6E] text-white' : 'bg-[#C88A36] text-white'}`}>
                  {modelsAgree ? <CheckCircle2 className="w-6 h-6" /> : <AlertTriangle className="w-6 h-6" />}
                </div>
                <div>
                  <div className="text-lg font-bold">
                    {modelsAgree ? '✓ Consensus Reached' : '⚠ Review Recommended'}
                  </div>
                  <div className="text-sm opacity-90 mt-0.5">
                    {modelsAgree 
                      ? `Both RASC-Net Proposed and Soft Voting Ensemble predicted: ${rascData.lesion_name}`
                      : `RASC-Net Proposed predicted ${rascData.lesion_name} while Soft Voting Ensemble predicted ${ensembleData.lesion_name}.`}
                  </div>
                </div>
              </div>
              <span className={`text-xs px-3.5 py-1.5 rounded-full font-bold uppercase border ${
                modelsAgree ? 'bg-[#D6E6D8] border-[#A8CDB0] text-[#2D5A38]' : 'bg-[#FBE8CD] border-[#EDD1A6] text-[#8C5D1E]'
              }`}>
                {modelsAgree ? 'High Consensus' : 'Divergent Models'}
              </span>
            </div>

            {/* Dual Prediction Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              
              {/* Card 1: ⭐ RASC-Net Proposed */}
              <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border-2 border-[#8B6B4A] space-y-6 relative">
                <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-4">
                  <div className="space-y-1">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded text-[11px] font-bold bg-[#8B6B4A] text-white uppercase tracking-wide">
                      ⭐ Recommended Model
                    </span>
                    <h3 className="text-xl font-bold text-[#3B2F2F]">RASC-Net Proposed</h3>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-extrabold text-[#8B6B4A]">{rascData.confidence_pct}%</div>
                    <span className="text-[11px] px-2 py-0.5 rounded-full font-bold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
                      {rascData.confidence_level} CONFIDENCE
                    </span>
                  </div>
                </div>

                <div className="bg-[#F8F5F0] p-4 rounded-xl border border-[#E7DDD2]">
                  <div className="text-xs text-[#7A624A] font-semibold uppercase tracking-wider">Primary Prediction</div>
                  <div className="text-xl font-bold text-[#3B2F2F] mt-1">{rascData.lesion_name}</div>
                  <div className="text-xs text-[#8B6B4A] font-mono mt-0.5">Code: {rascData.class_code?.toUpperCase() || ''}</div>
                </div>

                <div className="space-y-3">
                  <div className="text-sm font-semibold text-[#3B2F2F]">Top-3 Diagnostic Candidates</div>
                  {(rascData.top_3_predictions || []).map((cand, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-medium text-[#7A624A]">
                        <span>{cand.lesion_name} ({cand.class_code?.toUpperCase() || ''})</span>
                        <span className="font-bold text-[#8B6B4A]">{cand.confidence_pct}%</span>
                      </div>
                      <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
                        <div 
                          className="bg-[#8B6B4A] h-full rounded-full transition-all duration-700 animate-progress-bar" 
                          style={{ width: `${cand.confidence_pct}%` }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Card 2: Soft Voting Ensemble */}
              <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-6 relative">
                <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-4">
                  <div className="space-y-1">
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded text-[11px] font-bold bg-[#F4EFE6] text-[#7A624A] uppercase tracking-wide border border-[#E7DDD2]">
                      Baseline Ensemble
                    </span>
                    <h3 className="text-xl font-bold text-[#3B2F2F]">Soft Voting Ensemble</h3>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-extrabold text-[#C8A97E]">{ensembleData.confidence_pct}%</div>
                    <span className="text-[11px] px-2 py-0.5 rounded-full font-bold bg-[#F4EFE6] text-[#7A624A] border border-[#E7DDD2]">
                      {ensembleData.confidence_level} CONFIDENCE
                    </span>
                  </div>
                </div>

                <div className="bg-[#F8F5F0] p-4 rounded-xl border border-[#E7DDD2]">
                  <div className="text-xs text-[#7A624A] font-semibold uppercase tracking-wider">Primary Prediction</div>
                  <div className="text-xl font-bold text-[#3B2F2F] mt-1">{ensembleData.lesion_name}</div>
                  <div className="text-xs text-[#C8A97E] font-mono mt-0.5">Code: {ensembleData.class_code?.toUpperCase() || ''}</div>
                </div>

                <div className="space-y-3">
                  <div className="text-sm font-semibold text-[#3B2F2F]">Top-3 Diagnostic Candidates</div>
                  {(ensembleData.top_3_predictions || []).map((cand, idx) => (
                    <div key={idx} className="space-y-1">
                      <div className="flex justify-between text-xs font-medium text-[#7A624A]">
                        <span>{cand.lesion_name} ({cand.class_code?.toUpperCase() || ''})</span>
                        <span className="font-bold text-[#C8A97E]">{cand.confidence_pct}%</span>
                      </div>
                      <div className="w-full bg-[#F4EFE6] h-2.5 rounded-full overflow-hidden">
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

            {/* Grad-CAM Card */}
            <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-4">
              <GradcamErrorBoundary>
                <div className="flex items-center justify-between border-b border-[#F4EFE6] pb-4">
                  <div>
                    <h3 className="text-xl font-semibold text-[#3B2F2F] flex items-center space-x-2">
                      <Eye className="w-5 h-5 text-[#8B6B4A]" />
                      <span>Grad-CAM Visual Explainability</span>
                    </h3>
                    <p className="text-xs text-[#7A624A] mt-0.5">
                      Spatial activation map generated strictly for RASC-Net Proposed architecture
                    </p>
                  </div>

                  <button
                    onClick={() => fetchGradcam(selectedFile)}
                    disabled={gradcamLoading}
                    className="px-4 py-2 bg-[#F4EFE6] hover:bg-[#E7DDD2] text-[#8B6B4A] rounded-xl border border-[#E7DDD2] text-xs font-semibold flex items-center space-x-2 transition-all shadow-xs"
                  >
                    <RefreshCw className={`w-3.5 h-3.5 ${gradcamLoading ? 'animate-spin' : ''}`} />
                    <span>{gradcamLoading ? 'Generating...' : 'Generate Grad-CAM'}</span>
                  </button>
                </div>

                {typeof gradcamImage === 'string' && gradcamImage.startsWith('data:image/png;base64,') ? (
                  <div className="bg-[#F8F5F0] p-6 rounded-2xl border border-[#E7DDD2] text-center space-y-3">
                    <img 
                      src={gradcamImage} 
                      alt="RASC-Net Proposed Grad-CAM" 
                      className="max-w-[450px] w-full mx-auto rounded-xl border border-[#E7DDD2] shadow-md object-cover" 
                    />
                    <p className="text-xs text-[#7A624A] max-w-lg mx-auto leading-relaxed">
                      Highlighted red and warm spatial activations pinpoint exact anatomical structures driving RASC-Net Proposed's diagnostic decision.
                    </p>
                  </div>
                ) : gradcamLoading ? (
                  <div className="h-56 bg-[#F8F5F0] rounded-2xl border border-[#E7DDD2] flex items-center justify-center text-sm text-[#7A624A]">
                    <div className="w-6 h-6 border-2 border-[#8B6B4A] border-t-transparent rounded-full animate-spin mr-3"></div>
                    Generating Grad-CAM Heatmap Overlay...
                  </div>
                ) : gradcamError ? (
                  <div className="p-4 bg-[#FBF0EF] border border-[#F2D6D3] rounded-xl text-[#C0564B] text-xs flex items-center justify-between">
                    <span>{gradcamError}</span>
                    <button onClick={() => fetchGradcam(selectedFile)} className="px-3 py-1 bg-[#C0564B] text-white rounded-lg text-xs font-bold shadow-xs">Retry</button>
                  </div>
                ) : null}
              </GradcamErrorBoundary>
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

            {/* Grad-CAM */}
            {typeof gradcamImage === 'string' && gradcamImage.startsWith('data:image/png;base64,') && (
              <div className="space-y-2 text-xs border-t pt-3">
                <h4 className="font-bold border-b pb-1">3. Grad-CAM Spatial Explainability Heatmap</h4>
                <div className="flex items-center space-x-4 bg-slate-50 p-3 rounded-xl border">
                  <img src={gradcamImage} alt="Grad-CAM Overlay" className="w-28 h-28 object-cover rounded-lg border" />
                  <p className="text-slate-600 leading-relaxed">
                    Red and warm spatial activations highlight exact anatomical structures driving RASC-Net Proposed diagnosis.
                  </p>
                </div>
              </div>
            )}

            {/* Risk Assessment */}
            {riskData && (
              <div className="space-y-2 text-xs bg-emerald-50/50 p-4 rounded-xl border border-emerald-200">
                <h4 className="font-bold border-b border-emerald-200 pb-1 text-emerald-900">4. Rule-Based Clinical Risk Assessment</h4>
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
