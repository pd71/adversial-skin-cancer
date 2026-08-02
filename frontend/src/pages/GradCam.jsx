import React, { useState } from 'react';
import axios from 'axios';
import { Eye, RefreshCw, AlertTriangle, Sparkles } from 'lucide-react';
import ImageUploadCard from '../components/ImageUploadCard';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

const GradCam = () => {
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

  const handleRunGradcam = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await axios.post(`${API_BASE}/api/gradcam`, formData);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'Failed to generate Grad-CAM visualization.');
    } finally {
      setLoading(false);
    }
  };

  const gradcamImage = results?.gradcam ?? results?.gradcam_image_base64 ?? null;
  const isImageValid = typeof gradcamImage === 'string' && gradcamImage.startsWith('data:image');

  return (
    <div className="min-h-screen bg-[#F8F5F0] text-[#3B2F2F] pb-16">
      <div className="max-w-[1200px] mx-auto px-6 py-8 space-y-8">
        
        {/* Header */}
        <div className="border-b border-[#E7DDD2] pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-[#8B6B4A]/10 text-[#8B6B4A] rounded-2xl border border-[#E7DDD2]">
                <Eye className="w-8 h-8" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-[#3B2F2F] tracking-tight">
                  Grad-CAM Spatial Visual Explainability
                </h1>
                <p className="text-sm text-[#7A624A] mt-1">
                  Gradient-weighted class activation mapping generated strictly for RASC-Net Proposed architecture
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-3">
            <span className="inline-flex items-center px-3.5 py-1.5 rounded-full text-xs font-semibold bg-[#F4EFE6] text-[#8B6B4A] border border-[#E7DDD2]">
              <Sparkles className="w-4 h-4 mr-1.5 text-[#8B6B4A]" />
              Target Layer: conv2d_20
            </span>
          </div>
        </div>

        {/* Universal Image Upload Card Component */}
        <div className="space-y-4">
          <ImageUploadCard
            title="Select Lesion Image for Explainability Analysis"
            description="Drag & drop, browse files, or paste (Ctrl + V) from clipboard"
            selectedFile={selectedFile}
            imagePreview={imagePreview}
            onImageSelected={handleImageSelected}
            onImageRemoved={handleImageRemoved}
          />

          {selectedFile && (
            <div className="bg-[#FFFDF9] rounded-2xl p-4 shadow-sm border border-[#E7DDD2] flex flex-col sm:flex-row items-center justify-between gap-4 animate-fade-in">
              <div className="text-xs text-[#7A624A]">
                Image loaded for Grad-CAM. Click below to generate spatial activation heatmap.
              </div>
              <button 
                onClick={handleRunGradcam} 
                disabled={loading}
                className={`px-6 py-2.5 rounded-xl font-bold text-white shadow-md transition-all flex items-center space-x-2 text-sm whitespace-nowrap ${
                  loading
                    ? 'bg-[#D8C3A5] cursor-not-allowed'
                    : 'bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] hover:from-[#6E5338] hover:to-[#3B2F2F] shadow-[#8B6B4A]/25'
                }`}
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                <span>{loading ? 'Generating...' : 'Generate Grad-CAM'}</span>
              </button>
            </div>
          )}

          {error && (
            <div className="p-4 bg-[#FBF0EF] border border-[#F2D6D3] rounded-xl text-[#C0564B] text-xs flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Result Card */}
        {results && (
          <div className="bg-[#FFFDF9] rounded-2xl p-6 shadow-md border border-[#E7DDD2] space-y-6 text-center animate-fade-in">
            <div className="border-b border-[#F4EFE6] pb-4">
              <h3 className="text-xl font-bold text-[#3B2F2F]">
                {results.model_name || 'RASC-Net Proposed'} Heatmap Overlay
              </h3>
            </div>
            
            {isImageValid ? (
              <div className="bg-[#F8F5F0] p-6 rounded-2xl border border-[#E7DDD2] inline-block max-w-[500px] w-full">
                <img 
                  src={gradcamImage} 
                  alt="RASC-Net Proposed Grad-CAM" 
                  className="max-w-[450px] w-full mx-auto rounded-xl border border-[#E7DDD2] shadow-md object-cover" 
                />
                <p className="text-xs text-[#7A624A] mt-4 leading-relaxed">
                  Red and warm spatial activations highlight exact anatomical structures driving RASC-Net Proposed diagnosis.
                </p>
              </div>
            ) : (
              <div className="p-8 text-[#7A624A] text-sm">No valid Grad-CAM image returned.</div>
            )}

            <div className="flex flex-wrap justify-center gap-6 pt-4 border-t border-[#F4EFE6] text-sm">
              <div className="px-4 py-2 bg-[#F8F5F0] rounded-xl border border-[#E7DDD2]">
                <span className="text-[#7A624A] font-medium">Predicted Class: </span>
                <strong className="text-[#3B2F2F]">{results.predicted_class ? results.predicted_class.toUpperCase() : 'N/A'}</strong>
              </div>
              <div className="px-4 py-2 bg-[#F8F5F0] rounded-xl border border-[#E7DDD2]">
                <span className="text-[#7A624A] font-medium">Confidence: </span>
                <strong className="text-[#8B6B4A]">{results.confidence_pct ? `${results.confidence_pct}%` : 'N/A'}</strong>
              </div>
              <div className="px-4 py-2 bg-[#F8F5F0] rounded-xl border border-[#E7DDD2]">
                <span className="text-[#7A624A] font-medium">Target Layer: </span>
                <strong className="text-[#C8A97E]">{results.last_conv_layer || 'conv2d_20'}</strong>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default GradCam;
