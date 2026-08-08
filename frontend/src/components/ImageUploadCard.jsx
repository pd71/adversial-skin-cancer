import React, { useState, useEffect, useRef } from 'react';
import { UploadCloud, Image as ImageIcon, X, RefreshCw, CheckCircle2, Clipboard } from 'lucide-react';

const ImageUploadCard = ({
  onImageSelected,
  onImageRemoved,
  title = "Upload Dermoscopy Image",
  description = "Drag & drop, browse files, or paste (Ctrl + V) from clipboard",
  selectedFile = null,
  imagePreview = null,
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [fileDetails, setFileDetails] = useState(null);
  const [pasteNotice, setPasteNotice] = useState(false);
  const fileInputRef = useRef(null);

  // Compute image resolution details when imagePreview changes
  useEffect(() => {
    if (selectedFile && imagePreview) {
      const img = new Image();
      img.src = imagePreview;
      img.onload = () => {
        setFileDetails({
          name: selectedFile.name,
          size: selectedFile.size > 1024 * 1024 
            ? `${(selectedFile.size / (1024 * 1024)).toFixed(2)} MB`
            : `${(selectedFile.size / 1024).toFixed(1)} KB`,
          resolution: `${img.width} × ${img.height} px`
        });
      };
    } else {
      setFileDetails(null);
    }
  }, [selectedFile, imagePreview]);

  // Global & Local Paste (Ctrl + V) Handler
  useEffect(() => {
    const handlePaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;

      for (let i = 0; i < items.length; i++) {
        if (items[i].type.indexOf('image') !== -1) {
          const file = items[i].getAsFile();
          if (file) {
            e.preventDefault();
            // Create proper File object if name is missing
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const renamedFile = new File([file], `clipboard_image_${timestamp}.png`, {
              type: file.type || 'image/png'
            });

            onImageSelected(renamedFile);
            setPasteNotice(true);
            setTimeout(() => setPasteNotice(false), 3000);
            break;
          }
        }
      }
    };

    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [onImageSelected]);

  // Drag & Drop Handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelected(file);
      }
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type.startsWith('image/')) {
        onImageSelected(file);
      }
    }
  };

  return (
    <div className="bg-[#EFE7DC] dark:bg-[#251E19] rounded-2xl p-6 shadow-md border border-[#D9CBB9] dark:border-[#3D3128] space-y-5 relative transition-all">
      
      {/* Header */}
      <div className="border-b border-[#D9CBB9] dark:border-[#3D3128] pb-4 flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold text-[#3B2F2F] dark:text-[#F5EFEB] flex items-center space-x-2">
            <UploadCloud className="w-5 h-5 text-[#8B6B4A] dark:text-[#D4AF37]" />
            <span>{title}</span>
          </h3>
          <p className="text-xs text-[#7A624A] dark:text-[#B8A594] mt-0.5">{description}</p>
        </div>
        <span className="text-[11px] font-bold px-2.5 py-1 rounded-full bg-[#E5DACE] dark:bg-[#322822] text-[#8B6B4A] dark:text-[#D4AF37] border border-[#D9CBB9] dark:border-[#4A3B30]">
          Multi-Source Upload
        </span>
      </div>

      {/* Paste Notification Toast */}
      {pasteNotice && (
        <div className="p-3 bg-[#E8F0E9] border border-[#C5DDC8] text-[#2D5A38] text-xs rounded-xl flex items-center space-x-2 animate-fade-in">
          <CheckCircle2 className="w-4 h-4 text-[#5F8D6E] flex-shrink-0" />
          <span className="font-semibold">✓ Image pasted from clipboard successfully!</span>
        </div>
      )}

      {/* Interactive Drop Zone & Preview */}
      {imagePreview ? (
        <div className="space-y-4">
          {/* Large ~300px Preview */}
          <div className="w-full max-w-[320px] h-[300px] mx-auto rounded-2xl overflow-hidden border border-[#D9CBB9] dark:border-[#3D3128] shadow-md bg-white relative group">
            <img src={imagePreview} alt="Selected Lesion Preview" className="w-full h-full object-cover" />
            <div className="absolute inset-0 bg-[#3B2F2F]/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-3 py-1.5 bg-[#FFFDF9] text-[#3B2F2F] text-xs font-bold rounded-lg shadow-sm hover:bg-[#F4EFE6]"
              >
                Replace
              </button>
              <button
                type="button"
                onClick={onImageRemoved}
                className="px-3 py-1.5 bg-[#C0564B] text-white text-xs font-bold rounded-lg shadow-sm hover:bg-[#A8483E]"
              >
                Remove
              </button>
            </div>
          </div>

          {/* File Metadata Details */}
          {fileDetails && (
            <div className="bg-[#E5DACE] dark:bg-[#1C1713] p-3.5 rounded-xl border border-[#D9CBB9] dark:border-[#3D3128] text-xs text-[#5C4A38] space-y-1.5 max-w-[320px] mx-auto">
              <div className="font-bold text-[#3B2F2F] dark:text-[#F5EFEB] truncate flex items-center space-x-1.5">
                <ImageIcon className="w-4 h-4 text-[#8B6B4A] dark:text-[#D4AF37] flex-shrink-0" />
                <span className="truncate">{fileDetails.name}</span>
              </div>
              <div className="flex justify-between text-[#7A624A] dark:text-[#B8A594] pt-1 border-t border-[#D9CBB9] dark:border-[#3D3128]">
                <span>Size: <strong className="text-[#3B2F2F] dark:text-[#F5EFEB]">{fileDetails.size}</strong></span>
                <span>Dimensions: <strong className="text-[#3B2F2F] dark:text-[#F5EFEB]">{fileDetails.resolution}</strong></span>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-center space-x-3 pt-2">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="px-4 py-2 bg-[#E5DACE] dark:bg-[#322822] hover:bg-[#D9CBB9] text-[#8B6B4A] dark:text-[#D4AF37] rounded-xl border border-[#D9CBB9] dark:border-[#4A3B30] text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-xs"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Replace Image</span>
            </button>
            <button
              type="button"
              onClick={onImageRemoved}
              className="px-4 py-2 bg-[#FBF0EF] hover:bg-[#F5DCDA] text-[#C0564B] rounded-xl border border-[#F2D6D3] text-xs font-semibold flex items-center space-x-1.5 transition-all shadow-xs"
            >
              <X className="w-3.5 h-3.5" />
              <span>Remove Image</span>
            </button>
          </div>
        </div>
      ) : (
        /* Dropzone Component */
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-2xl p-8 text-center transition-all cursor-pointer relative overflow-hidden shadow-xs ${
            isDragging
              ? 'border-[#8B6B4A] bg-[#E5DACE] scale-[1.02] shadow-md'
              : 'border-[#D9CBB9] dark:border-[#4A3B30] hover:border-[#8B6B4A]'
          }`}
          style={{
            backgroundImage: `linear-gradient(rgba(235, 222, 206, 0.90), rgba(218, 200, 180, 0.92)), url('/dermoscopy_bg.png')`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
          }}
        >
          <div className="space-y-4 max-w-sm mx-auto">
            <div className="w-16 h-16 bg-[#F4EFE6] text-[#8B6B4A] rounded-full flex items-center justify-center mx-auto shadow-xs border border-[#E7DDD2]">
              <UploadCloud className="w-8 h-8" />
            </div>

            <div>
              <div className="text-base font-bold text-[#3B2F2F]">
                {isDragging ? "Drop image here to upload" : "Upload Dermoscopy Image"}
              </div>
              <div className="text-xs text-[#7A624A] mt-1 space-y-0.5">
                <div>Drag & Drop directly here</div>
                <div className="font-semibold text-[#8B6B4A] flex items-center justify-center space-x-1 pt-0.5">
                  <Clipboard className="w-3.5 h-3.5" />
                  <span>or Paste image (Ctrl + V)</span>
                </div>
              </div>
            </div>

            <button
              type="button"
              className="px-5 py-2.5 bg-gradient-to-r from-[#8B6B4A] to-[#6E5338] text-white rounded-xl text-xs font-bold shadow-sm hover:from-[#6E5338] hover:to-[#3B2F2F] transition-all"
            >
              Browse Image
            </button>

            <div className="text-[11px] text-[#A67C52] font-semibold pt-1">
              Supported formats: PNG • JPG • JPEG • WebP
            </div>
          </div>
        </div>
      )}

      {/* Hidden File Input */}
      <input
        type="file"
        ref={fileInputRef}
        accept="image/*"
        onChange={handleFileInput}
        className="hidden"
      />

    </div>
  );
};

export default ImageUploadCard;
