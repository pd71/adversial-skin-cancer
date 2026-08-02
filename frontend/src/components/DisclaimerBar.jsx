import React, { useState, useEffect } from 'react';
import { ShieldAlert, X } from 'lucide-react';

const DisclaimerBar = () => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    const isDismissed = sessionStorage.getItem('disclaimer_dismissed');
    if (isDismissed === 'true') {
      setIsVisible(false);
    }
  }, []);

  const handleDismiss = () => {
    sessionStorage.setItem('disclaimer_dismissed', 'true');
    setIsVisible(false);
  };

  if (!isVisible) return null;

  return (
    <div className="bg-[#F4EFE6] border-b border-[#E7DDD2] text-[#3B2F2F] text-xs py-2 px-4 sticky top-0 z-50 transition-all shadow-xs">
      <div className="max-w-[1200px] mx-auto flex items-center justify-between gap-3 min-h-[24px]">
        <div className="flex items-center space-x-2 min-w-0">
          <ShieldAlert className="w-4 h-4 text-[#8B6B4A] flex-shrink-0" />
          <p className="truncate sm:whitespace-normal leading-tight font-medium text-[11px] sm:text-xs text-[#7A624A]">
            <strong className="font-bold text-[#3B2F2F]">Medical Disclaimer:</strong> This application provides AI-assisted predictions for research and educational purposes only. It is not a substitute for professional medical diagnosis or treatment. Always consult a qualified dermatologist or healthcare professional before making medical decisions.
          </p>
        </div>
        <button
          onClick={handleDismiss}
          className="p-1 hover:bg-[#E7DDD2] text-[#7A624A] hover:text-[#3B2F2F] rounded-lg transition-all flex-shrink-0"
          title="Dismiss Disclaimer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};

export default DisclaimerBar;
