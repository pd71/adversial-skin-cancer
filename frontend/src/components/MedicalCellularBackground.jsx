import React from 'react';

const MedicalCellularBackground = () => {
  return (
    <div 
      aria-hidden="true" 
      className="fixed inset-0 pointer-events-none z-0 overflow-hidden select-none"
    >
      {/* 1. Base Dermoscopy Skin Texture Image Layer */}
      <div 
        className="absolute inset-0 opacity-30 dark:opacity-25 transition-opacity duration-1000 mix-blend-multiply dark:mix-blend-overlay"
        style={{
          backgroundImage: `url('/dermoscopy_bg.png')`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundRepeat: 'no-repeat',
        }}
      />

      {/* 2. Enhanced Rich Theme Color Splashes (Vibrant Gold & Warm Bronze Gradients) */}
      {/* Top Left Splash */}
      <div className="absolute -top-40 -left-40 w-[48rem] h-[48rem] rounded-full bg-gradient-to-br from-[#8B6B4A]/40 via-[#C8A97E]/30 to-transparent dark:from-[#D4AF37]/35 dark:via-[#8B6B4A]/25 blur-3xl animate-float-slow" />

      {/* Top Right Splash */}
      <div className="absolute top-1/4 -right-40 w-[50rem] h-[50rem] rounded-full bg-gradient-to-tl from-[#C8A97E]/40 via-[#8B6B4A]/30 to-transparent dark:from-[#8B6B4A]/35 dark:via-[#D4AF37]/25 blur-3xl animate-float-slow-reverse" style={{ animationDelay: '-6s' }} />

      {/* Bottom Center Splash */}
      <div className="absolute -bottom-40 left-1/4 w-[46rem] h-[46rem] rounded-full bg-gradient-to-tr from-[#8B6B4A]/35 via-[#D4AF37]/30 to-transparent dark:from-[#D4AF37]/30 dark:via-[#8B6B4A]/20 blur-3xl animate-float-slow" style={{ animationDelay: '-12s' }} />

      {/* Center Ambient Glow Accent */}
      <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[36rem] h-[36rem] rounded-full bg-[#D4AF37]/20 dark:bg-[#D4AF37]/15 blur-3xl animate-pulse-glow" />

      {/* 3. Floating Medical Crosses (+) & Glowing Pulse Points */}
      <div className="absolute inset-0">
        {/* Medical Cross Plus 1 (Top Left) */}
        <div className="absolute top-[14%] left-[8%] text-[#8B6B4A]/45 dark:text-[#D4AF37]/55 animate-cellular-float shadow-sm" style={{ animationDuration: '22s' }}>
          <svg className="w-10 h-10" viewBox="0 0 24 24" fill="currentColor">
            <path d="M11 2h2v7h7v2h-7v7h-2v-7H4v-2h7V2z" />
          </svg>
        </div>

        {/* Medical Cross Plus 2 (Middle Right) */}
        <div className="absolute top-[40%] right-[9%] text-[#C8A97E]/45 dark:text-[#8B6B4A]/55 animate-cellular-float shadow-sm" style={{ animationDuration: '26s', animationDelay: '-6s' }}>
          <svg className="w-12 h-12" viewBox="0 0 24 24" fill="currentColor">
            <path d="M11 2h2v7h7v2h-7v7h-2v-7H4v-2h7V2z" />
          </svg>
        </div>

        {/* Medical Cross Plus 3 (Bottom Left) */}
        <div className="absolute bottom-[18%] left-[12%] text-[#8B6B4A]/40 dark:text-[#D4AF37]/50 animate-cellular-float shadow-sm" style={{ animationDuration: '20s', animationDelay: '-11s' }}>
          <svg className="w-9 h-9" viewBox="0 0 24 24" fill="currentColor">
            <path d="M11 2h2v7h7v2h-7v7h-2v-7H4v-2h7V2z" />
          </svg>
        </div>

        {/* Glowing Medical Pulse Points */}
        <div className="absolute top-[26%] left-[24%] w-3 h-3 rounded-full bg-[#8B6B4A]/50 dark:bg-[#D4AF37]/60 animate-ping" style={{ animationDuration: '3.5s' }} />
        <div className="absolute top-[60%] right-[20%] w-3.5 h-3.5 rounded-full bg-[#C8A97E]/50 dark:bg-[#8B6B4A]/60 animate-ping" style={{ animationDuration: '4.5s', animationDelay: '-2s' }} />
        <div className="absolute bottom-[22%] left-[30%] w-2.5 h-2.5 rounded-full bg-[#8B6B4A]/50 dark:bg-[#D4AF37]/60 animate-ping" style={{ animationDuration: '5.5s', animationDelay: '-3s' }} />
      </div>
    </div>
  );
};

export default MedicalCellularBackground;
