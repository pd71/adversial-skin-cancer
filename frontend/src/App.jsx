import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navigation from './components/Navigation';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Attacks from './pages/Attacks';
import Metrics from './pages/Metrics';

import { ThemeProvider } from './context/ThemeContext';

import MedicalCellularBackground from './components/MedicalCellularBackground';

const AppLayout = () => {
  const location = useLocation();
  // Hide Navbar only on Landing page ('/')
  const isLandingPage = location.pathname === '/';

  return (
    <div className="relative min-h-screen">
      {/* Global Background Medical Dermoscopy Cellular Animation Layer */}
      <MedicalCellularBackground />

      <div className="relative z-10">
        {!isLandingPage && <Navigation />}
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/attacks" element={<Attacks />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </div>
    </div>
  );
};

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AppLayout />
      </Router>
    </ThemeProvider>
  );
}

export default App;
