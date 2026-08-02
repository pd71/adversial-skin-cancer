import React from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navigation from './components/Navigation';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Attacks from './pages/Attacks';
import GradCam from './pages/GradCam';
import Metrics from './pages/Metrics';

const AppLayout = () => {
  const location = useLocation();
  // Hide Navbar only on Landing page ('/')
  const isLandingPage = location.pathname === '/';

  return (
    <>
      {!isLandingPage && <Navigation />}
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/predict" element={<Predict />} />
        <Route path="/attacks" element={<Attacks />} />
        <Route path="/gradcam" element={<GradCam />} />
        <Route path="/metrics" element={<Metrics />} />
      </Routes>
    </>
  );
};

function App() {
  return (
    <Router>
      <AppLayout />
    </Router>
  );
}

export default App;
