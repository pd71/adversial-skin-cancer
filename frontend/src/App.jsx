import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navigation from './components/Navigation';
import Landing from './pages/Landing';
import Predict from './pages/Predict';
import Attacks from './pages/Attacks';
import GradCam from './pages/GradCam';
import Metrics from './pages/Metrics';

function App() {
  return (
    <Router>
      <Navigation />
      <div className="container">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="/attacks" element={<Attacks />} />
          <Route path="/gradcam" element={<GradCam />} />
          <Route path="/metrics" element={<Metrics />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
