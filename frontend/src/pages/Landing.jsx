import React from 'react';
import { Link } from 'react-router-dom';
import { Upload, ShieldAlert, BarChart3, Eye } from 'lucide-react';

const Landing = () => {
  return (
    <div className="hero">
      <h1>Skin Cancer Classification AI</h1>
      <p>
        An advanced deep learning dashboard for dermatological diagnostics, featuring ensemble predictions, adversarial robustness evaluations, and explainable AI insights.
      </p>
      
      <div className="grid" style={{ marginTop: '3rem' }}>
        <div className="card">
          <Upload size={40} color="var(--secondary-color)" style={{ marginBottom: '1rem' }} />
          <h3>Diagnostic Prediction</h3>
          <p style={{ color: 'var(--text-secondary)', margin: '1rem 0' }}>Upload an image or use a webcam to get instant ensemble predictions from our MobileNetV2 and ResNet50 models.</p>
          <Link to="/predict" className="btn">Upload Image</Link>
        </div>
        
        <div className="card">
          <ShieldAlert size={40} color="#eab308" style={{ marginBottom: '1rem' }} />
          <h3>Adversarial Attacks</h3>
          <p style={{ color: 'var(--text-secondary)', margin: '1rem 0' }}>Evaluate model robustness against state-of-the-art adversarial attacks like FGSM, PGD, and CW.</p>
          <Link to="/attacks" className="btn">View Attacks</Link>
        </div>
        
        <div className="card">
          <Eye size={40} color="#8b5cf6" style={{ marginBottom: '1rem' }} />
          <h3>Grad-CAM Explainability</h3>
          <p style={{ color: 'var(--text-secondary)', margin: '1rem 0' }}>Understand model decisions with visual heatmaps highlighting the regions of interest used for prediction.</p>
          <Link to="/gradcam" className="btn">View Grad-CAM</Link>
        </div>
        
        <div className="card">
          <BarChart3 size={40} color="#10b981" style={{ marginBottom: '1rem' }} />
          <h3>Performance Metrics</h3>
          <p style={{ color: 'var(--text-secondary)', margin: '1rem 0' }}>Review comprehensive evaluation metrics including accuracy, confusion matrices, and defense results.</p>
          <Link to="/metrics" className="btn">View Metrics</Link>
        </div>
      </div>
    </div>
  );
};

export default Landing;
