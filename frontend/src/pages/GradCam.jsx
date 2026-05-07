import React, { useState } from 'react';
import axios from 'axios';

const GradCam = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleRunGradcam = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/api/gradcam', formData);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary-color)' }}>Grad-CAM Explainability</h2>
      
      <div className="card" style={{ marginBottom: '2rem' }}>
        <p style={{ marginBottom: '1rem' }}>Visualize the regions of the image that strongly influenced the model's prediction.</p>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'block' }} />
          <button className="btn" onClick={handleRunGradcam} disabled={!file || loading}>
            {loading ? 'Generating...' : 'Generate Grad-CAM'}
          </button>
        </div>
        {error && <div className="error-box">{error}</div>}
      </div>

      {results && (
        <div className="grid">
          <div className="card" style={{ textAlign: 'center' }}>
            <h3 style={{ marginBottom: '1rem' }}>MobileNetV2</h3>
            <img src={`data:image/jpeg;base64,${results.mobilenet.overlay}`} alt="MobileNetV2 Grad-CAM" className="image-preview" />
            <div style={{ marginTop: '1rem' }}>
              <p><strong>Predicted:</strong> {results.mobilenet.predicted_class.toUpperCase()}</p>
              <p><strong>Confidence:</strong> {(results.mobilenet.confidence * 100).toFixed(2)}%</p>
            </div>
          </div>
          
          <div className="card" style={{ textAlign: 'center' }}>
            <h3 style={{ marginBottom: '1rem' }}>ResNet50</h3>
            <img src={`data:image/jpeg;base64,${results.resnet.overlay}`} alt="ResNet50 Grad-CAM" className="image-preview" />
            <div style={{ marginTop: '1rem' }}>
              <p><strong>Predicted:</strong> {results.resnet.predicted_class.toUpperCase()}</p>
              <p><strong>Confidence:</strong> {(results.resnet.confidence * 100).toFixed(2)}%</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GradCam;
