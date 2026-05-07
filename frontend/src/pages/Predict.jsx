import React, { useState } from 'react';
import axios from 'axios';
import { UploadCloud, CheckCircle2, AlertCircle } from 'lucide-react';

const Predict = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary-color)' }}>Diagnostic Prediction</h2>
      
      <div className="grid">
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Upload Image</h3>
          <label className="file-upload-label">
            <UploadCloud size={48} style={{ marginBottom: '1rem' }} />
            <span>Click to upload or drag and drop</span>
            <span style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>JPG, PNG up to 10MB</span>
            <input type="file" accept="image/*" onChange={handleFileChange} />
          </label>
          
          {preview && (
            <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
              <img src={preview} alt="Preview" className="image-preview" />
              <div style={{ marginTop: '1rem' }}>
                <button 
                  className="btn" 
                  onClick={handlePredict} 
                  disabled={loading}
                  style={{ width: '100%' }}
                >
                  {loading ? <><div className="loader"></div> Processing...</> : 'Analyze Image'}
                </button>
              </div>
            </div>
          )}
          
          {error && (
            <div className="error-box">
              <AlertCircle size={20} style={{ display: 'inline', verticalAlign: 'text-bottom', marginRight: '0.5rem' }} />
              {error}
            </div>
          )}
        </div>

        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Results</h3>
          {!result && !loading && <p style={{ color: 'var(--text-secondary)' }}>Upload an image and run analysis to see results here.</p>}
          
          {loading && (
            <div style={{ textAlign: 'center', padding: '3rem 0' }}>
              <div className="loader" style={{ width: '40px', height: '40px', borderTopColor: 'var(--secondary-color)' }}></div>
              <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Running ensemble models...</p>
            </div>
          )}

          {result && !loading && (
            <div>
              <div className="result-box">
                <CheckCircle2 size={24} style={{ display: 'inline', verticalAlign: 'middle', marginRight: '0.5rem' }} />
                <strong style={{ fontSize: '1.25rem' }}>{result.class.toUpperCase()}</strong>
              </div>
              
              <div style={{ marginTop: '2rem' }}>
                <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
                <p><strong>Model Used:</strong> {result.model_used}</p>
                <p><strong>Ensemble:</strong> {result.ensemble_prediction ? "Yes" : "No"}</p>
              </div>

              <div style={{ marginTop: '2rem' }}>
                <h4 style={{ marginBottom: '0.5rem' }}>Probabilities</h4>
                {Object.entries(result.probabilities).sort((a,b) => b[1]-a[1]).map(([cls, prob]) => (
                  <div key={cls} style={{ marginBottom: '0.5rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                      <span>{cls.toUpperCase()}</span>
                      <span>{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ width: '100%', backgroundColor: 'var(--border-color)', borderRadius: '4px', height: '8px' }}>
                      <div style={{ width: `${prob * 100}%`, backgroundColor: 'var(--secondary-color)', height: '100%', borderRadius: '4px' }}></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Predict;
