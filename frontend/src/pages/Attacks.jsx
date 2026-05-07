import React, { useState } from 'react';
import axios from 'axios';

const Attacks = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleRunAttacks = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await axios.post('http://localhost:5000/api/attacks', formData);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary-color)' }}>Adversarial Attack Generation</h2>
      
      <div className="card" style={{ marginBottom: '2rem' }}>
        <p style={{ marginBottom: '1rem' }}>Generate FGSM, PGD, and CW adversarial examples on the fly using MobileNetV2.</p>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <input type="file" id="attack-file" accept="image/*" onChange={handleFileChange} style={{ display: 'block' }} />
          <button className="btn" onClick={handleRunAttacks} disabled={!file || loading}>
            {loading ? 'Generating Attacks...' : 'Run Attacks'}
          </button>
        </div>
        {error && <div className="error-box">{error}</div>}
      </div>

      {results && (
        <div className="grid">
          <div className="card" style={{ textAlign: 'center' }}>
            <h4>Clean Original</h4>
            <img src={`data:image/jpeg;base64,${results.original}`} alt="Original" className="image-preview" />
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <h4>FGSM Attack</h4>
            <img src={`data:image/jpeg;base64,${results.fgsm}`} alt="FGSM" className="image-preview" />
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <h4>PGD Attack</h4>
            <img src={`data:image/jpeg;base64,${results.pgd}`} alt="PGD" className="image-preview" />
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <h4>CW Attack (Simplified)</h4>
            <img src={`data:image/jpeg;base64,${results.cw}`} alt="CW" className="image-preview" />
          </div>
        </div>
      )}
    </div>
  );
};

export default Attacks;
