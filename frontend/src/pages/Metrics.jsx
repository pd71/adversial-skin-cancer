import React, { useEffect, useState } from 'react';
import axios from 'axios';

const Metrics = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/metrics');
        setData(response.data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchMetrics();
  }, []);

  if (loading) return <div>Loading metrics...</div>;
  if (error) return <div className="error-box">Error loading metrics: {error}</div>;

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary-color)' }}>Performance Metrics & Defenses</h2>
      
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h3>Adversarial Robustness Evaluation</h3>
        {data?.adversarial && data.adversarial.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Clean Acc</th>
                  <th>FGSM Acc</th>
                  <th>PGD Acc</th>
                  <th>CW Acc</th>
                </tr>
              </thead>
              <tbody>
                {data.adversarial.map((row, i) => (
                  <tr key={i}>
                    <td><strong>{row.Model}</strong></td>
                    <td>{(row.Clean_Accuracy * 100).toFixed(2)}%</td>
                    <td>{(row.FGSM_Accuracy * 100).toFixed(2)}%</td>
                    <td>{(row.PGD_Accuracy * 100).toFixed(2)}%</td>
                    <td>{(row.CW_Accuracy * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>No adversarial metrics found. Run the attack evaluation pipeline first.</p>
        )}
      </div>

      <div className="card">
        <h3>Defense Evaluation (Bit Depth + Blur + JPEG)</h3>
        {data?.defense && data.defense.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Attack</th>
                  <th>Before Defense Acc</th>
                  <th>After Defense Acc</th>
                </tr>
              </thead>
              <tbody>
                {data.defense.map((row, i) => (
                  <tr key={i}>
                    <td><strong>{row.Model}</strong></td>
                    <td>{row.Attack}</td>
                    <td>{(row.Accuracy_Before_Defense * 100).toFixed(2)}%</td>
                    <td style={{ color: row.Accuracy_After_Defense > row.Accuracy_Before_Defense ? 'green' : 'inherit' }}>
                      {(row.Accuracy_After_Defense * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)', marginTop: '1rem' }}>No defense metrics found. Run the defense evaluation pipeline first.</p>
        )}
      </div>
    </div>
  );
};

export default Metrics;
