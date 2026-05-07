import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity } from 'lucide-react';

const Navigation = () => {
  const location = useLocation();

  const isActive = (path) => {
    return location.pathname === path ? 'nav-link active' : 'nav-link';
  };

  return (
    <header className="header">
      <Link to="/" style={{ textDecoration: 'none', color: 'var(--primary-color)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 'bold', fontSize: '1.25rem' }}>
        <Activity size={24} color="var(--secondary-color)" />
        PreventIQ
      </Link>
      <nav className="nav-links">
        <Link to="/predict" className={isActive('/predict')}>Predict</Link>
        <Link to="/attacks" className={isActive('/attacks')}>Attacks</Link>
        <Link to="/gradcam" className={isActive('/gradcam')}>Grad-CAM</Link>
        <Link to="/metrics" className={isActive('/metrics')}>Metrics</Link>
      </nav>
    </header>
  );
};

export default Navigation;
