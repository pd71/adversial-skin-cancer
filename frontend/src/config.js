/** Dynamic API Base URL resolver.
 * Handles local development (localhost:5000) and automatic Dev Tunnel routing (-5173 -> -5000).
 */
export const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/+$/, '');
  }

  if (typeof window !== 'undefined' && window.location.hostname.includes('.devtunnels.ms')) {
    const protocol = window.location.protocol;
    // Replace -5173 with -5000 for backend devtunnel
    const hostname = window.location.hostname.replace('-5173', '-5000');
    return `${protocol}//${hostname}`;
  }

  return 'http://localhost:5000';
};
