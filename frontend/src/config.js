/** Dynamic API Base URL resolver.
 * Handles local development (localhost:5000) and automatic Dev Tunnel routing (-5173 -> -5000).
 */
export const getApiBaseUrl = () => {
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL.replace(/\/+$/, '');
  }

  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    const protocol = window.location.protocol;

    if (hostname.includes('.devtunnels.ms')) {
      if (hostname === 'dermshield.asse.devtunnels.ms') {
        return `${protocol}//dermshield-5000.asse.devtunnels.ms`;
      }
      return `${protocol}//${hostname.replace(/-80|-5173/, '-5000')}`;
    }

    if (hostname.includes('loca.lt')) {
      return `${protocol}//dermshield-api.loca.lt`;
    }
  }

  return 'http://localhost:5000';
};
