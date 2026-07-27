import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  UploadCloud, CheckCircle2, AlertCircle, Info, X, ShieldAlert, HeartPulse, Stethoscope, Sun, AlertTriangle 
} from 'lucide-react';

const LESION_DETAILS = {
  nv: {
    fullName: 'Melanocytic Nevi',
    shortCode: 'nv',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Melanocytic nevi (common moles) are benign neoplasms composed of melanocytes (pigment-producing skin cells). They appear as well-circumscribed, uniform spots or nodules ranging from skin-colored to dark brown.',
    symptoms: [
      'Uniform color (tan, brown, dark brown, or black)',
      'Defined, regular borders with a distinct round or oval shape',
      'Flat macule or slightly elevated smooth papule',
      'Typically smaller than 6mm (0.25 inches) in diameter'
    ],
    causes: [
      'Normal proliferation of melanocyte cells',
      'Genetic predisposition and family history of moles',
      'Sun exposure during childhood and adolescence'
    ],
    nextSteps: [
      'Perform regular monthly self-examinations using the ABCDE criteria',
      'Schedule routine annual skin checkups with a dermatologist',
      'Consult a doctor promptly if you notice rapid growth, color shifts, or irregular borders'
    ],
    prevention: [
      'Apply broad-spectrum SPF 30+ sunscreen daily on exposed skin',
      'Wear protective hats, sunglasses, and long-sleeved clothing outdoors',
      'Avoid peak solar radiation hours between 10 AM and 4 PM'
    ]
  },
  mel: {
    fullName: 'Melanoma',
    shortCode: 'mel',
    nature: 'Malignant',
    riskLevel: 'High',
    description: 'Melanoma is a serious and potentially aggressive form of skin cancer that originates in pigment-producing melanocytes. Early detection and immediate medical intervention are critical for high survival rates.',
    symptoms: [
      'Asymmetrical structure where two halves do not match',
      'Irregular, scalloped, notched, or poorly defined borders',
      'Varied color distribution (shades of brown, black, pink, red, or white)',
      'Diameter greater than 6mm (size of a pencil eraser)',
      'Evolving lesion: changes in size, shape, color, elevation, or starts bleeding/itching'
    ],
    causes: [
      'Severe, blistering sunburns and intense ultraviolet (UV) radiation',
      'Personal or family history of melanoma skin cancer',
      'High total mole count or presence of atypical (dysplastic) nevi',
      'Fair skin, light hair, blue/green eyes, or high freckling tendency',
      'Immunosuppression or suppressed immune function'
    ],
    nextSteps: [
      'Schedule an IMMEDIATE urgent consultation with a board-certified dermatologist',
      'Undergo professional dermoscopic examination and full biopsy',
      'Follow histopathological staging and structured treatment planning'
    ],
    prevention: [
      'Strict daily broad-spectrum SPF 50+ sun protection',
      'Avoid tanning beds and artificial UV tanning devices completely',
      'Perform thorough monthly full-body skin self-checks'
    ]
  },
  bkl: {
    fullName: 'Benign Keratosis-like Lesions',
    shortCode: 'bkl',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Benign keratosis-like lesions include seborrheic keratoses, solar lentigines, and lichen planus-like keratoses. They are harmless, non-cancerous epidermal growths common in older adults.',
    symptoms: [
      'Waxy, scaly, or "pasted-on" surface appearance',
      'Color range from light tan, yellow, olive, brown, to deep black',
      'Slightly elevated or raised flat texture',
      'Frequently found on face, chest, shoulders, or back'
    ],
    causes: [
      'Natural aging process and epithelial cell accumulation',
      'Cumulative sun and UV radiation exposure over time',
      'Genetic factors and family history'
    ],
    nextSteps: [
      'No urgent treatment required as these lesions pose no malignant threat',
      'Have a dermatologist verify the diagnosis to rule out atypical lesions',
      'Optional minor cryotherapy or removal if irritated by clothing or for cosmetic preference'
    ],
    prevention: [
      'Maintain daily broad-spectrum sunscreen application to minimize solar lentigines',
      'Moisturize skin regularly and avoid aggressive scratching or friction'
    ]
  },
  bcc: {
    fullName: 'Basal Cell Carcinoma',
    shortCode: 'bcc',
    nature: 'Malignant',
    riskLevel: 'Moderate',
    description: 'Basal cell carcinoma (BCC) is the most common form of skin cancer worldwide. It develops in the basal cells of the deepest epidermal layer, grows slowly, and rarely metastasizes, but can cause localized tissue destruction if untreated.',
    symptoms: [
      'Pearly, translucent, or shiny bump with visible tiny blood vessels (telangiectasias)',
      'Flat, firm, pinkish patch with slightly elevated rolled borders',
      'Open sore or ulcer that bleeds, oozes, crusts, and fails to heal',
      'Waxy or scar-like yellowish area with ill-defined borders'
    ],
    causes: [
      'Chronic, long-term ultraviolet (UV) radiation from sunlight',
      'Frequent exposure to indoor tanning beds',
      'Light skin tone, blue/green eyes, or history of radiation therapy'
    ],
    nextSteps: [
      'Schedule a prompt evaluation with a dermatologist',
      'Skin biopsy to confirm diagnosis and histological subtype',
      'Discuss removal options: Mohs micrographic surgery, surgical excision, or topical therapies'
    ],
    prevention: [
      'Apply broad-spectrum SPF 30+ sunscreen daily and reapply every 2 hours',
      'Wear protective UV-blocking sunglasses and broad-brimmed hats',
      'Schedule annual professional dermatological skin examinations'
    ]
  },
  akiec: {
    fullName: 'Actinic Keratoses / Intraepithelial Carcinoma',
    shortCode: 'akiec',
    nature: 'Pre-cancerous',
    riskLevel: 'Moderate',
    description: 'Actinic keratosis (AK) is a common pre-cancerous lesion caused by chronic sun exposure. Intraepithelial carcinoma (Bowen disease) represents early in-situ carcinoma. Prompt evaluation prevents progression into invasive squamous cell carcinoma.',
    symptoms: [
      'Rough, dry, scaly, or sand-paper-like rough patch on the skin',
      'Flat to slightly raised pink, red, or brownish crust',
      'Sensation of stinging, itching, tenderness, or burning',
      'Commonly occurs on sun-exposed areas (scalp, face, ears, forearms, hands)'
    ],
    causes: [
      'Cumulative solar ultraviolet (UV) radiation exposure over decades',
      'Fair skin, light-colored eyes, blonde/red hair, or outdoors occupation',
      'Age-related decrease in skin repair mechanisms or immune suppression'
    ],
    nextSteps: [
      'Consult a dermatologist for clinical mapping and evaluation',
      'Undergo targeted field treatment (cryotherapy, topical 5-fluorouracil, or photodynamic therapy)',
      'Follow-up monitoring every 6 months'
    ],
    prevention: [
      'Strict daily broad-spectrum SPF 50+ sunscreen use',
      'Avoid sun exposure during peak UV intensity hours',
      'Seek early medical treatment for any persistent scaly patches'
    ]
  },
  vasc: {
    fullName: 'Vascular Lesions',
    shortCode: 'vasc',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Vascular lesions encompass cherry angiomas, angiokeratomas, pyogenic granulomas, and port-wine stains. They are harmless malformations or benign proliferations of blood vessels in the skin layers.',
    symptoms: [
      'Bright red, purple, or deep blue papule or macule',
      'Blanches (fades temporarily) when light pressure is applied',
      'Smooth, dome-shaped, or slightly raised vascular spot',
      'May bleed easily if accidentally scratched or scraped'
    ],
    causes: [
      'Benign endothelial cell proliferation within capillaries',
      'Age-related vascular changes and hormonal fluctuations',
      'Genetic predisposition'
    ],
    nextSteps: [
      'Routine dermatologist checkup to confirm benign vascular diagnosis',
      'No medical intervention required unless bleeding occurs or for cosmetic preference',
      'Optional laser therapy or electrocautery if removal is desired'
    ],
    prevention: [
      'Protect skin from mechanical trauma, scratching, or harsh friction',
      'Standard sun protection measures'
    ]
  },
  df: {
    fullName: 'Dermatofibroma',
    shortCode: 'df',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Dermatofibroma is a harmless, common benign fibrous skin nodule typically found on the lower legs. It feels firm, like a hard pea beneath the skin, and characteristically dimples inward when pinched.',
    symptoms: [
      'Firm, hard nodule felt beneath the skin surface',
      'Reddish-brown, pink, or skin-colored appearance',
      'Characteristic "dimple sign" (dimples inward when gently squeezed from sides)',
      'Usually smaller than 1cm in diameter'
    ],
    causes: [
      'Benign fibrous tissue reaction following minor skin trauma (bug bite, splinter, minor cut)',
      'More prevalent in young and middle-aged adults'
    ],
    nextSteps: [
      'Dermatologist evaluation for dermoscopic confirmation',
      'No treatment required as these nodules are entirely benign',
      'Surgical excision only if painful, itchy, or repeatedly bumped/injured'
    ],
    prevention: [
      'Avoid picking or scratching insect bites, splinters, or minor skin wounds'
    ]
  }
};

const Predict = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalDetails, setModalDetails] = useState(null);

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

  const openModalForClass = (classCode) => {
    const normalizedCode = classCode?.toLowerCase();
    const details = LESION_DETAILS[normalizedCode] || LESION_DETAILS['nv'];
    setModalDetails(details);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  // Close modal on ESC key
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        closeModal();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const predictedCode = result?.class?.toLowerCase();
  const predictedInfo = LESION_DETAILS[predictedCode] || { fullName: result?.class || '', nature: 'Unknown', riskLevel: 'Low' };

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary-color)' }}>Diagnostic Skin Lesion Prediction</h2>
      
      <div className="grid">
        {/* Upload Card */}
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Upload Dermoscopic Image</h3>
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
                  {loading ? <><div className="loader"></div> Processing Ensemble Analysis...</> : 'Analyze Dermoscopic Image'}
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

        {/* Prediction Results Card */}
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Diagnostic Prediction Results</h3>
          {!result && !loading && (
            <p style={{ color: 'var(--text-secondary)' }}>Upload an image and run analysis to view detailed classification results here.</p>
          )}
          
          {loading && (
            <div style={{ textAlign: 'center', padding: '3rem 0' }}>
              <div className="loader" style={{ width: '40px', height: '40px', borderTopColor: 'var(--secondary-color)' }}></div>
              <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Running deep ensemble inference...</p>
            </div>
          )}

          {result && !loading && (
            <div>
              {/* Highlighted Result Box with Learn More Button */}
              <div className="result-box" style={{ padding: '1.25rem', borderRadius: '10px', background: '#f0fdf4', border: '1px solid #bbf7d0' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <CheckCircle2 size={26} style={{ color: '#166534' }} />
                    <div>
                      <span style={{ fontSize: '0.8rem', color: '#15803d', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Predicted Lesion
                      </span>
                      <h4 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#14532d', lineHeight: 1.2 }}>
                        {predictedInfo.fullName}
                      </h4>
                    </div>
                  </div>

                  {/* Learn More Button */}
                  <button 
                    onClick={() => openModalForClass(predictedCode)}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.4rem',
                      padding: '0.5rem 1rem',
                      borderRadius: '8px',
                      background: 'rgba(22, 101, 52, 0.1)',
                      border: '1px solid rgba(22, 101, 52, 0.25)',
                      color: '#14532d',
                      fontWeight: 600,
                      fontSize: '0.88rem',
                      cursor: 'pointer',
                      backdropFilter: 'blur(8px)',
                      transition: 'all 0.2s ease',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.04)'
                    }}
                    className="learn-more-btn"
                  >
                    <Info size={16} />
                    <span>Learn More</span>
                  </button>
                </div>
              </div>
              
              {/* Prediction Metadata */}
              <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '1rem' }}>
                <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b' }}>CONFIDENCE</div>
                  <div style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--secondary-color)' }}>
                    {(result.confidence * 100).toFixed(2)}%
                  </div>
                </div>

                <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b' }}>NATURE</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 800, color: predictedInfo.nature === 'Malignant' ? '#dc2626' : '#16a34a' }}>
                    {predictedInfo.nature === 'Malignant' ? '🔴 Malignant' : '🟢 Benign'}
                  </div>
                </div>

                <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 600, color: '#64748b' }}>MODEL USED</div>
                  <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#334155' }}>
                    {result.model_used}
                  </div>
                </div>
              </div>

              {/* Class Probability Distribution Breakdown */}
              <div style={{ marginTop: '1.75rem' }}>
                <h4 style={{ marginBottom: '0.85rem', fontSize: '1rem', fontWeight: 700, color: '#334155' }}>
                  Class Probability Distribution
                </h4>
                {Object.entries(result.probabilities).sort((a,b) => b[1]-a[1]).map(([clsCode, prob]) => {
                  const info = LESION_DETAILS[clsCode.toLowerCase()] || { fullName: clsCode.toUpperCase() };
                  const isTop = clsCode.toLowerCase() === predictedCode;

                  return (
                    <div key={clsCode} style={{ marginBottom: '0.65rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.25rem', fontWeight: isTop ? 700 : 500 }}>
                        <span style={{ color: isTop ? 'var(--primary-color)' : '#475569' }}>
                          {info.fullName} ({clsCode})
                        </span>
                        <span style={{ color: isTop ? 'var(--secondary-color)' : '#64748b' }}>
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div style={{ width: '100%', backgroundColor: '#e2e8f0', borderRadius: '999px', height: '8px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            width: `${prob * 100}%`, 
                            backgroundColor: isTop ? 'var(--secondary-color)' : '#94a3b8', 
                            height: '100%', 
                            borderRadius: '999px',
                            transition: 'width 0.4s ease'
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* INFORMATIONAL MODAL POPUP */}
      {isModalOpen && modalDetails && (
        <div 
          onClick={closeModal}
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(8px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem',
            animation: 'fadeIn 0.25s ease-out'
          }}
        >
          <div 
            onClick={(e) => e.stopPropagation()}
            style={{
              background: 'rgba(255, 255, 255, 0.98)',
              backdropFilter: 'blur(16px)',
              border: '1px solid rgba(226, 232, 240, 0.8)',
              borderRadius: '16px',
              maxWidth: '680px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
              padding: '2rem',
              position: 'relative',
              animation: 'scaleUp 0.25s ease-out'
            }}
          >
            {/* Close Button */}
            <button 
              onClick={closeModal}
              style={{
                position: 'absolute',
                top: '1.25rem',
                right: '1.25rem',
                background: '#f1f5f9',
                border: 'none',
                borderRadius: '50%',
                width: '36px',
                height: '36px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                color: '#64748b',
                transition: 'all 0.2s ease'
              }}
              aria-label="Close Modal"
            >
              <X size={20} />
            </button>

            {/* Modal Header */}
            <div style={{ marginBottom: '1.5rem', paddingRight: '2rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
                <h3 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--primary-color)', margin: 0 }}>
                  {modalDetails.fullName}
                </h3>
                <span style={{ fontSize: '0.85rem', fontWeight: 700, padding: '0.2rem 0.6rem', borderRadius: '999px', background: '#f1f5f9', color: '#475569' }}>
                  Code: {modalDetails.shortCode}
                </span>
              </div>

              {/* Badges: Benign/Malignant & Risk Level */}
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
                <span className={`badge ${modalDetails.nature === 'Malignant' ? 'badge-red' : 'badge-green'}`} style={{ fontSize: '0.85rem', padding: '0.35rem 0.75rem', fontWeight: 700 }}>
                  {modalDetails.nature === 'Malignant' ? '🔴 Malignant' : modalDetails.nature === 'Pre-cancerous' ? '🟠 Pre-cancerous' : '🟢 Benign'}
                </span>

                <span className={`badge ${modalDetails.riskLevel === 'High' ? 'badge-red' : modalDetails.riskLevel === 'Moderate' ? 'badge-amber' : 'badge-blue'}`} style={{ fontSize: '0.85rem', padding: '0.35rem 0.75rem', fontWeight: 700 }}>
                  Risk Level: {modalDetails.riskLevel}
                </span>
              </div>
            </div>

            {/* Description Section */}
            <div style={{ marginBottom: '1.5rem', background: '#f8fafc', padding: '1rem 1.25rem', borderRadius: '10px', borderLeft: '4px solid var(--secondary-color)' }}>
              <p style={{ fontSize: '0.95rem', color: '#334155', lineHeight: 1.6, margin: 0 }}>
                {modalDetails.description}
              </p>
            </div>

            {/* Common Symptoms */}
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-color)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <HeartPulse size={18} style={{ color: '#ef4444' }} /> Common Symptoms & Characteristics
              </h4>
              <ul style={{ paddingLeft: '1.25rem', margin: 0, fontSize: '0.9rem', color: '#475569', lineHeight: 1.6 }}>
                {modalDetails.symptoms.map((item, idx) => (
                  <li key={idx} style={{ marginBottom: '0.35rem' }}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Possible Causes / Risk Factors */}
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-color)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertTriangle size={18} style={{ color: '#f59e0b' }} /> Possible Causes & Risk Factors
              </h4>
              <ul style={{ paddingLeft: '1.25rem', margin: 0, fontSize: '0.9rem', color: '#475569', lineHeight: 1.6 }}>
                {modalDetails.causes.map((item, idx) => (
                  <li key={idx} style={{ marginBottom: '0.35rem' }}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Recommended Next Steps */}
            <div style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-color)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Stethoscope size={18} style={{ color: 'var(--secondary-color)' }} /> Recommended Next Steps
              </h4>
              <ul style={{ paddingLeft: '1.25rem', margin: 0, fontSize: '0.9rem', color: '#475569', lineHeight: 1.6 }}>
                {modalDetails.nextSteps.map((item, idx) => (
                  <li key={idx} style={{ marginBottom: '0.35rem' }}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Prevention Tips */}
            <div style={{ marginBottom: '1.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--primary-color)', marginBottom: '0.6rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sun size={18} style={{ color: '#10b981' }} /> Prevention & Skin Health Tips
              </h4>
              <ul style={{ paddingLeft: '1.25rem', margin: 0, fontSize: '0.9rem', color: '#475569', lineHeight: 1.6 }}>
                {modalDetails.prevention.map((item, idx) => (
                  <li key={idx} style={{ marginBottom: '0.35rem' }}>{item}</li>
                ))}
              </ul>
            </div>

            {/* Disclaimer Box */}
            <div style={{ background: '#fffbeb', border: '1px solid #fef3c7', borderRadius: '10px', padding: '1rem 1.25rem', display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
              <ShieldAlert size={22} style={{ color: '#b45309', flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong style={{ fontSize: '0.88rem', color: '#92400e', display: 'block', marginBottom: '0.2rem' }}>
                  Medical Disclaimer
                </strong>
                <p style={{ fontSize: '0.82rem', color: '#b45309', margin: 0, lineHeight: 1.5 }}>
                  This AI prediction is intended to assist and does not replace a professional medical diagnosis. Please consult a qualified dermatologist for confirmation and appropriate clinical treatment.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Predict;
