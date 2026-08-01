import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  UploadCloud, CheckCircle2, AlertCircle, Info, X, ShieldAlert, HeartPulse, 
  Stethoscope, Sun, AlertTriangle, User, Calendar, MapPin, Activity, FileText,
  Printer, Copy, Download, Award, Crosshair, FileCheck, HelpCircle
} from 'lucide-react';

const LESION_DETAILS = {
  nv: {
    fullName: 'Melanocytic Nevi',
    shortCode: 'nv',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Melanocytic nevi (common moles) are benign neoplasms composed of melanocytes. They appear as well-circumscribed, uniform spots or nodules ranging from skin-colored to dark brown.',
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
      'Wear protective hats, sunglasses, and long-sleeved clothing outdoors'
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
      'High total mole count or presence of atypical (dysplastic) nevi'
    ],
    nextSteps: [
      'Schedule an IMMEDIATE urgent consultation with a board-certified dermatologist',
      'Undergo professional dermoscopic examination and full biopsy',
      'Follow histopathological staging and structured treatment planning'
    ],
    prevention: [
      'Strict daily broad-spectrum SPF 50+ sun protection',
      'Avoid tanning beds and artificial UV tanning devices completely'
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
      'Slightly elevated or raised flat texture'
    ],
    causes: [
      'Natural aging process and epithelial cell accumulation',
      'Cumulative sun and UV radiation exposure over time'
    ],
    nextSteps: [
      'No urgent treatment required as these lesions pose no malignant threat',
      'Have a dermatologist verify the diagnosis to rule out atypical lesions'
    ],
    prevention: [
      'Maintain daily broad-spectrum sunscreen application to minimize solar lentigines'
    ]
  },
  bcc: {
    fullName: 'Basal Cell Carcinoma',
    shortCode: 'bcc',
    nature: 'Malignant',
    riskLevel: 'Moderate',
    description: 'Basal cell carcinoma (BCC) is the most common form of skin cancer worldwide. It develops in basal cells of the deepest epidermal layer, grows slowly, and rarely metastasizes, but can cause localized tissue destruction if untreated.',
    symptoms: [
      'Pearly, translucent, or shiny bump with visible tiny blood vessels',
      'Flat, firm, pinkish patch with slightly elevated rolled borders',
      'Open sore or ulcer that bleeds, oozes, crusts, and fails to heal'
    ],
    causes: [
      'Chronic, long-term ultraviolet (UV) radiation from sunlight',
      'Frequent exposure to indoor tanning beds'
    ],
    nextSteps: [
      'Schedule a prompt evaluation with a dermatologist',
      'Skin biopsy to confirm diagnosis and histological subtype'
    ],
    prevention: [
      'Apply broad-spectrum SPF 30+ sunscreen daily and reapply every 2 hours'
    ]
  },
  akiec: {
    fullName: 'Actinic Keratoses / Intraepithelial Carcinoma',
    shortCode: 'akiec',
    nature: 'Pre-cancerous',
    riskLevel: 'Moderate',
    description: 'Actinic keratosis (AK) is a common pre-cancerous lesion caused by chronic sun exposure. Intraepithelial carcinoma represents early in-situ carcinoma. Prompt evaluation prevents progression into invasive squamous cell carcinoma.',
    symptoms: [
      'Rough, dry, scaly, or sand-paper-like rough patch on the skin',
      'Flat to slightly raised pink, red, or brownish crust',
      'Sensation of stinging, itching, tenderness, or burning'
    ],
    causes: [
      'Cumulative solar ultraviolet (UV) radiation exposure over decades'
    ],
    nextSteps: [
      'Consult a dermatologist for clinical mapping and evaluation',
      'Undergo targeted field treatment (cryotherapy, topical 5-fluorouracil, or photodynamic therapy)'
    ],
    prevention: [
      'Strict daily broad-spectrum SPF 50+ sunscreen use'
    ]
  },
  vasc: {
    fullName: 'Vascular Lesions',
    shortCode: 'vasc',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Vascular lesions encompass cherry angiomas, angiokeratomas, pyogenic granulomas, and port-wine stains. They are harmless malformations or benign proliferations of blood vessels in skin layers.',
    symptoms: [
      'Bright red, purple, or deep blue papule or macule',
      'Blanches (fades temporarily) when light pressure is applied'
    ],
    causes: [
      'Benign endothelial cell proliferation within capillaries'
    ],
    nextSteps: [
      'Routine dermatologist checkup to confirm benign vascular diagnosis'
    ],
    prevention: [
      'Protect skin from mechanical trauma, scratching, or harsh friction'
    ]
  },
  df: {
    fullName: 'Dermatofibroma',
    shortCode: 'df',
    nature: 'Benign',
    riskLevel: 'Low',
    description: 'Dermatofibroma is a harmless, common benign fibrous skin nodule typically found on lower legs. It feels firm, like a hard pea beneath the skin, and characteristically dimples inward when pinched.',
    symptoms: [
      'Firm, hard nodule felt beneath skin surface',
      'Reddish-brown, pink, or skin-colored appearance'
    ],
    causes: [
      'Benign fibrous tissue reaction following minor skin trauma'
    ],
    nextSteps: [
      'Dermatologist evaluation for dermoscopic confirmation'
    ],
    prevention: [
      'Avoid picking or scratching insect bites or minor skin wounds'
    ]
  }
};

const SYMPTOM_OPTIONS = [
  'Itching', 'Bleeding', 'Pain', 'Rapid Growth', 
  'Change in Size', 'Change in Color', 'Ulceration', 'None'
];

const LOCATIONS = [
  'Face', 'Scalp', 'Neck', 'Chest', 'Abdomen', 'Back', 
  'Upper Arm', 'Forearm', 'Hand', 'Thigh', 'Leg', 'Foot', 'Other'
];

const DURATIONS = [
  'Less than 1 month', '1–3 months', '3–6 months', '6–12 months', 'More than 1 year'
];

const Predict = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [gradcamOverlay, setGradcamOverlay] = useState(null);
  const [error, setError] = useState(null);

  // Patient Metadata Form State
  const [metadata, setMetadata] = useState({
    age: 45,
    sex: 'Male',
    location: 'Back',
    duration: '3–6 months',
    symptoms: ['Change in Size'],
    family_history: 'No',
    previous_cancer: 'No',
    notes: ''
  });

  // Modal States
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalDetails, setModalDetails] = useState(null);

  const [isReportModalOpen, setIsReportModalOpen] = useState(false);
  const [copySuccess, setCopySuccess] = useState(false);

  const handleMetadataChange = (field, value) => {
    setMetadata(prev => ({ ...prev, [field]: value }));
  };

  const handleSymptomToggle = (symptom) => {
    setMetadata(prev => {
      let updated = [...prev.symptoms];
      if (symptom === 'None') {
        updated = ['None'];
      } else {
        updated = updated.filter(s => s !== 'None');
        if (updated.includes(symptom)) {
          updated = updated.filter(s => s !== symptom);
        } else {
          updated.push(symptom);
        }
      }
      if (updated.length === 0) updated = ['None'];
      return { ...prev, symptoms: updated };
    });
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult(null);
      setGradcamOverlay(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', file);
    formData.append('metadata', JSON.stringify(metadata));

    try {
      // 1. Run CNN Prediction + Clinical Risk Score
      const response = await axios.post('http://localhost:5000/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResult(response.data);

      // 2. Fetch Grad-CAM Overlay for Visual Verification
      try {
        const gradcamFormData = new FormData();
        gradcamFormData.append('image', file);
        const gradcamRes = await axios.post('http://localhost:5000/api/gradcam', gradcamFormData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        if (gradcamRes.data?.mobilenet?.overlay) {
          setGradcamOverlay(`data:image/jpeg;base64,${gradcamRes.data.mobilenet.overlay}`);
        }
      } catch (gErr) {
        console.warn('Grad-CAM overlay fetch skipped:', gErr);
      }

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
    setIsReportModalOpen(false);
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') closeModal();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Export Utilities
  const handlePrintReport = () => {
    window.print();
  };

  const generateReportText = () => {
    if (!result) return '';
    const risk = result.clinical_risk || {};
    const info = LESION_DETAILS[result.class?.toLowerCase()] || { fullName: result.class };

    const top3Text = (result.top_3_predictions || []).map((item, idx) => {
      const name = LESION_DETAILS[item.class_code.toLowerCase()]?.fullName || item.class_code.toUpperCase();
      return `  ${idx + 1}. ${name} (${item.class_code.toUpperCase()}): ${(item.probability * 100).toFixed(2)}%`;
    }).join('\n');

    return `
================================================================================
           GENERAL HOSPITAL DERMATOLOGY ASSESSMENT CENTER
             AI SKIN CANCER CLINICAL DECISION REPORT
================================================================================
Report Reference ID : DS-REPORT-${Math.floor(100000 + Math.random() * 900000)}
Generation Date     : ${new Date().toLocaleDateString()} at ${new Date().toLocaleTimeString()}
System Model        : ${result.model_used || 'RASC-Net (Residual Attention Skin Cancer Network)'}

--------------------------------------------------------------------------------
1. PATIENT DEMOGRAPHICS & CLINICAL HISTORY
--------------------------------------------------------------------------------
Patient Age                   : ${metadata.age} years
Biological Sex                : ${metadata.sex}
Lesion Anatomical Location    : ${metadata.location}
Duration of Lesion            : ${metadata.duration}
Reported Symptoms             : ${metadata.symptoms.join(', ')}
Family History of Skin Cancer : ${metadata.family_history}
Previous Cancer Diagnosis     : ${metadata.previous_cancer}
Additional Clinical Notes     : ${metadata.notes || 'None'}

--------------------------------------------------------------------------------
2. AI IMAGE ANALYSIS & TOP CNN PREDICTIONS (IMAGE ONLY)
--------------------------------------------------------------------------------
Primary Predicted Lesion      : ${info.fullName} (${result.class?.toUpperCase()})
Primary Classification Conf   : ${(result.confidence * 100).toFixed(2)}%
Confidence Category           : ${result.confidence_level || 'HIGH'}

Top 3 Diagnostic Candidates:
${top3Text}

Confidence Assessment Note:
${result.confidence_message}

--------------------------------------------------------------------------------
3. RULE-BASED CLINICAL RISK ASSESSMENT (DECISION SUPPORT)
--------------------------------------------------------------------------------
Risk Classification Category   : ${risk.risk_level || 'UNKNOWN'}
Cumulative Risk Score         : ${risk.score || 0} Points

Itemized Factor Calculation Breakdown:
${(risk.explanation || []).map(exp => `  + ${exp}`).join('\n')}

--------------------------------------------------------------------------------
4. CLINICAL RECOMMENDATIONS
--------------------------------------------------------------------------------
${risk.recommendation || 'Consult dermatologist.'}

--------------------------------------------------------------------------------
5. MEDICAL DISCLAIMER & COMPLIANCE NOTICE
--------------------------------------------------------------------------------
This report is generated by an AI decision-support system. It is intended solely
to assist qualified healthcare providers and does NOT constitute an independent
automated medical diagnosis.
================================================================================
`;
  };

  const handleCopyReport = () => {
    const text = generateReportText();
    navigator.clipboard.writeText(text);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2500);
  };

  const handleDownloadReport = () => {
    const text = generateReportText();
    const element = document.createElement("a");
    const fileBlob = new Blob([text], { type: 'text/plain' });
    element.href = URL.createObjectURL(fileBlob);
    element.download = `Hospital_Clinical_Report_${metadata.age}Y_${metadata.sex}_${result.class}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const predictedCode = result?.class?.toLowerCase();
  const predictedInfo = LESION_DETAILS[predictedCode] || { fullName: result?.class || '', nature: 'Unknown', riskLevel: 'Low' };
  const riskData = result?.clinical_risk;

  return (
    <div>
      <h2 style={{ marginBottom: '1.5rem', color: 'var(--primary-color)' }}>Clinical Decision-Support & Diagnostic Prediction</h2>

      {/* PATIENT METADATA FORM SECTION */}
      <div className="card" style={{ marginBottom: '2rem', background: 'linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1.25rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.75rem' }}>
          <User size={22} style={{ color: 'var(--secondary-color)' }} />
          <h3 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--primary-color)', margin: 0 }}>
            Patient Clinical Information
          </h3>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1.25rem' }}>
          {/* Age Input */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
              Patient Age (Years) *
            </label>
            <input 
              type="number" 
              min="1" 
              max="120"
              value={metadata.age}
              onChange={(e) => handleMetadataChange('age', e.target.value)}
              style={{ width: '100%', padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.95rem' }}
            />
          </div>

          {/* Biological Sex */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
              Biological Sex *
            </label>
            <select
              value={metadata.sex}
              onChange={(e) => handleMetadataChange('sex', e.target.value)}
              style={{ width: '100%', padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.95rem', background: 'white' }}
            >
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Lesion Location */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
              Lesion Anatomical Location *
            </label>
            <select
              value={metadata.location}
              onChange={(e) => handleMetadataChange('location', e.target.value)}
              style={{ width: '100%', padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.95rem', background: 'white' }}
            >
              {LOCATIONS.map(loc => (
                <option key={loc} value={loc}>{loc}</option>
              ))}
            </select>
          </div>

          {/* Lesion Duration */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
              Duration of Lesion *
            </label>
            <select
              value={metadata.duration}
              onChange={(e) => handleMetadataChange('duration', e.target.value)}
              style={{ width: '100%', padding: '0.6rem 0.8rem', borderRadius: '8px', border: '1px solid #cbd5e1', fontSize: '0.95rem', background: 'white' }}
            >
              {DURATIONS.map(dur => (
                <option key={dur} value={dur}>{dur}</option>
              ))}
            </select>
          </div>

          {/* Family History */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
              Family History of Skin Cancer?
            </label>
            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.4rem' }}>
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', cursor: 'pointer' }}>
                <input type="radio" name="fam" value="Yes" checked={metadata.family_history === 'Yes'} onChange={() => handleMetadataChange('family_history', 'Yes')} /> Yes
              </label>
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', cursor: 'pointer' }}>
                <input type="radio" name="fam" value="No" checked={metadata.family_history === 'No'} onChange={() => handleMetadataChange('family_history', 'No')} /> No
              </label>
            </div>
          </div>

          {/* Previous Cancer Diagnosis */}
          <div>
            <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.35rem' }}>
              Previous Skin Cancer Diagnosis?
            </label>
            <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.4rem' }}>
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', cursor: 'pointer' }}>
                <input type="radio" name="prev" value="Yes" checked={metadata.previous_cancer === 'Yes'} onChange={() => handleMetadataChange('previous_cancer', 'Yes')} /> Yes
              </label>
              <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.9rem', cursor: 'pointer' }}>
                <input type="radio" name="prev" value="No" checked={metadata.previous_cancer === 'No'} onChange={() => handleMetadataChange('previous_cancer', 'No')} /> No
              </label>
            </div>
          </div>
        </div>

        {/* Multi-Select Symptoms */}
        <div style={{ marginTop: '1.25rem' }}>
          <label style={{ display: 'block', fontSize: '0.85rem', fontWeight: 600, color: '#475569', marginBottom: '0.5rem' }}>
            Reported Symptoms & Characteristics
          </label>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
            {SYMPTOM_OPTIONS.map(sym => {
              const isSelected = metadata.symptoms.includes(sym);
              return (
                <button
                  key={sym}
                  type="button"
                  onClick={() => handleSymptomToggle(sym)}
                  style={{
                    padding: '0.4rem 0.85rem',
                    borderRadius: '20px',
                    border: isSelected ? '1px solid #2563eb' : '1px solid #cbd5e1',
                    background: isSelected ? '#eff6ff' : 'white',
                    color: isSelected ? '#1d4ed8' : '#64748b',
                    fontSize: '0.85rem',
                    fontWeight: isSelected ? 600 : 400,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {isSelected ? '✓ ' : '+ '}{sym}
                </button>
              );
            })}
          </div>
        </div>
      </div>

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
                  {loading ? <><div className="loader"></div> Running RASC-Net Neural Analysis...</> : 'Analyze Dermoscopic Image'}
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

        {/* Diagnostic & Risk Results Card */}
        <div className="card">
          <h3 style={{ marginBottom: '1rem' }}>Diagnostic Prediction & Risk Assessment</h3>
          {!result && !loading && (
            <p style={{ color: 'var(--text-secondary)' }}>Fill patient information, upload an image, and run analysis to view detailed clinical classification results here.</p>
          )}

          {loading && (
            <div style={{ textAlign: 'center', padding: '3rem 0' }}>
              <div className="loader" style={{ width: '40px', height: '40px', borderTopColor: 'var(--secondary-color)' }}></div>
              <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Executing RASC-Net neural classification...</p>
            </div>
          )}

          {result && !loading && (
            <div>
              {/* Highlighted Result Box */}
              <div className="result-box" style={{ padding: '1.25rem', borderRadius: '10px', background: '#f0fdf4', border: '1px solid #bbf7d0' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <CheckCircle2 size={26} style={{ color: '#166534' }} />
                    <div>
                      <span style={{ fontSize: '0.8rem', color: '#15803d', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Predicted Lesion (CNN Only)
                      </span>
                      <h4 style={{ fontSize: '1.35rem', fontWeight: 800, color: '#14532d', lineHeight: 1.2 }}>
                        {predictedInfo.fullName}
                      </h4>
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button 
                      onClick={() => openModalForClass(predictedCode)}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                        padding: '0.5rem 0.85rem',
                        borderRadius: '8px',
                        background: 'rgba(22, 101, 52, 0.1)',
                        border: '1px solid rgba(22, 101, 52, 0.25)',
                        color: '#14532d',
                        fontWeight: 600,
                        fontSize: '0.85rem',
                        cursor: 'pointer'
                      }}
                    >
                      <Info size={15} /> Learn More
                    </button>

                    <button 
                      onClick={() => setIsReportModalOpen(true)}
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.4rem',
                        padding: '0.5rem 0.85rem',
                        borderRadius: '8px',
                        background: '#2563eb',
                        color: 'white',
                        border: 'none',
                        fontWeight: 600,
                        fontSize: '0.85rem',
                        cursor: 'pointer'
                      }}
                    >
                      <FileText size={15} /> Hospital Report
                    </button>
                  </div>
                </div>
              </div>

              {/* CONFIDENCE LEVEL INDICATOR */}
              <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem 1rem', borderRadius: '8px', background: result.confidence_level === 'HIGH' ? '#f0fdf4' : result.confidence_level === 'MODERATE' ? '#fffbeb' : '#fef2f2', border: result.confidence_level === 'HIGH' ? '1px solid #bbf7d0' : result.confidence_level === 'MODERATE' ? '1px solid #fde68a' : '1px solid #fecaca' }}>
                <Activity size={18} style={{ color: result.confidence_level === 'HIGH' ? '#16a34a' : result.confidence_level === 'MODERATE' ? '#d97706' : '#dc2626' }} />
                <div>
                  <span style={{ fontSize: '0.82rem', fontWeight: 700, color: '#334155' }}>
                    AI CONFIDENCE LEVEL: 
                  </span>
                  <span style={{ 
                    fontSize: '0.82rem', 
                    fontWeight: 800, 
                    marginLeft: '0.4rem',
                    color: result.confidence_level === 'HIGH' ? '#16a34a' : result.confidence_level === 'MODERATE' ? '#d97706' : '#dc2626' 
                  }}>
                    {result.confidence_level} ({(result.confidence * 100).toFixed(1)}%)
                  </span>
                  <p style={{ fontSize: '0.8rem', color: '#64748b', margin: 0, marginTop: '2px' }}>
                    {result.confidence_message}
                  </p>
                </div>
              </div>

              {/* TOP 3 PREDICTIONS LIST */}
              <div style={{ marginTop: '1.25rem', background: '#f8fafc', padding: '1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 700, color: '#334155', marginBottom: '0.65rem' }}>
                  Top 3 Diagnostic Candidates
                </h4>
                {(result.top_3_predictions || []).map((item, idx) => {
                  const info = LESION_DETAILS[item.class_code.toLowerCase()] || { fullName: item.class_code.toUpperCase() };
                  const pct = (item.probability * 100).toFixed(1);
                  return (
                    <div key={item.class_code} style={{ marginBottom: '0.5rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.83rem', fontWeight: idx === 0 ? 700 : 500, color: idx === 0 ? '#0f172a' : '#475569' }}>
                        <span>{idx + 1}. {info.fullName} ({item.class_code.toUpperCase()})</span>
                        <span>{pct}%</span>
                      </div>
                      <div style={{ width: '100%', backgroundColor: '#cbd5e1', borderRadius: '999px', height: '6px', overflow: 'hidden', marginTop: '2px' }}>
                        <div style={{ width: `${pct}%`, backgroundColor: idx === 0 ? '#2563eb' : '#94a3b8', height: '100%' }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* RULE-BASED CLINICAL RISK DISPLAY */}
              {riskData && (
                <div style={{ marginTop: '1.25rem', padding: '1.1rem', borderRadius: '10px', background: riskData.risk_level === 'HIGH RISK' ? '#fef2f2' : riskData.risk_level === 'MODERATE RISK' ? '#fffbeb' : '#f0fdf4', border: riskData.risk_level === 'HIGH RISK' ? '1px solid #fecaca' : riskData.risk_level === 'MODERATE RISK' ? '1px solid #fde68a' : '1px solid #bbf7d0' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#334155' }}>
                      CLINICAL RISK ASSESSMENT (Decision Support)
                    </span>
                    <span style={{ 
                      fontSize: '0.85rem', 
                      fontWeight: 800, 
                      padding: '0.25rem 0.65rem', 
                      borderRadius: '6px', 
                      background: riskData.risk_level === 'HIGH RISK' ? '#dc2626' : riskData.risk_level === 'MODERATE RISK' ? '#d97706' : '#16a34a', 
                      color: 'white' 
                    }}>
                      {riskData.risk_level} ({riskData.score} pts)
                    </span>
                  </div>

                  <p style={{ fontSize: '0.88rem', color: '#475569', marginBottom: '0.5rem', lineHeight: 1.5 }}>
                    <strong>Recommendation:</strong> {riskData.recommendation}
                  </p>

                  {riskData.explanation && riskData.explanation.length > 0 && (
                    <div style={{ fontSize: '0.8rem', color: '#64748b' }}>
                      <strong>Factor Breakdown:</strong> {riskData.explanation.join(' | ')}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* HOSPITAL-GRADE CLINICAL REPORT MODAL */}
      {isReportModalOpen && result && (
        <div 
          onClick={() => setIsReportModalOpen(false)}
          style={{
            position: 'fixed',
            top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.75)',
            backdropFilter: 'blur(8px)',
            zIndex: 1100,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem'
          }}
        >
          <div 
            onClick={(e) => e.stopPropagation()}
            className="printable-report"
            style={{
              background: 'white',
              borderRadius: '16px',
              maxWidth: '800px',
              width: '100%',
              maxHeight: '92vh',
              overflowY: 'auto',
              padding: '2.5rem',
              position: 'relative',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)'
            }}
          >
            {/* Action Bar (No Print) */}
            <div className="no-print" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '1rem' }}>
              <h3 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--primary-color)', margin: 0 }}>
                AI Clinical Assessment Report
              </h3>

              <div style={{ display: 'flex', gap: '0.6rem' }}>
                <button onClick={handlePrintReport} className="btn btn-outline" style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
                  <Printer size={15} /> Print Report
                </button>

                <button onClick={handleCopyReport} className="btn btn-outline" style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
                  <Copy size={15} /> {copySuccess ? 'Copied!' : 'Copy Text'}
                </button>

                <button onClick={handleDownloadReport} className="btn" style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem' }}>
                  <Download size={15} /> Download PDF / TXT
                </button>

                <button onClick={() => setIsReportModalOpen(false)} style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.4rem' }}>
                  <X size={20} color="#64748b" />
                </button>
              </div>
            </div>

            {/* Printable Hospital Header */}
            <div style={{ textAlign: 'center', marginBottom: '2rem', borderBottom: '3px solid #0f172a', paddingBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
                <Award size={28} style={{ color: '#2563eb' }} />
                <h1 style={{ fontSize: '1.75rem', fontWeight: 900, color: '#0f172a', letterSpacing: '-0.02em', margin: 0 }}>
                  GENERAL HOSPITAL DERMATOLOGY CENTER
                </h1>
              </div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#2563eb', margin: 0 }}>
                AI SKIN CANCER CLINICAL ASSESSMENT REPORT
              </h3>
              <p style={{ fontSize: '0.82rem', color: '#64748b', marginTop: '0.4rem' }}>
                Report Reference ID: <strong>DS-REPORT-{Math.floor(100000 + Math.random() * 900000)}</strong> | Generated on {new Date().toLocaleDateString()} at {new Date().toLocaleTimeString()}
              </p>
            </div>

            {/* 1. Patient Demographics & History */}
            <div style={{ marginBottom: '1.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#0f172a', borderBottom: '1px solid #cbd5e1', paddingBottom: '0.4rem', marginBottom: '0.75rem' }}>
                1. Patient Clinical Demographics & History
              </h4>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem 1.5rem', fontSize: '0.9rem', color: '#334155' }}>
                <div><strong>Patient Age:</strong> {metadata.age} years</div>
                <div><strong>Biological Sex:</strong> {metadata.sex}</div>
                <div><strong>Anatomical Location:</strong> {metadata.location}</div>
                <div><strong>Lesion Duration:</strong> {metadata.duration}</div>
                <div><strong>Reported Symptoms:</strong> {metadata.symptoms.join(', ')}</div>
                <div><strong>Family History Cancer:</strong> {metadata.family_history}</div>
                <div><strong>Previous Diagnosis:</strong> {metadata.previous_cancer}</div>
                {metadata.notes && <div style={{ gridColumn: '1 / -1' }}><strong>Notes:</strong> {metadata.notes}</div>}
              </div>
            </div>

            {/* 2. AI Image Analysis & Top 3 Predictions */}
            <div style={{ marginBottom: '1.75rem' }}>
              <h4 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#0f172a', borderBottom: '1px solid #cbd5e1', paddingBottom: '0.4rem', marginBottom: '0.75rem' }}>
                2. AI Image Analysis & Top Predictions (CNN Only)
              </h4>
              <div style={{ background: '#f8fafc', padding: '1.1rem', borderRadius: '8px', border: '1px solid #e2e8f0', fontSize: '0.9rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                  <div>
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#64748b' }}>PRIMARY CLASSIFICATION:</span>
                    <h4 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#0f172a', margin: 0 }}>
                      {predictedInfo.fullName} ({result.class?.toUpperCase()})
                    </h4>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontSize: '0.8rem', fontWeight: 700, color: '#64748b' }}>CONFIDENCE:</span>
                    <div style={{ fontSize: '1.2rem', fontWeight: 800, color: '#2563eb' }}>
                      {(result.confidence * 100).toFixed(2)}% ({result.confidence_level})
                    </div>
                  </div>
                </div>

                {/* Top 3 List in Report */}
                <div style={{ marginTop: '0.85rem', paddingTop: '0.85rem', borderTop: '1px solid #cbd5e1' }}>
                  <strong style={{ fontSize: '0.85rem', color: '#334155', display: 'block', marginBottom: '0.4rem' }}>
                    Top 3 Diagnostic Candidates:
                  </strong>
                  {(result.top_3_predictions || []).map((item, idx) => {
                    const info = LESION_DETAILS[item.class_code.toLowerCase()] || { fullName: item.class_code.toUpperCase() };
                    return (
                      <div key={item.class_code} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.83rem', color: '#475569', marginBottom: '0.2rem' }}>
                        <span>{idx + 1}. {info.fullName} ({item.class_code.toUpperCase()})</span>
                        <span><strong>{(item.probability * 100).toFixed(2)}%</strong></span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* 3. Grad-CAM Visualization Section */}
            {gradcamOverlay && (
              <div style={{ marginBottom: '1.75rem' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#0f172a', borderBottom: '1px solid #cbd5e1', paddingBottom: '0.4rem', marginBottom: '0.75rem' }}>
                  3. Grad-CAM Spatial Explainability Heatmap
                </h4>
                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'center', background: '#f8fafc', padding: '1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                  <img src={gradcamOverlay} alt="Grad-CAM Heatmap" style={{ width: '160px', height: '160px', borderRadius: '8px', objectFit: 'cover', border: '2px solid #cbd5e1' }} />
                  <div>
                    <strong style={{ fontSize: '0.9rem', color: '#0f172a', display: 'block', marginBottom: '0.3rem' }}>
                      Visual Feature Activation Map
                    </strong>
                    <p style={{ fontSize: '0.85rem', color: '#475569', margin: 0, lineHeight: 1.5 }}>
                      "The highlighted red/warm regions indicate specific anatomical sub-structures that most influenced the CNN model's diagnostic classification."
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* 4. Clinical Risk Assessment */}
            {riskData && (
              <div style={{ marginBottom: '1.75rem' }}>
                <h4 style={{ fontSize: '1.05rem', fontWeight: 800, color: '#0f172a', borderBottom: '1px solid #cbd5e1', paddingBottom: '0.4rem', marginBottom: '0.75rem' }}>
                  4. Rule-Based Clinical Risk Assessment
                </h4>
                <div style={{ background: '#f0fdf4', padding: '1.1rem', borderRadius: '8px', border: '1px solid #bbf7d0', fontSize: '0.9rem' }}>
                  <div style={{ marginBottom: '0.4rem' }}>
                    <strong>Risk Category:</strong> <span style={{ fontWeight: 800, color: riskData.risk_level === 'HIGH RISK' ? '#dc2626' : riskData.risk_level === 'MODERATE RISK' ? '#d97706' : '#16a34a' }}>{riskData.risk_level} ({riskData.score} Points)</span>
                  </div>
                  <div style={{ marginBottom: '0.4rem' }}><strong>Itemized Factor Breakdown:</strong></div>
                  <ul style={{ paddingLeft: '1.25rem', margin: '0 0 0.5rem 0', fontSize: '0.85rem', color: '#475569' }}>
                    {(riskData.explanation || []).map((exp, idx) => (
                      <li key={idx}>+ {exp}</li>
                    ))}
                  </ul>
                  <div><strong>Clinical Recommendation:</strong> {riskData.recommendation}</div>
                </div>
              </div>
            )}

            {/* Medical Disclaimer */}
            <div style={{ marginTop: '2rem', paddingTop: '1rem', borderTop: '1px solid #e2e8f0', fontSize: '0.78rem', color: '#64748b', textAlign: 'center', lineHeight: 1.5 }}>
              <strong>Medical Disclaimer:</strong> This report is generated by an artificial intelligence decision-support tool. It is not an automated medical diagnosis. All predictions and risk scores must be reviewed and confirmed by a licensed dermatologist or medical practitioner.
            </div>
          </div>
        </div>
      )}

      {/* INFORMATIONAL MODAL POPUP */}
      {isModalOpen && modalDetails && (
        <div 
          onClick={closeModal}
          style={{
            position: 'fixed',
            top: 0, left: 0, right: 0, bottom: 0,
            backgroundColor: 'rgba(15, 23, 42, 0.65)',
            backdropFilter: 'blur(8px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '1.5rem'
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
              padding: '2rem',
              position: 'relative'
            }}
          >
            <button onClick={closeModal} style={{ position: 'absolute', top: '1.25rem', right: '1.25rem', background: '#f1f5f9', border: 'none', borderRadius: '50%', width: '36px', height: '36px', cursor: 'pointer' }}>
              <X size={20} color="#64748b" />
            </button>

            <h3 style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--primary-color)', marginBottom: '0.5rem' }}>
              {modalDetails.fullName}
            </h3>
            <p style={{ fontSize: '0.95rem', color: '#334155', lineHeight: 1.6 }}>{modalDetails.description}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default Predict;
