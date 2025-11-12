// ImageWorkflow.tsx
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Eye, Layers, Wand2, Microscope } from 'lucide-react';

interface ImageWorkflowProps {
  file: File;
  onBack: () => void;
}

const BACKEND_URL = "https://flexifyai.onrender.com";

export const ImageWorkflow: React.FC<ImageWorkflowProps> = ({ file, onBack }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<any | null>(null); // basic
  const [features, setFeatures] = useState<any | null>(null); // feature extraction
  const [explain, setExplain] = useState<any | null>(null); // explanation
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string>('basic');

  // Advanced-specific state
  const [suggestedActions, setSuggestedActions] = useState<any[] | null>(null);
  const [actionResults, setActionResults] = useState<Record<string, any>>({});

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  // Generic POST helper for file + form fields
  const postFile = async (endpoint: string, formFields?: Record<string,string|number>) => {
    const formData = new FormData();
    formData.append("file", file);
    if (formFields) {
      for (const k of Object.keys(formFields)) {
        formData.append(k, String((formFields as any)[k]));
      }
    }

    const res = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      // try to parse JSON error if present
      let txt = await res.text();
      try {
        const j = JSON.parse(txt);
        throw new Error(j.error || j.detail || txt);
      } catch {
        throw new Error(`Server returned ${res.status}: ${txt}`);
      }
    }

    const data = await res.json();
    if (data.error) throw new Error(data.error);
    return data;
  };

  // Individual calls (used by Analyze button)
  const runBasic = async () => {
    const data = await postFile("/process-image/");
    setResults(data);
    setFeatures(null);
    setExplain(null);
    setSuggestedActions(null);
    setActionResults({});
  };

  const runFeatures = async () => {
    const data = await postFile("/extract-image-features/", { k_colors: 4 });
    setFeatures(data);
    // keep others cleared
    setResults(null);
    setExplain(null);
    setSuggestedActions(null);
    setActionResults({});
  };

  const runExplain = async (level: "short" | "detailed" = "short") => {
    const data = await postFile("/explain-image/", { explanation_level: level });
    setExplain(data);
    setResults(null);
    setFeatures(null);
    setSuggestedActions(null);
    setActionResults({});
  };

  // Advanced: call advanced-analysis endpoint (backend returns features + explanation + suggested actions)
  const runAdvanced = async () => {
    const data = await postFile("/advanced-analysis/", { k_colors: 6 });
    // backend returns { features, explanation, ocr_text, domain_suggestions, suggested_actions }
    if (data.features) setFeatures(data.features);
    if (data.explanation) setExplain(data.explanation);
    if (data.ocr_text) {
      // you may want to show OCR in basic panel; keep results for OCR only
      setResults({ dimensions: { width: data.features?.width, height: data.features?.height }, format: data.features?.format || "PNG", mode: data.features?.mode || "RGB", extracted_text: data.ocr_text });
    } else {
      setResults(null);
    }

    setSuggestedActions(Array.isArray(data.suggested_actions) ? data.suggested_actions : []);
    setActionResults({});
  };

  // Analyze button handler
  const onAnalyze = async () => {
    setError(null);
    setIsProcessing(true);

    try {
      if (selectedAnalysis === 'basic') {
        await runBasic();
      } else if (selectedAnalysis === 'features') {
        await runFeatures();
      } else if (selectedAnalysis === 'explain') {
        await runExplain("short");
      } else if (selectedAnalysis === 'advanced') {
        await runAdvanced();
      }
    } catch (err: any) {
      setError(err?.message || "Failed to run analysis");
    } finally {
      setIsProcessing(false);
    }
  };

  // Run a suggested advanced action (calls action.endpoint)
  const runSuggestedAction = async (action: any) => {
    setError(null);
    setIsProcessing(true);
    try {
      const data = await postFile(action.endpoint);
      setActionResults(prev => ({ ...prev, [action.id]: data }));
    } catch (err: any) {
      setError(err?.message || "Suggested action failed");
    } finally {
      setIsProcessing(false);
    }
  };

  const renderAnalysisOptions = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <button
        onClick={() => setSelectedAnalysis('basic')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'basic' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-300'}`}
      >
        <Eye className="h-6 w-6 mb-2" />
        <span className="font-medium">Basic Analysis</span>
      </button>

      <button
        onClick={() => setSelectedAnalysis('features')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'features' ? 'border-purple-500 bg-purple-50' : 'border-gray-200 hover:border-purple-300'}`}
      >
        <Layers className="h-6 w-6 mb-2" />
        <span className="font-medium">Feature Extraction</span>
      </button>

      <button
        onClick={() => setSelectedAnalysis('explain')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'explain' ? 'border-green-500 bg-green-50' : 'border-gray-200 hover:border-green-300'}`}
      >
        <Wand2 className="h-6 w-6 mb-2" />
        <span className="font-medium">Explainable AI</span>
      </button>

      <button
        onClick={() => setSelectedAnalysis('advanced')}
        className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${selectedAnalysis === 'advanced' ? 'border-indigo-500 bg-indigo-50' : 'border-gray-200 hover:border-indigo-300'}`}
      >
        <Microscope className="h-6 w-6 mb-2" />
        <span className="font-medium">Advanced Analysis</span>
      </button>
    </div>
  );

  const renderResults = () => {
    if (!results && !features && !explain && (!suggestedActions || suggestedActions.length === 0)) return null;

    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

          {/* Basic results */}
          {results && (
            <>
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Image Properties</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Dimensions</span>
                    <span className="font-medium">{results.dimensions?.width} × {results.dimensions?.height}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Format</span>
                    <span className="font-medium">{results.format}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Color Mode</span>
                    <span className="font-medium">{results.mode}</span>
                  </div>
                </div>
              </div>

              {results.extracted_text && (
                <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm col-span-2">
                  <h3 className="text-lg font-semibold mb-4">Extracted Text (OCR)</h3>
                  <pre className="whitespace-pre-wrap text-sm">{results.extracted_text}</pre>
                </div>
              )}
            </>
          )}

          {/* Feature extraction UI */}
          {features && (
            <div className="col-span-3 space-y-4">
              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Feature Summary</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-600">Avg Brightness</div>
                    <div className="font-medium">{features.avg_brightness}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Contrast</div>
                    <div className="font-medium">{features.contrast}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Dimensions</div>
                    <div className="font-medium">{features.width} × {features.height}</div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-600">Format</div>
                    <div className="font-medium">{features.format}</div>
                  </div>
                </div>
              </div>

              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Dominant Colors</h3>
                <div className="flex gap-3 items-center">
                  {features.dominant_colors?.map((c: string, i: number) => (
                    <div key={i} className="flex flex-col items-center">
                      <div style={{background: c, width: 48, height: 48, borderRadius: 8, border: "1px solid rgba(0,0,0,0.08)"}}/>
                      <div className="text-xs mt-2">{c}</div>
                    </div>
                  ))}
                </div>

                <div className="mt-6">
                  <h4 className="text-sm text-gray-700 mb-2">Palette</h4>
                  {features.palette_image && <img src={`data:image/png;base64,${features.palette_image}`} alt="palette" className="h-12 rounded-md shadow-sm" />}
                </div>

                <div className="mt-6">
                  <h4 className="text-sm text-gray-700 mb-2">Color Histogram</h4>
                  {features.histogram_image && <img src={`data:image/png;base64,${features.histogram_image}`} alt="hist" className="w-full max-w-xl" />}
                </div>

                <div className="mt-6">
                  <h4 className="text-sm text-gray-700 mb-2">Edge Map</h4>
                  {features.edge_image && <img src={`data:image/png;base64,${features.edge_image}`} alt="edges" className="w-full max-w-xl" />}
                </div>
              </div>
            </div>
          )}

          {/* Explanation UI */}
          {explain && (
            <div className="col-span-3 space-y-4">
              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Explainable Summary</h3>
                <div className="text-gray-700">
                  {explain.explanation_steps?.map((s: string, i: number) => <p key={i} className="mb-2">{s}</p>)}
                </div>
              </div>

              <div className="bg-white/80 rounded-xl p-6 border shadow-sm">
                <h3 className="text-lg font-semibold mb-4">Saliency / Edge Map</h3>
                {explain.saliency_image && <img src={`data:image/png;base64,${explain.saliency_image}`} alt="saliency" className="w-full max-w-xl" />}
                <div className="mt-4">
                  <h4 className="text-sm text-gray-700">Feature Summary</h4>
                  <pre className="text-xs bg-gray-50 p-3 rounded">{JSON.stringify(explain.feature_summary || explain, null, 2)}</pre>
                </div>
              </div>
            </div>
          )}

        </div>

        {/* Advanced: Suggested actions UI */}
        { suggestedActions && suggestedActions.length > 0 && (
          <div className="mt-6 bg-white/80 rounded-xl p-6 border shadow-sm">
            <h3 className="text-lg font-semibold mb-4">Recommended follow-up analyses</h3>
            <div className="grid gap-4">
              {suggestedActions.map((a: any) => (
                <div key={a.id} className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 p-3 border rounded">
                  <div>
                    <div className="font-medium">{a.label}</div>
                    <div className="text-sm text-gray-600">{a.description}</div>
                    <div className="text-xs text-gray-400 mt-1">Endpoint: {a.endpoint}</div>
                  </div>

                  <div className="flex flex-col items-end gap-2">
                    <button
                      onClick={() => runSuggestedAction(a)}
                      disabled={isProcessing}
                      className="px-3 py-2 bg-blue-600 text-white rounded-md"
                    >
                      Run
                    </button>
                    {actionResults[a.id] && (
                      <button
                        onClick={() => setActionResults(prev => { const p = {...prev}; delete p[a.id]; return p; })}
                        className="text-xs mt-2 text-gray-600 underline"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>

            {/* Render action results */}
            <div className="mt-4 space-y-3">
              {Object.keys(actionResults).map(key => (
                <div key={key} className="bg-gray-50 p-3 rounded">
                  <div className="font-medium mb-1">{key}</div>
                  <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(actionResults[key], null, 2)}</pre>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center mb-8">
        <button onClick={onBack} className="flex items-center px-4 py-2 bg-white/50 rounded-xl border mr-6">
          <ArrowLeft className="h-4 w-4 mr-2" /> Back
        </button>
        <div>
          <h1 className="text-3xl font-bold">Image Analysis</h1>
          <p className="text-gray-600">Processing: {file.name}</p>
        </div>
      </div>

      <div className="space-y-8">
        <div className="bg-white/70 rounded-2xl p-8 border">
          {preview && (
            <div className="mb-8">
              <img src={preview} alt="Preview" className="w-full h-auto max-w-2xl mx-auto" />
            </div>
          )}

          {renderAnalysisOptions()}

          <div className="text-center">
            <button
              onClick={onAnalyze}
              disabled={isProcessing}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl"
            >
              {isProcessing ? "Processing..." : "Analyze Image"}
            </button>

            {error && <div className="mt-4 text-red-500 bg-red-50 p-4 rounded-lg">{error}</div>}
          </div>

          {renderResults()}
        </div>
      </div>
    </div>
  );
};








// import React, { useState } from 'react';
// import { ArrowLeft, Eye, Download, Image, Chart, Wand2, Layers, Microscope } from 'lucide-react';

// interface ImageWorkflowProps {
//   file: File;
//   onBack: () => void;
// }

// interface ImageAnalysisResults {
//   dimensions: {
//     width: number;
//     height: number;
//   };
//   format: string;
//   mode: string;
//   extracted_text: string;
// }

// interface ImageFeaturesResults {
//   mean_color?: { R: number; G: number; B: number };
//   stddev_color?: { R: number; G: number; B: number };
//   mean_gray?: number;
//   stddev_gray?: number;
//   histogram_preview?: number[];
// }

// interface ExplainResult {
//   explanation_steps: string[];
//   shap_value_stub: { area: string; importance: number }[];
//   note: string;
// }

// interface AdvancedAnalysisResult {
//   aspect_ratio: number | null;
//   brightness: number;
//   contrast: number;
//   entropy?: number | null;
// }


// type AnalysisResults = ImageAnalysisResults | ImageFeaturesResults | ExplainResult | AdvancedAnalysisResult; // Union type




// const BACKEND_URL = "http://localhost:8000";

// export const ImageWorkflow: React.FC<ImageWorkflowProps> = ({ file, onBack }) => {
//   const [isProcessing, setIsProcessing] = useState(false);
//   const [results, setResults] = useState<AnalysisResults | null>(null);
//   const [preview, setPreview] = useState<string | null>(null);
//   const [error, setError] = useState<string | null>(null);
//   const [selectedAnalysis, setSelectedAnalysis] = useState<string>('basic');

//   React.useEffect(() => {
//     const objectUrl = URL.createObjectURL(file);
//     setPreview(objectUrl);
//     return () => URL.revokeObjectURL(objectUrl);
//   }, [file]);

  

//   const processImage = async () => {
//     setIsProcessing(true);
//     setError(null);

//     const formData = new FormData();
//     formData.append("file", file);

    
//     let endpoint = "/process-image/";
//     if (selectedAnalysis === "features") endpoint = "/image-features/";
//     else if (selectedAnalysis === "explain") endpoint = "/image-explain/";
//     else if (selectedAnalysis === "advanced") endpoint = "/image-advanced/";

//     try {
//       const response = await fetch(`${BACKEND_URL}${endpoint}`, {
//         method: "POST",
//         body: formData
//       });
//       const data = await response.json();
//       if (data.error) {
//         setError(data.error);
//       } else {
//         setResults(data);
//       }
//     } catch (err) {
//       setError("Failed to process image");
//     }
//     setIsProcessing(false);
//   };


//   const renderAnalysisOptions = () => (
//     <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
//       <button
//         onClick={() => setSelectedAnalysis('basic')}
//         className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${
//           selectedAnalysis === 'basic' 
//             ? 'border-blue-500 bg-blue-50' 
//             : 'border-gray-200 hover:border-blue-300'
//         }`}
//       >
//         <Eye className="h-6 w-6 mb-2" />
//         <span className="font-medium">Basic Analysis</span>
//       </button>
      
//       <button
//         onClick={() => setSelectedAnalysis('features')}
//         className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${
//           selectedAnalysis === 'features' 
//             ? 'border-purple-500 bg-purple-50' 
//             : 'border-gray-200 hover:border-purple-300'
//         }`}
//       >
//         <Layers className="h-6 w-6 mb-2" />
//         <span className="font-medium">Feature Extraction</span>
//       </button>

//       <button
//         onClick={() => setSelectedAnalysis('explain')}
//         className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${
//           selectedAnalysis === 'explain' 
//             ? 'border-green-500 bg-green-50' 
//             : 'border-gray-200 hover:border-green-300'
//         }`}
//       >
//         <Wand2 className="h-6 w-6 mb-2" />
//         <span className="font-medium">Explainable AI</span>
//       </button>

//       <button
//         onClick={() => setSelectedAnalysis('advanced')}
//         className={`p-4 rounded-xl border-2 transition-all duration-200 flex flex-col items-center ${
//           selectedAnalysis === 'advanced' 
//             ? 'border-indigo-500 bg-indigo-50' 
//             : 'border-gray-200 hover:border-indigo-300'
//         }`}
//       >
//         <Microscope className="h-6 w-6 mb-2" />
//         <span className="font-medium">Advanced Analysis</span>
//       </button>
//     </div>
//   );

//   const renderFeatureResults = (features: ImageFeaturesResults) => (
//     <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm">
//       <h3 className="text-lg font-semibold mb-4 text-gray-800">Color Features</h3>
//       {features.mean_color && (
//         <div>
//           <strong>Mean Color:</strong>{" "}
//           R: {features.mean_color.R}, G: {features.mean_color.G}, B: {features.mean_color.B}
//         </div>
//       )}
//       {features.stddev_color && (
//         <div>
//           <strong>Color Std Dev:</strong>{" "}
//           R: {features.stddev_color.R}, G: {features.stddev_color.G}, B: {features.stddev_color.B}
//         </div>
//       )}
//       {features.mean_gray && <div>Mean Gray: {features.mean_gray}</div>}
//       {features.stddev_gray && <div>Std Dev Gray: {features.stddev_gray}</div>}
//       {features.histogram_preview && (
//         <div>
//           <strong>Histogram (first 32 bins):</strong> {features.histogram_preview.join(", ")}
//         </div>
//       )}
//     </div>
//   );

//   const renderExplainResults = (data: ExplainResult) => (
//     <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm">
//       <h3 className="text-lg font-semibold mb-4 text-gray-800">AI Explanation</h3>
//       <ul className="list-decimal list-inside mb-4 text-gray-700">
//         {data.explanation_steps.map((step, idx) => (
//           <li key={idx}>{step}</li>
//         ))}
//       </ul>
//       <h4 className="font-medium mb-2 text-gray-800">SHAP Importance by Area:</h4>
//       <ul className="list-disc list-inside text-gray-700">
//         {data.shap_value_stub.map((item, idx) => (
//           <li key={idx}><strong>{item.area}:</strong> {item.importance}</li>
//         ))}
//       </ul>
//       <p className="mt-4 italic text-gray-600">{data.note}</p>
//     </div>
//   );

//   const renderAdvancedResults = (data: AdvancedAnalysisResult) => (
//     <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm space-y-2 text-gray-700">
//       <h3 className="text-lg font-semibold mb-4 text-gray-800">Advanced Analysis</h3>
//       <div><strong>Aspect Ratio:</strong> {data.aspect_ratio ?? "N/A"}</div>
//       <div><strong>Brightness:</strong> {data.brightness}</div>
//       <div><strong>Contrast:</strong> {data.contrast}</div>
//       {data.entropy !== undefined && data.entropy !== null && (
//         <div><strong>Entropy:</strong> {data.entropy}</div>
//       )}
//     </div>
//   );


//   function isExplainResult(data : any): data is ExplainResult {
//     return data && Array.isArray(data.explanation_steps) && Array.isArray(data.shap_value_stub);
//   }

//   function isAdvancedResult(data : any): data is AdvancedAnalysisResult {
//     return data && ('aspect_ratio' in data) && ('brightness' in data) && ('contrast' in data);
//   }


//   const renderResults = () => {
//     if (!results) return null;

//     if (selectedAnalysis === "features") {
//       return renderFeatureResults(results as ImageFeaturesResults);
//     }

//     if (selectedAnalysis === 'explain' && isExplainResult(results)) {
//       return renderExplainResults(results);
//     }

//     if (selectedAnalysis === 'advanced' && isAdvancedResult(results)) {
//       return renderAdvancedResults(results);
//     }


//     return (
//       <div className="space-y-6">
//         <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
//           <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm">
//             <h3 className="text-lg font-semibold mb-4 text-gray-800">Image Properties</h3>
//             <div className="space-y-3">
//               <div className="flex justify-between items-center">
//                 <span className="text-gray-600">Dimensions</span>
//                 <span className="font-medium">{results.dimensions.width} × {results.dimensions.height}</span>
//               </div>
//               <div className="flex justify-between items-center">
//                 <span className="text-gray-600">Format</span>
//                 <span className="font-medium">{results.format}</span>
//               </div>
//               <div className="flex justify-between items-center">
//                 <span className="text-gray-600">Color Mode</span>
//                 <span className="font-medium">{results.mode}</span>
//               </div>
//             </div>
//           </div>

//           {results.extracted_text && (
//             <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm col-span-2">
//               <h3 className="text-lg font-semibold mb-4 text-gray-800">Extracted Text</h3>
//               <p className="text-gray-700 whitespace-pre-wrap">{results.extracted_text}</p>
//             </div>
//           )}
//         </div>

//         {selectedAnalysis === 'explain' && (
//           <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-white/30 shadow-sm">
//             <h3 className="text-lg font-semibold mb-4 text-gray-800">AI Explanation</h3>
//             <p className="text-gray-700 mb-4">
//               The model's analysis process can be broken down into these steps:
//             </p>
//             <ol className="list-decimal list-inside space-y-2 text-gray-700">
//               <li>Image preprocessing and normalization</li>
//               <li>Feature extraction using convolutional layers</li>
//               <li>Text detection and OCR processing</li>
//               <li>Final analysis and result compilation</li>
//             </ol>
//           </div>
//         )}
//       </div>
//     );
//   };

//   return (
//     <div className="max-w-6xl mx-auto">
//       <div className="flex items-center mb-8">
//         <button
//           onClick={onBack}
//           className="flex items-center px-4 py-2 bg-white/50 hover:bg-white/80 rounded-xl border border-white/30 transition-all duration-200 mr-6"
//         >
//           <ArrowLeft className="h-4 w-4 mr-2" />
//           Back
//         </button>
//         <div>
//           <h1 className="text-3xl font-bold text-gray-800">Image Analysis</h1>
//           <p className="text-gray-600">Processing: {file.name}</p>
//         </div>
//       </div>

//       <div className="space-y-8">
//         <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
//           {preview && (
//             <div className="mb-8">
//               <div className="max-w-2xl mx-auto rounded-lg overflow-hidden shadow-lg">
//                 <img
//                   src={preview}
//                   alt="Preview"
//                   className="w-full h-auto"
//                 />
//               </div>
//             </div>
//           )}

//           {renderAnalysisOptions()}

//           <div className="text-center">
//             <button
//               onClick={processImage}
//               disabled={isProcessing}
//               className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
//             >
//               {isProcessing ? (
//                 <span className="flex items-center">
//                   <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
//                     <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
//                     <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
//                   </svg>
//                   Processing...
//                 </span>
//               ) : 'Analyze Image'}
//             </button>

//             {error && (
//               <div className="mt-4 text-red-500 bg-red-50 p-4 rounded-lg">
//                 {error}
//               </div>
//             )}
//           </div>

//           {results && renderResults()}
//         </div>
//       </div>
//     </div>
//   );
// };


