import React, { useState } from 'react';
import { ArrowLeft, Plus, Trash2, Play, Download, BarChart3 } from 'lucide-react';
import { useEffect } from 'react';

interface Column {
  id: string;
  name: string;
  type: 'categorical' | 'numerical';
}

interface EDAWorkflowProps {
  file: File;
  onBack: () => void;
}

// const BACKEND_URL = "https://flexifyai.onrender.com"; 
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '0.0.0.0';
const BACKEND_URL = isLocal ? 'http://0.0.0.0:10000' : 'https://flexifyai.onrender.com';


export const EDAWorkflow: React.FC<EDAWorkflowProps> = ({ file, onBack }) => {
  const [columns, setColumns] = useState<Column[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [edaImage, setEdaImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;
    async function fetchColumnsInfo() {
      setIsAnalyzing(true); 
      setError(null);
      setColumns([]); 
      try {
        const formData = new FormData();
        formData.append("file", file);
        
        const resp = await fetch(`${BACKEND_URL}/analyze-csv/`, {
          method: "POST",
          body: formData,
        });
        const data = await resp.json();
        if (data.error) {
          setError(data.error); setIsAnalyzing(false); return;
        }
       
        setColumns(data.columns.map(
          (col: {name: string, type: string}, idx: number) => ({
            id: `${col.name}-${idx}-${Math.random().toString(36).substring(2, 7)}`,
            name: col.name,
            type: col.type === 'numerical' ? 'numerical' : 'categorical'
          })
        ));
      } catch (err) {
        setError("Could not analyze CSV file columns.");  
      }
      setIsAnalyzing(false);
    }
    fetchColumnsInfo();
  }, [file]);

  const addColumn = () => {
    const newColumn: Column = {
      id: Math.random().toString(36).substr(2, 9),
      name: '',
      type: 'numerical'
    };
    setColumns([...columns, newColumn]);
  };

  const updateColumn = (id: string, field: 'name' | 'type', value: string) => {
    setColumns(columns.map(col =>
      col.id === id ? { ...col, [field]: value } : col
    ));
  };

  const removeColumn = (id: string) => {
    setColumns(columns.filter(col => col.id !== id));
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    setError(null);
    setEdaImage(null);

   
    const columnsInfo = columns
      .filter(col => col.name.trim() !== '')
      .map(col => ({ name: col.name.trim(), type: col.type }));

    if (columnsInfo.length === 0) {
      setError("Please specify at least one column.");
      setIsAnalyzing(false);
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("columns", JSON.stringify(columnsInfo));

    try {
      const response = await fetch(`${BACKEND_URL}/eda/`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
        setIsAnalyzing(false);
        return;
      }
      setEdaImage(data.image || null);
      
      setResults({
        summary: {
          totalRows: data.totalRows ?? 0,
          totalColumns: columns.length,
          missingValues: data.missingValues ?? 0,
          duplicateRows: data.duplicateRows ?? 0
        },
        correlations: [],
        distributions: []
      });
      setAnalysisComplete(true);
    } catch (err) {
      setError("Failed to analyze data. Please try again.");
    }
    setIsAnalyzing(false);

  };

  const downloadReport = () => {
    
    const blob = new Blob(['EDA Report Content'], { type: 'application/pdf' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file.name.split('.')[0]}_eda_report.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex items-center mb-8">
        <button
          onClick={onBack}
          className="flex items-center px-4 py-2 bg-white/50 hover:bg-white/80 rounded-xl border border-white/30 transition-all duration-200 mr-6"
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </button>
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Exploratory Data Analysis</h1>
          <p className="text-gray-600">Analyzing: {file.name}</p>
        </div>
      </div>

      {!analysisComplete ? (
        <div className="space-y-8">
          {/* Column Configuration */}
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Configure Columns</h2>
              <button
                onClick={addColumn}
                className="flex items-center px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-600 text-white rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all duration-200"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Column
              </button>
            </div>

            {columns.length === 0 ? (
              <div className="text-center py-12">
                <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Add columns to start your analysis</p>
              </div>
            ) : (
              <div className="space-y-4">
                {columns.map((column) => (
                  <div key={column.id} className="flex items-center space-x-4 p-4 bg-white/50 rounded-xl border border-white/30">
                    <input
                      type="text"
                      placeholder="Column name"
                      value={column.name}
                      onChange={(e) => updateColumn(column.id, 'name', e.target.value)}
                      className="flex-1 px-4 py-2 bg-white/70 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    />
                    <select
                      value={column.type}
                      onChange={(e) => updateColumn(column.id, 'type', e.target.value)}
                      className="px-4 py-2 bg-white/70 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                    >
                      <option value="numerical">Numerical</option>
                      <option value="categorical">Categorical</option>
                    </select>
                    <button
                      onClick={() => removeColumn(column.id)}
                      className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors duration-200"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                ))}
              </div>
            )}
            {error && (
              <div className="mt-4 text-red-500 font-medium">{error}</div>
            )}
          </div>

          {/* Run Analysis Button */}
          {columns.length > 0 && (
            <div className="text-center">
              <button
                onClick={runAnalysis}
                disabled={isAnalyzing || columns.some(col => !col.name)}
                className="px-8 py-4 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                {isAnalyzing ? (
                  <>
                    <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing Data...
                  </>
                ) : (
                  <>
                    <Play className="inline h-4 w-4 mr-2" />
                    Run EDA Analysis
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      ) : (
        /* Results Display */
        <div className="space-y-8">
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Analysis Results</h2>
              <button
                onClick={downloadReport}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-lg"
              >
                <Download className="h-4 w-4 mr-2" />
                Download PDF Report
              </button>
            </div>

            {/* EDA Image from backend */}
            {edaImage && (
              <div className="flex justify-center mb-8">
                <img
                  src={`data:image/png;base64,${edaImage}`}
                  alt="EDA Visualization"
                  className="rounded-xl shadow-lg max-w-full"
                  style={{ maxHeight: 400 }}
                />
              </div>
            )}

            {/* Optionally, show summary stats if backend returns them */}
            {results && (
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
                  <h3 className="font-semibold text-blue-800 mb-2">Total Rows</h3>
                  <p className="text-3xl font-bold text-blue-600">{results.summary.totalRows.toLocaleString()}</p>
                </div>
                <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
                  <h3 className="font-semibold text-green-800 mb-2">Total Columns</h3>
                  <p className="text-3xl font-bold text-green-600">{results.summary.totalColumns}</p>
                </div>
                <div className="bg-gradient-to-br from-yellow-50 to-yellow-100 p-6 rounded-xl">
                  <h3 className="font-semibold text-yellow-800 mb-2">Missing Values</h3>
                  <p className="text-3xl font-bold text-yellow-600">{results.summary.missingValues}</p>
                </div>
                <div className="bg-gradient-to-br from-red-50 to-red-100 p-6 rounded-xl">
                  <h3 className="font-semibold text-red-800 mb-2">Duplicate Rows</h3>
                  <p className="text-3xl font-bold text-red-600">{results.summary.duplicateRows}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
