
import { ArrowLeft, Settings, Target, RefreshCw, Download, Brain, CheckCircle } from 'lucide-react';
import React, { useState, useEffect } from 'react';

interface MLWorkflowProps {
  file: File;
  onBack: () => void;
}

type MLModel =
  | 'decision_tree'
  | 'random_forest'
  | 'logistic_regression'
  | 'svm'
  | 'gradient_boosting'
  | 'knn';

type Step = 'model' | 'preprocessing' | 'training' | 'results';

// const BACKEND_URL = 'https://flexifyai.onrender.com';
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '0.0.0.0';
const BACKEND_URL = isLocal ? 'http://0.0.0.0:10000' : 'https://flexifyai.onrender.com';


export const MLWorkflow: React.FC<MLWorkflowProps> = ({ file, onBack }) => {
  const [currentStep, setCurrentStep] = useState<Step>('model');
  const [selectedModel, setSelectedModel] = useState<MLModel>('random_forest');
  const [hyperparameters, setHyperparameters] = useState<Record<string, any>>({});
  const [targetColumn, setTargetColumn] = useState('');
  const [preprocessingOptions, setPreprocessingOptions] = useState({
    handleMissing: 'drop',
    handleOutliers: 'none',
    scaleFeatures: true,
    removeCorrelated: true
  });
  const [isTraining, setIsTraining] = useState(false);
  const [trainingComplete, setTrainingComplete] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [columns, setColumns] = useState<string[]>([]);

  useEffect(() => {
    if (!file) return;

    async function fetchColumns() {
      try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch(`${BACKEND_URL}/analyze-csv/`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.error) {
          setError(data.error);
          return;
        }
        if (data.columns && Array.isArray(data.columns)) {
          
          setColumns(data.columns.map((col: {name: string}) => col.name));
        }
      } catch (err) {
        setError("Failed to fetch columns for target selection.");
      }
    }

    fetchColumns();
  }, [file]);
  
 
  const [comparisonResults, setComparisonResults] = useState<any[]>([]);
  const [isComparing, setIsComparing] = useState(false);

  const modelConfigs: Record<MLModel, {
    name: string;
    icon: string;
    hyperparameters: string[];
  }> = {
    decision_tree: {
      name: 'Decision Tree',
      icon: '🌳',
      hyperparameters: ['max_depth', 'min_samples_split']
    },
    random_forest: {
      name: 'Random Forest',
      icon: '🌲',
      hyperparameters: ['n_estimators', 'max_depth', 'min_samples_split']
    },
    logistic_regression: {
      name: 'Logistic Regression',
      icon: '📈',
      hyperparameters: ['C', 'max_iter']
    },
    svm: {
      name: 'Support Vector Machine',
      icon: '🎯',
      hyperparameters: ['C', 'kernel', 'gamma']
    },
    gradient_boosting: {
      name: 'Gradient Boosting',
      icon: '⚡',
      hyperparameters: ['n_estimators', 'learning_rate', 'max_depth']
    },
    knn: {
      name: 'K-Nearest Neighbors',
      icon: '👥',
      hyperparameters: ['n_neighbors']
    }
  };

  const stepTitles = {
    model: 'Model Selection',
    preprocessing: 'Data Preprocessing',
    training: 'Training Configuration',
    results: 'Training Results'
  };

  const startTraining = async () => {
    setIsTraining(true);
    setError(null);
    setResults(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', selectedModel);
    formData.append('target_column', targetColumn);
    formData.append('model_params', JSON.stringify(hyperparameters));
    formData.append('handle_missing', preprocessingOptions.handleMissing);
    formData.append('handle_outliers', preprocessingOptions.handleOutliers);

    try {
      const response = await fetch(`${BACKEND_URL}/upload/`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
        setIsTraining(false);
        return;
      }
      setResults(data);
      setIsTraining(false);
      setTrainingComplete(true);
      setCurrentStep('results');

     
      const toSave = {
        key: Math.random().toString(36),
        model: selectedModel,
        name: modelConfigs[selectedModel].name,
        icon: modelConfigs[selectedModel].icon,
        params: { ...hyperparameters },
        results: data
      };
      if (isComparing) {
        setComparisonResults(prev => [...prev, toSave]);
      } else {
        setComparisonResults([toSave]);
      }
      setIsComparing(false);
    } catch (err) {
      setError('Failed to train model. Please try again.');
      setIsTraining(false);
    }
  };

  const downloadResults = () => {
   
    const blob = new Blob(['ML Training Results'], { type: 'application/pdf' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${file.name.split('.')[0]}_ml_results.pdf`;
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
          <h1 className="text-3xl font-bold text-gray-800">Machine Learning Training</h1>
          <p className="text-gray-600">Training on: {file.name}</p>
        </div>
      </div>

      {/* Progress Steps */}
      <div className="mb-8">
        <div className="flex items-center justify-between bg-white/70 backdrop-blur-sm rounded-2xl p-6 border border-white/30">
          {Object.entries(stepTitles).map(([step, title], index) => (
            <div key={step} className="flex items-center">
              <div className={`flex items-center justify-center w-10 h-10 rounded-full ${
                currentStep === step 
                  ? 'bg-gradient-to-r from-purple-500 to-indigo-600 text-white' 
                  : index < Object.keys(stepTitles).indexOf(currentStep)
                  ? 'bg-green-500 text-white'
                  : 'bg-gray-200 text-gray-500'
              }`}>
                {index < Object.keys(stepTitles).indexOf(currentStep) ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <span className="text-sm font-semibold">{index + 1}</span>
                )}
              </div>
              <span className={`ml-3 font-medium ${
                currentStep === step ? 'text-purple-600' : 'text-gray-600'
              }`}>
                {title}
              </span>
              {index < Object.keys(stepTitles).length - 1 && (
                <div className="w-20 h-0.5 bg-gray-300 mx-6"></div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Step Content */}
      {currentStep === 'model' && (
        <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Select ML Algorithm</h2>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(modelConfigs).map(([key, config]) => (
              <div
                key={key}
                onClick={() => setSelectedModel(key as MLModel)}
                className={`p-6 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
                  selectedModel === key
                    ? 'border-purple-500 bg-purple-50/50 shadow-lg'
                    : 'border-gray-200 bg-white/50 hover:border-purple-300'
                }`}
              >
                <div className="text-3xl mb-3">{config.icon}</div>
                <h3 className="text-lg font-bold text-gray-800 mb-2">{config.name}</h3>
                <p className="text-sm text-gray-600">
                  Parameters: {config.hyperparameters.join(', ')}
                </p>
              </div>
            ))}
          </div>
          <div className="mt-8 text-center">
            <button
              onClick={() => setCurrentStep('preprocessing')}
              className="px-8 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold rounded-xl hover:from-purple-600 hover:to-indigo-700 transition-all duration-200"
            >
              Next: Configure Preprocessing
            </button>
          </div>
        </div>
      )}

      {currentStep === 'preprocessing' && (
        <div className="space-y-8">
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Data Preprocessing</h2>
            <div className="grid md:grid-cols-2 gap-8">
              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Missing Values</h3>
                <div className="space-y-3">
                  {[
                    { value: 'drop', label: 'Drop rows with missing values' },
                    { value: 'mean', label: 'Impute with mean/mode' }
                  ].map(option => (
                    <label key={option.value} className="flex items-center">
                      <input
                        type="radio"
                        name="missing"
                        value={option.value}
                        checked={preprocessingOptions.handleMissing === option.value}
                        onChange={(e) => setPreprocessingOptions({
                          ...preprocessingOptions,
                          handleMissing: e.target.value
                        })}
                        className="mr-3"
                      />
                      {option.label}
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-4">Outlier Detection</h3>
                <div className="space-y-3">
                  {[
                    { value: 'none', label: 'No outlier handling' },
                    { value: 'iqr', label: 'IQR method' },
                    { value: 'zscore', label: 'Z-score method' }
                  ].map(option => (
                    <label key={option.value} className="flex items-center">
                      <input
                        type="radio"
                        name="outliers"
                        value={option.value}
                        checked={preprocessingOptions.handleOutliers === option.value}
                        onChange={(e) => setPreprocessingOptions({
                          ...preprocessingOptions,
                          handleOutliers: e.target.value
                        })}
                        className="mr-3"
                      />
                      {option.label}
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>
          <div className="flex justify-between">
            <button
              onClick={() => setCurrentStep('model')}
              className="px-6 py-3 bg-white/50 hover:bg-white/80 rounded-xl border border-white/30 transition-all duration-200"
            >
              Previous
            </button>
            <button
              onClick={() => setCurrentStep('training')}
              className="px-8 py-3 bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold rounded-xl hover:from-purple-600 hover:to-indigo-700 transition-all duration-200"
            >
              Next: Training Setup
            </button>
          </div>
        </div>
      )}

      {currentStep === 'training' && (
        <div className="space-y-8">
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Training Configuration</h2>
            <div className="grid md:grid-cols-2 gap-8 mb-8">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Select Target Column
                </label>
                <div className="flex flex-wrap gap-4">
                  {columns.length === 0 && (
                    <p className="text-gray-500">Loading columns ...</p>
                  )}
                  {columns.map((col) => (
                    <label
                      key={col}
                      className="flex items-center space-x-2 cursor-pointer"
                    >
                      <input
                        type="radio"
                        name="targetColumn"
                        value={col}
                        checked={targetColumn === col}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        className="form-radio text-purple-600"
                      />
                      <span>{col}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Selected Model
                </label>
                <div className="flex items-center p-3 bg-purple-50 rounded-lg border border-purple-200">
                  <span className="text-2xl mr-3">{modelConfigs[selectedModel].icon}</span>
                  <span className="font-semibold text-purple-800">{modelConfigs[selectedModel].name}</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-700 mb-4">Hyperparameters</h3>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                {modelConfigs[selectedModel].hyperparameters.map((param) => (
                  <div key={param}>
                    <label className="block text-sm font-medium text-gray-600 mb-1">
                      {param.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </label>
                    <input
                      type="text"
                      placeholder="Auto"
                      value={hyperparameters[param] || ''}
                      onChange={(e) => setHyperparameters({
                        ...hyperparameters,
                        [param]: e.target.value
                      })}
                      className="w-full px-3 py-2 bg-white/70 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent text-sm"
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="text-center">
            <button
              onClick={startTraining}
              disabled={isTraining || !targetColumn}
              className="px-12 py-4 bg-gradient-to-r from-purple-500 to-indigo-600 text-white font-semibold rounded-xl hover:from-purple-600 hover:to-indigo-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              {isTraining ? (
                <>
                  <div className="inline-block animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                  Training Model...
                </>
              ) : (
                <>
                  <Brain className="inline h-5 w-5 mr-3" />
                  Start Training
                </>
              )}
            </button>
          </div>
          {isTraining && (
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
              <div className="text-center">
                <div className="mb-6">
                  <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-full mb-4">
                    <RefreshCw className="h-10 w-10 text-white animate-spin" />
                  </div>
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-2">Training in Progress</h3>
                <p className="text-gray-600 mb-6">
                  Preprocessing data and training your {modelConfigs[selectedModel].name} model...
                </p>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-gradient-to-r from-purple-500 to-indigo-600 h-2 rounded-full animate-pulse" style={{width: '60%'}}></div>
                </div>
              </div>
            </div>
          )}
          {error && (
            <div className="mt-4 text-red-500 font-medium text-center">{error}</div>
          )}
        </div>
      )}

      {currentStep === 'results' && trainingComplete && (
        <div>
          {/* Show current training result */}
          {results && (
            <div className="space-y-8">
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-gray-800">Training Results</h2>
                  <button
                    onClick={downloadResults}
                    className="flex items-center px-6 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-200 shadow-lg"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Report
                  </button>
                </div>
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                  <div className="bg-gradient-to-br from-green-50 to-green-100 p-6 rounded-xl">
                    <h3 className="font-semibold text-green-800 mb-2">Accuracy</h3>
                    <p className="text-3xl font-bold text-green-600">{results.accuracy ? `${results.accuracy.toFixed(2)}%` : '-'}</p>
                  </div>
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
                    <h3 className="font-semibold text-blue-800 mb-2">Std. Dev</h3>
                    <p className="text-3xl font-bold text-blue-600">{results.std_dev ? `${results.std_dev.toFixed(2)}%` : '-'}</p>
                  </div>
                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl">
                    <h3 className="font-semibold text-purple-800 mb-2">Rows Before</h3>
                    <p className="text-3xl font-bold text-purple-600">{results.rows_before ?? '-'}</p>
                  </div>
                  <div className="bg-gradient-to-br from-orange-50 to-orange-100 p-6 rounded-xl">
                    <h3 className="font-semibold text-orange-800 mb-2">Rows After</h3>
                    <p className="text-3xl font-bold text-orange-600">{results.rows_after ?? '-'}</p>
                  </div>
                </div>
                <div className="grid lg:grid-cols-2 gap-8">
                  <div className="bg-white/50 p-6 rounded-xl">
                    <h3 className="text-xl font-bold text-gray-800 mb-4">Data Cleaning Summary</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-white/70 rounded-lg">
                        <span>Rows Removed</span>
                        <span className="font-semibold">{results.rows_removed ?? '-'}</span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/70 rounded-lg">
                        <span>Columns</span>
                        <span className="font-semibold text-green-600">{results.columns ?? '-'}</span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/70 rounded-lg">
                        <span>Missing Handling</span>
                        <span className="font-semibold">{results.handle_missing ?? '-'}</span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/70 rounded-lg">
                        <span>Outlier Handling</span>
                        <span className="font-semibold">{results.handle_outliers ?? '-'}</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-white/50 p-6 rounded-xl">
                    <h3 className="text-xl font-bold text-gray-800 mb-4">Model & Params</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center p-3 bg-white/70 rounded-lg">
                        <span>Model Used</span>
                        <span className="font-semibold">{results.model_used ?? '-'}</span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/70 rounded-lg">
                        <span>Params</span>
                        <span className="font-semibold text-xs break-all">{JSON.stringify(results.params_used) ?? '-'}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <div className="mt-8 p-6 bg-gradient-to-br from-purple-50 to-indigo-50 rounded-xl border border-purple-200">
                  <h3 className="text-lg font-bold text-purple-800 mb-2">Model Performance Summary</h3>
                  <p className="text-purple-700">
                    Your {modelConfigs[selectedModel].name} model achieved an accuracy of <strong>{results.accuracy ? `${results.accuracy.toFixed(2)}%` : '-'}</strong>
                    with a standard deviation of <strong>{results.std_dev ? `${results.std_dev.toFixed(2)}%` : '-'}</strong>.
                  </p>
                </div>
                {/* Trigger another model comparison */}
                <div className="text-center mt-8">
                  <button
                    className="px-8 py-3 bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-semibold rounded-xl hover:from-indigo-600 hover:to-purple-700 transition-all duration-200"
                    onClick={() => {
                      setIsComparing(true);
                      setCurrentStep('model');
                      setTrainingComplete(false);
                      setResults(null);
                    }}
                  >
                    Train Using Another Model To Compare
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Comparison Table */}
          {comparisonResults.length > 1 && (
            <div className="mt-12 bg-white/80 p-8 rounded-2xl border border-purple-200 shadow-md">
              <h2 className="text-2xl font-bold text-purple-800 mb-6">Model Comparison</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full table-auto text-left">
                  <thead>
                    <tr>
                      <th className="p-3 font-semibold">Model</th>
                      <th className="p-3 font-semibold">Accuracy</th>
                      <th className="p-3 font-semibold">Std. Dev</th>
                      <th className="p-3 font-semibold">Params</th>
                      <th className="p-3 font-semibold">Rows Before</th>
                      <th className="p-3 font-semibold">Rows After</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonResults.map((cr) => (
                      <tr key={cr.key}>
                        <td className="p-3">{cr.icon} {cr.name}</td>
                        <td className="p-3">{cr.results.accuracy ? `${cr.results.accuracy.toFixed(2)}%` : '-'}</td>
                        <td className="p-3">{cr.results.std_dev ? `${cr.results.std_dev.toFixed(2)}%` : '-'}</td>
                        <td className="p-3 text-xs break-all">{JSON.stringify(cr.params)}</td>
                        <td className="p-3">{cr.results.rows_before ?? '-'}</td>
                        <td className="p-3">{cr.results.rows_after ?? '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};



