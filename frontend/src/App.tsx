import React, { useState } from 'react';
import { FileUpload } from './components/FileUpload';
import { WorkflowSelector } from './components/WorkflowSelector';
import { EDAWorkflow } from './components/EDAWorkflow';
import { MLWorkflow } from './components/MLWorkflow';
import { Header } from './components/Header';
import { Dataset, WorkflowType } from '../types';
import { TextWorkflow } from './components/TextWorkflow';
import { ImageWorkflow } from './components/ImageWorkflow';

function App() {
  const [uploadedDataset, setUploadedDataset] = useState<Dataset | null>(null);
  const [selectedWorkflow, setSelectedWorkflow] = useState<WorkflowType>(null);

  const handleFileUpload = (dataset: Dataset) => {
    setUploadedDataset(dataset);
    setSelectedWorkflow(null);
  };

  const handleWorkflowSelect = (workflow: WorkflowType) => {
    setSelectedWorkflow(workflow);
  };

  const handleReset = () => {
    setUploadedDataset(null);
    setSelectedWorkflow(null);
  };

 
  const shouldShowWorkflow = uploadedDataset?.type === 'csv';

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 relative overflow-hidden">
      {/* Abstract Background Elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-gradient-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl"></div>
        <div className="absolute -bottom-40 -left-40 w-96 h-96 bg-gradient-to-tr from-indigo-400/20 to-cyan-600/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-r from-purple-400/10 to-pink-600/10 rounded-full blur-2xl"></div>
      </div>

      <div className="relative z-10">
        <Header onReset={handleReset} />
        
        <main className="container mx-auto px-4 py-8">
          {!uploadedDataset ? (
            <FileUpload onFileUpload={handleFileUpload} />
          ) : uploadedDataset.type === 'csv' && !selectedWorkflow ? (
            <WorkflowSelector
              fileName={uploadedDataset.file.name}
              onWorkflowSelect={handleWorkflowSelect}
            />
          ) : uploadedDataset.type === 'csv' && selectedWorkflow === 'eda' ? (
            <EDAWorkflow file={uploadedDataset.file} onBack={() => setSelectedWorkflow(null)} />
          ) : uploadedDataset.type === 'csv' && selectedWorkflow === 'ml' ? (
            <MLWorkflow file={uploadedDataset.file} onBack={() => setSelectedWorkflow(null)} />
          ) : uploadedDataset.type === 'text' ? (
            <TextWorkflow file={uploadedDataset.file} onBack={handleReset} />
          ) : uploadedDataset.type === 'image' ? (
            <ImageWorkflow file={uploadedDataset.file} onBack={handleReset} />
          ) : (
            <div className="text-center py-12">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">
                Unsupported File Type
              </h2>
              <p className="text-gray-600">
                The selected file type is not supported.
              </p>
              <button
                onClick={handleReset}
                className="mt-6 px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
              >
                Upload Different File
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;

