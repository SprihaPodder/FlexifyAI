export type DatasetType = 'csv' | 'text' | 'image';
export type WorkflowType = 'eda' | 'ml' | null;

export interface Dataset {
  file: File;
  type: DatasetType;
}

import React, { useCallback, useState } from 'react';
import { Upload, File, CheckCircle, BarChart3, Image, FileText, Table } from 'lucide-react';
import { DatasetType, Dataset } from '../types';

interface FileUploadProps {
  onFileUpload: (dataset: Dataset) => void;
}

interface DatasetOption {
  type: DatasetType;
  label: string;
  icon: React.ReactNode;
  accept: string;
  description: string;
}

export const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedType, setSelectedType] = useState<DatasetType>('csv');
  const [error, setError] = useState<string | null>(null);

  const datasetOptions: DatasetOption[] = [
    {
      type: 'csv',
      label: 'Tabular Data (CSV)',
      icon: <Table className="h-8 w-8" />,
      accept: '.csv',
      description: 'Upload structured data in CSV format for ML and EDA'
    },
    {
      type: 'text',
      label: 'Text Data',
      icon: <FileText className="h-8 w-8" />,
      accept: '.txt,.doc,.docx',
      description: 'Upload text documents for NLP analysis'
    },
    {
      type: 'image',
      label: 'Image Data',
      icon: <Image className="h-8 w-8" />,
      accept: '.jpg,.jpeg,.png',
      description: 'Upload images for computer vision tasks'
    }
  ];

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const file = files[0];
      if (isValidFile(file)) {
        handleFileUpload(file);
      } else {
        setError(`Invalid file type. Please upload ${getAcceptedFileTypes()}`);
      }
    }
  }, [selectedType]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (isValidFile(file)) {
        handleFileUpload(file);
      } else {
        setError(`Invalid file type. Please upload ${getAcceptedFileTypes()}`);
      }
    }
  }, [selectedType]);

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setError(null);
    
    try {
      
      await new Promise(resolve => setTimeout(resolve, 1000)); 
      onFileUpload({ file, type: selectedType });
    } catch (err) {
      setError('Failed to process file. Please try again.');
    }
    
    setIsUploading(false);
  };

  const isValidFile = (file: File) => {
    const option = datasetOptions.find(opt => opt.type === selectedType);
    if (!option) return false;
    
    const validExtensions = option.accept.split(',');
    return validExtensions.some(ext => 
      file.name.toLowerCase().endsWith(ext.replace('.', '').toLowerCase())
    );
  };

  const getAcceptedFileTypes = () => {
    const option = datasetOptions.find(opt => opt.type === selectedType);
    return option?.accept || '';
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <h2 className="text-4xl font-bold text-gray-800 mb-4">
          Upload Your Dataset
        </h2>
        <p className="text-xl text-gray-600">
          Start your data science journey by uploading your dataset
        </p>
      </div>

      {/* Dataset Type Selection */}
      <div className="grid md:grid-cols-3 gap-6 mb-8">
        {datasetOptions.map((option) => (
          <div
            key={option.type}
            onClick={() => setSelectedType(option.type)}
            className={`p-6 rounded-xl border-2 cursor-pointer transition-all duration-200 ${
              selectedType === option.type
                ? 'border-blue-500 bg-blue-50/50 shadow-lg'
                : 'border-gray-200 bg-white/50 hover:border-blue-300'
            }`}
          >
            <div className="flex flex-col items-center text-center">
              <div className={`mb-4 ${
                selectedType === option.type ? 'text-blue-500' : 'text-gray-400'
              }`}>
                {option.icon}
              </div>
              <h3 className="font-bold mb-2">{option.label}</h3>
              <p className="text-sm text-gray-600">{option.description}</p>
            </div>
          </div>
        ))}
      </div>

      <div
        className={`relative border-2 border-dashed rounded-2xl p-12 text-center transition-all duration-300 ${
          isDragOver
            ? 'border-blue-500 bg-blue-50/50 scale-105'
            : 'border-gray-300 bg-white/50 hover:bg-white/70'
        } ${isUploading ? 'pointer-events-none' : ''}`}
        onDrop={handleDrop}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={(e) => {
          e.preventDefault();
          setIsDragOver(false);
        }}
      >
        {isUploading ? (
          <div className="flex flex-col items-center space-y-4">
            <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500"></div>
            <p className="text-lg font-medium text-gray-700">Processing your dataset...</p>
          </div>
        ) : (
          <>
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mb-4">
                <Upload className="h-10 w-10 text-white" />
              </div>
            </div>

            <h3 className="text-2xl font-bold text-gray-800 mb-2">
              Drag & drop your {selectedType.toUpperCase()} file here
            </h3>
            <p className="text-gray-600 mb-6">
              or click to browse your files
            </p>

            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept={getAcceptedFileTypes()}
              onChange={handleFileInput}
            />
            <label
              htmlFor="file-upload"
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-200 cursor-pointer shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              <File className="mr-2 h-5 w-5" />
              Select {selectedType.toUpperCase()} File
            </label>

            {error && (
              <div className="mt-4 text-red-500 font-medium">{error}</div>
            )}

            <div className="mt-8 text-sm text-gray-500">
              <p>Accepted format: {getAcceptedFileTypes()}</p>
              <p>Maximum file size: 100MB</p>
            </div>
          </>
        )}
      </div>
      {/* Features Preview */}
      <div className="mt-16 grid md:grid-cols-2 gap-8">
        <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-xl flex items-center justify-center mr-4">
              <BarChart3 className="h-6 w-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-800">Exploratory Data Analysis</h3>
          </div>
          <p className="text-gray-600">
            Generate comprehensive visualizations and statistical summaries of your dataset with customizable column types and interactive plots.
          </p>
        </div>

        <div className="bg-white/60 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
          <div className="flex items-center mb-4">
            <div className="w-12 h-12 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center mr-4">
              <CheckCircle className="h-6 w-6 text-white" />
            </div>
            <h3 className="text-xl font-bold text-gray-800">ML Model Training</h3>
          </div>
          <p className="text-gray-600">
            Train machine learning models with advanced preprocessing, hyperparameter tuning, and comprehensive performance evaluation.
          </p>
        </div>
      </div>
    </div>
  );
};






