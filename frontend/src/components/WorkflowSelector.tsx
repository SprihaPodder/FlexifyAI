import React from 'react';
import { BarChart3, Brain, ChevronRight, FileText } from 'lucide-react';
import { WorkflowType } from '../App';

interface WorkflowSelectorProps {
  fileName: string;
  onWorkflowSelect: (workflow: WorkflowType) => void;
}

export const WorkflowSelector: React.FC<WorkflowSelectorProps> = ({
  fileName,
  onWorkflowSelect,
}) => {
  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-12">
        <div className="inline-flex items-center bg-white/60 backdrop-blur-sm rounded-full px-6 py-3 border border-white/30 mb-6">
          <FileText className="h-5 w-5 text-green-600 mr-2" />
          <span className="font-medium text-gray-700">Dataset uploaded:</span>
          <span className="font-bold text-gray-800 ml-2">{fileName}</span>
        </div>
        <h2 className="text-4xl font-bold text-gray-800 mb-4">
          Choose Your Analysis Path
        </h2>
        <p className="text-xl text-gray-600">
          Select how you'd like to analyze your dataset
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* EDA Option */}
        <div
          onClick={() => onWorkflowSelect('eda')}
          className="group relative bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30 cursor-pointer transition-all duration-300 hover:bg-white/90 hover:shadow-2xl hover:scale-105 hover:-translate-y-2"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-green-500/10 to-emerald-600/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          
          <div className="relative">
            <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <BarChart3 className="h-8 w-8 text-white" />
            </div>
            
            <h3 className="text-2xl font-bold text-gray-800 mb-4">
              Exploratory Data Analysis
            </h3>
            
            <p className="text-gray-600 mb-6 leading-relaxed">
              Dive deep into your data with comprehensive statistical analysis, visualizations, and insights. Perfect for understanding data patterns and distributions.
            </p>
            
            <ul className="space-y-2 mb-8">
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                Column type specification (categorical/numerical)
              </li>
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                Interactive data visualizations
              </li>
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                Statistical summaries and correlations
              </li>
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-green-500 rounded-full mr-3"></div>
                Downloadable PDF report
              </li>
            </ul>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-green-600 bg-green-50 px-3 py-1 rounded-full">
                Data Exploration
              </span>
              <ChevronRight className="h-6 w-6 text-gray-400 group-hover:text-green-600 group-hover:translate-x-1 transition-all duration-200" />
            </div>
          </div>
        </div>

        {/* ML Option */}
        <div
          onClick={() => onWorkflowSelect('ml')}
          className="group relative bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30 cursor-pointer transition-all duration-300 hover:bg-white/90 hover:shadow-2xl hover:scale-105 hover:-translate-y-2"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 to-indigo-600/10 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
          
          <div className="relative">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-2xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
              <Brain className="h-8 w-8 text-white" />
            </div>
            
            <h3 className="text-2xl font-bold text-gray-800 mb-4">
              Machine Learning Training
            </h3>
            
            <p className="text-gray-600 mb-6 leading-relaxed">
              Train and evaluate machine learning models with advanced preprocessing, hyperparameter tuning, and comprehensive performance metrics.
            </p>
            
            <ul className="space-y-2 mb-8">
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full mr-3"></div>
                Multiple ML algorithms to choose from
              </li>
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full mr-3"></div>
                Advanced data preprocessing pipeline
              </li>
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full mr-3"></div>
                Hyperparameter optimization
              </li>
              <li className="flex items-center text-sm text-gray-600">
                <div className="w-2 h-2 bg-purple-500 rounded-full mr-3"></div>
                Model evaluation and downloadable results
              </li>
            </ul>
            
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-purple-600 bg-purple-50 px-3 py-1 rounded-full">
                Model Training
              </span>
              <ChevronRight className="h-6 w-6 text-gray-400 group-hover:text-purple-600 group-hover:translate-x-1 transition-all duration-200" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};