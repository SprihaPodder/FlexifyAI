import React from 'react';
import { BarChart3, RefreshCw } from 'lucide-react';

interface HeaderProps {
  onReset: () => void;
}

export const Header: React.FC<HeaderProps> = ({ onReset }) => {
  return (
    <header className="bg-white/70 backdrop-blur-md border-b border-white/20 sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
              <BarChart3 className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Flexify.ai
              </h1>
              <p className="text-sm text-gray-600">Flex the future of Machine Learning</p>
            </div>
          </div>
          <button
            onClick={onReset}
            className="flex items-center space-x-2 px-4 py-2 bg-white/50 hover:bg-white/80 rounded-xl border border-white/30 transition-all duration-200 hover:shadow-lg"
          >
            <RefreshCw className="h-4 w-4 text-gray-600" />
            <span className="text-sm font-medium text-gray-700">Reset</span>
          </button>
        </div>
      </div>
    </header>
  );
};