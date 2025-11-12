import React, { useState } from 'react';
import { ArrowLeft, FileText, BarChart2, Brain, Book, MessageCircle } from 'lucide-react';


interface TextAnalysisResults {
  statistics: {
    words: number;
    lines: number;
    characters: number;
    sentences: number;
    detected_encoding: string;
  };
  sentiment: {
    label: string;
    score: number;
  };
  preview: string;
}

interface TextWorkflowProps {
  file: File;
  onBack: () => void;
}

interface ReadabilityResults {
  flesch_reading_ease: number;
  flesch_kincaid_grade: number;
  gunning_fog: number;
  smog_index: number;
  automated_readability_index: number;
  coleman_liau_index: number;
  dale_chall_readability_score: number;
  difficult_words: number;
  linsear_write_formula: number;
  text_standard: string;
}

interface TopicsResults {
  noun_phrases: string[];
  common_words: [string, number][];
  main_topics: string[];
}

interface EntitiesResults {
  PERSON: string[];
  ORG: string[];
  GPE: string[];
  DATE: string[];
  TIME: string[];
  MONEY: string[];
  PERCENT: string[];
}

// const BACKEND_URL = "https://flexifyai.onrender.com";
const isLocal = window.location.hostname === 'localhost' || window.location.hostname === '0.0.0.0';
const BACKEND_URL = isLocal ? 'http://0.0.0.0:10000' : 'https://flexifyai.onrender.com';


export const TextWorkflow: React.FC<TextWorkflowProps> = ({ file, onBack }) => {
  
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<TextAnalysisResults | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);
  const [readabilityResults, setReadabilityResults] = useState<ReadabilityResults | null>(null);
  const [topicsResults, setTopicsResults] = useState<TopicsResults | null>(null);
  const [entitiesResults, setEntitiesResults] = useState<EntitiesResults | null>(null);
  const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(false);

  
  const processText = async () => {
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${BACKEND_URL}/process-text/`, {
        method: "POST",
        body: formData,
        headers: {
          'Accept': 'application/json',
        }
      });
      
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResults(data);
      }
    } catch (err) {
      setError("Failed to process text file");
    }
    
    setIsProcessing(false);
  };

  const performReadabilityAnalysis = async () => {
    setIsLoadingAnalysis(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${BACKEND_URL}/analyze-readability/`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setReadabilityResults(data);
      }
    } catch (err) {
      setError("Failed to perform readability analysis");
    }
    setIsLoadingAnalysis(false);
  };

  const performTopicExtraction = async () => {
    setIsLoadingAnalysis(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${BACKEND_URL}/extract-topics/`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setTopicsResults(data);
      }
    } catch (err) {
      setError("Failed to extract topics");
    }
    setIsLoadingAnalysis(false);
  };

  const performEntityRecognition = async () => {
    setIsLoadingAnalysis(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${BACKEND_URL}/extract-entities/`, {
        method: "POST",
        body: formData
      });
      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setEntitiesResults(data);
      }
    } catch (err) {
      setError("Failed to extract entities");
    }
    setIsLoadingAnalysis(false);
  };


  const handleAnalysisSelect = async (type: string) => {
    setSelectedAnalysis(type);
    setError(null);
    setIsLoadingAnalysis(true);

    try {
      switch (type) {
          case 'readability':
            if (!readabilityResults) {
                await performReadabilityAnalysis();
            }
            break;
          case 'topics':
            if (!topicsResults) {
                await performTopicExtraction();
            }
            break;
          case 'entities':
            if (!entitiesResults) {
                await performEntityRecognition();
            }
            break;
        }
      } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to perform analysis');
      } finally {
          setIsLoadingAnalysis(false);
      }
  };

  const getSentimentColor = (label: string) => {
    switch (label.toLowerCase()) {
      case 'positive': return 'text-green-600';
      case 'negative': return 'text-red-600';
      default: return 'text-yellow-600';
    }
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat().format(num);
  };

  const sanitizeText = (text: string): string => {
    return text
      .replace(/[^\x20-\x7E\n\t]/g, '') 
      .trim();
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
          <h1 className="text-3xl font-bold text-gray-800">Text Analysis</h1>
          <p className="text-gray-600">Analyzing: {file.name}</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="space-y-8">
        {!results ? (
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
            <div className="text-center">
              <div className="mb-6">
                <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full mb-4">
                  <FileText className="h-10 w-10 text-white" />
                </div>
              </div>
              
              <button
                onClick={processText}
                disabled={isProcessing}
                className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-semibold rounded-xl hover:from-blue-600 hover:to-purple-700 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProcessing ? (
                  <>
                    <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Analyzing Text...
                  </>
                ) : (
                  'Start Text Analysis'
                )}
              </button>

              {error && (
                <div className="mt-4 text-red-500 font-medium">{error}</div>
              )}
            </div>
          </div>
        ) : (
          <>
            {/* Statistics Cards */}
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/30">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600">Words</h3>
                  <BarChart2 className="h-5 w-5 text-blue-500" />
                </div>
                <p className="text-3xl font-bold text-gray-800">{formatNumber(results.statistics.words)}</p>
              </div>
              <div className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/30">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600">Sentences</h3>
                  <MessageCircle className="h-5 w-5 text-purple-500" />
                </div>
                <p className="text-3xl font-bold text-gray-800">{formatNumber(results.statistics.sentences)}</p>
              </div>
              <div className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/30">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600">Characters</h3>
                  <Book className="h-5 w-5 text-green-500" />
                </div>
                <p className="text-3xl font-bold text-gray-800">{formatNumber(results.statistics.characters)}</p>
              </div>
              <div className="bg-white/70 backdrop-blur-sm rounded-xl p-6 border border-white/30">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-gray-600">Sentiment</h3>
                  <Brain className="h-5 w-5 text-rose-500" />
                </div>
                <div className="flex items-baseline">
                  <p className={`text-2xl font-bold ${getSentimentColor(results.sentiment.label)}`}>
                    {results.sentiment.label}
                  </p>
                  <span className="ml-2 text-sm text-gray-500">
                    ({(results.sentiment.score * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            </div>

            {/* Further Analysis Options */}
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
              <h2 className="text-xl font-bold text-gray-800 mb-6">Advanced Analysis Options</h2>
              <div className="grid md:grid-cols-3 gap-6">
                <button
                  onClick={() => handleAnalysisSelect('readability')}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                    selectedAnalysis === 'readability' 
                      ? 'border-purple-400 bg-purple-50' 
                      : 'border-purple-200 hover:border-purple-400 bg-white/50'
                  }`}
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Readability Analysis</h3>
                  <p className="text-sm text-gray-600">Calculate reading level, complexity scores, and comprehension metrics</p>
                </button>

                <button
                  onClick={() => handleAnalysisSelect('topics')}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                    selectedAnalysis === 'topics' 
                      ? 'border-blue-400 bg-blue-50' 
                      : 'border-blue-200 hover:border-blue-400 bg-white/50'
                  }`}
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Topic Extraction</h3>
                  <p className="text-sm text-gray-600">Identify key topics, themes, and subject matter</p>
                </button>

                <button
                  onClick={() => handleAnalysisSelect('entities')}
                  className={`p-6 rounded-xl border-2 transition-all duration-200 ${
                    selectedAnalysis === 'entities' 
                      ? 'border-green-400 bg-green-50' 
                      : 'border-green-200 hover:border-green-400 bg-white/50'
                  }`}
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-2">Entity Recognition</h3>
                  <p className="text-sm text-gray-600">Extract names, places, organizations, and dates</p>
                </button>
              </div>
            </div>

            
            
            <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Text Preview</h2>
              <div className="bg-gray-50 rounded-lg p-6 font-mono text-sm overflow-auto max-h-96">
                {results?.preview ? (
                  <pre className="whitespace-pre-wrap break-words">
                    {sanitizeText(results.preview)}
                  </pre>
                ) : (
                  <p className="text-gray-500">No preview available</p>
                )}
              </div>
            </div>

            {/* Selected Analysis Results */}
            {selectedAnalysis && (
              <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-8 border border-white/30">
                {isLoadingAnalysis ? (
                  <div className="flex items-center justify-center py-12">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    <span className="ml-3 text-gray-600">Loading analysis...</span>
                  </div>
                ) : selectedAnalysis === 'readability' && readabilityResults ? (
                  <div>
                    <h2 className="text-xl font-bold text-gray-800 mb-6">Readability Analysis</h2>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-2">Flesch Reading Ease</h3>
                        <p className="text-3xl font-bold text-blue-600">{readabilityResults.flesch_reading_ease.toFixed(1)}</p>
                        <p className="text-sm text-gray-600 mt-2">
                          {readabilityResults.flesch_reading_ease > 90 ? "Very Easy" :
                          readabilityResults.flesch_reading_ease > 80 ? "Easy" :
                          readabilityResults.flesch_reading_ease > 70 ? "Fairly Easy" :
                          readabilityResults.flesch_reading_ease > 60 ? "Standard" :
                          readabilityResults.flesch_reading_ease > 50 ? "Fairly Difficult" :
                          readabilityResults.flesch_reading_ease > 30 ? "Difficult" : "Very Difficult"}
                        </p>
                      </div>
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-2">Grade Level</h3>
                        <p className="text-3xl font-bold text-purple-600">
                          {readabilityResults.flesch_kincaid_grade.toFixed(1)}
                        </p>
                        <p className="text-sm text-gray-600 mt-2">Flesch-Kincaid Grade Level</p>
                      </div>
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-2">Text Standard</h3>
                        <p className="text-xl font-bold text-green-600">{readabilityResults.text_standard}</p>
                        <p className="text-sm text-gray-600 mt-2">Recommended education level</p>
                      </div>
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-2">Difficult Words</h3>
                        <p className="text-3xl font-bold text-red-600">{readabilityResults.difficult_words}</p>
                        <p className="text-sm text-gray-600 mt-2">Complex words found</p>
                      </div>
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-2">Other Indices</h3>
                        <div className="space-y-2">
                          <div className="flex justify-between">
                            <span className="text-gray-600">Gunning Fog:</span>
                            <span className="font-medium">{readabilityResults.gunning_fog.toFixed(1)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">SMOG:</span>
                            <span className="font-medium">{readabilityResults.smog_index.toFixed(1)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-gray-600">Coleman-Liau:</span>
                            <span className="font-medium">{readabilityResults.coleman_liau_index.toFixed(1)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : selectedAnalysis === 'topics' && topicsResults ? (
                  <div>
                    <h2 className="text-xl font-bold text-gray-800 mb-6">Topic Analysis</h2>
                    <div className="grid md:grid-cols-2 gap-6">
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-4">Main Topics</h3>
                        <div className="flex flex-wrap gap-2">
                          {topicsResults.main_topics.map((topic, i) => (
                            <span key={i} className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                              {topic}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50">
                        <h3 className="text-lg font-medium text-gray-800 mb-4">Common Words</h3>
                        <div className="space-y-2">
                          {topicsResults.common_words.map(([word, count], i) => (
                            <div key={i} className="flex justify-between items-center">
                              <span className="text-gray-800">{word}</span>
                              <span className="px-2 py-1 bg-gray-100 rounded-full text-sm text-gray-600">
                                {count}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                      <div className="bg-white/80 rounded-xl p-6 border border-white/50 md:col-span-2">
                        <h3 className="text-lg font-medium text-gray-800 mb-4">Key Phrases</h3>
                        <div className="flex flex-wrap gap-2">
                          {topicsResults.noun_phrases.slice(0, 15).map((phrase, i) => (
                            <span key={i} className="px-3 py-1 bg-purple-100 text-purple-800 rounded-full text-sm">
                              {phrase}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : selectedAnalysis === 'entities' && entitiesResults ? (
                  <div>
                    <h2 className="text-xl font-bold text-gray-800 mb-6">Named Entity Recognition</h2>
                    <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                      {Object.entries(entitiesResults).map(([type, entities]) => 
                        entities.length > 0 && (
                          <div key={type} className="bg-white/80 rounded-xl p-6 border border-white/50">
                            <h3 className="text-lg font-medium text-gray-800 mb-4">{type}</h3>
                            <div className="flex flex-wrap gap-2">
                              {entities.map((entity, i) => (
                                <span key={i} className={`px-3 py-1 rounded-full text-sm
                                  ${type === 'PERSON' ? 'bg-red-100 text-red-800' :
                                    type === 'ORG' ? 'bg-blue-100 text-blue-800' :
                                    type === 'GPE' ? 'bg-green-100 text-green-800' :
                                    type === 'DATE' ? 'bg-yellow-100 text-yellow-800' :
                                    type === 'MONEY' ? 'bg-emerald-100 text-emerald-800' :
                                    'bg-gray-100 text-gray-800'}`}>
                                  {entity}
                                </span>
                              ))}
                            </div>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                ) : null}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};



