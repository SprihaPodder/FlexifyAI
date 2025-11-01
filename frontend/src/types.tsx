
export type DatasetType = 'csv' | 'text' | 'image';
export type WorkflowType = 'eda' | 'ml' | null;

export interface Dataset {
  file: File;
  type: DatasetType;
  metadata?: {
    size: number;
    lastModified: number;
    name: string;
    type: string;
  };
}


export type MLModel =
  | 'decision_tree'
  | 'random_forest'
  | 'logistic_regression'
  | 'svm'
  | 'gradient_boosting'
  | 'knn';


export interface Column {
  id: string;
  name: string;
  type: 'categorical' | 'numerical';
}


export interface MLResults {
  accuracy: number;
  std_dev: number;
  model_used: MLModel;
  params_used: Record<string, any>;
  rows_before: number;
  rows_after: number;
  rows_removed: number;
  columns: number;
  handle_missing: string;
  handle_outliers: string;
}

export interface EDAResults {
  image: string | null;
  totalRows: number;
  totalColumns: number;
  missingValues: number;
  duplicateRows: number;
}


export interface FileUploadProps {
  onFileUpload: (dataset: Dataset) => void;
}

export interface WorkflowSelectorProps {
  fileName: string;
  onWorkflowSelect: (workflow: WorkflowType) => void;
}

export interface EDAWorkflowProps {
  file: File;
  onBack: () => void;
}

export interface MLWorkflowProps {
  file: File;
  onBack: () => void;
}

export interface HeaderProps {
  onReset: () => void;
}