// FILE: src/utils/workflowData.ts
import type { Node, Edge } from '@xyflow/react';
import { VOCAB, ONE_HOT_R } from './matrixCalculations';

export const initialNodes: Node[] = [
  // Timestep 1: Input 'R'
  { 
    id: 't1_input', 
    type: 'wordVector', 
    position: { x: 0, y: 0 }, 
    data: { 
      label: "Input: x₁ ('R')", 
      matrix: ONE_HOT_R, 
      description: "One-hot vector for the first character",
      vocabulary: VOCAB 
    } 
  },
];

export const initialEdges: Edge[] = [
  // No connections yet
];