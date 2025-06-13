// FILE: src/utils/workflowData.ts
import type { Node, Edge } from '@xyflow/react';
import { 
  VOCAB, ONE_HOT_R,
  W_xh, W_hh, b_h,
  H1
} from './matrixCalculations';

// Node Positions
const col_static = -800;
const col_timestep_1 = 0;

const row_input = 0;
const row_h_prev = 400;
const row_calc_h = 800;

export const initialNodes: Node[] = [
  // --- Static Weight & Bias Matrices ---
  { id: 'w_xh', type: 'matrix', position: { x: col_static, y: row_input }, data: { label: 'Weights W_xh', matrix: W_xh, description: 'Input-to-Hidden (3x4)' } },
  { id: 'w_hh', type: 'matrix', position: { x: col_static, y: row_h_prev }, data: { label: 'Weights W_hh', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' } },
  { id: 'b_h', type: 'matrix', position: { x: col_static, y: row_calc_h }, data: { label: 'Bias b_h', matrix: b_h, description: 'Hidden Bias (1x4)' } },

  // --- Timestep 1: Input 'R' ---
  { 
    id: 't1_input', 
    type: 'wordVector', 
    position: { x: col_timestep_1 - 300, y: row_input }, 
    data: { 
      label: "Input: x₁ ('R')", 
      matrix: ONE_HOT_R, 
      description: "One-hot vector for the first character",
      vocabulary: VOCAB 
    } 
  },
  { 
    id: 't1_h_prev', 
    type: 'matrix', 
    position: { x: col_timestep_1 - 500, y: row_h_prev + 800}, 
    data: { 
      label: 'Prev. Hidden State: h₀', 
      matrix: [[0, 0, 0, 0]], 
      description: 'Starts as zeros' 
    } 
  },
  {
    id: 't1_calc_h',
    type: 'calculation',
    position: { x: col_timestep_1 + 220, y: row_calc_h - 250},
    data: {
      label: 'Calculate Hidden State h₁',
      formula: "h₁=tanh(x₁⋅Wxh + h₀⋅Whh + bh)",
      expectedMatrix: H1,
      hint: 'Combine current input and previous memory, then apply tanh.'
    }
  },
];

export const initialEdges: Edge[] = [
  // Connections from static weights to the calculation node
  { id: 'e-wxh-t1', source: 'w_xh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-whh-t1', source: 'w_hh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-bh-t1', source: 'b_h', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },

  // Connections from timestep inputs to the calculation node
  { id: 'e-x1-t1', source: 't1_input', target: 't1_calc_h', animated: true },
  { id: 'e-h0-t1', source: 't1_h_prev', target: 't1_calc_h', animated: true },
];