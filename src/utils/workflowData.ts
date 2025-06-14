// FILE: src/utils/workflowData.ts
import type { Node, Edge } from '@xyflow/react';
import { 
  VOCAB, ONE_HOT_A, ONE_HOT_B,
  W_xh, W_hh, W_hy, b_h, b_y,
  H1, Y1, PRED1, H2, Y2, PRED2 // Only H1/Y1/PRED1 and H2/Y2/PRED2
} from './matrixCalculations';

// Node Positions - Spacing for 2 timesteps
const col_timestep_1 = 0;
const col_timestep_2 = 3000; // Spacing for the second timestep

const row_input = 0;
const row_h_prev = 400;
const row_calc_h = 800;

// Consistent weight matrix positions for all timesteps
const weight_offset_x = -800; // Distance from timestep center to weights
const output_weight_offset_x = 700; // Distance from timestep center to output weights

// Position for the Intro Node, significantly to the left
const col_intro = -1500; 
const row_intro_mid = (row_input + row_calc_h) / 2; // Vertically centered

export const initialNodes: Node[] = [
  // --- Intro Node ---
  { 
    id: 'intro-rnn', 
    type: 'introNode', 
    position: { x: col_intro, y: row_intro_mid }, 
    data: { 
      label: "Recurrent Neural Networks (RNN)",
      description: "Explore the core mechanism of RNNs by manually performing a forward pass, character by character.",
      task: "Predict the next character in a sequence given the current input and the network's 'memory' (hidden state).",
      exampleSequence: "'A' -> 'B'", // Updated sequence
      focusAreas: [
        "Hidden State: The RNN's 'memory' from previous steps.",
        "Weight Reuse: The same weights are applied at every timestep.",
        "Sequential Processing: Data flows one step at a time."
      ]
    },
    zIndex: 10 
  },

  // --- Timestep 1: Input 'A' ---
  { 
    id: 't1_input', 
    type: 'wordVector', 
    position: { x: col_timestep_1 - 300, y: row_input }, 
    data: { 
      label: "Input: x₁ ('A')", 
      matrix: ONE_HOT_A, 
      description: "One-hot vector for 'A'",
      vocabulary: VOCAB 
    },
    zIndex: 1 
  },
  { 
    id: 't1_h_prev', 
    type: 'matrix', 
    position: { x: col_timestep_1 - 500, y: row_h_prev + 800}, 
    data: { 
      label: 'Prev. Hidden State: h₀', 
      matrix: [[0, 0, 0, 0]], 
      description: 'Starts as zeros (1x4)' 
    },
    zIndex: 1 
  },
  // Weight matrices for Timestep 1
  { id: 't1_w_xh', type: 'matrix', position: { x: col_timestep_1 + weight_offset_x, y: row_input }, data: { label: 'W_xh (T1)', matrix: W_xh, description: 'Input-to-Hidden (2x4)' }, zIndex: 1 },
  { id: 't1_w_hh', type: 'matrix', position: { x: col_timestep_1 + weight_offset_x, y: row_h_prev }, data: { label: 'W_hh (T1)', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 't1_b_h', type: 'matrix', position: { x: col_timestep_1 + weight_offset_x, y: row_calc_h }, data: { label: 'b_h (T1)', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  { id: 't1_w_hy', type: 'matrix', position: { x: col_timestep_1 + output_weight_offset_x, y: row_input }, data: { label: 'W_hy (T1)', matrix: W_hy, description: 'Hidden-to-Output (4x2)' }, zIndex: 1 },
  { id: 't1_b_y', type: 'matrix', position: { x: col_timestep_1 + output_weight_offset_x, y: row_h_prev }, data: { label: 'b_y (T1)', matrix: b_y, description: 'Output Bias (1x2)' }, zIndex: 1 },
  
  {
    id: 't1_calc_h',
    type: 'calculation',
    position: { x: col_timestep_1 + 220, y: row_calc_h - 250},
    data: {
      label: 'Calculate Hidden State h₁',
      formula: "h₁=tanh(x₁⋅Wxh + h₀⋅Whh + bh)",
      expectedMatrix: H1,
      hint: 'Combine current input and previous memory, then apply tanh.'
    },
    zIndex: 1
  },
  {
    id: 't1_calc_y',
    type: 'calculation',
    position: { x: col_timestep_1 + 220 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Calculate Output Logits y₁',
      formula: "y₁=h₁⋅Why + by",
      expectedMatrix: Y1,
      hint: 'Multiply hidden state h₁ by W_hy, then add bias b_y.',
      vocabulary: VOCAB
    },
    zIndex: 1
  },
  {
    id: 't1_pred',
    type: 'activation',
    position: { x: col_timestep_1 + 220 + 500 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Predict Next Char (ŷ₁)',
      formula: "ŷ₁=Softmax(y₁)",
      expectedMatrix: PRED1,
      description: 'Probability distribution over vocabulary',
      vocabulary: VOCAB,
      highlightMax: true
    },
    zIndex: 1
  },

  // --- Timestep 2: Input 'B' ---
  { 
    id: 't2_input', 
    type: 'wordVector', 
    position: { x: col_timestep_2 - 300, y: row_input }, 
    data: { 
      label: "Input: x₂ ('B')", 
      matrix: ONE_HOT_B, 
      description: "One-hot vector for 'B'",
      vocabulary: VOCAB 
    },
    zIndex: 1 
  },
  { 
    id: 't2_h_prev', 
    type: 'matrix', 
    position: { x: col_timestep_2 - 500, y: row_h_prev + 800}, 
    data: { 
      label: 'Prev. Hidden State: h₁', 
      matrix: H1, 
      description: 'Hidden state from T1 (1x4)' 
    },
    zIndex: 1 
  },
  // Weight matrices for Timestep 2 (same as T1, demonstrating weight sharing)
  { id: 't2_w_xh', type: 'matrix', position: { x: col_timestep_2 + weight_offset_x, y: row_input }, data: { label: 'W_xh (T2)', matrix: W_xh, description: 'Input-to-Hidden (2x4)' }, zIndex: 1 },
  { id: 't2_w_hh', type: 'matrix', position: { x: col_timestep_2 + weight_offset_x, y: row_h_prev }, data: { label: 'W_hh (T2)', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 't2_b_h', type: 'matrix', position: { x: col_timestep_2 + weight_offset_x, y: row_calc_h }, data: { label: 'b_h (T2)', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  { id: 't2_w_hy', type: 'matrix', position: { x: col_timestep_2 + output_weight_offset_x, y: row_input }, data: { label: 'W_hy (T2)', matrix: W_hy, description: 'Hidden-to-Output (4x2)' }, zIndex: 1 },
  { id: 't2_b_y', type: 'matrix', position: { x: col_timestep_2 + output_weight_offset_x, y: row_h_prev }, data: { label: 'b_y (T2)', matrix: b_y, description: 'Output Bias (1x2)' }, zIndex: 1 },
  
  {
    id: 't2_calc_h',
    type: 'calculation',
    position: { x: col_timestep_2 + 220, y: row_calc_h - 250},
    data: {
      label: 'Calculate Hidden State h₂',
      formula: "h₂=tanh(x₂⋅Wxh + h₁⋅Whh + bh)",
      expectedMatrix: H2,
      hint: 'Combine current input and previous memory, then apply tanh.'
    },
    zIndex: 1
  },
  {
    id: 't2_calc_y',
    type: 'calculation',
    position: { x: col_timestep_2 + 220 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Calculate Output Logits y₂',
      formula: "y₂=h₂⋅Why + by",
      expectedMatrix: Y2,
      hint: 'Multiply hidden state h₂ by W_hy, then add bias b_y.',
      vocabulary: VOCAB
    },
    zIndex: 1
  },
  {
    id: 't2_pred',
    type: 'activation',
    position: { x: col_timestep_2 + 220 + 500 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Predict Next Char (ŷ₂)',
      formula: "ŷ₂=Softmax(y₂)",
      expectedMatrix: PRED2,
      description: 'Probability distribution over vocabulary',
      vocabulary: VOCAB,
      highlightMax: true
    },
    zIndex: 1
  },
];

export const initialEdges: Edge[] = [
  // --- Timestep 1 Connections ---
  { id: 'e-t1wxh-t1h', source: 't1_w_xh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t1whh-t1h', source: 't1_w_hh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t1bh-t1h', source: 't1_b_h', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-x1-t1h', source: 't1_input', target: 't1_calc_h', animated: true },
  { id: 'e-h0-t1h', source: 't1_h_prev', target: 't1_calc_h', animated: true },
  { id: 'e-h1-t1y', source: 't1_calc_h', target: 't1_calc_y', animated: true },
  { id: 'e-t1why-t1y', source: 't1_w_hy', target: 't1_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-t1by-t1y', source: 't1_b_y', target: 't1_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-y1-t1p', source: 't1_calc_y', target: 't1_pred', animated: true },

  // --- Timestep 2 Connections ---
  { id: 'e-t2wxh-t2h', source: 't2_w_xh', target: 't2_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t2whh-t2h', source: 't2_w_hh', target: 't2_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t2bh-t2h', source: 't2_b_h', target: 't2_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-x2-t2h', source: 't2_input', target: 't2_calc_h', animated: true },
  { id: 'e-h1-t2h', source: 't2_h_prev', target: 't2_calc_h', animated: true }, // Connects to t2_h_prev
  { id: 'e-t1h-t2prev', source: 't1_calc_h', target: 't2_h_prev', animated: true, style: { stroke: '#ff6b6b', strokeWidth: 3 } }, // Hidden state flow from t1_calc_h
  { id: 'e-h2-t2y', source: 't2_calc_h', target: 't2_calc_y', animated: true },
  { id: 'e-t2why-t2y', source: 't2_w_hy', target: 't2_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-t2by-t2y', source: 't2_b_y', target: 't2_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-y2-t2p', source: 't2_calc_y', target: 't2_pred', animated: true },
];