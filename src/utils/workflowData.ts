// FILE: src/utils/workflowData.ts
import type { Node, Edge } from '@xyflow/react';
import { 
  VOCAB, ONE_HOT_R, ONE_HOT_N, ONE_HOT_EXCLAMATION,
  W_xh, W_hh, W_hy, b_h, b_y,
  H1, Y1, PRED1, H2, Y2, PRED2, H3, Y3, PRED3, H4, Y4, PRED4
} from './matrixCalculations';

// Node Positions - INCREASED SPACING BETWEEN TIMESTEPS
const col_static = -800;
const col_timestep_1 = 0;
const col_timestep_2 = 2400; // Increased from 1600 to 2400 (800px more spacing)
const col_timestep_3 = 4800; // Increased from 3200 to 4800 (1600px more spacing)
const col_timestep_4 = 7200; // Increased from 4800 to 7200 (2400px more spacing)

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
      exampleSequence: "'R' -> 'N' -> 'N' -> '!'",
      focusAreas: [
        "Hidden State: The RNN's 'memory' from previous steps.",
        "Weight Reuse: The same weights are applied at every timestep.",
        "Sequential Processing: Data flows one step at a time."
      ]
    },
    zIndex: 10 
  },

  // --- Timestep 1: Input 'R' ---
  { 
    id: 't1_input', 
    type: 'wordVector', 
    position: { x: col_timestep_1 - 300, y: row_input }, 
    data: { 
      label: "Input: x₁ ('R')", 
      matrix: ONE_HOT_R, 
      description: "One-hot vector for 'R'",
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
      description: 'Starts as zeros' 
    },
    zIndex: 1 
  },
  // Weight matrices for Timestep 1
  { id: 't1_w_xh', type: 'matrix', position: { x: col_timestep_1 + weight_offset_x, y: row_input }, data: { label: 'W_xh (T1)', matrix: W_xh, description: 'Input-to-Hidden (3x4)' }, zIndex: 1 },
  { id: 't1_w_hh', type: 'matrix', position: { x: col_timestep_1 + weight_offset_x, y: row_h_prev }, data: { label: 'W_hh (T1)', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 't1_b_h', type: 'matrix', position: { x: col_timestep_1 + weight_offset_x, y: row_calc_h }, data: { label: 'b_h (T1)', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  { id: 't1_w_hy', type: 'matrix', position: { x: col_timestep_1 + output_weight_offset_x, y: row_input }, data: { label: 'W_hy (T1)', matrix: W_hy, description: 'Hidden-to-Output (4x3)' }, zIndex: 1 },
  { id: 't1_b_y', type: 'matrix', position: { x: col_timestep_1 + output_weight_offset_x, y: row_h_prev }, data: { label: 'b_y (T1)', matrix: b_y, description: 'Output Bias (1x3)' }, zIndex: 1 },
  
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

  // --- Timestep 2: Input 'N' ---
  { 
    id: 't2_input', 
    type: 'wordVector', 
    position: { x: col_timestep_2 - 300, y: row_input }, 
    data: { 
      label: "Input: x₂ ('N')", 
      matrix: ONE_HOT_N, 
      description: "One-hot vector for 'N'",
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
      description: 'Hidden state from previous timestep' 
    },
    zIndex: 1 
  },
  // Weight matrices for Timestep 2
  { id: 't2_w_xh', type: 'matrix', position: { x: col_timestep_2 + weight_offset_x, y: row_input }, data: { label: 'W_xh (T2)', matrix: W_xh, description: 'Input-to-Hidden (3x4)' }, zIndex: 1 },
  { id: 't2_w_hh', type: 'matrix', position: { x: col_timestep_2 + weight_offset_x, y: row_h_prev }, data: { label: 'W_hh (T2)', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 't2_b_h', type: 'matrix', position: { x: col_timestep_2 + weight_offset_x, y: row_calc_h }, data: { label: 'b_h (T2)', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  { id: 't2_w_hy', type: 'matrix', position: { x: col_timestep_2 + output_weight_offset_x, y: row_input }, data: { label: 'W_hy (T2)', matrix: W_hy, description: 'Hidden-to-Output (4x3)' }, zIndex: 1 },
  { id: 't2_b_y', type: 'matrix', position: { x: col_timestep_2 + output_weight_offset_x, y: row_h_prev }, data: { label: 'b_y (T2)', matrix: b_y, description: 'Output Bias (1x3)' }, zIndex: 1 },
  
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

  // --- Timestep 3: Input 'N' ---
  { 
    id: 't3_input', 
    type: 'wordVector', 
    position: { x: col_timestep_3 - 300, y: row_input }, 
    data: { 
      label: "Input: x₃ ('N')", 
      matrix: ONE_HOT_N, 
      description: "One-hot vector for 'N'",
      vocabulary: VOCAB 
    },
    zIndex: 1 
  },
  { 
    id: 't3_h_prev', 
    type: 'matrix', 
    position: { x: col_timestep_3 - 500, y: row_h_prev + 800}, 
    data: { 
      label: 'Prev. Hidden State: h₂', 
      matrix: H2, 
      description: 'Hidden state from previous timestep' 
    },
    zIndex: 1 
  },
  // Weight matrices for Timestep 3
  { id: 't3_w_xh', type: 'matrix', position: { x: col_timestep_3 + weight_offset_x, y: row_input }, data: { label: 'W_xh (T3)', matrix: W_xh, description: 'Input-to-Hidden (3x4)' }, zIndex: 1 },
  { id: 't3_w_hh', type: 'matrix', position: { x: col_timestep_3 + weight_offset_x, y: row_h_prev }, data: { label: 'W_hh (T3)', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 't3_b_h', type: 'matrix', position: { x: col_timestep_3 + weight_offset_x, y: row_calc_h }, data: { label: 'b_h (T3)', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  { id: 't3_w_hy', type: 'matrix', position: { x: col_timestep_3 + output_weight_offset_x, y: row_input }, data: { label: 'W_hy (T3)', matrix: W_hy, description: 'Hidden-to-Output (4x3)' }, zIndex: 1 },
  { id: 't3_b_y', type: 'matrix', position: { x: col_timestep_3 + output_weight_offset_x, y: row_h_prev }, data: { label: 'b_y (T3)', matrix: b_y, description: 'Output Bias (1x3)' }, zIndex: 1 },
  
  {
    id: 't3_calc_h',
    type: 'calculation',
    position: { x: col_timestep_3 + 220, y: row_calc_h - 250},
    data: {
      label: 'Calculate Hidden State h₃',
      formula: "h₃=tanh(x₃⋅Wxh + h₂⋅Whh + bh)",
      expectedMatrix: H3,
      hint: 'Combine current input and previous memory, then apply tanh.'
    },
    zIndex: 1
  },
  {
    id: 't3_calc_y',
    type: 'calculation',
    position: { x: col_timestep_3 + 220 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Calculate Output Logits y₃',
      formula: "y₃=h₃⋅Why + by",
      expectedMatrix: Y3,
      hint: 'Multiply hidden state h₃ by W_hy, then add bias b_y.',
      vocabulary: VOCAB
    },
    zIndex: 1
  },
  {
    id: 't3_pred',
    type: 'activation',
    position: { x: col_timestep_3 + 220 + 500 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Predict Next Char (ŷ₃)',
      formula: "ŷ₃=Softmax(y₃)",
      expectedMatrix: PRED3,
      description: 'Probability distribution over vocabulary',
      vocabulary: VOCAB,
      highlightMax: true
    },
    zIndex: 1
  },

  // --- Timestep 4: Input '!' ---
  { 
    id: 't4_input', 
    type: 'wordVector', 
    position: { x: col_timestep_4 - 300, y: row_input }, 
    data: { 
      label: "Input: x₄ ('!')", 
      matrix: ONE_HOT_EXCLAMATION, 
      description: "One-hot vector for '!'",
      vocabulary: VOCAB 
    },
    zIndex: 1 
  },
  { 
    id: 't4_h_prev', 
    type: 'matrix', 
    position: { x: col_timestep_4 - 500, y: row_h_prev + 800}, 
    data: { 
      label: 'Prev. Hidden State: h₃', 
      matrix: H3, 
      description: 'Hidden state from previous timestep' 
    },
    zIndex: 1 
  },
  // Weight matrices for Timestep 4
  { id: 't4_w_xh', type: 'matrix', position: { x: col_timestep_4 + weight_offset_x, y: row_input }, data: { label: 'W_xh (T4)', matrix: W_xh, description: 'Input-to-Hidden (3x4)' }, zIndex: 1 },
  { id: 't4_w_hh', type: 'matrix', position: { x: col_timestep_4 + weight_offset_x, y: row_h_prev }, data: { label: 'W_hh (T4)', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 't4_b_h', type: 'matrix', position: { x: col_timestep_4 + weight_offset_x, y: row_calc_h }, data: { label: 'b_h (T4)', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  { id: 't4_w_hy', type: 'matrix', position: { x: col_timestep_4 + output_weight_offset_x, y: row_input }, data: { label: 'W_hy (T4)', matrix: W_hy, description: 'Hidden-to-Output (4x3)' }, zIndex: 1 },
  { id: 't4_b_y', type: 'matrix', position: { x: col_timestep_4 + output_weight_offset_x, y: row_h_prev }, data: { label: 'b_y (T4)', matrix: b_y, description: 'Output Bias (1x3)' }, zIndex: 1 },
  
  {
    id: 't4_calc_h',
    type: 'calculation',
    position: { x: col_timestep_4 + 220, y: row_calc_h - 250},
    data: {
      label: 'Calculate Hidden State h₄',
      formula: "h₄=tanh(x₄⋅Wxh + h₃⋅Whh + bh)",
      expectedMatrix: H4,
      hint: 'Combine current input and previous memory, then apply tanh.'
    },
    zIndex: 1
  },
  {
    id: 't4_calc_y',
    type: 'calculation',
    position: { x: col_timestep_4 + 220 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Calculate Output Logits y₄',
      formula: "y₄=h₄⋅Why + by",
      expectedMatrix: Y4,
      hint: 'Multiply hidden state h₄ by W_hy, then add bias b_y.',
      vocabulary: VOCAB
    },
    zIndex: 1
  },
  {
    id: 't4_pred',
    type: 'activation',
    position: { x: col_timestep_4 + 220 + 500 + 500 + 300, y: row_calc_h - 250 },
    data: {
      label: 'Predict Next Char (ŷ₄)',
      formula: "ŷ₄=Softmax(y₄)",
      expectedMatrix: PRED4,
      description: 'Probability distribution over vocabulary',
      vocabulary: VOCAB,
      highlightMax: true
    },
    zIndex: 1
  },
];

export const initialEdges: Edge[] = [
  // --- Timestep 1 Connections ---
  // Connections from duplicated weights to t1_calc_h
  { id: 'e-t1wxh-t1h', source: 't1_w_xh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t1whh-t1h', source: 't1_w_hh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t1bh-t1h', source: 't1_b_h', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  
  // Timestep inputs to t1_calc_h
  { id: 'e-x1-t1h', source: 't1_input', target: 't1_calc_h', animated: true },
  { id: 'e-h0-t1h', source: 't1_h_prev', target: 't1_calc_h', animated: true },
  
  // t1_calc_y connections
  { id: 'e-h1-t1y', source: 't1_calc_h', target: 't1_calc_y', animated: true },
  { id: 'e-t1why-t1y', source: 't1_w_hy', target: 't1_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-t1by-t1y', source: 't1_b_y', target: 't1_calc_y', style: { strokeDasharray: '5 5' } },
  
  // t1_pred connection
  { id: 'e-y1-t1p', source: 't1_calc_y', target: 't1_pred', animated: true },

  // --- Timestep 2 Connections ---
  // Connections from duplicated weights to t2_calc_h
  { id: 'e-t2wxh-t2h', source: 't2_w_xh', target: 't2_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t2whh-t2h', source: 't2_w_hh', target: 't2_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t2bh-t2h', source: 't2_b_h', target: 't2_calc_h', style: { strokeDasharray: '5 5' } },
  
  // Timestep inputs to t2_calc_h
  { id: 'e-x2-t2h', source: 't2_input', target: 't2_calc_h', animated: true },
  { id: 'e-h1-t2h', source: 't2_h_prev', target: 't2_calc_h', animated: true },
  
  // Hidden state flow from t1 to t2
  { id: 'e-t1h-t2prev', source: 't1_calc_h', target: 't2_h_prev', animated: true, style: { stroke: '#ff6b6b', strokeWidth: 3 } },
  
  // t2_calc_y connections
  { id: 'e-h2-t2y', source: 't2_calc_h', target: 't2_calc_y', animated: true },
  { id: 'e-t2why-t2y', source: 't2_w_hy', target: 't2_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-t2by-t2y', source: 't2_b_y', target: 't2_calc_y', style: { strokeDasharray: '5 5' } },
  
  // t2_pred connection
  { id: 'e-y2-t2p', source: 't2_calc_y', target: 't2_pred', animated: true },

  // --- Timestep 3 Connections ---
  // Connections from duplicated weights to t3_calc_h
  { id: 'e-t3wxh-t3h', source: 't3_w_xh', target: 't3_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t3whh-t3h', source: 't3_w_hh', target: 't3_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t3bh-t3h', source: 't3_b_h', target: 't3_calc_h', style: { strokeDasharray: '5 5' } },
  
  // Timestep inputs to t3_calc_h
  { id: 'e-x3-t3h', source: 't3_input', target: 't3_calc_h', animated: true },
  { id: 'e-h2-t3h', source: 't3_h_prev', target: 't3_calc_h', animated: true },
  
  // Hidden state flow from t2 to t3
  { id: 'e-t2h-t3prev', source: 't2_calc_h', target: 't3_h_prev', animated: true, style: { stroke: '#ff6b6b', strokeWidth: 3 } },
  
  // t3_calc_y connections
  { id: 'e-h3-t3y', source: 't3_calc_h', target: 't3_calc_y', animated: true },
  { id: 'e-t3why-t3y', source: 't3_w_hy', target: 't3_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-t3by-t3y', source: 't3_b_y', target: 't3_calc_y', style: { strokeDasharray: '5 5' } },
  
  // t3_pred connection
  { id: 'e-y3-t3p', source: 't3_calc_y', target: 't3_pred', animated: true },

  // --- Timestep 4 Connections ---
  // Connections from duplicated weights to t4_calc_h
  { id: 'e-t4wxh-t4h', source: 't4_w_xh', target: 't4_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t4whh-t4h', source: 't4_w_hh', target: 't4_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-t4bh-t4h', source: 't4_b_h', target: 't4_calc_h', style: { strokeDasharray: '5 5' } },
  
  // Timestep inputs to t4_calc_h
  { id: 'e-x4-t4h', source: 't4_input', target: 't4_calc_h', animated: true },
  { id: 'e-h3-t4h', source: 't4_h_prev', target: 't4_calc_h', animated: true },
  
  // Hidden state flow from t3 to t4
  { id: 'e-t3h-t4prev', source: 't3_calc_h', target: 't4_h_prev', animated: true, style: { stroke: '#ff6b6b', strokeWidth: 3 } },
  
  // t4_calc_y connections
  { id: 'e-h4-t4y', source: 't4_calc_h', target: 't4_calc_y', animated: true },
  { id: 'e-t4why-t4y', source: 't4_w_hy', target: 't4_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-t4by-t4y', source: 't4_b_y', target: 't4_calc_y', style: { strokeDasharray: '5 5' } },
  
  // t4_pred connection
  { id: 'e-y4-t4p', source: 't4_calc_y', target: 't4_pred', animated: true },
];