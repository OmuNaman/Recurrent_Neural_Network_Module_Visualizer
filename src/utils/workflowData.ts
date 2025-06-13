// FILE: src/utils/workflowData.ts
import type { Node, Edge } from '@xyflow/react';
import { 
  VOCAB, ONE_HOT_R,
  W_xh, W_hh, W_hy, b_h, b_y, // Added W_hy and b_y for completeness, though not used in this specific set of nodes yet
  H1, Y1, PRED1 // Added Y1 and PRED1 for completeness
} from './matrixCalculations';

// Node Positions (FROM YOUR PROVIDED EXAMPLE)
const col_static = -800;
const col_timestep_1 = 0;

const row_input = 0;
const row_h_prev = 400;
const row_calc_h = 800;

// Additional row positions for output and prediction based on your previous structures
const row_calc_y = row_calc_h + 450; // Adding space for y calculation
const row_pred_y = row_calc_y + 350; // Adding space for prediction

// Row positions for additional static weights (W_hy, b_y) if needed for future steps
// Aligning these to match the style of the other static weights
const row_static_w_hy = row_calc_h + 400; 
const row_static_b_y = row_calc_h + 800;

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

  // --- Static Weight & Bias Matrices (Using YOUR coordinates) ---
  { id: 'w_xh', type: 'matrix', position: { x: col_static, y: row_input }, data: { label: 'Weights W_xh', matrix: W_xh, description: 'Input-to-Hidden (3x4)' }, zIndex: 1 },
  { id: 'w_hh', type: 'matrix', position: { x: col_static, y: row_h_prev }, data: { label: 'Weights W_hh', matrix: W_hh, description: 'Hidden-to-Hidden (4x4)' }, zIndex: 1 },
  { id: 'b_h', type: 'matrix', position: { x: col_static, y: row_calc_h }, data: { label: 'Bias b_h', matrix: b_h, description: 'Hidden Bias (1x4)' }, zIndex: 1 },
  
  // Adding W_hy and b_y for completeness, positioned consistently
  { id: 'w_hy', type: 'matrix', position: { x: col_static + 1500 - 120, y: row_static_w_hy - 1000}, data: { label: 'W_hy', matrix: W_hy, description: 'Hidden-to-Output (4x3)' }, zIndex: 1 },
  { id: 'b_y', type: 'matrix', position: { x: col_static + 1500 - 120, y: row_static_b_y - 700}, data: { label: 'b_y', matrix: b_y, description: 'Output Bias (1x3)' }, zIndex: 1 },

  // --- Timestep 1: Input 'R' (Using YOUR coordinates for the existing nodes) ---
  { 
    id: 't1_input', 
    type: 'wordVector', 
    position: { x: col_timestep_1 - 300, y: row_input }, 
    data: { 
      label: "Input: x₁ ('R')", 
      matrix: ONE_HOT_R, 
      description: "One-hot vector for the first character",
      vocabulary: VOCAB 
    },
    zIndex: 1 
  },
  { 
    id: 't1_h_prev', 
    type: 'matrix', 
    position: { x: col_timestep_1 - 500, y: row_h_prev + 800}, // As per your example
    data: { 
      label: 'Prev. Hidden State: h₀', 
      matrix: [[0, 0, 0, 0]], 
      description: 'Starts as zeros' 
    },
    zIndex: 1 
  },
  {
    id: 't1_calc_h',
    type: 'calculation',
    position: { x: col_timestep_1 + 220, y: row_calc_h - 250}, // As per your example
    data: {
      label: 'Calculate Hidden State h₁',
      formula: "h₁=tanh(x₁⋅Wxh + h₀⋅Whh + bh)",
      expectedMatrix: H1,
      hint: 'Combine current input and previous memory, then apply tanh.'
    },
    zIndex: 1
  },

  // Adding the output calculation and prediction nodes, positioned relative to t1_calc_h
  {
    id: 't1_calc_y',
    type: 'calculation',
    position: { x: col_timestep_1 + 220 + 500 + 300, y: row_calc_h - 250 }, // Shifted right from t1_calc_h
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
    position: { x: col_timestep_1 + 220 + 500 + 500 + 300, y: row_calc_h - 250 }, // Shifted further right
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
];

export const initialEdges: Edge[] = [
  // Intro Node to Timestep 1 Inputs

  // Connections from static weights to the t1_calc_h node
  { id: 'e-wxh-t1h', source: 'w_xh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-whh-t1h', source: 'w_hh', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },
  { id: 'e-bh-t1h', source: 'b_h', target: 't1_calc_h', style: { strokeDasharray: '5 5' } },

  // Connections from timestep inputs to the t1_calc_h node
  { id: 'e-x1-t1h', source: 't1_input', target: 't1_calc_h', animated: true },
  { id: 'e-h0-t1h', source: 't1_h_prev', target: 't1_calc_h', animated: true },

  // Connections for t1_calc_y
  { id: 'e-h1-t1y', source: 't1_calc_h', target: 't1_calc_y', animated: true },
  { id: 'e-why-t1y', source: 'w_hy', target: 't1_calc_y', style: { strokeDasharray: '5 5' } },
  { id: 'e-by-t1y', source: 'b_y', target: 't1_calc_y', style: { strokeDasharray: '5 5' } },

  // Connection from t1_calc_y to t1_pred
  { id: 'e-y1-t1p', source: 't1_calc_y', target: 't1_pred', animated: true },
];