// FILE: src/components/SelfAttentionWorkflow.tsx
import { useState, useCallback, useMemo, useEffect } from 'react';
import {
  ReactFlow,
  addEdge,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  Connection,
  ReactFlowProvider,
  useReactFlow // Import useReactFlow for fitView
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { ThemeProvider } from '@/components/ThemeProvider';
import { ThemeToggle } from '@/components/ThemeToggle';
import { Button } from '@/components/ui/button';
import { RotateCcw, Cpu } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'; // Import Dialog components

// Import all custom node types
import { WordVectorNode } from '@/components/workflow/WordVectorNode';
import { MatrixNode } from '@/components/workflow/MatrixNode';
import { CalculationNode } from '@/components/workflow/CalculationNode';
import { ActivationNode } from '@/components/workflow/ActivationNode'; // ADD THIS IMPORT
import { IntroNode } from '@/components/workflow/IntroNode';             // ADD THIS IMPORT

import { initialNodes as rawInitialNodes, initialEdges } from '@/utils/workflowData';

// Define all custom node types used in the workflow
const nodeTypes = {
  wordVector: WordVectorNode,
  matrix: MatrixNode,
  calculation: CalculationNode,
  activation: ActivationNode, // ADD THIS LINE
  introNode: IntroNode,       // ADD THIS LINE
};

// Define the sequence of nodes for all timesteps
const ALL_TIMESTEP_NODES = [
  'intro-rnn',
  // Timestep 1
  't1_input', 't1_h_prev', 't1_w_xh', 't1_w_hh', 't1_b_h', 't1_calc_h', 't1_w_hy', 't1_b_y', 't1_calc_y', 't1_pred',
  // Timestep 2  
  't2_input', 't2_h_prev', 't2_w_xh', 't2_w_hh', 't2_b_h', 't2_calc_h', 't2_w_hy', 't2_b_y', 't2_calc_y', 't2_pred',
  // Timestep 3
  't3_input', 't3_h_prev', 't3_w_xh', 't3_w_hh', 't3_b_h', 't3_calc_h', 't3_w_hy', 't3_b_y', 't3_calc_y', 't3_pred',
  // Timestep 4
  't4_input', 't4_h_prev', 't4_w_xh', 't4_w_hh', 't4_b_h', 't4_calc_h', 't4_w_hy', 't4_b_y', 't4_calc_y', 't4_pred',
];

function RNNWorkflowContent({ isDark, onToggleTheme }: { isDark: boolean; onToggleTheme: (isDark: boolean) => void }) {
  const [completedNodeIds, setCompletedNodeIds] = useState<Set<string>>(new Set());
  const [showIntroModal, setShowIntroModal] = useState(true); // State to control the intro modal
  const reactFlowInstance = useReactFlow(); // Get React Flow instance for fitting view

  const handleNodeComplete = useCallback((nodeId: string) => {
    setCompletedNodeIds(prev => new Set(prev).add(nodeId));
  }, []);

  const processedNodes = useMemo(() => {
    return rawInitialNodes.map(node => {
      // Handle the intro node separately
      if (node.id === 'intro-rnn') {
        // Intro node is only active if not yet completed and no other nodes are enabled yet
        const isIntroActive = !completedNodeIds.has('intro-rnn') && completedNodeIds.size === 0;
        return { 
          ...node, 
          data: { 
            ...node.data, 
            onComplete: handleNodeComplete, 
            disabled: !isIntroActive 
          } 
        };
      }

      // Logic for all timesteps
      let disabled = true;
      const isIntroCompleted = completedNodeIds.has('intro-rnn');
      
      // Timestep 1 logic
      const isT1InputCompleted = completedNodeIds.has('t1_input') && completedNodeIds.has('t1_h_prev');
      const isT1WeightsCompleted = completedNodeIds.has('t1_w_xh') && completedNodeIds.has('t1_w_hh') && completedNodeIds.has('t1_b_h');
      const isT1CalcHCompleted = completedNodeIds.has('t1_calc_h');
      const isT1OutputWeightsCompleted = completedNodeIds.has('t1_w_hy') && completedNodeIds.has('t1_b_y');
      const isT1CalcYCompleted = completedNodeIds.has('t1_calc_y');
      const isT1PredCompleted = completedNodeIds.has('t1_pred');
      
      // Timestep 2 logic
      const isT2InputCompleted = completedNodeIds.has('t2_input') && completedNodeIds.has('t2_h_prev');
      const isT2WeightsCompleted = completedNodeIds.has('t2_w_xh') && completedNodeIds.has('t2_w_hh') && completedNodeIds.has('t2_b_h');
      const isT2CalcHCompleted = completedNodeIds.has('t2_calc_h');
      const isT2OutputWeightsCompleted = completedNodeIds.has('t2_w_hy') && completedNodeIds.has('t2_b_y');
      const isT2CalcYCompleted = completedNodeIds.has('t2_calc_y');
      const isT2PredCompleted = completedNodeIds.has('t2_pred');
      
      // Timestep 3 logic
      const isT3InputCompleted = completedNodeIds.has('t3_input') && completedNodeIds.has('t3_h_prev');
      const isT3WeightsCompleted = completedNodeIds.has('t3_w_xh') && completedNodeIds.has('t3_w_hh') && completedNodeIds.has('t3_b_h');
      const isT3CalcHCompleted = completedNodeIds.has('t3_calc_h');
      const isT3OutputWeightsCompleted = completedNodeIds.has('t3_w_hy') && completedNodeIds.has('t3_b_y');
      const isT3CalcYCompleted = completedNodeIds.has('t3_calc_y');
      const isT3PredCompleted = completedNodeIds.has('t3_pred');
      
      // Timestep 4 logic
      const isT4InputCompleted = completedNodeIds.has('t4_input') && completedNodeIds.has('t4_h_prev');
      const isT4WeightsCompleted = completedNodeIds.has('t4_w_xh') && completedNodeIds.has('t4_w_hh') && completedNodeIds.has('t4_b_h');
      const isT4CalcHCompleted = completedNodeIds.has('t4_calc_h');
      const isT4OutputWeightsCompleted = completedNodeIds.has('t4_w_hy') && completedNodeIds.has('t4_b_y');
      const isT4CalcYCompleted = completedNodeIds.has('t4_calc_y');

      switch (node.id) {
        // Timestep 1
        case 't1_input':
        case 't1_h_prev':
        case 't1_w_xh':
        case 't1_w_hh':
        case 't1_b_h':
          disabled = !isIntroCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_calc_h':
          disabled = !isT1InputCompleted || !isT1WeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_w_hy':
        case 't1_b_y':
          disabled = !isT1CalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_calc_y':
          disabled = !isT1CalcHCompleted || !isT1OutputWeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't1_pred':
          disabled = !isT1CalcYCompleted || completedNodeIds.has(node.id);
          break;
          
        // Timestep 2 (enabled only after timestep 1 is complete)
        case 't2_input':
        case 't2_h_prev':
        case 't2_w_xh':
        case 't2_w_hh':
        case 't2_b_h':
          disabled = !isT1PredCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_calc_h':
          disabled = !isT2InputCompleted || !isT2WeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_w_hy':
        case 't2_b_y':
          disabled = !isT2CalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_calc_y':
          disabled = !isT2CalcHCompleted || !isT2OutputWeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't2_pred':
          disabled = !isT2CalcYCompleted || completedNodeIds.has(node.id);
          break;
          
        // Timestep 3 (enabled only after timestep 2 is complete)
        case 't3_input':
        case 't3_h_prev':
        case 't3_w_xh':
        case 't3_w_hh':
        case 't3_b_h':
          disabled = !isT2PredCompleted || completedNodeIds.has(node.id);
          break;
        case 't3_calc_h':
          disabled = !isT3InputCompleted || !isT3WeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't3_w_hy':
        case 't3_b_y':
          disabled = !isT3CalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't3_calc_y':
          disabled = !isT3CalcHCompleted || !isT3OutputWeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't3_pred':
          disabled = !isT3CalcYCompleted || completedNodeIds.has(node.id);
          break;
          
        // Timestep 4 (enabled only after timestep 3 is complete)
        case 't4_input':
        case 't4_h_prev':
        case 't4_w_xh':
        case 't4_w_hh':
        case 't4_b_h':
          disabled = !isT3PredCompleted || completedNodeIds.has(node.id);
          break;
        case 't4_calc_h':
          disabled = !isT4InputCompleted || !isT4WeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't4_w_hy':
        case 't4_b_y':
          disabled = !isT4CalcHCompleted || completedNodeIds.has(node.id);
          break;
        case 't4_calc_y':
          disabled = !isT4CalcHCompleted || !isT4OutputWeightsCompleted || completedNodeIds.has(node.id);
          break;
        case 't4_pred':
          disabled = !isT4CalcYCompleted || completedNodeIds.has(node.id);
          break;
          
        default:
          disabled = true; // All other future nodes are disabled for now
          break;
      }

      // Keep nodes disabled if they are already completed
      if (completedNodeIds.has(node.id)) {
        disabled = true;
      }
      
      return { ...node, data: { ...node.data, onComplete: handleNodeComplete, disabled: disabled } };
    });
  }, [completedNodeIds, handleNodeComplete]);

  const [nodes, setNodes, onNodesChange] = useNodesState(processedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

  // Reset function to clear all progress and re-show modal
  const resetWorkflow = () => {
    setCompletedNodeIds(new Set());
    setShowIntroModal(true);
    // Fit view to intro node after reset
    setTimeout(() => {
        reactFlowInstance.fitView({
            nodes: [{ id: 'intro-rnn' }],
            duration: 800,
            padding: 0.2
        });
    }, 100);
  };

  // Effect to fit view to the intro node on initial load
  useEffect(() => {
    if (showIntroModal) {
      reactFlowInstance.fitView({
        nodes: [{ id: 'intro-rnn' }],
        duration: 800,
        padding: 0.2
      });
    }
  }, [showIntroModal, reactFlowInstance]);

  // Effect to auto-navigate view as user progresses through timesteps
  useEffect(() => {
    const currentCompletedCount = completedNodeIds.size;
    
    // Navigate to different timesteps based on progress
    if (completedNodeIds.has('t1_pred') && !completedNodeIds.has('t2_input')) {
      // Just completed timestep 1, focus on timestep 2
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 't2_input' }, { id: 't2_h_prev' }, { id: 't2_w_xh' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    } else if (completedNodeIds.has('t2_pred') && !completedNodeIds.has('t3_input')) {
      // Just completed timestep 2, focus on timestep 3
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 't3_input' }, { id: 't3_h_prev' }, { id: 't3_w_xh' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    } else if (completedNodeIds.has('t3_pred') && !completedNodeIds.has('t4_input')) {
      // Just completed timestep 3, focus on timestep 4
      setTimeout(() => {
        reactFlowInstance.fitView({
          nodes: [{ id: 't4_input' }, { id: 't4_h_prev' }, { id: 't4_w_xh' }],
          duration: 1000,
          padding: 0.3
        });
      }, 500);
    }
  }, [completedNodeIds, reactFlowInstance]);

  return (
    <div className={`h-screen w-full flex flex-col transition-colors duration-300 relative overflow-hidden ${
       isDark ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-900'
    }`}>
      {/* Intro Modal */}
      <Dialog open={showIntroModal} onOpenChange={setShowIntroModal}>
        <DialogContent className="sm:max-w-[550px] bg-background/95 backdrop-blur-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-2xl">
              <Cpu className="text-purple-500" />
              Welcome to the RNN Time-Traveler Lab!
            </DialogTitle>
            <DialogDescription className="pt-2 text-base">
              You're about to explore the fundamental mechanism of Recurrent Neural Networks.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4 text-sm text-muted-foreground">
            <p>We'll process the sequence <span className="font-bold text-lg text-orange-400 font-mono">"R" → "N" → "N" → "!"</span> character by character. At each step, you'll calculate the network's internal <strong>Hidden State</strong> (its memory) and its <strong>Prediction</strong> for the next character.</p>
            <p>Notice how the <strong>same weight matrices</strong> appear at each timestep - this demonstrates the core RNN principle of <em>weight sharing</em> across time!</p>
            <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-lg">
              <p className="text-purple-400 font-medium">🎯 Your Mission:</p>
              <p className="text-sm">Complete all 4 timesteps to see how an RNN processes sequential data with shared weights and maintains memory across time!</p>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={() => setShowIntroModal(false)} className="w-full">
              Let's Start the Journey! 🚀
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Header */}
      <div className={`shrink-0 sticky top-4 mx-4 z-50 flex items-center justify-between backdrop-blur-md shadow-lg rounded-lg p-4 transition-colors duration-300 ${
        isDark ? 'bg-slate-800/70 border-slate-700/50' : 'bg-white/80 border-slate-300/60'
      }`}>
        <h1 className="text-xl font-bold bg-gradient-to-r from-amber-500 to-red-500 bg-clip-text text-transparent">
          RNN with Weight Sharing: Complete Forward Pass (R→N→N→!)
        </h1>
        <div className="flex items-center gap-4">
          <div className="text-sm text-muted-foreground">
            Progress: {completedNodeIds.size - 1}/{ALL_TIMESTEP_NODES.length - 1} nodes
          </div>
          <Button onClick={resetWorkflow} variant="outline" size="sm" className="flex items-center gap-2">
            <RotateCcw className="w-4 h-4" /> Reset
          </Button>
          <ThemeToggle isDark={isDark} onToggle={onToggleTheme} />
        </div>
      </div>

      <div className="flex-grow w-full h-full relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          nodeTypes={nodeTypes}
          fitView // Ensure fitView is enabled
        >
          <Background gap={32} size={1.5} color={isDark ? '#334155' : '#cbd5e1'} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  );
}

// Wrapper component to provide React Flow context
export function SelfAttentionWorkflow() {
  const [isDark, setIsDark] = useState(true);
  const handleThemeToggle = (newIsDark: boolean) => setIsDark(newIsDark);
  return (
    <ReactFlowProvider>
      <ThemeProvider isDark={isDark}>
        <RNNWorkflowContent isDark={isDark} onToggleTheme={handleThemeToggle} />
      </ThemeProvider>
    </ReactFlowProvider>
  );
}